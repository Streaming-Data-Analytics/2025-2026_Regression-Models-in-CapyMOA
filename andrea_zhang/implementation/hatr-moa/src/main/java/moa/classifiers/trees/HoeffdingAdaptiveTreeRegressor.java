package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.MultiChoiceOption;

import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Measurement;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.hatr.*;

import java.util.*;

/**
 * Hoeffding Adaptive Tree Regressor (HATR)
 *
 * A regression Hoeffding Tree with ADWIN-based concept drift detection.
 * At each internal node, an ADWIN detector monitors prediction error.
 * On drift, an alternate subtree grows in background; a z-test determines
 * when to swap the current subtree with the (better) alternate one.
 *
 * Reference: Bifet & Gavalda, "Adaptive Learning from Evolving Data Streams",
 *   IDA 2009. River's implementation was the primary algorithmic reference.
 */
public class HoeffdingAdaptiveTreeRegressor extends AbstractClassifier {

    // MOA Options 

    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g',
        "Number of instances a leaf should observe between split attempts.", 200, 1, Integer.MAX_VALUE);

    public FloatOption deltaOption = new FloatOption("delta", 'd',
        "Significance level for the Hoeffding bound (1 - delta = confidence).", 1e-7, 0.0, 1.0);

    public FloatOption tauOption = new FloatOption("tau", 't',
        "Threshold below which a split will be forced to break ties.", 0.05, 0.0, 1.0);

    public MultiChoiceOption leafPredictionOption = new MultiChoiceOption("leafPrediction", 'l',
        "Prediction mechanism at leaf nodes.",
        new String[]{"MEAN", "MODEL", "ADAPTIVE"},
        new String[]{"Target mean", "Online linear model (Perceptron)", "Adaptive (mean vs. model)"}, 2);

    public FloatOption modelSelectorDecayOption = new FloatOption("modelSelectorDecay", 'e',
        "EWMA decay for model selector in ADAPTIVE mode.", 0.95, 0.0, 1.0);

    public FlagOption bootstrapSamplingOption = new FlagOption("bootstrapSampling", 'b',
        "Enable Poisson(1) bootstrap sampling in leaf updates.");

    public IntOption driftWindowThresholdOption = new IntOption("driftWindowThreshold", 'w',
        "Min examples alternate tree must observe before z-test comparison.", 300, 1, Integer.MAX_VALUE);

    public FloatOption switchSignificanceOption = new FloatOption("switchSignificance", 's',
        "p-value threshold for the z-test swap decision.", 0.05, 0.0, 1.0);

    public IntOption minSamplesSplitOption = new IntOption("minSamplesSplit", 'm',
        "Minimum samples per branch for a split candidate to be valid.", 5, 1, Integer.MAX_VALUE);

    public IntOption maxDepthOption = new IntOption("maxDepth", 'D',
        "Maximum tree depth (-1 = unlimited).", -1, -1, Integer.MAX_VALUE);

    // NOTE: char 'r' is reserved by MOA's AbstractClassifier for randomSeed; use 'z' here
    // to avoid an option-flag collision (which would break CLI parsing / CapyMOA wrappers).
    public IntOption tebstDigitsOption = new IntOption("tebstDigits", 'z',
        "Rounding digits for Truncated E-BST (numeric attribute observer).", 1, 0, 10);

    public FlagOption meritPrePruneOption = new FlagOption("meritPrePrune", 'p',
        "Enable merit-based pre-pruning (null split option).");

    public FloatOption adwinDeltaOption = new FloatOption("adwinDelta", 'A',
        "ADWIN delta parameter for drift detection.", 0.002, 0.0, 1.0);

    public FloatOption perceptronLROption = new FloatOption("perceptronLR", 'L',
        "Learning rate for the Perceptron leaf model.", 0.01, 0.0, 1.0);

    // Internal state (package-accessible for node classes) 

    public HANode root;
    public int nActiveLeaves;
    public int nInactiveLeaves;
    public int nAlternateTrees;
    public int nPrunedAlternateTrees;
    public int nSwitchAlternateTrees;

    // Resolved options (cached for efficiency — avoids Option.getValue() per sample)
    public int gracePeriod;
    public double delta;
    public double tau;
    public int leafPredMode;   // 0=MEAN, 1=MODEL, 2=ADAPTIVE
    public double modelSelectorDecay;
    public boolean bootstrapSampling;
    public int driftWindowThreshold;
    public double switchSignificance;
    public int minSamplesSplit;
    public int maxDepth;
    public boolean meritPreprune;
    public double adwinDelta;
    public double perceptronLR;
    public List<Integer> nominalAttributeIndices; // null = infer from Instance

    /** Prototype drift detector — cloned for each new node. */
    public ADWINDetector driftDetectorProto;

    /** Prototype attribute observer — cloned for each new leaf's numeric observers. */
    public HAAttributeObserver numericObserverProto;

    private Random rng;

    // AbstractClassifier lifecycle 

    @Override
    public void resetLearningImpl() {
        // Resolve options into fields
        gracePeriod = gracePeriodOption.getValue();
        delta = deltaOption.getValue();
        tau = tauOption.getValue();
        leafPredMode = leafPredictionOption.getChosenIndex();
        modelSelectorDecay = modelSelectorDecayOption.getValue();
        bootstrapSampling = bootstrapSamplingOption.isSet();
        driftWindowThreshold = driftWindowThresholdOption.getValue();
        switchSignificance = switchSignificanceOption.getValue();
        minSamplesSplit = minSamplesSplitOption.getValue();
        maxDepth = maxDepthOption.getValue() < 0 ? Integer.MAX_VALUE : maxDepthOption.getValue();
        meritPreprune = meritPrePruneOption.isSet();
        adwinDelta = adwinDeltaOption.getValue();
        perceptronLR = perceptronLROption.getValue();

        driftDetectorProto = new ADWINDetector(adwinDelta, 32, 5, 5, 10);
        numericObserverProto = new HATEBSTObserver(tebstDigitsOption.getValue());

        root = null;
        nActiveLeaves = 0;
        nInactiveLeaves = 0;
        nAlternateTrees = 0;
        nPrunedAlternateTrees = 0;
        nSwitchAlternateTrees = 0;

        rng = new Random(this.randomSeedOption.getValue());
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (root == null) {
            root = newLeaf(null, 0);
            nActiveLeaves = 1;
        }

        if (root instanceof AdaLeafNode) {
            ((AdaLeafNode) root).adaLearnOne(inst, this, null, -1);
        } else if (root instanceof AdaBranchNode) {
            ((AdaBranchNode) root).adaLearnOne(inst, this, null, -1);
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (root == null) return new double[]{0.0};

        List<HALeafNode> leaves;
        if (root instanceof AdaBranchNode) {
            leaves = ((AdaBranchNode) root).traverseWithAlternate(inst);
        } else if (root instanceof HABranchNode) {
            HALeafNode leaf = ((HABranchNode) root).traverseToLeaf(inst);
            leaves = leaf != null ? Collections.singletonList(leaf) : Collections.emptyList();
        } else {
            leaves = Collections.singletonList((HALeafNode) root);
        }

        if (leaves.isEmpty()) return new double[]{0.0};
        double pred = 0;
        for (HALeafNode l : leaves) pred += l.getPrediction(inst);
        return new double[]{pred / leaves.size()};
    }

    @Override
    public boolean isRandomizable() { return true; }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
            new Measurement("tree nodes", root != null ? root.getNNodes() : 0),
            new Measurement("tree leaves", root != null ? root.getNLeaves() : 0),
            new Measurement("active leaves", nActiveLeaves),
            new Measurement("inactive leaves", nInactiveLeaves),
            new Measurement("alternate trees", nAlternateTrees),
            new Measurement("pruned alt trees", nPrunedAlternateTrees),
            new Measurement("switched alt trees", nSwitchAlternateTrees),
        };
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        String pad = " ".repeat(indent);
        out.append(pad).append("HoeffdingAdaptiveTreeRegressor\n");
        out.append(pad).append(" leafPrediction=").append(leafPredictionOption.getChosenLabel()).append("\n");
        out.append(pad).append(" gracePeriod=").append(gracePeriod).append("\n");
        out.append(pad).append(" delta=").append(delta).append("\n");
        out.append(pad).append(" nodes=").append(root != null ? root.getNNodes() : 0).append("\n");
        out.append(pad).append(" alternateTrees=").append(nAlternateTrees).append("\n");
    }

    // ── Internal factory methods (called by node classes) ────────────────────

    /**
     * Create a new adaptive leaf at the given depth.
     * Inherits fmse values from parent if parent is AdaLeafAdaptive.
     */
    public HALeafNode newLeaf(HALeafNode parent, int depth) {
        ADWINDetector det = driftDetectorProto.createNew();

        // River: a child inherits a deep copy of the parent leaf's model; a leaf
        // with no model parent (root, or alternate-tree leaf grown from a branch)
        // starts from a fresh model prototype. lr and intercept_lr both default to
        // perceptronLR (River's SGD lr and intercept_lr defaults are both 0.01).
        HAPerceptron model = null;
        if (leafPredMode > 0) {
            if (parent instanceof AdaLeafModel) {
                model = ((AdaLeafModel) parent).model.copy();
            } else if (parent instanceof AdaLeafAdaptive) {
                model = ((AdaLeafAdaptive) parent).model.copy();
            } else {
                model = new HAPerceptron(64, perceptronLR, perceptronLR, 0.0);
            }
        }

        switch (leafPredMode) {
            case 0: // MEAN
                return new AdaLeafMean(null, depth, numericObserverProto.createNew(), det, rng);
            case 1: // MODEL
                return new AdaLeafModel(null, depth, numericObserverProto.createNew(), det, rng, model);
            default: { // ADAPTIVE
                AdaLeafAdaptive leaf = new AdaLeafAdaptive(null, depth,
                    numericObserverProto.createNew(), det, rng, model);
                if (parent instanceof AdaLeafAdaptive) {
                    leaf.fmseMean  = ((AdaLeafAdaptive) parent).fmseMean;
                    leaf.fmseModel = ((AdaLeafAdaptive) parent).fmseModel;
                }
                return leaf;
            }
        }
    }

    /**
     * Attempt to split a leaf using the Hoeffding bound criterion.
     * Creates an adaptive branch node if the split is warranted.
     * Called by AdaLeafNode.adaLearnOne after grace period.
     */
    public void attemptToSplit(HALeafNode leaf, HABranchNode parent, int parentBranch, ADWINDetector branchDriftDet) {
        List<SplitCandidate> candidates = leaf.bestSplitCandidates(this);
        Collections.sort(candidates);

        if (candidates.isEmpty()) return;

        boolean shouldSplit;
        double hoeffdingBound = 0;
        SplitCandidate best = candidates.get(candidates.size() - 1);

        if (candidates.size() < 2) {
            shouldSplit = !best.isNullSplit();
        } else {
            SplitCandidate secondBest = candidates.get(candidates.size() - 2);
            double n = leaf.getTotalWeight();
            hoeffdingBound = Math.sqrt((Math.log(1.0 / delta)) / (2.0 * n));

            shouldSplit = best.merit > 0.0 && (
                (secondBest.merit / best.merit) < (1 - hoeffdingBound)
                || hoeffdingBound < tau
            );

            // Remove poor attributes
            if (shouldSplit) {
                double bestRatio = secondBest.merit / best.merit;
                for (SplitCandidate c : candidates) {
                    if (!c.isNullSplit() && c.merit / best.merit < bestRatio - 2 * hoeffdingBound) {
                        leaf.disableAttribute(c.attIndex);
                    }
                }
            }
        }

        if (!shouldSplit) {
            // Manage EBST memory
            if (candidates.size() >= 2) {
                SplitCandidate secondBest = candidates.get(candidates.size() - 2);
                if (best.merit > 0 && secondBest.merit > 0) {
                    pruneLeafObservers(leaf, best.merit, secondBest.merit / best.merit, hoeffdingBound);
                }
            }
            return;
        }

        if (best.isNullSplit()) {
            // Pre-pruning
            leaf.deactivate();
            nInactiveLeaves++; nActiveLeaves--;
            return;
        }

        // Create child leaves from split statistics
        List<VarStats> childStats = best.childrenStats;
        HALeafNode[] childLeaves = new HALeafNode[childStats.size()];
        for (int i = 0; i < childStats.size(); i++) {
            childLeaves[i] = newLeaf(leaf, leaf.depth + 1);
            childLeaves[i].stats = childStats.get(i).copy();
            // River sets last_split_attempt_at = total_weight at leaf construction
            // (HTLeaf.__init__). Here stats are assigned after construction, so we must
            // re-sync it; otherwise children keep last_split_attempt_at = 0 and re-attempt
            // splits far too early (causing systematic over-splitting vs River).
            childLeaves[i].lastSplitAttemptAt = childLeaves[i].getTotalWeight();
        }

        // Create the adaptive branch node
        AdaBranchNode branch;
        if (best.isNumeric) {
            branch = new AdaNumBinaryBranch(leaf.stats, best.attIndex, best.numericThreshold,
                leaf.depth, childLeaves[0], childLeaves[1], branchDriftDet);
        } else {
            branch = new AdaNomBinaryBranch(leaf.stats, best.attIndex, best.nominalValue,
                leaf.depth, childLeaves[0], childLeaves[1], branchDriftDet);
        }

        nActiveLeaves--;
        nActiveLeaves += childLeaves.length;

        if (parent == null) {
            root = branch;
        } else {
            parent.children.set(parentBranch, branch);
        }
    }

    private void pruneLeafObservers(HALeafNode leaf, double bestVr, double lastRatio, double lastE) {
        if (leaf.observers == null) return;
        for (HAAttributeObserver obs : leaf.observers.values()) {
            if (obs instanceof HAEBSTObserver) {
                ((HAEBSTObserver) obs).pruneBadSplits(leaf.stats, lastRatio, bestVr, lastE, minSamplesSplit);
            }
        }
    }

    // CapyMOA compatibility 

    /** Convenience: set nominalAttributeIndices from a list. */
    public void setNominalAttributeIndices(List<Integer> indices) {
        this.nominalAttributeIndices = indices;
    }

    @Override
    public String getPurposeString() {
        return "Hoeffding Adaptive Tree Regressor (HATR): online regression tree with "
            + "ADWIN concept drift detection and alternate subtree management.";
    }
}
