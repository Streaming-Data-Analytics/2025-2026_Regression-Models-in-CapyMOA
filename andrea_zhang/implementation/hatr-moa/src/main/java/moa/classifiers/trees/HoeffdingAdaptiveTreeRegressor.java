package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.MultiChoiceOption;

import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Measurement;
import moa.core.SizeOf;
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
    // NOTE: char 'r' is reserved by MOA's AbstractClassifier for randomSeed

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

    public IntOption tebstDigitsOption = new IntOption("tebstDigits", 'z',
        "Rounding digits for Truncated E-BST (numeric attribute observer).", 1, 0, 10);

    public FlagOption meritPrePruneOption = new FlagOption("meritPrePrune", 'p',
        "Enable merit-based pre-pruning (null split option).");

    public FloatOption adwinDeltaOption = new FloatOption("adwinDelta", 'A',
        "ADWIN delta parameter for drift detection.", 0.002, 0.0, 1.0);

    public FloatOption maxSizeMBOption = new FloatOption("maxSizeMB", 'M',
        "Maximum memory consumed by the tree (MB). Requires SizeOf agent; silently disabled otherwise.",
        500.0, 0.0, Float.MAX_VALUE);

    public IntOption memoryEstimatePeriodOption = new IntOption("memoryEstimatePeriod", 'E',
        "Number of instances between memory consumption checks.", 1000000, 1, Integer.MAX_VALUE);

    public FlagOption stopMemManagementOption = new FlagOption("stopMemManagement", 'S',
        "Stop tree growth (rather than deactivate leaves) when memory limit is hit.");

    public FlagOption removePoorAttrsOption = new FlagOption("removePoorAttrs", 'R',
        "Disable poor attributes to save memory.");

    public HANode root;
    public int nActiveLeaves;
    public int nInactiveLeaves;
    public int nAlternateTrees;
    public int nPrunedAlternateTrees;
    public int nSwitchAlternateTrees;

    // Resolved options (cached to avoid Option.getValue() overhead per sample)
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
    public boolean removePoorAttrs;
    public double adwinDelta;
    public ADWINDetector driftDetectorProto;

    public HAAttributeObserver numericObserverProto;

    public long maxByteSize;
    public int memoryEstimatePeriod;
    public boolean stopMemManagement;
    public boolean growthAllowed;
    public double trainWeightSeenByModel;
    public double activeLeafByteSizeEstimate;
    public double inactiveLeafByteSizeEstimate;
    public double byteSizeEstimateOverheadFraction;

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
        removePoorAttrs = removePoorAttrsOption.isSet();
        adwinDelta = adwinDeltaOption.getValue();
        driftDetectorProto = new ADWINDetector(adwinDelta, 32, 5, 5, 10);
        numericObserverProto = new HATEBSTObserver(tebstDigitsOption.getValue());

        maxByteSize = (long) (maxSizeMBOption.getValue() * 1024 * 1024);
        memoryEstimatePeriod = memoryEstimatePeriodOption.getValue();
        stopMemManagement = stopMemManagementOption.isSet();
        growthAllowed = true;
        trainWeightSeenByModel = 0;
        activeLeafByteSizeEstimate = 0;
        inactiveLeafByteSizeEstimate = 0;
        byteSizeEstimateOverheadFraction = 1.0;

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

        trainWeightSeenByModel += inst.weight();
        if (trainWeightSeenByModel % memoryEstimatePeriod == 0) {
            estimateModelByteSizes();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (root == null) return new double[]{0.0};

        List<HALeafNode> leaves;
        if (root instanceof AdaBranchNode) {
            leaves = ((AdaBranchNode) root).traverseWithAlternate(inst);
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

    // Internal factory methods (called by node classes)

    public HALeafNode newLeaf(HALeafNode parent, int depth) {
        ADWINDetector det = driftDetectorProto.createNew();

        HAPerceptron model = null;
        if (leafPredMode > 0) {
            if (parent instanceof AdaLeafModel) {
                model = ((AdaLeafModel) parent).model.copy();
            } else if (parent instanceof AdaLeafAdaptive) {
                model = ((AdaLeafAdaptive) parent).model.copy();
            } else {
                model = new HAPerceptron(64, 0.01, 0.01, 0.0);
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

    /** Hoeffding bound split check; replaces leaf with an adaptive branch if split is justified. */
    public void attemptToSplit(HALeafNode leaf, HABranchNode parent, int parentBranch, ADWINDetector branchDriftDet) {
        List<SplitCandidate> candidates = leaf.bestSplitCandidates(this);
        Collections.sort(candidates);

        if (candidates.isEmpty()) return;

        boolean shouldSplit;
        double hoeffdingBound = 0;
        SplitCandidate best = candidates.get(candidates.size() - 1);

        if (candidates.size() < 2) {
            shouldSplit = true;
        } else {
            SplitCandidate secondBest = candidates.get(candidates.size() - 2);
            double n = leaf.getTotalWeight();
            hoeffdingBound = Math.sqrt((Math.log(1.0 / delta)) / (2.0 * n));

            shouldSplit = best.merit > 0.0 && (
                (secondBest.merit / best.merit) < (1 - hoeffdingBound)
                || hoeffdingBound < tau
            );

            // Remove poor attributes independently of should_split
            if (removePoorAttrs) {
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
            childLeaves[i].lastSplitAttemptAt = childLeaves[i].getTotalWeight();
        }

        // Create the adaptive branch node
        AdaBranchNode branch;
        if (best.isNumeric) {
            branch = new AdaNumBinaryBranch(leaf.stats, best.attIndex, best.numericThreshold,
                leaf.depth, childLeaves[0], childLeaves[1], branchDriftDet);
        } else if (best.isMultiway()) {
            branch = new AdaNomMultiwayBranch(leaf.stats, best.attIndex, best.nominalValues,
                leaf.depth, branchDriftDet, childLeaves);
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

    /** Estimates memory footprint; calls enforceTrackerLimit() if over budget. No-op without SizeOf agent. */
    public void estimateModelByteSizes() {
        if (root == null) return;
        List<HALeafNode> leaves = root.iterLeaves();
        long totalActiveSize = 0, totalInactiveSize = 0;
        for (HALeafNode leaf : leaves) {
            if (leaf.isActive()) 
                totalActiveSize += SizeOf.fullSizeOf(leaf);
            else                 
                totalInactiveSize += SizeOf.fullSizeOf(leaf);
        }
        if (totalActiveSize > 0 && nActiveLeaves > 0)
            activeLeafByteSizeEstimate = (double) totalActiveSize / nActiveLeaves;
        if (totalInactiveSize > 0 && nInactiveLeaves > 0)
            inactiveLeafByteSizeEstimate = (double) totalInactiveSize / nInactiveLeaves;
        long actualModelSize = SizeOf.fullSizeOf(this);
        double estimatedModelSize = nActiveLeaves * activeLeafByteSizeEstimate + nInactiveLeaves * inactiveLeafByteSizeEstimate;
        if (estimatedModelSize > 0)
            byteSizeEstimateOverheadFraction = (double) actualModelSize / estimatedModelSize;
        if (actualModelSize > maxByteSize)
            enforceTrackerLimit();
    }

    /** Deactivates least-promising leaves (deepest first) until the tree fits in maxByteSize. */
    public void enforceTrackerLimit() {
        if (nInactiveLeaves > 0 ||
                (nActiveLeaves * activeLeafByteSizeEstimate
                 + nInactiveLeaves * inactiveLeafByteSizeEstimate)
                 * byteSizeEstimateOverheadFraction > maxByteSize) {
            if (stopMemManagement) {
                growthAllowed = false;
                return;
            }
        }
        List<HALeafNode> leaves = root.iterLeaves();
        leaves.sort(Comparator.comparingInt(HALeafNode::calculatePromise));
        int maxActive = 0;
        while (maxActive < leaves.size()) {
            maxActive++;
            if ((maxActive * activeLeafByteSizeEstimate
                 + (leaves.size() - maxActive) * inactiveLeafByteSizeEstimate)
                 * byteSizeEstimateOverheadFraction > maxByteSize) {
                maxActive--;
                break;
            }
        }
        int cutoff = leaves.size() - maxActive;
        for (int i = 0; i < cutoff; i++) {
            if (leaves.get(i).isActive()) {
                leaves.get(i).deactivate();
                nActiveLeaves--; nInactiveLeaves++;
            }
        }
        for (int i = cutoff; i < leaves.size(); i++) {
            if (!leaves.get(i).isActive()) {
                leaves.get(i).activate();
                nActiveLeaves++; nInactiveLeaves--;
            }
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

    @Override
    public String getPurposeString() {
        return "Hoeffding Adaptive Tree Regressor (HATR): online regression tree with "
            + "ADWIN concept drift detection and alternate subtree management.";
    }
}
