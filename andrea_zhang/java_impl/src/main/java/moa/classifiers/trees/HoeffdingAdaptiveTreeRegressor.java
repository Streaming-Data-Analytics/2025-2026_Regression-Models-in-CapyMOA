package moa.classifiers.trees;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;
import moa.core.SizeOf;

import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class HoeffdingAdaptiveTreeRegressor extends AbstractClassifier implements Regressor {

    private static final long serialVersionUID = 1L;

    //region === OPTIONS (MOA CLI / CapyMOA configuration) ===

    public IntOption gracePeriodOption = new IntOption(
            "gracePeriod", 'g',
            "Number of instances a leaf should observe between split attempts.",
            200, 1, Integer.MAX_VALUE);

    public IntOption maxDepthOption = new IntOption(
        "maxDepth", 'x',
        "Maximum tree depth. 0 means unlimited.",
        0, 0, Integer.MAX_VALUE);

    public FloatOption deltaOption = new FloatOption(
            "delta", 'c',
            "Significance level for the Hoeffding bound (1 - delta = confidence).",
            1e-7, 0.0, 1.0);

    public FloatOption tauOption = new FloatOption(
            "tau", 't',
            "Threshold below which a split will be forced to break ties.",
            0.05, 0.0, 1.0);

    public MultiChoiceOption leafPredictionOption = new MultiChoiceOption(
            "leafPrediction", 'l',
            "Prediction strategy used at leaves.",
            new String[]{"MEAN", "MODEL", "ADAPTIVE"},
            new String[]{
                "Target mean",
                "Linear regression model",
                "Adaptive: chooses between MEAN and MODEL via FMSE tracking"
            }, 2); // default: ADAPTIVE

    public FloatOption modelSelectorDecayOption = new FloatOption(
            "modelSelectorDecay", 'q',
            "Exponential decay factor for FMSE tracking in ADAPTIVE leaf mode.",
            0.95, 0.0, 1.0);
    
    public StringOption nominalAttributesOption = new StringOption(
            "nominalAttributes", 'n',
            "Comma-separated list of 0-based attribute indices to treat as nominal "
            + "(River: nominal_attributes=[...]). Empty string means use schema type only.",
            "");

    public IntOption minSamplesSplitOption = new IntOption(
            "minSamplesSplit", 'm',
            "Minimum number of samples each branch resulting from a split must have.",
            5, 1, Integer.MAX_VALUE);

    public FlagOption noBootstrapSamplingOption = new FlagOption(
            "noBootstrapSampling", 'b',
            "Disable Poisson bootstrap sampling at leaves (River default: enabled).");

    public IntOption driftWindowThresholdOption = new IntOption(
            "driftWindowThreshold", 'w',
            "Minimum examples an alternate tree must observe before being considered for replacement.",
            300, 1, Integer.MAX_VALUE);

    public FloatOption switchSignificanceOption = new FloatOption(
            "switchSignificance", 'z',
            "Significance level (p-value threshold) for the z-test when swapping alternate trees.",
            0.05, 0.0, 1.0);

    public FlagOption binarySplitOption = new FlagOption(
        "binarySplit", 'B',
        "Force binary splits for nominal features (River: binary_split=True). "
        + "Uses optimal binary partition (sort by mean, try all K-1 consecutive splits).");

    public FloatOption maxSizeMiBOption = new FloatOption(
            "maxSize", 'M',
            "Maximum tree size in mebibytes (MiB).",
            500.0, 1.0, Double.MAX_VALUE);

    public IntOption memoryEstimatePeriodOption = new IntOption(
            "memoryEstimatePeriod", 'e',
            "Number of instances between memory size checks.",
            1000000, 1, Integer.MAX_VALUE);

    public FlagOption stopMemManagementOption = new FlagOption(
            "stopMemManagement", 's',
            "Stop growing the tree when the memory limit is reached.");

    public FlagOption removePoorAttrsOption = new FlagOption(
            "removePoorAttrs", 'p',
            "Disable attributes whose split merit is significantly worse than the best candidate.");

    public FlagOption noPrePruneOption = new FlagOption(
            "noPrePrune", 'u',
            "Disable merit-based pre-pruning (River default: enabled). "
            + "When set, leaves are never deactivated because no attribute improves variance.");

    public FloatOption learningRateOption = new FloatOption(
            "learningRate", 'r',
            "Learning rate for the SGD linear model at leaves.",
            0.01, 0.0, 1.0);

    public FloatOption l2Option = new FloatOption(
            "l2", '2',
            "L2 regularization strength for the SGD linear model (River default: 0.0). Mutually exclusive with l1.",
            0.0, 0.0, Double.MAX_VALUE);

    public FloatOption l1Option = new FloatOption(
            "l1", '1',
            "L1 regularization strength for the SGD linear model (River default: 0.0). Mutually exclusive with l2.",
            0.0, 0.0, Double.MAX_VALUE);

    public IntOption tebstDigitsOption = new IntOption(
            "tebstDigits", 'k',
            "Number of decimal digits for TEBST rounding (River default: 1).",
            1, 0, 10);

    public FlagOption numericalMultiwayOption = new FlagOption(
        "numericalMultiway", 'N',
        "Enable multiway splits on numerical features using radius-based binning "
        + "(slot = floor(x / radius)). Uses RadiusSplitter instead of TEBSTSplitter. "
        + "Scale features before use; mirrors River QOSplitter(allow_multiway_splits=True).");

    public FloatOption numericalMultiwayRadiusOption = new FloatOption(
        "numericalMultiwayRadius", 'R',
        "Bin width for numerical multiway splits. Values mapped to slot = floor(x / radius). "
        + "River QOSplitter default: 0.25.",
        0.25, Double.MIN_VALUE, Double.MAX_VALUE);

    public FloatOption adwinDeltaOption = new FloatOption(
            "adwinDelta", 'a',
            "Delta parameter for all ADWIN drift detectors.",
            0.002, 0.0, 1.0);


    //endregion === OPTIONS (MOA CLI / CapyMOA configuration) ===

    protected Node root;
    protected double trainWeightSeen = 0;
    protected int nActiveLeaves = 0;
    protected int nInactiveLeaves = 0;
    protected boolean growthAllowed = true;
    private double sizeEstimateOverhead = 1.0;
    private double activeLeafSizeEstimate = 0.0;
    private double inactiveLeafSizeEstimate = 0.0;

    // Global feature statistics for LinearModel normalization (mirrors FIMTDD.sumOfAttrValues/Squares).
    protected double[] sumOfAttrValues;
    protected double[] sumOfAttrSqValues;

    protected int nAlternateTrees = 0;
    protected int nSwitchAlternateTrees = 0;
    protected int nPrunedAlternateTrees = 0;

    //region === Convenience fields read from options at reset time ===
    protected int gracePeriod;
    protected double delta;
    protected double tau;
    protected int minSamplesSplit;
    protected int driftWindowThreshold;
    protected double switchSignificance;
    protected LeafPrediction leafPrediction;
    protected double learningRate;
    protected double l2;
    protected double l1;
    protected double modelSelectorDecay;
    protected boolean binarySplit;
    protected boolean bootstrapSampling;
    protected int tebstDigits;
    protected boolean numericalMultiway;
    protected double numericalMultiwayRadius;
    protected int maxDepth;
    protected boolean removePoorAttrs;
    protected boolean stopMemManagement;
    protected double maxSizeMiB;
    protected int memoryEstimatePeriod;
    protected double adwinDelta;
    protected boolean meritPreprune;
    protected Set<Integer> nominalAttributeIndices;
    //endregion === Convenience fields read from options at reset time ===

    public enum LeafPrediction { MEAN, MODEL, ADAPTIVE }

    private static final NormalDistribution NORM = new NormalDistribution();

    //region === AbstractClassifier INTERFACE ===

    @Override
    public String getPurposeString() {
        return "Hoeffding Adaptive Tree Regressor: Java port of River's HATR."
             + "Uses ADWIN at each node for drift detection and grows alternate subtrees in background.";
    }

    @Override
    public boolean isRandomizable() { return true; }

    @Override
    public void resetLearningImpl() {
        gracePeriod = gracePeriodOption.getValue();
        delta = deltaOption.getValue();
        tau = tauOption.getValue();
        minSamplesSplit = minSamplesSplitOption.getValue();
        driftWindowThreshold = driftWindowThresholdOption.getValue();
        switchSignificance = switchSignificanceOption.getValue();
        learningRate = learningRateOption.getValue();
        l2 = l2Option.getValue();
        l1 = l1Option.getValue();
        if (l1 > 0 && l2 > 0)
            throw new IllegalArgumentException("l1 and l2 regularization cannot both be non-zero (River behaviour).");
        modelSelectorDecay = modelSelectorDecayOption.getValue();
        binarySplit = binarySplitOption.isSet();
        bootstrapSampling = !noBootstrapSamplingOption.isSet();
        tebstDigits = tebstDigitsOption.getValue();
        numericalMultiway = numericalMultiwayOption.isSet();
        numericalMultiwayRadius = numericalMultiwayRadiusOption.getValue();
        maxDepth = maxDepthOption.getValue() == 0 ? Integer.MAX_VALUE : maxDepthOption.getValue();
        removePoorAttrs = removePoorAttrsOption.isSet();
        stopMemManagement = stopMemManagementOption.isSet();
        maxSizeMiB = maxSizeMiBOption.getValue();
        memoryEstimatePeriod = memoryEstimatePeriodOption.getValue();
        adwinDelta = adwinDeltaOption.getValue();
        meritPreprune = !noPrePruneOption.isSet();

        String nomStr = nominalAttributesOption.getValue().trim();
        if (nomStr.isEmpty()) {
            nominalAttributeIndices = null;
        } else {
            nominalAttributeIndices = new HashSet<>();
            for (String token : nomStr.split(",")) {
                String t = token.trim();
                if (!t.isEmpty())
                    nominalAttributeIndices.add(Integer.parseInt(t));
            }
        }

        switch (leafPredictionOption.getChosenIndex()) {
            case 0: leafPrediction = LeafPrediction.MEAN;  break;
            case 1: leafPrediction = LeafPrediction.MODEL; break;
            default: leafPrediction = LeafPrediction.ADAPTIVE;
        }

        root = null;
        trainWeightSeen = 0;
        nActiveLeaves = 0;
        nInactiveLeaves = 0;
        growthAllowed = true;
        sizeEstimateOverhead = 1.0;
        activeLeafSizeEstimate = 0.0;
        inactiveLeafSizeEstimate = 0.0;
        nAlternateTrees = 0;
        nSwitchAlternateTrees = 0;
        nPrunedAlternateTrees = 0;
        sumOfAttrValues = null;
        sumOfAttrSqValues = null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("HoeffdingAdaptiveTreeRegressor\n");
        out.append("active leaves: ").append(nActiveLeaves).append("\n");
        out.append("inactive leaves: ").append(nInactiveLeaves).append("\n");
        out.append("instances seen: ").append((long) trainWeightSeen).append("\n");
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
            new Measurement("active leaves", nActiveLeaves),
            new Measurement("inactive leaves", nInactiveLeaves),
            new Measurement("alternate trees", nAlternateTrees),
            new Measurement("switch alternate trees", nSwitchAlternateTrees),
            new Measurement("pruned alternate trees", nPrunedAlternateTrees),
        };
    }

    // TRAINING AND PREDICTION

    // Mirrors River's StandardScaler.transform_one: z-score normalization (no factor 3).
    // Applied upstream so all components (splits, ADWIN, leaf models) see normalized values.
    public Instance normalizeInstance(Instance inst) {
        if (sumOfAttrValues == null || trainWeightSeen < 2) return inst;
        Instance copy = inst.copy();
        int nAttrs = inst.numAttributes() - 1;
        for (int i = 0; i < nAttrs; i++) {
            if (inst.attribute(i).isNominal() || inst.isMissing(i)) continue;
            double mean = sumOfAttrValues[i] / trainWeightSeen;
            double variance = sumOfAttrSqValues[i] / trainWeightSeen - mean * mean;
            if (variance <= 0) { copy.setValue(i, 0.0); continue; }
            double sd = Math.sqrt(variance);
            copy.setValue(i, sd < 1e-10 ? 0.0 : (inst.value(i) - mean) / sd);
        }
        return copy;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainWeightSeen += inst.weight();

        // Update global feature statistics before learning (mirrors FIMTDD).
        int nAttrs = inst.numAttributes() - 1;
        if (sumOfAttrValues == null) {
            sumOfAttrValues   = new double[nAttrs];
            sumOfAttrSqValues = new double[nAttrs];
        }
        double w = inst.weight();
        for (int i = 0; i < nAttrs; i++) {
            if (!inst.attribute(i).isNominal() && !inst.isMissing(i)) {
                double v = inst.value(i);
                sumOfAttrValues[i]   += v * w;
                sumOfAttrSqValues[i] += v * v * w;
            }
        }

        if (root == null) {
            root = newLeaf(0);
            nActiveLeaves = 1;
        }
        root.learn(normalizeInstance(inst), this, null, -1);

        if (trainWeightSeen % memoryEstimatePeriod == 0) {
            estimateModelSize();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (root == null)
            return new double[]{0.0};
        Instance normInst = normalizeInstance(inst);
        List<Node> leaves = new ArrayList<>();
        root.collectLeaves(normInst, leaves);
        if (leaves.isEmpty())
            return new double[]{0.0};
        double sum = 0;
        for (Node leaf : leaves)
            sum += leaf.predict(normInst, this);
        return new double[]{sum / leaves.size()};
    }

    //endregion === AbstractClassifier INTERFACE ===

    //region === MEMORY MANAGEMENT ===

    private void estimateModelSize() {
        if (root == null) 
            return;
        List<LeafNode> all = new ArrayList<>();
        root.collectAllLeaves(all);
        if (all.isEmpty()) 
            return;

        long activeTotal = 0; 
        int activeCount = 0;
        long inactiveTotal = 0; 
        int inactiveCount = 0;
        for (LeafNode l : all) {
            long sz = l.estimateByteSize();
            if (l.isActive()) { 
                activeTotal += sz; 
                activeCount++; 
            }
            else { 
                inactiveTotal += sz; 
                inactiveCount++; 
            }
        }
        if (activeCount > 0) 
            activeLeafSizeEstimate = (double) activeTotal / activeCount;
        if (inactiveCount > 0) 
            inactiveLeafSizeEstimate = (double) inactiveTotal / inactiveCount;

        double actualBytes = SizeOf.fullSizeOf(this);
        double estimateBytes = nActiveLeaves * activeLeafSizeEstimate + nInactiveLeaves * inactiveLeafSizeEstimate;
        if (estimateBytes > 0) 
            sizeEstimateOverhead = actualBytes / estimateBytes;
        if (actualBytes > maxSizeMiB * 1024.0 * 1024.0)
            enforceTreeSizeLimit();
    }

    private void enforceTreeSizeLimit() {
        double maxBytes = maxSizeMiB * 1024.0 * 1024.0;

        // Mirrors River: enter only when inactive leaves exist or size is exceeded
        double treeSize = sizeEstimateOverhead * (nActiveLeaves * activeLeafSizeEstimate + nInactiveLeaves * inactiveLeafSizeEstimate);
        if (nInactiveLeaves == 0 && treeSize <= maxBytes) 
            return;

        if (stopMemManagement) {
            growthAllowed = false;
            return;
        }

        List<LeafNode> leaves = new ArrayList<>();
        if (root != null) 
            root.collectAllLeaves(leaves);
        if (leaves.isEmpty()) 
            return;
        leaves.sort(java.util.Comparator.comparingDouble(LeafNode::calculatePromise));

        // Find the maximum number of active leaves that fits in the budget
        int maxActive = 0;
        while (maxActive < leaves.size()) {
            maxActive++;
            double projected = (maxActive * activeLeafSizeEstimate+ (leaves.size() - maxActive) * inactiveLeafSizeEstimate) * sizeEstimateOverhead;
            if (projected > maxBytes) {
                maxActive--;
                break;
            }
        }

        int cutoff = leaves.size() - maxActive;

        // Deactivate worst-promise leaves below the cutoff
        for (int i = 0; i < cutoff; i++) {
            LeafNode leaf = leaves.get(i);
            if (leaf.isActive()) {
                leaf.deactivate();
                nInactiveLeaves++;
                nActiveLeaves--;
            }
        }

        // Reactivate best-promise leaves above the cutoff that were previously deactivated
        for (int i = cutoff; i < leaves.size(); i++) {
            LeafNode leaf = leaves.get(i);
            if (!leaf.isActive() && leaf.depth < maxDepth) {
                leaf.activate();
                nActiveLeaves++;
                nInactiveLeaves--;
            }
        }
    }

    //endregion === MEMORY MANAGEMENT ===

    //region === ATTEMPT TO SPLIT (mirrors River's HoeffdingTreeRegressor._attempt_to_split) ===

    // Called only from LeafNode.learn() after grace period and depth checks.
    protected void attemptToSplit(LeafNode leaf, Node parent, int parentBranch) {
        List<SplitSuggestion> bestSplitSuggestions = leaf.bestSplitSuggestions(this);
        Collections.sort(bestSplitSuggestions);
        boolean shouldSplit = false;
        double hb = Double.NaN;
        if (bestSplitSuggestions.size() < 2) {
            shouldSplit = !bestSplitSuggestions.isEmpty();
        } else {
            hb = hoeffdingBound(1.0, delta, leaf.stats.getN());
            SplitSuggestion best = bestSplitSuggestions.get(bestSplitSuggestions.size() - 1);
            SplitSuggestion secondBest = bestSplitSuggestions.get(bestSplitSuggestions.size() - 2);
            if (best.merit > 0.0 && (
                    secondBest.merit / best.merit < 1 - hb || hb < tau)) {
                shouldSplit = true;
            }
            if (removePoorAttrs) {
                double bestRatio = secondBest.merit / best.merit;
                for (SplitSuggestion suggestion : bestSplitSuggestions) {
                    if (suggestion.feature != -1
                            && suggestion.merit / best.merit < bestRatio - 2 * hb) {
                        leaf.disableAttribute(suggestion.feature);
                    }
                }
            }
        }
        if (shouldSplit) {
            SplitSuggestion splitDecision = bestSplitSuggestions.get(bestSplitSuggestions.size() - 1);
            if (splitDecision.feature == -1) {
                leaf.deactivate();
                nInactiveLeaves++;
                nActiveLeaves--;
            } else {
                AdaSplitNode newSplit = splitDecision.assemble(leaf.depth, adwinDelta);
                for (int b = 0; b < splitDecision.childrenStats.length; b++) {
                    newSplit.children[b] = newLeaf(leaf.depth + 1, splitDecision.childrenStats[b], leaf);
                }
                nActiveLeaves--;
                nActiveLeaves += splitDecision.childrenStats.length;
                if (parent == null)
                    root = newSplit;
                else
                    ((SplitNode) parent).children[parentBranch] = newSplit;
            }
            enforceTreeSizeLimit();
        } else if (
                bestSplitSuggestions.size() >= 2
                && bestSplitSuggestions.get(bestSplitSuggestions.size() - 1).merit > 0
                && bestSplitSuggestions.get(bestSplitSuggestions.size() - 2).merit > 0) {
            double lastCheckRatio = bestSplitSuggestions.get(bestSplitSuggestions.size() - 2).merit / bestSplitSuggestions.get(bestSplitSuggestions.size() - 1).merit;
            double lastCheckVR = bestSplitSuggestions.get(bestSplitSuggestions.size() - 1).merit;
            leaf.manageMemory(lastCheckRatio, lastCheckVR, hb, minSamplesSplit);
        }
    }

    //endregion === ATTEMPT TO SPLIT ===

    //region === STATIC HELPERS ===

    public static double hoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt((range * range * Math.log(1.0 / confidence)) / (2.0 * n));
    }

    private static int poisson(double rate, java.util.Random rng) {
        double L = Math.exp(-rate);
        int k = 0; 
        double p = 1.0;
        do { 
            k++; 
            p *= rng.nextDouble(); 
        } while (p > L);
        return k - 1;
    }

    //endregion === STATIC HELPERS ===

    //region === NODE BASE ===

    public abstract static class Node {
        protected int depth;

        public Node(int depth) { this.depth = depth; }

        public abstract void learn(Instance inst, HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch);
        public abstract double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree);
        public Var getErrorTracker() { return null; }
        public abstract void collectLeaves(Instance inst, List<Node> result);
        public abstract void collectAllLeaves(List<LeafNode> result);
        // Mirrors River's total_weight and n_leaves properties.
        public abstract double totalWeight();
        public abstract int nLeaves();
        public void killTreeChildren(HoeffdingAdaptiveTreeRegressor tree) {}
    }

    //endregion === NODE BASE ===


    //region === LEAF NODE ===

    public abstract static class LeafNode extends Node {

        protected Var stats = new Var();
        protected double weightSeenAtLastSplitEval = 0;

        // ADWIN + error tracker live here because learn() uses them
        protected ADWIN driftDetector;
        protected Var errorTracker;
        private double maxObservedError = 1.0;

        /** Per-attribute splitters (numeric: TEBSTSplitter or RadiusSplitter, nominal: NominalSplitter); null → inactive. */
        protected Map<Integer, Splitter> splitters = null;
        protected Set<Integer> disabledAttrs = new HashSet<>();

        // Stored locally so updateStatsBase can use it without an outer-class reference.
        private int tebstDigits;

        // Mirrors River's AdaLeafRegressor.rng: reference to the shared tree RNG.
        protected java.util.Random rng;

        public LeafNode(int depth, double adwinDelta, int tebstDigits) {
            super(depth);
            this.tebstDigits = tebstDigits;
            this.driftDetector = new ADWIN(adwinDelta);
            this.errorTracker = new Var();
            this.splitters = new HashMap<>();
        }

        public Var getErrorTracker() { return errorTracker; }

        public boolean isActive() { return splitters != null; }
        public void activate() {
            if (splitters == null) 
                splitters = new HashMap<>();
        }
        public void deactivate() { splitters = null; }

        public double calculatePromise() { return -depth; }

        public long estimateByteSize() { return SizeOf.fullSizeOf(this); }

        public void disableAttribute(int attrIdx) {
            if (splitters != null && splitters.remove(attrIdx) != null)
                disabledAttrs.add(attrIdx);
        }

        // Mirrors River's HTLeaf.learn_one: update_stats then update_splitters.
        private void updateStatsBase(Instance inst, double w, HoeffdingAdaptiveTreeRegressor tree) {
            double y = inst.classValue();
            stats.update(y, w);
            if (isActive()) {
                int numAttrs = inst.numAttributes() - 1;
                for (int i = 0; i < numAttrs; i++) {
                    if (disabledAttrs.contains(i) || inst.isMissing(i))
                        continue;
                    boolean isNom = inst.attribute(i).isNominal() || (tree.nominalAttributeIndices != null && tree.nominalAttributeIndices.contains(i));
                    splitters.computeIfAbsent(i, k -> isNom ? new NominalSplitter() : (tree.numericalMultiway ? new RadiusSplitter(tree.numericalMultiwayRadius) : new TEBSTSplitter(tebstDigits))).update(inst.value(i), y, w);
                }
            }
        }

        public double getMean() { return stats.getMean(); }
        public double getVariance() { return stats.get(); }

        @Override public double totalWeight() { return stats.getN(); }
        @Override public int nLeaves() { return 1; }

        @Override public void collectLeaves(Instance inst, List<Node> result) { result.add(this); }
        @Override public void collectAllLeaves(List<LeafNode> result) { result.add(this); }

        // Hook: called BEFORE updateStatsBase() — mirrors River's LeafAdaptive.learn_one (FMSE update).
        protected void beforeUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {}

        // Hook: called AFTER updateStatsBase() — mirrors River's LeafModel.learn_one (model update).
        protected void afterUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {}

        /**
         * Single shared learn() — mirrors River's AdaLeafRegressor.learn_one.
         *
         * All leaf types inherit this unchanged; they only differ in prediction()
         * and beforeUpdate() / afterUpdate().
         */
        @Override
        public final void learn(Instance inst, HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch) {

            double y = inst.classValue();
            double y_pred = predict(inst, tree);   // pre-update prediction

            // Bootstrap sampling (River: w *= Poisson(1) if k > 0)
            double w = inst.weight();
            if (tree.bootstrapSampling) {
                int k = poisson(1.0, rng);
                if (k > 0)
                    w *= k;
            }

            // Drift detection + error tracking
            double error = Math.abs(y - y_pred);
            double oldMean = errorTracker.getMean();
            maxObservedError = Math.max(maxObservedError, error);
            driftDetector.setInput(error / maxObservedError);
            errorTracker.update(error, 1.0);
            if (driftDetector.getChange() && errorTracker.getMean() < oldMean)
                errorTracker = new Var();

            double preMean = getMean();

            // River: LeafAdaptive.learn_one — FMSE update with pre-update predictions
            beforeUpdate(inst, w, y, preMean, tree);

            // River: HTLeaf.learn_one — stats + splitters update
            updateStatsBase(inst, w, tree);

            // River: LeafModel.learn_one — model update
            afterUpdate(inst, w, y, preMean, tree);

            // Mirrors River AdaLeafRegressor.learn_one: grace period + depth check here,
            // not inside attemptToSplit.
            double weightSeen = stats.getN();
            if (weightSeen - weightSeenAtLastSplitEval >= tree.gracePeriod) {
                if (depth >= tree.maxDepth) {
                    deactivate();
                    tree.nActiveLeaves--;
                    tree.nInactiveLeaves++;
                } else if (isActive()) {
                    tree.attemptToSplit(this, parent, parentBranch);
                    weightSeenAtLastSplitEval = weightSeen;
                }
            }
        }

        // Mirrors River's HTLeaf.best_split_suggestions(split_criterion, tree).
        List<SplitSuggestion> bestSplitSuggestions(HoeffdingAdaptiveTreeRegressor tree) {
            List<SplitSuggestion> suggestions = new ArrayList<>();
            if (tree.meritPreprune)
                suggestions.add(SplitSuggestion.NULL);

            for (Map.Entry<Integer, Splitter> entry : splitters.entrySet()) {
                SplitSuggestion s = entry.getValue().bestEvaluatedSplitSuggestion(
                    stats, entry.getKey(), tree.binarySplit, tree.minSamplesSplit);
                if (s != null)
                    suggestions.add(s);
            }

            return suggestions;
        }

        // Mirrors River's LeafMean.manage_memory.
        void manageMemory(double lastCheckRatio, double lastCheckVR, double lastCheckE, int minSamplesSplit) {
            for (Splitter s : splitters.values()) {
                s.removeBadSplits(lastCheckRatio, lastCheckVR, lastCheckE, minSamplesSplit, stats);
            }
        }
    }

    //endregion === LEAF NODE ===

    //region === ADA LEAF MEAN ===

    public static class AdaLeafMean extends LeafNode {

        public AdaLeafMean(int depth, double adwinDelta, int tebstDigits) {
            super(depth, adwinDelta, tebstDigits);
        }

        /** River: LeafMean.prediction -> stats.mean.get(). */
        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            return getMean();
        }
    }

    //endregion === ADA LEAF MEAN ===

    //region === ADA LEAF MODEL ===

    public static class AdaLeafModel extends AdaLeafMean {

        protected LinearModel leafModel;

        public AdaLeafModel(int depth, double adwinDelta, int tebstDigits, double learningRate, double l2, double l1) {
            super(depth, adwinDelta, tebstDigits);
            this.leafModel = new LinearModel(learningRate, l2, l1);
        }

        // River: LeafModel.prediction → leaf_model.predict_one(x).
        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            return leafModel.predict(inst, tree);
        }

        // River: LeafModel.learn_one → leaf_model.learn_one(x, y, w).
        @Override
        protected void afterUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {
            leafModel.update(inst, w, tree);
        }
    }

    //endregion === ADA LEAF MEAN 

    //region === ADA LEAF ADAPTIVE ===

    public static class AdaLeafAdaptive extends AdaLeafModel {

        private double fmseMean  = 0.0;
        private double fmseModel = 0.0;

        public AdaLeafAdaptive(int depth, double adwinDelta, int tebstDigits, double learningRate, double l2, double l1) {
            super(depth, adwinDelta, tebstDigits, learningRate, l2, l1);
        }

        // River: LeafAdaptive.prediction
        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            if (fmseMean < fmseModel)
                return getMean();
            return super.predict(inst, tree);
        }

        // River: LeafAdaptive.learn_one — FMSE update with pre-update predictions (before stats/model).
        @Override
        protected void beforeUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {
            double preModel = leafModel.predict(inst, tree);
            double d = tree.modelSelectorDecay;
            fmseMean = d * fmseMean + (y - preMean) * (y - preMean);
            fmseModel = d * fmseModel + (y - preModel) * (y - preModel);
        }

        // River: LeafModel.learn_one — model update (delegated to super).
        @Override
        protected void afterUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {
            super.afterUpdate(inst, w, y, preMean, tree);
        }
    }
    //endregion === ADA LEAF ADAPTIVE ===

    
    //region === SPLIT NODE ===

    public abstract static class SplitNode extends Node {

        protected Node[] children;

        public SplitNode(int numChildren, int depth) {
            super(depth);
            this.children = new Node[numChildren];
        }

        public abstract int getBranchIndex(Instance inst);

        // Mirrors River's most_common_path (defined per-subclass in branch.py): finds child with highest total_weight.
        protected int mostCommonPath() {
            int best = 0;
            double bestW = children[0] != null ? children[0].totalWeight() : -1;
            for (int i = 1; i < children.length; i++) {
                double w = children[i] != null ? children[i].totalWeight() : -1;
                if (w > bestW) { 
                    bestW = w; 
                    best = i; 
                }
            }
            return best;
        }

        public Node routeChild(Instance inst) {
            int branch = getBranchIndex(inst);
            if (branch < 0 || branch >= children.length) 
                branch = mostCommonPath();
            return children[branch];
        }

        // Mirrors River's DTBranch.total_weight: recursive sum over children.
        @Override
        public double totalWeight() {
            double sum = 0;
            for (Node child : children)
                if (child != null) sum += child.totalWeight();
            return sum;
        }

        // Mirrors River's DTBranch.n_leaves: recursive count of all leaves.
        @Override
        public int nLeaves() {
            int sum = 0;
            for (Node child : children)
                if (child != null) sum += child.nLeaves();
            return sum;
        }

        @Override
        public void collectAllLeaves(List<LeafNode> result) {
            for (Node child : children)
                if (child != null)
                    child.collectAllLeaves(result);
        }
    }

    //endregion === SPLIT NODE ===

    //region === ADA SPLIT NODE ===

    public abstract static class AdaSplitNode extends SplitNode {

        protected ADWIN driftDetector;
        protected Var  errorTracker;
        protected Node alternateTree;
        private double maxObservedError = 1.0;

        protected Var branchStats = new Var();

        public AdaSplitNode(int numChildren, int depth, double adwinDelta) {
            super(numChildren, depth);
            this.driftDetector = new ADWIN(adwinDelta);
            this.errorTracker = new Var();
        }

        public Var getErrorTracker() { return errorTracker; }

        public double getBranchMean() { return branchStats.getMean(); }

        @Override
        public void collectLeaves(Instance inst, List<Node> result) {
            if (alternateTree != null) 
                alternateTree.collectLeaves(inst, result);
            Node child = routeChild(inst);
            if (child != null) 
                child.collectLeaves(inst, result);
        }

        @Override
        public void collectAllLeaves(List<LeafNode> result) {
            super.collectAllLeaves(result);
            if (alternateTree != null) 
                alternateTree.collectAllLeaves(result);
        }

        @Override
        public void learn(Instance inst, HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch) {
            double y = inst.classValue();
            double w = inst.weight();

            // Get prediction from the leaf this instance would reach (not alternate subtrees)
            Node curr = routeChild(inst);
            while (curr instanceof SplitNode) curr = ((SplitNode) curr).routeChild(inst);
            double yPred = (curr != null) ? curr.predict(inst, tree) : getBranchMean();

            // Update branch stats with original (un-sampled) weight, mirrors River's self.stats.update(y, w)
            branchStats.update(y, w);

            double driftInput = Math.abs(y - yPred);
            double oldMean = errorTracker.getMean();
            maxObservedError = Math.max(maxObservedError, driftInput);
            driftDetector.setInput(driftInput / maxObservedError);
            errorTracker.update(driftInput, 1.0);
            boolean errorChange = driftDetector.getChange();

            // Error is decreasing — keep things as they are, reset tracker
            if (errorChange && errorTracker.getMean() < oldMean) {
                errorTracker = new Var();
                errorChange = false;
            }

            if (errorChange && alternateTree == null) {
                errorTracker = new Var();
                // New alternate tree starts at same depth as this branch
                alternateTree = tree.newLeaf(depth);
                tree.nAlternateTrees++;
            } else if (alternateTree != null) {
                compareAlternate(tree, parent, parentBranch);
            }

            if (alternateTree != null) {
                alternateTree.learn(inst, tree, parent, parentBranch);
            }

            // Route to child
            int branchIdx = getBranchIndex(inst);
            if (branchIdx >= 0 && branchIdx < children.length && children[branchIdx] != null) {
                children[branchIdx].learn(inst, tree, this, branchIdx);
            } else if (this instanceof AdaNomMultiwaySplitNode) {
                AdaNomMultiwaySplitNode multi = (AdaNomMultiwaySplitNode) this;
                if (!inst.isMissing(multi.attrIdx)) {
                    // New category seen — add branch dynamically
                    int catVal = (int) inst.value(multi.attrIdx);
                    int newBranch = multi.addCategory(catVal);
                    children = java.util.Arrays.copyOf(children, children.length + 1);
                    LeafNode newLeaf = tree.newLeaf(depth + 1);
                    children[newBranch] = newLeaf;
                    tree.nActiveLeaves++;
                    newLeaf.learn(inst, tree, this, newBranch);
                } else {
                    int best = mostCommonPath();
                    if (children[best] != null)
                        children[best].learn(inst, tree, this, best);
                }
            } else if (this instanceof AdaNumMultiwaySplitNode) {
                AdaNumMultiwaySplitNode multi = (AdaNumMultiwaySplitNode) this;
                if (!inst.isMissing(multi.attrIdx)) {
                    // New slot seen — add branch dynamically (mirrors NumericMultiwayBranch.add_child)
                    int newBranch = multi.addSlot(inst.value(multi.attrIdx));
                    children = java.util.Arrays.copyOf(children, children.length + 1);
                    LeafNode newLeaf = tree.newLeaf(depth + 1);
                    children[newBranch] = newLeaf;
                    tree.nActiveLeaves++;
                    newLeaf.learn(inst, tree, this, newBranch);
                } else {
                    int best = mostCommonPath();
                    if (children[best] != null)
                        children[best].learn(inst, tree, this, best);
                }
            } else {
                // Missing feature — use most common path
                int best = mostCommonPath();
                if (children[best] != null)
                    children[best].learn(inst, tree, this, best);
            }
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            Node child = routeChild(inst);
            return (child != null) ? child.predict(inst, tree) : getBranchMean();
        }

        // Mirrors River's AdaBranchRegressor.learn_one switch/prune logic.
        private void compareAlternate(HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch) {
            if (alternateTree == null) return;
            Var altErr = alternateTree.getErrorTracker();
            if (altErr == null)
                return;

            double altN = altErr.getN();
            double curN = errorTracker.getN();

            if (altN <= tree.driftWindowThreshold || curN <= tree.driftWindowThreshold)
                return;

            double altMean = altErr.getMean();
            double curMean = errorTracker.getMean();
            double altVar = altErr.get();
            double curVar = errorTracker.get();

            double denom = Math.sqrt(altVar / altN + curVar / curN);
            if (denom < 1e-10) 
                return;

            double z = (altMean - curMean) / denom;
            double pValue = 2.0 * NORM.cumulativeProbability(-Math.abs(z));

            if (pValue > tree.switchSignificance) 
                return;

            if (altMean < curMean) {
                // Mirrors River: tree._n_active_leaves -= self.n_leaves; += alt.n_leaves; kill_tree_children
                tree.nActiveLeaves -= this.nLeaves();
                tree.nActiveLeaves += alternateTree.nLeaves();
                this.killTreeChildren(tree);
                if (parent != null) {
                    ((SplitNode) parent).children[parentBranch] = alternateTree;
                } else {
                    tree.root = alternateTree;
                }
                alternateTree = null;
                tree.nSwitchAlternateTrees++;
            } else {
                // Mirrors River: if isinstance(DTBranch): kill_tree_children; alternate = None
                alternateTree.killTreeChildren(tree);
                alternateTree = null;
                tree.nPrunedAlternateTrees++;
            }
        }

        // Mirrors River's AdaBranchRegressor.kill_tree_children.
        @Override
        public void killTreeChildren(HoeffdingAdaptiveTreeRegressor tree) {
            for (Node child : children) {
                if (child == null) 
                    continue;
                if (child instanceof AdaSplitNode) {
                    AdaSplitNode branch = (AdaSplitNode) child;
                    if (branch.alternateTree != null) {
                        branch.alternateTree.killTreeChildren(tree);
                        tree.nPrunedAlternateTrees++;
                        branch.alternateTree = null;
                    }
                    branch.killTreeChildren(tree);
                } else if (child instanceof LeafNode) {
                    if (((LeafNode) child).isActive()) 
                        tree.nActiveLeaves--;
                    else 
                        tree.nInactiveLeaves--;
                }
            }
        }
    }

    //endregion === ADA SPLIT NODE ===

    //region === FACTORY METHODS ===

    /**
     * Creates the correct leaf type based on leafPrediction — mirrors River's
     * _new_leaf() which instantiates AdaLeafRegMean / AdaLeafRegModel / AdaLeafRegAdaptive.
     */
    protected LeafNode newLeaf(int depth) {
        LeafNode leaf;
        switch (leafPrediction) {
            case MEAN: leaf = new AdaLeafMean(depth, adwinDelta, tebstDigits); break;
            case MODEL: leaf = new AdaLeafModel(depth, adwinDelta, tebstDigits, learningRate, l2, l1); break;
            default: leaf = new AdaLeafAdaptive(depth, adwinDelta, tebstDigits, learningRate, l2, l1);
        }
        leaf.rng = classifierRandom;
        return leaf;
    }

    protected LeafNode newLeaf(int depth, Var initStats) {
        LeafNode leaf = newLeaf(depth);
        if (initStats != null) {
            leaf.stats = initStats.plus(new Var());
            leaf.weightSeenAtLastSplitEval = initStats.getN();
        }
        return leaf;
    }

    // Mirrors River's _new_leaf(initial_stats, parent=leaf): propagates leaf_model and fmse from parent.
    protected LeafNode newLeaf(int depth, Var initStats, LeafNode parentLeaf) {
        LeafNode leaf = newLeaf(depth, initStats);
        if (parentLeaf instanceof AdaLeafModel && leaf instanceof AdaLeafModel) {
            ((AdaLeafModel) leaf).leafModel = ((AdaLeafModel) parentLeaf).leafModel.clone();
        }
        if (parentLeaf instanceof AdaLeafAdaptive && leaf instanceof AdaLeafAdaptive) {
            AdaLeafAdaptive src = (AdaLeafAdaptive) parentLeaf;
            AdaLeafAdaptive dst = (AdaLeafAdaptive) leaf;
            dst.fmseMean = src.fmseMean;
            dst.fmseModel = src.fmseModel;
        }
        return leaf;
    }

    //endregion === FACTORY METHODS ===

    //region === CONCRETE ADA SPLIT NODE SUBCLASSES ===
    // (mirror River's AdaNumBinaryBranchReg, AdaNomBinaryBranchReg, AdaNomMultiwayBranchReg, AdaNumMultiwayBranchReg)

    public static class AdaNumBinarySplitNode extends AdaSplitNode {
        private final int attrIdx;
        private final double threshold;

        public AdaNumBinarySplitNode(int depth, int attrIdx, double threshold, double adwinDelta) {
            super(2, depth, adwinDelta);
            this.attrIdx = attrIdx;
            this.threshold = threshold;
        }

        public int getBranchIndex(Instance inst) {
            if (inst.isMissing(attrIdx)) 
                return -1;
            return inst.value(attrIdx) <= threshold ? 0 : 1;
        }
    }

    public static class AdaNomBinarySplitNode extends AdaSplitNode {
        private final int attrIdx;
        private final Set<Integer> leftCats;
        private final Set<Integer> rightCats;

        public AdaNomBinarySplitNode(int depth, int attrIdx, Set<Integer> leftCats, Set<Integer> rightCats, double adwinDelta) {
            super(2, depth, adwinDelta);
            this.attrIdx   = attrIdx;
            this.leftCats  = leftCats;
            this.rightCats = rightCats;
        }

        public int getBranchIndex(Instance inst) {
            if (inst.isMissing(attrIdx)) 
                return -1;
            int cat = (int) inst.value(attrIdx);
            if (leftCats.contains(cat))  
                return 0;
            if (rightCats.contains(cat)) 
                return 1;
            return -1;
        }
    }

    public static class AdaNomMultiwaySplitNode extends AdaSplitNode {
        final int attrIdx;
        private final List<Integer> categoryValues;
        private final Map<Integer,Integer> categoryToBranch;

        public AdaNomMultiwaySplitNode(int depth, int attrIdx, int[] initialCategories, double adwinDelta) {
            super(initialCategories.length, depth, adwinDelta);
            this.attrIdx = attrIdx;
            this.categoryValues = new ArrayList<>();
            this.categoryToBranch = new HashMap<>();
            for (int i = 0; i < initialCategories.length; i++) {
                categoryValues.add(initialCategories[i]);
                categoryToBranch.put(initialCategories[i], i);
            }
        }

        public int getBranchIndex(Instance inst) {
            if (inst.isMissing(attrIdx)) return -1;
            Integer branch = categoryToBranch.get((int) inst.value(attrIdx));
            return branch != null ? branch : -1;
        }

        public int addCategory(int catIndex) {
            int branch = categoryValues.size();
            categoryValues.add(catIndex);
            categoryToBranch.put(catIndex, branch);
            return branch;
        }
    }

    // Mirrors River's AdaNumMultiwayBranchReg + NumericMultiwayBranch.
    // Slots are computed as floor(x / radius); each slot maps to a branch index.
    // New slots seen at learn-time are added dynamically (mirrors add_child).
    public static class AdaNumMultiwaySplitNode extends AdaSplitNode {
        final int attrIdx;
        final double radius;
        private final Map<Integer, Integer> slotToBranch;
        private final Map<Integer, Integer> branchToSlot;

        public AdaNumMultiwaySplitNode(int depth, int attrIdx, double radius, int[] initialSlotIds, double adwinDelta) {
            super(initialSlotIds.length, depth, adwinDelta);
            this.attrIdx = attrIdx;
            this.radius = radius;
            this.slotToBranch = new HashMap<>();
            this.branchToSlot = new HashMap<>();
            for (int i = 0; i < initialSlotIds.length; i++) {
                slotToBranch.put(initialSlotIds[i], i);
                branchToSlot.put(i, initialSlotIds[i]);
            }
        }

        @Override
        public int getBranchIndex(Instance inst) {
            if (inst.isMissing(attrIdx)) return -1;
            int slot = (int) Math.floor(inst.value(attrIdx) / radius);
            Integer branch = slotToBranch.get(slot);
            return branch != null ? branch : -1;
        }

        public int addSlot(double featureVal) {
            int slot = (int) Math.floor(featureVal / radius);
            int branch = slotToBranch.size();
            slotToBranch.put(slot, branch);
            branchToSlot.put(branch, slot);
            return branch;
        }
    }

    //endregion === CONCRETE ADA SPLIT NODE SUBCLASSES ===

    //region === SPLITTER INTERFACE ===

    // Mirrors River's Splitter base class: uniform update + best_evaluated_split_suggestion.
    public interface Splitter {
        void update(double attVal, double y, double w);
        SplitSuggestion bestEvaluatedSplitSuggestion(Var preSplit, int attrIdx, boolean binaryOnly, int minSamples);
        default void removeBadSplits(double lastCheckRatio, double lastCheckVR, double lastCheckE, int minSamplesSplit, Var preSplit) {}
    }

    //endregion === SPLITTER INTERFACE ===

    //region === NOMINAL SPLITTER ====

    public static class NominalSplitter implements Splitter {

        private final Map<Integer, Var> catStats = new HashMap<>();

        @Override
        public void update(double attVal, double y, double w) {
            catStats.computeIfAbsent((int) attVal, k -> new Var()).update(y, w);
        }

        @Override
        public SplitSuggestion bestEvaluatedSplitSuggestion(Var preSplit, int attrIdx, boolean binaryOnly, int minSamples) {
            SplitCandidate res = binaryOnly ? bestBinarySplit(preSplit, minSamples) : bestSplit(preSplit, minSamples);
            if (res == null) return null;
            SplitSuggestion s = new SplitSuggestion();
            s.merit = res.merit;
            s.feature = attrIdx;
            s.numericalFeature = false;
            s.multiwaySplit = !binaryOnly && res.multiwaySplit;
            s.childrenStats = res.postSplitDists;
            if (!binaryOnly && res.multiwaySplit) {
                s.sortedCats = getSortedCategories();
            } else {
                int[][] parts = getBinarySplitCategories(res.splitCatIdx);
                s.leftSet = new HashSet<>();
                s.rightSet = new HashSet<>();
                for (int c : parts[0]) s.leftSet.add(c);
                for (int c : parts[1]) s.rightSet.add(c);
            }
            return s;
        }

        // Returns best split: tries multiway (when > 2 categories) and all one-vs-rest binary splits.
        // Sets multiwaySplit=true on the result when the multiway option wins.
        public SplitCandidate bestSplit(Var preSplit, int minSamplesSplit) {
            if (catStats.size() < 2) return null;
            SplitCandidate best = new SplitCandidate();

            // Multiway: all categories as separate branches
            if (catStats.size() > 2) {
                int[] sorted = getSortedCategories();
                Var[] dists = new Var[sorted.length];
                boolean enough = true;
                for (int i = 0; i < sorted.length; i++) {
                    dists[i] = catStats.get(sorted[i]);
                    if (dists[i].getN() < minSamplesSplit) { enough = false; break; }
                }
                if (enough) {
                    double merit = varianceReductionMulti(preSplit, dists);
                    if (merit > best.merit) {
                        best.merit = merit;
                        best.postSplitDists = dists;
                        best.splitCatIdx = -1;
                        best.multiwaySplit = true;
                    }
                }
            }

            // Binary: category X vs rest
            for (Map.Entry<Integer, Var> entry : catStats.entrySet()) {
                Var leftDist = entry.getValue();
                if (leftDist.getN() < minSamplesSplit) continue;
                Var rightDist = preSplit.minus(leftDist);
                if (rightDist.getN() < minSamplesSplit) continue;
                double merit = varianceReduction(preSplit, leftDist, rightDist);
                if (merit > best.merit) {
                    best.merit = merit;
                    best.postSplitDists = new Var[]{ leftDist, rightDist };
                    best.splitCatIdx = entry.getKey();
                    best.multiwaySplit = false;
                }
            }

            if (best.merit <= 0 || best.postSplitDists == null) return null;
            return best;
        }

        public SplitCandidate bestBinarySplit(Var preSplit, int minSamplesSplit) {
            if (catStats.size() < 2) return null;
            SplitCandidate best = new SplitCandidate();

            for (Map.Entry<Integer, Var> entry : catStats.entrySet()) {
                Var leftDist = entry.getValue();
                if (leftDist.getN() < minSamplesSplit) continue;
                Var rightDist = preSplit.minus(leftDist);
                if (rightDist.getN() < minSamplesSplit) continue;
                double merit = varianceReduction(preSplit, leftDist, rightDist);
                if (merit > best.merit) {
                    best.merit = merit;
                    best.postSplitDists = new Var[]{ leftDist, rightDist };
                    best.splitCatIdx = entry.getKey();
                    best.multiwaySplit = false;
                }
            }

            if (best.merit <= 0 || best.postSplitDists == null) return null;
            return best;
        }

        public int[] getSortedCategories() {
            int[] cats = new int[catStats.size()];
            int i = 0;
            for (int k : catStats.keySet()) cats[i++] = k;
            java.util.Arrays.sort(cats);
            return cats;
        }

        // Returns {leftCats, rightCats} for the binary split on splitCat (left={splitCat}, right=rest).
        public int[][] getBinarySplitCategories(int splitCat) {
            List<Integer> right = new ArrayList<>();
            for (int cat : catStats.keySet()) {
                if (cat != splitCat) right.add(cat);
            }
            int[] rightArr = new int[right.size()];
            for (int i = 0; i < right.size(); i++) rightArr[i] = right.get(i);
            return new int[][]{ new int[]{ splitCat }, rightArr };
        }

        public Var[] getChildrenStats(int[] sortedCats) {
            Var[] stats = new Var[sortedCats.length];
            for (int i = 0; i < sortedCats.length; i++)
                stats[i] = catStats.getOrDefault(sortedCats[i], new Var());
            return stats;
        }

        private static double varianceReduction(Var parent, Var left, Var right) {
            double n = parent.getN();
            if (n == 0) 
                return 0;
            return parent.get() - (left.getN() / n * left.get() + right.getN() / n * right.get());
        }

        private static double varianceReductionMulti(Var parent, Var[] children) {
            double n = parent.getN();
            if (n == 0) 
                return 0;
            double contrib = 0;
            for (Var c : children) contrib += c.getN() / n * c.get();
            return parent.get() - contrib;
        }
    }

    //endregion === NOMINAL SPLITTER ===

    //region === SPLIT CANDIDATE ===

    public static final class SplitCandidate {
        public double merit = Double.NEGATIVE_INFINITY;
        public double splitVal = Double.NaN;
        public Var[] postSplitDists = null;
        public int splitCatIdx = -1;
        public boolean multiwaySplit = false;
    }

    //endregion === SPLIT CANDIDATE ===

    //region === SPLIT SUGGESTION ===

    // Mirrors River's BranchFactory: wraps a split candidate with attribute identity,
    // children distributions, and provides assemble() to build the AdaSplitNode.
    public static final class SplitSuggestion implements Comparable<SplitSuggestion> {
        static final SplitSuggestion NULL = new SplitSuggestion(); // merit_preprune null split

        double merit = Double.NEGATIVE_INFINITY;
        int feature = -1;         // -1 = null split (River: feature=None)
        boolean numericalFeature;
        boolean multiwaySplit;
        Var[] childrenStats;
        double splitVal;           // numeric binary
        Set<Integer> leftSet;      // nominal binary
        Set<Integer> rightSet;     // nominal binary
        int[] sortedCats;          // nominal multiway
        double radius;             // numeric multiway
        int[] slotIds;             // numeric multiway

        @Override
        public int compareTo(SplitSuggestion o) {
            return Double.compare(this.merit, o.merit);
        }

        // Mirrors River's split_decision.assemble(branch, leaf.stats, leaf.depth, *leaves, **kwargs).
        AdaSplitNode assemble(int depth, double adwinDelta) {
            if (numericalFeature && multiwaySplit)
                return new AdaNumMultiwaySplitNode(depth, feature, radius, slotIds, adwinDelta);
            if (numericalFeature)
                return new AdaNumBinarySplitNode(depth, feature, splitVal, adwinDelta);
            if (multiwaySplit)
                return new AdaNomMultiwaySplitNode(depth, feature, sortedCats, adwinDelta);
            return new AdaNomBinarySplitNode(depth, feature, leftSet, rightSet, adwinDelta);
        }
    }

    //endregion === SPLIT SUGGESTION ===

    //region === TEBST SPLITTER ===

    public static class TEBSTSplitter implements Splitter {

        private final double roundFactor;
        private EBSTNode root = null;

        public TEBSTSplitter(int digits) {
            this.roundFactor = Math.pow(10, digits);
        }

        private double round(double v) {
            return Math.round(v * roundFactor) / roundFactor;
        }

        @Override
        public void update(double attVal, double y, double w) {
            attVal = round(attVal);
            if (root == null)
                root = new EBSTNode(attVal, y, w);
            else
                root.insertValue(attVal, y, w);
        }

        @Override
        public SplitSuggestion bestEvaluatedSplitSuggestion(Var preSplit, int attrIdx, boolean binaryOnly, int minSamples) {
            SplitCandidate res = bestSplit(preSplit, minSamples);
            if (res == null) return null;
            SplitSuggestion s = new SplitSuggestion();
            s.merit = res.merit;
            s.feature = attrIdx;
            s.numericalFeature = true;
            s.splitVal = res.splitVal;
            s.childrenStats = new Var[]{ res.postSplitDists[0], preSplit.minus(res.postSplitDists[0]) };
            return s;
        }

        public SplitCandidate bestSplit(Var preSplit, int minSamplesSplit) {
            if (root == null || preSplit.getN() < 2 * minSamplesSplit)
                return null;
            SplitCandidate best = new SplitCandidate();
            findBestSplit(root, best, new Var(), preSplit, minSamplesSplit);
            if (Double.isNaN(best.splitVal) || best.merit <= 0)
                return null;
            return best;
        }

        // In-order traversal mirroring River's EBSTSplitter._find_best_split.
        // aux accumulates the combined estimator of all nodes whose right subtree
        // is currently being explored (Chan's parallel variance formula).
        private void findBestSplit(EBSTNode node, SplitCandidate best, Var aux, Var preSplit, int minSplit) {
            if (node.left != null)
                findBestSplit(node.left, best, aux, preSplit, minSplit);

            Var leftDist = node.estimator.plus(aux);
            Var rightDist = preSplit.minus(leftDist);

            if (leftDist.getN() >= minSplit && rightDist.getN() >= minSplit) {
                double merit = varianceReduction(preSplit, leftDist, rightDist);
                if (merit > best.merit) {
                    best.merit = merit;
                    best.splitVal = node.attVal;
                    best.postSplitDists = new Var[]{ leftDist, rightDist };
                }
            }

            if (node.right != null) {
                aux.addInPlace(node.estimator);
                findBestSplit(node.right, best, aux, preSplit, minSplit);
                aux.subtractInPlace(node.estimator);
            }
        }

        @Override
        public void removeBadSplits(double lastCheckRatio, double lastCheckVR, double epsilon, int minSamplesSplit, Var preSplit) {
            if (root == null || lastCheckVR <= 0)
                return;
            root = removeBadSplitNodes(root, new Var(), preSplit, lastCheckRatio, lastCheckVR, epsilon, minSamplesSplit);
        }

        // Post-order pruning mirroring River's EBSTSplitter._remove_bad_split_nodes.
        private EBSTNode removeBadSplitNodes(EBSTNode node, Var aux, Var preSplit, double lastCheckRatio, double lastCheckVR, double lastCheckE, int minSamplesSplit) {
            if (node == null)
                return null;

            boolean isBad;
            if (node.left != null) {
                node.left = removeBadSplitNodes(node.left, aux, preSplit, lastCheckRatio, lastCheckVR, lastCheckE, minSamplesSplit);
                isBad = (node.left == null);
            } else {
                isBad = true;
            }

            if (isBad) {
                if (node.right != null) {
                    aux.addInPlace(node.estimator);
                    node.right = removeBadSplitNodes(node.right, aux, preSplit, lastCheckRatio, lastCheckVR, lastCheckE, minSamplesSplit);
                    aux.subtractInPlace(node.estimator);
                    isBad = (node.right == null);
                } else {
                    isBad = true;
                }
            }

            if (isBad) {
                Var leftDist = node.estimator.plus(aux);
                Var rightDist = preSplit.minus(leftDist);
                double merit = 0.0;
                if (leftDist.getN() >= minSamplesSplit && rightDist.getN() >= minSamplesSplit)
                    merit = varianceReduction(preSplit, leftDist, rightDist);
                if (merit / lastCheckVR < lastCheckRatio - 2 * lastCheckE)
                    return null;
            }

            return node;
        }

        private static double varianceReduction(Var parent, Var left, Var right) {
            double n = parent.getN();
            if (n == 0) return 0;
            return parent.get() - (left.getN() / n * left.get() + right.getN() / n * right.get());
        }
    }

    //endregion === TEBST SPLITTER ===

    //region === RADIUS SPLITTER ===

    // Mirrors River's QOSplitter(allow_multiway_splits=True). Used when numericalMultiway=true.
    // Bins numerical values into slots via slot = floor(x / radius).
    // Evaluates both a multiway candidate (one branch per slot, when > 2 slots and !binaryOnly)
    // and the best binary candidate (midpoint between adjacent sorted slots), returns the better one.
    public static class RadiusSplitter implements Splitter {

        private final double radius;
        private final TreeMap<Integer, Var> slotStats = new TreeMap<>();

        public RadiusSplitter(double radius) {
            this.radius = radius;
        }

        private int slot(double v) {
            return (int) Math.floor(v / radius);
        }

        @Override
        public void update(double attVal, double y, double w) {
            slotStats.computeIfAbsent(slot(attVal), k -> new Var()).update(y, w);
        }

        @Override
        public SplitSuggestion bestEvaluatedSplitSuggestion(Var preSplit, int attrIdx, boolean binaryOnly, int minSamples) {
            if (slotStats.size() < 2) return null;

            List<Integer> sortedSlots = new ArrayList<>(slotStats.keySet());
            SplitSuggestion best = null;

            // Multiway candidate: one branch per slot.
            if (!binaryOnly && slotStats.size() > 2) {
                Var[] dists = new Var[sortedSlots.size()];
                boolean enough = true;
                for (int i = 0; i < sortedSlots.size(); i++) {
                    dists[i] = slotStats.get(sortedSlots.get(i));
                    if (dists[i].getN() < minSamples) { enough = false; break; }
                }
                if (enough) {
                    double merit = varianceReductionMulti(preSplit, dists);
                    if (merit > 0) {
                        SplitSuggestion s = new SplitSuggestion();
                        s.merit = merit;
                        s.feature = attrIdx;
                        s.numericalFeature = true;
                        s.multiwaySplit = true;
                        s.radius = radius;
                        s.slotIds = sortedSlots.stream().mapToInt(Integer::intValue).toArray();
                        s.childrenStats = dists;
                        best = s;
                    }
                }
            }

            // Binary candidate: best midpoint between adjacent sorted slots.
            Var leftAcc = new Var();
            for (int i = 0; i < sortedSlots.size() - 1; i++) {
                leftAcc.addInPlace(slotStats.get(sortedSlots.get(i)));
                Var right = preSplit.minus(leftAcc);
                if (leftAcc.getN() < minSamples || right.getN() < minSamples) continue;
                double merit = varianceReduction(preSplit, leftAcc, right);
                if (best == null || merit > best.merit) {
                    SplitSuggestion s = new SplitSuggestion();
                    s.merit = merit;
                    s.feature = attrIdx;
                    s.numericalFeature = true;
                    s.multiwaySplit = false;
                    s.splitVal = (sortedSlots.get(i) + 1) * radius;
                    s.childrenStats = new Var[]{ leftAcc.plus(new Var()), preSplit.minus(leftAcc) };
                    best = s;
                }
            }

            return best;
        }

        private static double varianceReduction(Var parent, Var left, Var right) {
            double n = parent.getN();
            if (n == 0) return 0;
            return parent.get() - (left.getN() / n * left.get() + right.getN() / n * right.get());
        }

        private static double varianceReductionMulti(Var parent, Var[] children) {
            double n = parent.getN();
            if (n == 0) return 0;
            double contrib = 0;
            for (Var c : children) contrib += c.getN() / n * c.get();
            return parent.get() - contrib;
        }
    }

    //endregion === RADIUS SPLITTER ===

    //region === VAR (Welford's online weighted variance, mirrors River's stats.Var with ddof=1) ===

    public static class Var {
        private double mean = 0.0;
        private double S = 0.0;
        private double n = 0.0;

        public void update(double x, double w) {
            n += w;
            double delta = x - mean;
            mean += (w / n) * delta;
            S += w * delta * (x - mean);
        }

        public double getMean() { return mean; }
        public double getN() { return n; }
        public double get() { return n > 1.0 ? S / (n - 1.0) : 0.0; }

        /** Chan's parallel algorithm: returns a new Var = this + other. */
        public Var plus(Var other) {
            if (this.n == 0) 
                return other.copy();
            if (other.n == 0) 
                return this.copy();
            Var r = new Var();
            r.n = this.n + other.n;
            double delta = other.mean - this.mean;
            r.mean = (this.n * this.mean + other.n * other.mean) / r.n;
            r.S = this.S + other.S + delta * delta * (this.n * other.n / r.n);
            return r;
        }

        /** Returns a new Var = this - other (removes other's contribution from this). */
        public Var minus(Var other) {
            if (other.n == 0) 
                return this.copy();
            double nA = this.n - other.n;
            if (nA <= 0) 
                return new Var();
            Var r = new Var();
            r.n = nA;
            r.mean = (this.n * this.mean - other.n * other.mean) / nA;
            double delta = other.mean - r.mean;
            r.S = Math.max(0.0, this.S - other.S - delta * delta * (nA * other.n / this.n));
            return r;
        }

        /** In-place: this += other. */
        public void addInPlace(Var other) {
            if (other.n == 0) 
                return;
            double newN = this.n + other.n;
            double delta = other.mean - this.mean;
            S += other.S + delta * delta * (this.n * other.n / newN);
            mean = (this.n * this.mean + other.n * other.mean) / newN;
            n = newN;
        }

        /** In-place: this -= other (removes other's contribution). */
        public void subtractInPlace(Var other) {
            if (other.n == 0) 
                return;
            double nC = this.n;
            double nA = nC - other.n;
            if (nA <= 0) { 
                n = 0; mean = 0; 
                S = 0; 
                return; 
            }
            double meanA = (nC * this.mean - other.n * other.mean) / nA;
            double delta = other.mean - meanA;
            S = Math.max(0.0, this.S - other.S - delta * delta * (nA * other.n / nC));
            mean = meanA;
            n = nA;
        }

        private Var copy() {
            Var v = new Var(); 
            v.n = n; 
            v.mean = mean; 
            v.S = S; 
            return v;
        }
    }

    //endregion === VAR ===

    //region === EBST NODE ====

    public static class EBSTNode {
        double attVal;
        EBSTNode left, right;

        // univariate: non-null when target is a scalar double
        Var estimator;
        // multivariate: non-null when target is Map<Integer, Double>
        Map<Integer, Var> estimators;

        EBSTNode(double attVal, double y, double w) {
            this.attVal = attVal;
            this.estimator = new Var();
            updateEstimatorUnivariate(y, w);
        }

        EBSTNode(double attVal, Map<Integer, Double> target, double w) {
            this.attVal = attVal;
            this.estimators = new HashMap<>();
            updateEstimatorMultivariate(target, w);
        }

        void updateEstimatorUnivariate(double y, double w) {
            estimator.update(y, w);
        }

        void updateEstimatorMultivariate(Map<Integer, Double> target, double w) {
            for (Map.Entry<Integer, Double> e : target.entrySet())
                estimators.computeIfAbsent(e.getKey(), k -> new Var()).update(e.getValue(), w);
        }

        void insertValue(double attVal, double y, double w) {
            EBSTNode current = this, ante = null;
            boolean  isRight = false;
            while (current != null) {
                ante = current;
                if (attVal == current.attVal) {
                    current.updateEstimatorUnivariate(y, w);
                    return;
                } else if (attVal < current.attVal) {
                    current.updateEstimatorUnivariate(y, w);
                    current = current.left;
                    isRight = false;
                } else {
                    current = current.right;
                    isRight = true;
                }
            }
            EBSTNode newNode = new EBSTNode(attVal, y, w);
            if (isRight) 
                ante.right = newNode;
            else 
                ante.left = newNode;
        }

        void insertValue(double attVal, Map<Integer, Double> target, double w) {
            EBSTNode current = this, ante = null;
            boolean isRight = false;
            while (current != null) {
                ante = current;
                if (attVal == current.attVal) {
                    current.updateEstimatorMultivariate(target, w);
                    return;
                } else if (attVal < current.attVal) {
                    current.updateEstimatorMultivariate(target, w);
                    current = current.left;
                    isRight = false;
                } else {
                    current = current.right;
                    isRight = true;
                }
            }
            EBSTNode newNode = new EBSTNode(attVal, target, w);
            if (isRight) 
                ante.right = newNode;
            else         
                ante.left = newNode;
        }
    }

    //endregion === EBST NODE ===

    //region === LINEAR MODEL  (SGD, aligned to River's LinearRegression defaults) ===

    public static class LinearModel {
        private static final double CLIP_GRADIENT = 1e12;
        private static final double INTERCEPT_LR  = 0.01;

        private final Map<Integer, Double> weights  = new HashMap<>();
        private final Map<Integer, Double> cumL1map = new HashMap<>();
        private double maxCumL1 = 0;
        private double bias;
        private final double lr, l2, l1;

        public LinearModel(double lr, double l2, double l1) {
            this.lr = lr;
            this.l2 = l2;
            this.l1 = l1;
        }

        public int getNumWeights() { return weights.size(); }

        public LinearModel clone() {
            LinearModel m = new LinearModel(lr, l2, l1);
            m.weights.putAll(weights);
            m.cumL1map.putAll(cumL1map);
            m.maxCumL1 = maxCumL1;
            m.bias = bias;
            return m;
        }

        // Instance is already normalized upstream by normalizeInstance() — use values directly.
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            double p = bias;
            for (int j = 0; j < inst.numValues(); j++) {
                int i = inst.index(j);
                if (i == inst.classIndex() || inst.attribute(i).isNominal())
                    continue;
                p += weights.getOrDefault(i, 0.0) * inst.valueSparse(j);
            }
            return p;
        }

        public void update(Instance inst, double w, HoeffdingAdaptiveTreeRegressor tree) {
            double rawGradient = (predict(inst, tree) - inst.classValue()) * w;
            double gradient = Math.max(-CLIP_GRADIENT, Math.min(CLIP_GRADIENT, rawGradient));

            for (int j = 0; j < inst.numValues(); j++) {
                int i = inst.index(j);
                if (i == inst.classIndex() || inst.attribute(i).isNominal())
                    continue;
                double x = inst.valueSparse(j);
                double wi = weights.getOrDefault(i, 0.0);
                weights.put(i, wi - lr * (gradient * x + l2 * wi));
            }
            bias -= INTERCEPT_LR * gradient;

            if (l1 > 0) {
                maxCumL1 += l1 * lr;
                for (int j = 0; j < inst.numValues(); j++) {
                    int i = inst.index(j);
                    if (i == inst.classIndex() || inst.attribute(i).isNominal()) continue;
                    double wOld = weights.getOrDefault(i, 0.0);
                    double wNew = wOld;
                    if (wOld > 0)
                        wNew = Math.max(0.0, wOld - (maxCumL1 + cumL1map.getOrDefault(i, 0.0)));
                    else if (wOld < 0)
                        wNew = Math.min(0.0, wOld + (maxCumL1 - cumL1map.getOrDefault(i, 0.0)));
                    weights.put(i, wNew);
                    cumL1map.merge(i, wNew - wOld, Double::sum);
                }
            }
        }
    }

    //endregion === LINEAR MODEL  (SGD, aligned to River's LinearRegression defaults) ===

}
