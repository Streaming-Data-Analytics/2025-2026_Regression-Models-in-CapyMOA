package moa.classifiers.trees;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;
import moa.core.SizeOf;

import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

    // no leaf model, è un LinearRegressor all'interno di questa classe 

    public FloatOption modelSelectorDecayOption = new FloatOption(
            "modelSelectorDecay", 'q',
            "Exponential decay factor for FMSE tracking in ADAPTIVE leaf mode.",
            0.95, 0.0, 1.0);
    
    // lista nominal attributes ??

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

    public FloatOption adwinDeltaOption = new FloatOption(
            "adwinDelta", 'a',
            "Delta parameter for all ADWIN drift detectors.",
            0.002, 0.0, 1.0);


    //endregion === OPTIONS (MOA CLI / CapyMOA configuration) ===

    protected Node    root;
    protected double  trainWeightSeen  = 0;
    protected int     nActiveLeaves    = 0;
    protected int     nInactiveLeaves  = 0;
    protected boolean growthAllowed    = true;
    private   double  sizeEstimateOverhead     = 1.0;
    private   double  activeLeafSizeEstimate   = 0.0;
    private   double  inactiveLeafSizeEstimate = 0.0;

    //region === Convenience fields read from options at reset time ===
    protected int            gracePeriod;
    protected double         delta;
    protected double         tau;
    protected int            minSamplesSplit;
    protected int            driftWindowThreshold;
    protected double         switchSignificance;
    protected LeafPrediction leafPrediction;
    protected double         learningRate;
    protected double         l2;
    protected double         l1;
    protected double         modelSelectorDecay;
    protected boolean        binarySplit;
    protected boolean        bootstrapSampling;
    protected int            tebstDigits;
    protected int            maxDepth;
    protected boolean        removePoorAttrs;
    protected boolean        stopMemManagement;
    protected double         maxSizeMiB;
    protected int            memoryEstimatePeriod;
    protected double         adwinDelta;
    protected boolean        meritPreprune;
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
        maxDepth = maxDepthOption.getValue() == 0 ? Integer.MAX_VALUE : maxDepthOption.getValue();
        removePoorAttrs = removePoorAttrsOption.isSet();
        stopMemManagement = stopMemManagementOption.isSet();
        maxSizeMiB = maxSizeMiBOption.getValue();
        memoryEstimatePeriod = memoryEstimatePeriodOption.getValue();
        adwinDelta = adwinDeltaOption.getValue();
        meritPreprune = !noPrePruneOption.isSet();

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
            new Measurement("active leaves",   nActiveLeaves),
            new Measurement("inactive leaves", nInactiveLeaves),
        };
    }

    // TRAINING AND PREDICTION

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainWeightSeen += inst.weight();
        if (root == null) {
            root = newLeaf(0);
            nActiveLeaves = 1;
        }
        root.learn(inst, this, null, -1);

        if ((long) trainWeightSeen % memoryEstimatePeriod == 0) {
            estimateModelSize();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (root == null) 
            return new double[]{0.0};
        List<Node> leaves = new ArrayList<>();
        root.collectLeaves(inst, leaves);
        if (leaves.isEmpty()) 
            return new double[]{0.0};
        double sum = 0;
        for (Node leaf : leaves) 
            sum += leaf.predict(inst, this);
        return new double[]{sum / leaves.size()};
    }

    //endregion === AbstractClassifier INTERFACE ===

    //region === MEMORY MANAGEMENT ===

    private void estimateModelSize() {
        
    }

    void enforceTreeSizeLimit() {
        
    }

    //endregion === MEMORY MANAGEMENT ===

    //region === STATIC HELPERS ===

    public static double hoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt((range * range * Math.log(1.0 / confidence)) / (2.0 * n));
    }

    /**
     * Normalizes an absolute prediction error to [0,1] using the 3σ empirical rule.
     * Required because MOA's ADWIN expects bounded input in [0,1].
     */
    private static double normalizeForADWIN(double error, ErrorEstimator stats) {
        if (stats.getCount() < 2) 
            return 0.0;
        double mean = stats.getMean();
        double std  = Math.sqrt(Math.max(0, stats.getVariance()));
        if (std < 1e-10) 
            return 0.0;
        double lo = mean - 3.0 * std, hi = mean + 3.0 * std;
        return Math.min(1.0, Math.max(0.0, (error - lo) / (hi - lo)));
    }

    private static int poisson(double rate, java.util.Random rng) {
        double L = Math.exp(-rate);
        int k = 0; double p = 1.0;
        do { 
            k++; p *= rng.nextDouble(); 
        } while (p > L);
        return k - 1;
    }

    //endregion === STATIC HELPERS ===

    //region === NODE BASE ===

    public abstract static class Node {
        protected int depth;

        public Node(int depth) { this.depth = depth; }

        public abstract void learn(Instance inst, HoeffdingAdaptiveTreeRegressor tree,
                                     Node parent, int parentBranch);
        public abstract double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree);
        public ErrorEstimator getErrorTracker() { return null; }
        public abstract void collectLeaves(Instance inst, List<Node> result);
        public abstract void collectAllActiveLeaves(List<LeafNode> result);
        public abstract void collectAllLeaves(List<LeafNode> result);
    }

    //endregion === NODE BASE ===


    //region === LEAF NODE ===

    // LEAF NODE  (abstract base — mirrors River's AdaLeafRegressor)
    //
    // Contains the single shared learn() that all leaf types inherit, identical
    // to River's AdaLeafRegressor.learn_one: bootstrap sampling, drift detection
    // via ADWIN, error tracking, then delegates to afterUpdate() for the
    // subclass-specific part (leafModel update and/or FMSE), and finally
    // calls attemptSplit().
    //
    // Each concrete subclass only overrides:
    //   - prediction()   → what to return as a prediction (mean / model / adaptive)
    //   - afterUpdate()  → what extra work to do after stats are updated
    //                      (no-op for MEAN, model update for MODEL/ADAPTIVE)

    public abstract static class LeafNode extends Node {

        protected double sumY = 0;
        protected double sumYSq = 0;
        protected double weightSeen = 0;
        protected double weightSeenAtLastSplitEval = 0;

        // ADWIN + error tracker live here because learn() uses them
        protected ADWINDetector driftDetector;
        protected ErrorEstimator errorTracker;

        /** Per-attribute TEBST splitters; null → leaf is inactive. */
        protected Map<Integer, TEBSTSplitter> splitters = null;
        protected Map<Integer, NominalSplitter> nominalSplitters = null;
        protected Set<Integer> disabledAttrs = new HashSet<>();

        // Stored locally so updateStatsBase can use it without an outer-class reference.
        private int tebstDigits;

        public LeafNode(int depth, double adwinDelta, int tebstDigits) {
            super(depth);
            this.tebstDigits = tebstDigits;
            this.driftDetector = new ADWINDetector(adwinDelta);
            this.errorTracker = new ErrorEstimator();
            this.splitters = new HashMap<>();
            this.nominalSplitters = new HashMap<>();
        }

        @Override public ErrorEstimator getErrorTracker() { return errorTracker; }

        public boolean isActive() { return splitters != null; }
        public void activate() {
            if (splitters == null) { splitters = new HashMap<>(); nominalSplitters = new HashMap<>(); }
        }
        public void deactivate() { splitters = null; nominalSplitters = null; }

        public double calculatePromise() { return -depth; }

        public long estimateByteSize() { return SizeOf.fullSizeOf(this); }

        public void disableAttribute(int attrIdx) {
            if (splitters != null) splitters.remove(attrIdx);
            if (nominalSplitters != null) nominalSplitters.remove(attrIdx);
            disabledAttrs.add(attrIdx);
        }

        // Updates target statistics and splitters only — no leafModel.
        // Mirrors River's LeafMean.learn_one (the deepest super() in the MRO chain).
        private void updateStatsBase(Instance inst, double w) {
            double y = inst.classValue();
            sumY += w * y;
            sumYSq += w * y * y;
            weightSeen += w;
            if (isActive()) {
                int numAttrs = inst.numAttributes() - 1;
                for (int i = 0; i < numAttrs; i++) {
                    if (disabledAttrs.contains(i)) continue;
                    if (inst.attribute(i).isNominal()) {
                        nominalSplitters.computeIfAbsent(i, k -> new NominalSplitter()).update((int) inst.value(i), y, w);
                    } else {
                        splitters.computeIfAbsent(i, k -> new TEBSTSplitter(tebstDigits)).update(inst.value(i), y, w);
                    }
                }
            }
        }

        public double getMean() { return weightSeen > 0 ? sumY / weightSeen : 0; }
        public double getVariance() {
            if (weightSeen < 2) 
                return 0;
            return Math.max(0, (sumYSq - sumY * sumY / weightSeen) / (weightSeen - 1));
        }

        @Override public void collectLeaves(Instance inst, List<Node> result) { result.add(this); }
        @Override public void collectAllActiveLeaves(List<LeafNode> result) { if (isActive()) result.add(this); }
        @Override public void collectAllLeaves(List<LeafNode> result) { result.add(this); }

        /**
         * Hook called after updateStatsBase() in the shared learn().
         * Subclasses override to update the leaf model and/or FMSE trackers.
         * Default is a no-op (AdaLeafMean needs nothing extra).
         *
         * @param preMean  target mean captured BEFORE updateStatsBase() ran
         */
        protected void afterUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {}

        /**
         * Single shared learn() — mirrors River's AdaLeafRegressor.learn_one.
         *
         * All leaf types inherit this unchanged; they only differ in prediction()
         * and afterUpdate().
         */
        @Override
        public final void learn(Instance inst, HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch) {

            double y = inst.classValue();
            double y_pred = predict(inst, tree);   // pre-update prediction

            // Bootstrap sampling (River: w *= Poisson(1) if k > 0)
            double w = inst.weight();
            if (tree.bootstrapSampling) {
                int k = poisson(1.0, tree.classifierRandom);
                if (k > 0) w *= k;
            }

            // Drift detection + error tracking
            double error = Math.abs(y - y_pred);
            double oldMean = errorTracker.getMean();
            driftDetector.update(normalizeForADWIN(error, errorTracker));
            errorTracker.update(error);
            if (driftDetector.detectedChange() && errorTracker.getMean() < oldMean)
                errorTracker.reset();

            // Capture pre-update mean BEFORE stats change (needed by AdaLeafAdaptive)
            double preMean = getMean();

            // Stats + splitters update (River: LeafMean.learn_one via super chain)
            updateStatsBase(inst, w);

            // Subclass hook: model update, FMSE, etc.
            afterUpdate(inst, w, y, preMean, tree);

            if (tree.growthAllowed) attemptSplit(tree, parent, parentBranch);
        }

        protected void attemptSplit(HoeffdingAdaptiveTreeRegressor tree,Node parent, int parentBranch) {
            if (!isActive()) 
                return;
            if (weightSeen - weightSeenAtLastSplitEval < tree.gracePeriod) 
                return;

            // Depth-based pre-pruning — mirrors River: checked INSIDE the grace-period
            // block, so deactivation only happens every grace_period instances.
            if (depth >= tree.maxDepth) {
                deactivate(); 
                tree.nActiveLeaves--; 
                tree.nInactiveLeaves++; 
                return;
            }
            weightSeenAtLastSplitEval = weightSeen;

            double parentVariance = getVariance();
            if (parentVariance <= 0) 
                return;

            double bestVR = Double.NEGATIVE_INFINITY;
            double secondVR = Double.NEGATIVE_INFINITY;
            int    bestAttr = -1;
            double bestThresh = 0;
            int    bestBinaryIdx = -1;
            boolean bestIsNominal = false;
            int    nCandidates = 0;
            double bestLeftSumY = 0, bestLeftSumYSq = 0, bestLeftWeight = 0;
            Map<Integer, double[]> perAttrBest = new HashMap<>();

            for (Map.Entry<Integer, TEBSTSplitter> entry : splitters.entrySet()) {
                int attrIdx = entry.getKey();
                double[] res = entry.getValue().bestSplit(parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplit);
                if (res == null) continue;
                nCandidates++;
                perAttrBest.put(attrIdx, res);
                double vr = res[1];
                if (vr > bestVR) {
                    secondVR = bestVR; bestVR = vr;
                    bestAttr = attrIdx; bestThresh = res[0]; bestIsNominal = false;
                    bestLeftSumY = res[3]; bestLeftSumYSq = res[4]; bestLeftWeight = res[5];
                } else if (vr > secondVR) { secondVR = vr; }
            }

            for (Map.Entry<Integer, NominalSplitter> entry : nominalSplitters.entrySet()) {
                int attrIdx = entry.getKey();
                double[] res = tree.binarySplit ? entry.getValue().bestBinarySplit(parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplit) : entry.getValue().bestSplit(parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplit);
                if (res == null) continue;
                nCandidates++;
                double vr = res[1];
                if (vr > bestVR) {
                    secondVR = bestVR; bestVR = vr;
                    bestAttr = attrIdx; bestIsNominal = true;
                    bestBinaryIdx = tree.binarySplit ? (int) res[0] : -1;
                    if( tree.binarySplit) {
                        bestLeftSumY = res[2]; bestLeftSumYSq = res[3]; bestLeftWeight = res[4];
                    }
                } else if (vr > secondVR) { 
                    secondVR = vr; 
                }
            }

            // Mirror River: merit_preprune adds a null split (VR=0) as candidate.
            // If no real candidate improves variance, the null split wins → deactivate.
            // With merit_preprune=False no null split is added, so this path never fires.
            if (nCandidates == 0 || bestVR <= 0) {
                if (tree.meritPreprune) {
                    deactivate(); tree.nActiveLeaves--; tree.nInactiveLeaves++;
                    tree.enforceTreeSizeLimit();
                }
                return;
            }

            double epsilon = hoeffdingBound(1.0, tree.delta, weightSeen);
            boolean shouldSplit;
            if (nCandidates == 1) {
                shouldSplit = bestVR > 0;
            } else {
                shouldSplit = bestVR > 0 && (
                        (secondVR / bestVR < 1.0 - epsilon) || (epsilon < tree.tau));
            }

            if (shouldSplit) {
                AdaSplitNode newSplit;
                int numBranches;
                if (bestIsNominal) {
                    if (tree.binarySplit) {
                        int[][] parts = nominalSplitters.get(bestAttr).getBinarySplitCategories(bestBinaryIdx);
                        Set<Integer> leftSet  = new HashSet<>();
                        Set<Integer> rightSet = new HashSet<>();
                        for (int c : parts[0]) leftSet.add(c);
                        for (int c : parts[1]) rightSet.add(c);
                        newSplit = new AdaNomBinarySplitNode(depth, bestAttr, leftSet, rightSet, tree.adwinDelta);
                        numBranches = 2;
                    } else {
                        int[] cats = nominalSplitters.get(bestAttr).getSortedCategories();
                        newSplit = new AdaNomMultiwaySplitNode(depth, bestAttr, cats, tree.adwinDelta);
                        numBranches = cats.length;
                    }
                } else {
                    newSplit = new AdaNumBinarySplitNode(depth, bestAttr, bestThresh, tree.adwinDelta);
                    numBranches = 2;
                }
                if (bestIsNominal && !tree.binarySplit) {
                    // multiway: una branch per categoria, stats dirette da catStats
                    int[] cats = nominalSplitters.get(bestAttr).getSortedCategories();
                    double[][] catStats = nominalSplitters.get(bestAttr).getChildrenStats(cats);

                    for (int b = 0; b < numBranches; b++) {
                        newSplit.children[b] = tree.newLeaf(
                            depth + 1,
                            catStats[b][0],
                            catStats[b][1],
                            catStats[b][2]
                        );
                    }
                } else {
                    // binario (numerico o nominal binary): branch 0 = left, branch 1 = right
                    double rightSumY = sumY - bestLeftSumY;
                    double rightSumYSq = sumYSq - bestLeftSumYSq;
                    double rightWeight = weightSeen - bestLeftWeight;

                    newSplit.children[0] = tree.newLeaf(
                        depth + 1,
                        bestLeftSumY,
                        bestLeftSumYSq,
                        bestLeftWeight
                    );

                    newSplit.children[1] = tree.newLeaf(
                        depth + 1,
                        rightSumY,
                        rightSumYSq,
                        rightWeight
                    );
                }

                if (parent == null) tree.root = newSplit;
                else ((SplitNode) parent).children[parentBranch] = newSplit;

                tree.nActiveLeaves--;
                tree.nActiveLeaves += numBranches;
                tree.enforceTreeSizeLimit();

            } else if (nCandidates >= 2 && bestVR > 0 && secondVR > 0) {
                double secondRatio = secondVR / bestVR;
                for (Map.Entry<Integer, double[]> entry : perAttrBest.entrySet()) {
                    int attrIdx = entry.getKey();
                    double vr   = entry.getValue()[1];
                    TEBSTSplitter s = splitters.get(attrIdx);
                    if (s == null) continue;
                    s.removeBadSplits(secondRatio, bestVR, epsilon, parentVariance, sumY, sumYSq, weightSeen);
                    if (tree.removePoorAttrs && vr / bestVR < secondRatio - 2 * epsilon)
                        disableAttribute(attrIdx);
                }
                if (tree.removePoorAttrs) {
                    for (Map.Entry<Integer, NominalSplitter> entry : nominalSplitters.entrySet()) {
                        int attrIdx = entry.getKey();
                        double[] res = tree.binarySplit ? entry.getValue().bestBinarySplit(parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplit) : entry.getValue().bestSplit(parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplit);
                        if (res == null) continue;
                        double vr = res[1];
                        if (vr / bestVR < secondRatio - 2 * epsilon)
                            disableAttribute(attrIdx);
                    }
                }
            }
        }
    }

    //endregion === LEAF NODE ===

    //region === ADA LEAF MEAN ===

    // ADA LEAF — MEAN  (mirrors River's AdaLeafRegMean = AdaLeafRegressor + LeafMean)
    //
    // Inherits learn() from LeafNode unchanged.
    // afterUpdate() not overridden -> no-op (no leafModel to update).

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

    // ADA LEAF — MODEL  (mirrors River's AdaLeafRegModel = AdaLeafRegressor + LeafModel)
    //
    // Extends AdaLeafMean (mirrors River: LeafModel extends LeafMean).
    // Inherits learn() from LeafNode unchanged.
    // afterUpdate() initialises and updates the LinearModel after stats are updated.

    public static class AdaLeafModel extends AdaLeafMean {

        protected LinearModel leafModel;

        public AdaLeafModel(int depth, double adwinDelta, int tebstDigits) {
            super(depth, adwinDelta, tebstDigits);
        }

        /** River: LeafModel.prediction → leaf_model.predict_one(x). */
        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            return leafModel != null ? leafModel.predict(inst) : getMean();
        }

        /**
         * River: LeafModel.learn_one → leaf_model.learn_one(x, y, w).
         * Stats are already updated by the time this hook runs.
         */
        @Override
        protected void afterUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {
            if (leafModel == null)
                leafModel = new LinearModel(tree.learningRate, tree.l2, tree.l1);
            leafModel.update(inst, w);
        }
    }

    //endregion === ADA LEAF MEAN 

    //region === ADA LEAF ADAPTIVE ===

    // ADA LEAF — ADAPTIVE  (mirrors River's AdaLeafRegAdaptive = AdaLeafRegressor + LeafAdaptive)
    //
    // Extends AdaLeafModel (mirrors River: LeafAdaptive extends LeafModel).
    // Inherits learn() from LeafNode and leafModel field from AdaLeafModel.
    // afterUpdate() captures preModel BEFORE calling super.afterUpdate() (which
    // updates leafModel), then computes FMSE — matching River's LeafAdaptive.learn_one order.

    public static class AdaLeafAdaptive extends AdaLeafModel {

        private double fmseMean  = 0.0;
        private double fmseModel = 0.0;

        public AdaLeafAdaptive(int depth, double adwinDelta, int tebstDigits) {
            super(depth, adwinDelta, tebstDigits);
        }

        /**
         * River: LeafAdaptive.prediction.
         *   fmse_mean < fmse_model -> mean wins  (regression tree mode)
         *   otherwise -> super.prediction() (LeafModel: leafModel or mean)
         */
        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            if (fmseMean < fmseModel) return getMean();
            return super.predict(inst, tree);
        }

        /**
         * River: LeafAdaptive.learn_one captures pre-update predictions, then
         * calls super().learn_one() which updates leafModel (LeafModel.learn_one).
         *
         * Here: capture preModel BEFORE super.afterUpdate() updates leafModel,
         * then update FMSE with pre-update predictions.
         */
        @Override
        protected void afterUpdate(Instance inst, double w, double y, double preMean, HoeffdingAdaptiveTreeRegressor tree) {
            double preModel = (leafModel != null) ? leafModel.predict(inst) : preMean;

            super.afterUpdate(inst, w, y, preMean, tree);  // initialises + updates leafModel

            double d = tree.modelSelectorDecay;
            fmseMean  = d * fmseMean  + (y - preMean)  * (y - preMean);
            fmseModel = d * fmseModel + (y - preModel) * (y - preModel);
        }
    }
    //endregion === ADA LEAF ADAPTIVE ===

    
    //region === SPLIT NODE ===

    public abstract static class SplitNode extends Node {

        protected Node[]   children;
        protected double[] childVisits;

        public SplitNode(int numChildren, int depth) {
            super(depth);
            this.children    = new Node[numChildren];
            this.childVisits = new double[numChildren];
        }

        public abstract int getBranchIndex(Instance inst);

        protected int mostCommonPath() {
            int best = 0;
            for (int i = 1; i < childVisits.length; i++)
                if (childVisits[i] > childVisits[best]) 
                    best = i;
            return best;
        }

        public Node routeChild(Instance inst) {
            int branch = getBranchIndex(inst);
            if (branch < 0 || branch >= children.length) branch = mostCommonPath();
            return children[branch];
        }

        @Override
        public void collectAllActiveLeaves(List<LeafNode> result) {
            for (Node child : children) 
                if (child != null) 
                    child.collectAllActiveLeaves(result);
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

        protected ADWINDetector  driftDetector;
        protected ErrorEstimator errorTracker;
        protected Node           alternateTree;

        protected double branchSumY = 0, branchSumYSq = 0, branchWeight = 0;

        public AdaSplitNode(int numChildren, int depth, double adwinDelta) {
            super(numChildren, depth);
            this.driftDetector = new ADWINDetector(adwinDelta);
            this.errorTracker  = new ErrorEstimator();
        }

        @Override public ErrorEstimator getErrorTracker() { return errorTracker; }

        public void collectAllBranchNodes(List<AdaSplitNode> result) {
            result.add(this);
            for (Node child : children)
                if (child instanceof AdaSplitNode)
                    ((AdaSplitNode) child).collectAllBranchNodes(result);
            if (alternateTree instanceof AdaSplitNode)
                ((AdaSplitNode) alternateTree).collectAllBranchNodes(result);
        }

        public double getBranchMean() { return branchWeight > 0 ? branchSumY / branchWeight : 0; }

        @Override
        public void collectLeaves(Instance inst, List<Node> result) {
            if (alternateTree != null) alternateTree.collectLeaves(inst, result);
            Node child = routeChild(inst);
            if (child != null) child.collectLeaves(inst, result);
        }

        @Override
        public void collectAllActiveLeaves(List<LeafNode> result) {
            super.collectAllActiveLeaves(result);
            if (alternateTree != null) 
                alternateTree.collectAllActiveLeaves(result);
        }

        @Override
        public void collectAllLeaves(List<LeafNode> result) {
            super.collectAllLeaves(result);
            if (alternateTree != null) 
                alternateTree.collectAllLeaves(result);
        }

        @Override
        public void learn(Instance inst, HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch) {

            
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
            Node child = routeChild(inst);
            return (child != null) ? child.predict(inst, tree) : getBranchMean();
        }

        private void compareAlternate(HoeffdingAdaptiveTreeRegressor tree, Node parent, int parentBranch) {
            
        }

        private void killSubtree(Node n, HoeffdingAdaptiveTreeRegressor tree) {
            
        }
    }

    //endregion === ADA SPLIT NODE ===

    //region === FACTORY METHODS ===

    /**
     * Creates the correct leaf type based on leafPrediction — mirrors River's
     * _new_leaf() which instantiates AdaLeafRegMean / AdaLeafRegModel / AdaLeafRegAdaptive.
     */
    protected LeafNode newLeaf(int depth) {
        switch (leafPrediction) {
            case MEAN: return new AdaLeafMean(depth, adwinDelta, tebstDigits);
            case MODEL: return new AdaLeafModel(depth, adwinDelta, tebstDigits);
            default: return new AdaLeafAdaptive(depth, adwinDelta, tebstDigits);
        }
    }

    protected LeafNode newLeaf(int depth, double initSumY, double initSumYSq, double initWeight) {
        LeafNode leaf = newLeaf(depth);
        leaf.sumY = initSumY;
        leaf.sumYSq = initSumYSq;
        leaf.weightSeen = initWeight;
        leaf.weightSeenAtLastSplitEval = initWeight;
        return leaf;
    }

    //endregion === FACTORY METHODS ===

    //region === CONCRETE ADA SPLIT NODE SUBCLASSES ===
    // (mirror River's AdaNumBinaryBranchReg, AdaNomBinaryBranchReg, AdaNomMultiwayBranchReg)

    public static class AdaNumBinarySplitNode extends AdaSplitNode {
        private final int    attrIdx;
        private final double threshold;

        public AdaNumBinarySplitNode(int depth, int attrIdx, double threshold, double adwinDelta) {
            super(2, depth, adwinDelta);
            this.attrIdx   = attrIdx;
            this.threshold = threshold;
        }

        @Override
        public int getBranchIndex(Instance inst) {
            if (inst.isMissing(attrIdx)) return -1;
            return inst.value(attrIdx) <= threshold ? 0 : 1;
        }
    }

    public static class AdaNomBinarySplitNode extends AdaSplitNode {
        private final int          attrIdx;
        private final Set<Integer> leftCats;
        private final Set<Integer> rightCats;

        public AdaNomBinarySplitNode(int depth, int attrIdx, Set<Integer> leftCats, Set<Integer> rightCats, double adwinDelta) {
            super(2, depth, adwinDelta);
            this.attrIdx   = attrIdx;
            this.leftCats  = leftCats;
            this.rightCats = rightCats;
        }

        @Override
        public int getBranchIndex(Instance inst) {
            if (inst.isMissing(attrIdx)) return -1;
            int cat = (int) inst.value(attrIdx);
            if (leftCats.contains(cat))  return 0;
            if (rightCats.contains(cat)) return 1;
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

        @Override
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

    //endregion === CONCRETE ADA SPLIT NODE SUBCLASSES ===

    //region === NOMINAL SPLITTER ====

    public static class NominalSplitter {

        private final Map<Integer, double[]> catStats = new HashMap<>();

        public void update(int catIndex, double y, double w) {
        }

        public double[] bestSplit(double parentVariance, double totalSumY, double totalSumYSq, double totalCount, int minSamplesSplit) {
            return new double[]{};
        }

        public int[] getSortedCategories() {
            int[] cats = new int[catStats.size()];
            int i = 0;
            for (int k : catStats.keySet()) cats[i++] = k;
            java.util.Arrays.sort(cats);
            return cats;
        }

        public double[] bestBinarySplit(double parentVariance, double totalSumY, double totalSumYSq, double totalCount, int minSamplesSplit) {
            return new double[]{};
        }

        public int[][] getBinarySplitCategories(int splitIndex) {
            return new int[][]{};
        }

        public double[][] getChildrenStats(int[] sortedCats) {
            return new double[sortedCats.length][];
        }
    }

    //endregion === NOMINAL SPLITTER ===

    //region === TEBST SPLITTER ===

    public static class TEBSTSplitter {

        private final double roundFactor;
        private EBSTNode root = null;

        public TEBSTSplitter(int digits) { this.roundFactor = Math.pow(10, digits); }

        private double round(double v) { return Math.round(v * roundFactor) / roundFactor; }

        public void update(double attVal, double y, double w) {
            attVal = round(attVal);
            if (root == null) 
                root = new EBSTNode(attVal, y, w);
            else              
                root.insertValue(attVal, y, w);
        }

        public double[] bestSplit(double parentVariance, double totalSumY, double totalSumYSq, double totalCount, int minSamplesSplit) {
            if (root == null || totalCount < 2 * minSamplesSplit) return null;
            double[] result = {Double.NaN, 0.0, 0.0, 0.0, 0.0, 0.0}; // result[3..5] sono le left-branch stats al momento del best threshold trovato.
            double[] aux    = {0.0, 0.0, 0.0};
            findBestSplit(root, result, aux, parentVariance, totalSumY, totalSumYSq, totalCount, minSamplesSplit);
            return Double.isNaN(result[0]) ? null : result;
        }

        private void findBestSplit(EBSTNode node, double[] result, double[] aux, double pVar, double tY, double tYSq, double tN, int minSplit) {
        }

        public void removeBadSplits(double lastCheckRatio, double lastCheckVR, double epsilon, double parentVariance, double totalSumY, double totalSumYSq, double totalCount) {
            if (root == null || lastCheckVR <= 0) return;
            double[] aux = {0.0, 0.0, 0.0};
            root = removeNode(root, aux, lastCheckRatio, lastCheckVR, epsilon, parentVariance, totalSumY, totalSumYSq, totalCount);
        }

        private EBSTNode removeNode(EBSTNode node, double[] aux, double ratio, double bestVR, double eps, double pVar, double tY, double tYSq, double tN) {
            if (node == null) return null;
            return node;
        }
    }

    //endregion === TEBST SPLITTER ===

    //region === EBST NODE ====

    public static class EBSTNode {
        double   attVal;
        double   sumY, sumYSq, count;
        EBSTNode left, right;

        EBSTNode(double attVal, double y, double w) {
            this.attVal = attVal; this.sumY = w * y; this.sumYSq = w * y * y; this.count = w;
        }

        void insertValue(double attVal, double y, double w) {
            
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
            this.lr = lr; this.l2 = l2; this.l1 = l1;
        }

        public int getNumWeights() { return weights.size(); }

        public double predict(Instance inst) {
            double p = bias;
            for (int j = 0; j < inst.numValues(); j++) {
                int i = inst.index(j);
                if (i == inst.classIndex() || inst.attribute(i).isNominal()) 
                    continue;
                p += weights.getOrDefault(i, 0.0) * inst.valueSparse(j);
            }
            return p;
        }

        public void update(Instance inst, double w) {
            double rawGradient = (predict(inst) - inst.classValue()) * w;
            double gradient    = Math.max(-CLIP_GRADIENT, Math.min(CLIP_GRADIENT, rawGradient));

            for (int j = 0; j < inst.numValues(); j++) {
                int i = inst.index(j);
                if (i == inst.classIndex() || inst.attribute(i).isNominal()) 
                    continue;
                double wi = weights.getOrDefault(i, 0.0);
                weights.put(i, wi - lr * (gradient * inst.valueSparse(j) + l2 * wi));
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

    //region === ADWIN DETECTOR ===

    public interface DriftDetector {
        void update(double value);
        boolean detectedChange();
        void reset();
    }

    public static class ADWINDetector implements DriftDetector {
        private final double delta;
        private ADWIN adwin;
        private boolean changed = false;

        public ADWINDetector(double delta) { this.delta = delta; this.adwin = new ADWIN(delta); }

        @Override public void update(double v) { changed = adwin.setInput(v); }
        @Override public boolean detectedChange() { return changed; }
        @Override public void reset() { adwin = new ADWIN(delta); changed = false; }
    }

    //endregion === ADWIN DETECTOR ===
 
    //region === ERROR ESTIMATOR  (Welford online mean + variance) ===

    public static class ErrorEstimator {
        private double mean = 0, M2 = 0;
        private int n = 0;

        public void update(double v) {
            n++;
            double delta = v - mean; mean += delta / n; M2 += delta * (v - mean);
        }

        public double getMean() { return mean; }
        public double getVariance() { return n < 2 ? 0 : Math.max(0, M2 / (n - 1)); }
        public int    getCount() { return n; }
        public void   reset() { mean = 0; M2 = 0; n = 0; }
    }

    //endregion === ERROR ESTIMATOR  (Welford online mean + variance) ===
}
