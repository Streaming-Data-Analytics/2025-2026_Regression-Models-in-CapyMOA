package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.MultiChoiceOption;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.distribution.NormalDistribution;

public class HoeffdingAdaptiveTreeRegressor extends AbstractClassifier implements Regressor {
    private static final long serialVersionUID = 1L;

    // all parameters in River's HoeffdingTree
    protected Node root;
    protected double trainWeightSeen = 0; 
    protected int nActiveLeaves = 0; 
    protected int nInactiveLeaves = 0; 
    protected boolean growthAllowed = true; 
    private double sizeEstimateOverhead = 1.0; // fattore di correzione usato per stimare meglio quanta memoria occupa davvero l’albero. Parte da 1.0 perché all’inizio non sa ancora quanto la stima sia imprecisa, quindi assume temporaneamente che la stima sia perfetta.
    private   double  activeLeafSizeEstimate   = 0.0;
    private   double  inactiveLeafSizeEstimate = 0.0;

    //region === OPTIONS ===
    public IntOption maxDepthOption = new IntOption(
        "maxDepth", 'x',
        "Maximum tree depth. 0 means unlimited.",
        0, 0, Integer.MAX_VALUE); 
    
    public FlagOption binarySplitOption = new FlagOption(
        "binarySplit", 'B',
        "Force binary splits for nominal features (River: binary_split=True). "
        + "Uses optimal binary partition (sort by mean, try all K-1 consecutive splits).");

    // max size, max byte size ? 

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
    
    public FlagOption meritPrepruneOption = new FlagOption(
            "meritPreprune", 'u',
            "Enable merit-based pre-pruning: deactivate leaf when no split candidate has positive merit.");
    
    // all parameters in River's HoeffdingTreeRegressor
    
    public IntOption gracePeriodOption = new IntOption(
            "gracePeriod", 'g',
            "Number of instances a leaf should observe between split attempts.",
            200, 1, Integer.MAX_VALUE);
    
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
    
    // splitter ??

    // all parameters in River's HoeffdingAdptiveTreeRegressor

    public FlagOption bootstrapSamplingOption = new FlagOption(
            "bootstrapSampling", 'b',
            "Enable Poisson bootstrap sampling at leaves (default: on).");
    
    public IntOption driftWindowThresholdOption = new IntOption(
            "driftWindowThreshold", 'w',
            "Minimum examples an alternate tree must observe before being considered for replacement.",
            300, 1, Integer.MAX_VALUE);
    
    // drift detector ??

    public FloatOption switchSignificanceOption = new FloatOption(
            "switchSignificance", 'z',
            "Significance level (p-value threshold) for the z-test when swapping alternate trees.",
            0.05, 0.0, 1.0);

    // seed ??

    public IntOption tebstDigitsOption = new IntOption(
            "tebstDigits", 'k',
            "Number of decimal digits for TEBST rounding (River default: 1).",
            1, 0, 10); 

    private static final NormalDistribution NORM = new NormalDistribution();

    //endregion === OPTIONS ===

    //region === METHODS ===

    public static double hoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt((range * range * Math.log(1.0 / confidence)) / (2.0 * n));
    }

    @Override
    public void resetLearningImpl() {
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
    public boolean isRandomizable() {
        return true; // MOA will manage classifierRandom and randomSeed
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("HoeffdingAdaptiveTreeRegressor\n");
        out.append("  active leaves   : ").append(nActiveLeaves).append("\n");
        out.append("  inactive leaves : ").append(nInactiveLeaves).append("\n");
        out.append("  instances seen  : ").append((long) trainWeightSeen).append("\n");
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainWeightSeen += inst.weight();
        if (root == null) {
		root = newLeaf(null, 0);
		nActiveLeaves = 1;
	}
        root.learn(inst, this, null, -1);
        if ((long) trainWeightSeen % memoryEstimatePeriodOption.getValue() == 0) {
                estimateModelSize();
	}
    }

    @Override
    public double[] getVotesForInstance(Instance inst) { // multi path, main + alternate 
        if (root == null) return new double[]{0.0};
        List<Node> leaves = new ArrayList<>();
        root.collectLeaves(inst, leaves);
        if (leaves.isEmpty()) return new double[]{0.0};
        double sum = 0;
        for (Node leaf : leaves) sum += leaf.predict(inst, this);
        return new double[]{sum / leaves.size()};
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{};
    }

    private void estimateModelSize() {}

    protected LeafNode newLeaf(Node parent, int depth) {
        switch (leafPredictionOption.getChosenIndex()) {
            case 0:  return new AdaLeafMean();
            case 1: return new AdaLeafModel();
            default: return new AdaLeafAdaptive();
        }
    }

    protected LeafNode newLeaf(Node parent, int depth, double initSumY, double initSumYSq, double initWeight){
        LeafNode leaf = newLeaf(parent,depth);
        leaf.sumY = initSumY;
        leaf.sumYSq = initSumYSq;
        leaf.weightSeen = initWeight;
        leaf.weightSeenAtLastSplitEval = initWeight;
        return leaf;

    }

    protected SplitNode newSplit(Node parent, InstanceConditionalTest t, int numChildren, int depth) {
        return new AdaSplitNode(t, numChildren);
    }

    private static int poisson(double rate, java.util.Random rng) {
        double L = Math.exp(-rate);
        int k = 0; double p = 1.0;
        do { k++; p *= rng.nextDouble(); } while (p > L);
        return k - 1;
    }

    private static double normalizeForADWIN(double error, ErrorEstimator stats) {
        if (stats.getCount() < 2) return 0.0;
        double mean = stats.getMean();
        double std  = Math.sqrt(Math.max(0, stats.getVariance()));
        if (std < 1e-10) return 0.0;
        double lo = mean - 3.0 * std, hi = mean + 3.0 * std;
        return Math.min(1.0, Math.max(0.0, (error - lo) / (hi - lo)));
    }
    
    //endregion === METHODS ===

    //region === CLASSES ===
    public abstract class Node {
        protected Node parent;
        protected int depth;

        public abstract void learn(
                Instance inst, 
                HoeffdingAdaptiveTreeRegressor tree, 
                Node parent, 
                int parentBranch
        );
        public abstract void collectLeaves(
                Instance inst, 
                List<Node> result
        );
        public abstract double predict(
                Instance inst, 
                HoeffdingAdaptiveTreeRegressor tree
        );
    }

    public abstract class LeafNode extends Node {

        protected double sumY = 0;
        protected double sumYSq = 0;
        protected double weightSeen = 0;
        protected double weightSeenAtLastSplitEval = 0;

        protected Map<Integer, TEBSTSplitter> splitters = null;
        protected Map<Integer, NominalSplitter> nominalSplitters = null;

        protected Set<Integer> disabledAttrs = new HashSet<>();
        
        protected ADWINDetector driftDetector;
        protected ErrorEstimator errorTracker;

        @Override
        public final void learn(
                Instance inst, 
                HoeffdingAdaptiveTreeRegressor tree, 
                Node parent, 
                int parentBranch
        ) 
        {
                double y = inst.classValue();
                double y_pred = predict(inst, tree);
                
                // bootstrap sampling 
                double w = inst.weight();
                if (tree.bootstrapSamplingOption.isSet()) {
                        int k = poisson(1.0, tree.classifierRandom);
                        if (k > 0) w *= k;
                }

                // Drift detection + error tracking
                double error   = Math.abs(y - y_pred);
                double oldMean = errorTracker.getMean();
                driftDetector.update(normalizeForADWIN(error, errorTracker));
                errorTracker.update(error);
                if (driftDetector.detectedChange() && errorTracker.getMean() < oldMean)
                        errorTracker.reset();

                double preMean = getMean();

                updateStatsBase(inst, w);

                // afterUpdate(inst, w, y, preMean, tree);

                if (tree.growthAllowed) attemptSplit(tree, parent, parentBranch);

        }

        private void updateStatsBase(Instance inst, double w) {
            double y = inst.classValue();
            sumY += w * y;
            sumYSq += w * y * y;
            weightSeen += w;
            // update splitters
            if(isActive()) {
                int numAttrs = inst.numAttributes() - 1;
                for (int i = 0; i < numAttrs; i++) {
                    if (disabledAttrs.contains(i)) continue;
                    if (inst.attribute(i).isNominal()) {
                        // nominal splitter
                        nominalSplitters
                        .computeIfAbsent(i, k -> new NominalSplitter())
                        .update((int) inst.value(i), y, w);
                    } else {
                        // tebst splitter
                        splitters
                        .computeIfAbsent(i, k -> new TEBSTSplitter(tebstDigitsOption.getValue()))
                        .update(inst.value(i), y, w);
                    }
                }
            }

        }

        public boolean isActive() {
                return splitters != null;
        }

        public void activate() {
            if (splitters == null) { 
                splitters = new HashMap<>(); 
                nominalSplitters = new HashMap<>();
            }
        }

        public void deactivate() { 
                splitters = null; 
                nominalSplitters = null; 
        }

        public double getMean() {
                return 0.0;
        }

        public double getVariance() {
            if (weightSeen < 2) return 0;
            return Math.max(0, (sumYSq - sumY * sumY / weightSeen) / (weightSeen - 1));
        }

        public void disableAttribute(int attrIdx) {
                
        }

        protected void attemptSplit(
                HoeffdingAdaptiveTreeRegressor tree,
                Node parent, int parentBranch
        ) {
                if (!isActive()) return;
                if (weightSeen - weightSeenAtLastSplitEval < tree.gracePeriodOption.getValue()) return;

                // Depth-based pre-pruning — mirrors River: checked INSIDE the grace-period
                // block, so deactivation only happens every grace_period instances.
                if (depth >= tree.maxDepthOption.getValue()) {
                        deactivate(); tree.nActiveLeaves--; tree.nInactiveLeaves++; return;
                }
                weightSeenAtLastSplitEval = weightSeen;

                // da qui in giù vedere HTR
                double parentVariance = getVariance();
                if (parentVariance <= 0) return;

                double bestVR        = Double.NEGATIVE_INFINITY;
                double secondVR      = Double.NEGATIVE_INFINITY;
                int    bestAttr      = -1;
                double bestThresh    = 0;
                int    bestBinaryIdx = -1;
                boolean bestIsNominal = false;
                int    nCandidates   = 0;
                double bestLeftSumY = 0, bestLeftSumYSq = 0, bestLeftWeight = 0;
                Map<Integer, double[]> perAttrBest = new HashMap<>();

                for (Map.Entry<Integer, TEBSTSplitter> entry : splitters.entrySet()) {
                        int attrIdx = entry.getKey();
                        double[] res = entry.getValue().bestSplit(
                                parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplitOption.getValue());
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
                        double[] res = tree.binarySplitOption.isSet()
                                ? entry.getValue().bestBinarySplit(
                                        parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplitOption.getValue())
                                : entry.getValue().bestSplit(
                                        parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplitOption.getValue());
                        if (res == null) continue;
                        nCandidates++;
                        double vr = res[1];
                        if (vr > bestVR) {
                                secondVR = bestVR; bestVR = vr;
                                bestAttr = attrIdx; bestIsNominal = true;
                                bestBinaryIdx = tree.binarySplitOption.isSet() ? (int) res[0] : -1;
                                if( tree.binarySplitOption.isSet()) {
                                        bestLeftSumY = res[2]; bestLeftSumYSq = res[3]; bestLeftWeight = res[4];
                                }
                        } else if (vr > secondVR) { secondVR = vr; }
                }

                if (nCandidates == 0) return;
                if (bestVR <= 0) {
                        // null split winds: no attribute improves variance - always deactivate (River)
                        deactivate(); tree.nActiveLeaves--; tree.nInactiveLeaves++;
                        tree.enforceTreeSizeLimit();
                        return;
                }

                double epsilon = hoeffdingBound(1.0, tree.deltaOption.getValue(), weightSeen);
                boolean shouldSplit;
                if (nCandidates == 1) {
                        shouldSplit = bestVR > 0;
                } else {
                        shouldSplit = bestVR > 0 && (
                                (secondVR / bestVR < 1.0 - epsilon) || (epsilon < tree.tauOption.getValue()));
                }

                if (shouldSplit) {
                        InstanceConditionalTest test;
                        int numBranches;
                        if (bestIsNominal) {
                                if (tree.binarySplitOption.isSet()) {
                                        int[][] parts = nominalSplitters.get(bestAttr).getBinarySplitCategories(bestBinaryIdx);
                                        Set<Integer> leftSet  = new HashSet<>();
                                        Set<Integer> rightSet = new HashSet<>();
                                        for (int c : parts[0]) leftSet.add(c);
                                        for (int c : parts[1]) rightSet.add(c);
                                        test = new NominalBinaryTest(bestAttr, leftSet, rightSet);
                                        numBranches = 2;
                                } else {
                                        int[] cats = nominalSplitters.get(bestAttr).getSortedCategories();
                                        test = new NominalMultiwayTest(bestAttr, cats);
                                        numBranches = cats.length;
                                }
                        } else {
                                test = new NumericThresholdTest(bestAttr, bestThresh);
                                numBranches = 2;
                        }
                        AdaSplitNode newSplit = (AdaSplitNode) tree.newSplit(parent, test, numBranches, depth);
                        if (bestIsNominal && !tree.binarySplitOption.isSet()) {
                                // multiway: una branch per categoria, stats dirette da catStats
                                int[] cats = nominalSplitters.get(bestAttr).getSortedCategories();
                                double[][] catStats = nominalSplitters.get(bestAttr).getChildrenStats(cats);

                                for (int b = 0; b < numBranches; b++) {
                                        newSplit.children[b] = tree.newLeaf(
                                        newSplit,
                                        depth + 1,
                                        catStats[b][0],
                                        catStats[b][1],
                                        catStats[b][2]
                                        );
                                }
                                } else {
                                // binario (numerico o nominal binary): branch 0 = left, branch 1 = right
                                double rightSumY   = sumY - bestLeftSumY;
                                double rightSumYSq = sumYSq - bestLeftSumYSq;
                                double rightWeight = weightSeen - bestLeftWeight;

                                newSplit.children[0] = tree.newLeaf(
                                        newSplit,
                                        depth + 1,
                                        bestLeftSumY,
                                        bestLeftSumYSq,
                                        bestLeftWeight
                                );

                                newSplit.children[1] = tree.newLeaf(
                                        newSplit,
                                        depth + 1,
                                        rightSumY,
                                        rightSumYSq,
                                        rightWeight
                                );
                        }

                        if (parent == null) tree.root = newSplit;
                        else ((SplitNode) parent).children[parentBranch] = newSplit;
                        newSplit.parent = parent;

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
                                if (tree.removePoorAttrsOption.isSet() && vr / bestVR < secondRatio - 2 * epsilon)
                                        disableAttribute(attrIdx);
                        }
                        if (tree.removePoorAttrsOption.isSet()) {
                                for (Map.Entry<Integer, NominalSplitter> entry : nominalSplitters.entrySet()) {
                                        int attrIdx = entry.getKey();
                                        double[] res = tree.binarySplitOption.isSet()
                                                ? entry.getValue().bestBinarySplit(
                                                        parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplitOption.getValue())
                                                : entry.getValue().bestSplit(
                                                        parentVariance, sumY, sumYSq, weightSeen, tree.minSamplesSplitOption.getValue());
                                        if (res == null) continue;
                                        double vr = res[1];
                                        if (vr / bestVR < secondRatio - 2 * epsilon)
                                                disableAttribute(attrIdx);
                                }
                        }
                }
        }

    }

    public abstract class SplitNode extends Node {
        protected InstanceConditionalTest splitTest;
        protected Node[] children;

        public SplitNode(InstanceConditionalTest test, int numBranches) {
            this.splitTest = test;
            this.children = new Node[numBranches];
        }

    }

    public class AdaLeafMean extends LeafNode {

        @Override
        public void collectLeaves(Instance inst, List<Node> result) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'collectLeaves'");
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
                return getMean();
        }
        
    }

    public class AdaLeafModel extends AdaLeafMean {

        private LinearModel leafModel;

        @Override
        public void collectLeaves(Instance inst, List<Node> result) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'collectLeaves'");
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
                return leafModel.predict(inst);
        }
        
    }

    public class AdaLeafAdaptive extends AdaLeafModel {

        private LinearModel leafModel;
        private double fmseMean  = 0.0;
        private double fmseModel = 0.0;

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
                if (fmseMean < fmseModel) return getMean();
                return leafModel.predict(inst);
        }
        
    }

    public class AdaSplitNode extends SplitNode {

        public AdaSplitNode(InstanceConditionalTest test, int numBranches) {
                super(test, numBranches);
                //TODO Auto-generated constructor stub
        }

        @Override
        public void learn(
                Instance inst, 
                HoeffdingAdaptiveTreeRegressor tree, 
                Node parent, 
                int parentBranch
        ) 
        {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'learn'");
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'predict'");
        }

        @Override
        public void collectLeaves(Instance inst, List<Node> result) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'collectLeaves'");
        }
        
    }

    public static class LinearModel {

        public double predict(Instance inst) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'predict'");
        }
        
    }

    public static class ErrorEstimator {
        private double mean = 0, M2 = 0;
        private int n = 0;

        public void update(double v) {
                n ++;
                double delta = v - mean;
                mean += delta / n;
                M2 += delta * (v - mean);
        }

        public double getMean() {
                return mean;
        }

        public double getVariance() {
                return n < 2 ? 0: Math.max(0, M2 / (n-1));
        }

        public int getCount() {
                return n;
        }

        public void reset() {
                mean = 0;
                M2 = 0;
                n = 0;
        }
    }

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

        @Override public void update(double v){ changed = adwin.setInput(v); }
        @Override public boolean detectedChange() { return changed; }
        @Override public void reset() { adwin = new ADWIN(delta); changed = false; }
    }

    public static class NumericThresholdTest extends InstanceConditionalTest{

        public NumericThresholdTest(int attrIdx, double threshold) {}

        @Override
        public void getDescription(StringBuilder sb, int indent) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getDescription'");
        }

        @Override
        public int branchForInstance(Instance inst) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'branchForInstance'");
        }

        @Override
        public int maxBranches() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'maxBranches'");
        }

        @Override
        public String describeConditionForBranch(int branch, InstancesHeader context) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'describeConditionForBranch'");
        }

        @Override
        public int[] getAttsTestDependsOn() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getAttsTestDependsOn'");
        }
    }

    public static class NominalBinaryTest extends InstanceConditionalTest {

        public NominalBinaryTest(int attrIdx, Set<Integer> leftSet, Set<Integer> rightSet) {}

        @Override
        public void getDescription(StringBuilder sb, int indent) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getDescription'");
        }

        @Override
        public int branchForInstance(Instance inst) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'branchForInstance'");
        }

        @Override
        public int maxBranches() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'maxBranches'");
        }

        @Override
        public String describeConditionForBranch(int branch, InstancesHeader context) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'describeConditionForBranch'");
        }

        @Override
        public int[] getAttsTestDependsOn() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getAttsTestDependsOn'");
        }}

    public static class NominalMultiwayTest extends InstanceConditionalTest {

        public NominalMultiwayTest(int attrIdx, int[] sortedCategories) {}

        @Override
        public void getDescription(StringBuilder sb, int indent) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getDescription'");
        }

        @Override
        public int branchForInstance(Instance inst) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'branchForInstance'");
        }

        @Override
        public int maxBranches() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'maxBranches'");
        }

        @Override
        public String describeConditionForBranch(int branch, InstancesHeader context) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'describeConditionForBranch'");
        }

        @Override
        public int[] getAttsTestDependsOn() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getAttsTestDependsOn'");
        }}


    public static class TEBSTSplitter {
        private final double roundFactor;
        private EBSTNode root = null;

        public TEBSTSplitter(int digits) { this.roundFactor = Math.pow(10, digits); }

        public double[] bestSplit(double parentVariance, double totalSumY, double totalSumYSq, double totalCount, int minSamplesSplit) {
                if (root == null || totalCount < 2 * minSamplesSplit) return null;
                double[] result = {Double.NaN, 0.0, 0.0, 0.0, 0.0, 0.0};
                double[] aux    = {0.0, 0.0, 0.0};
                findBestSplit(root, result, aux, parentVariance, totalSumY, totalSumYSq, totalCount, minSamplesSplit);
                return Double.isNaN(result[0]) ? null : result;
        }

        private void findBestSplit(EBSTNode root2, double[] result, double[] aux, double parentVariance,
                        double totalSumY, double totalSumYSq, double totalCount, int minSamplesSplit) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'findBestSplit'");
        }

        public void removeBadSplits(double secondRatio, double bestVR, double epsilon, double parentVariance,
                        double sumY, double sumYSq, double weightSeen) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'removeBadSplits'");
        }

        public void update(double attVal, double y, double w) {}
    }

    public static class EBSTNode {}

    public static class NominalSplitter {
        public void update(int catIndex, double y, double w) {}

        public double[][] getChildrenStats(int[] cats) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getChildrenStats'");
        }

        public int[][] getBinarySplitCategories(int bestBinaryIdx) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getBinarySplitCategories'");
        }

        public int[] getSortedCategories() {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'getSortedCategories'");
        }

        public double[] bestSplit(double parentVariance, double sumY, double sumYSq, double weightSeen, int value) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'bestSplit'");
        }

        public double[] bestBinarySplit(double parentVariance, double sumY, double sumYSq, double weightSeen,
                        int value) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'bestBinarySplit'");
        }
    }

    public void enforceTreeSizeLimit() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'enforceTreeSizeLimit'");
    }

    //endregion === CLASSES ===
    
}
