package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.MultiChoiceOption;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor.AdaLeafAdaptive;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor.AdaLeafMean;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor.AdaLeafModel;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor.ErrorEstimator;
import moa.core.Measurement;

import java.util.ArrayList;
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
            default:    return new AdaLeafAdaptive();
        }
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
                    } else {
                        // tebst splitter
                    }
                }
            }

        }

        public boolean isActive() {
                return splitters != null;
        }

        public double getMean() {
                return 0.0;
        }

        protected void attemptSplit(
                HoeffdingAdaptiveTreeRegressor tree,
                Node parent, int parentBranch
        ) {
                if (!isActive()) return;
        }

    }

    public abstract class SplitNode extends Node {}

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

    public class AdaLeafModel extends LeafNode {

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

    public class AdaLeafAdaptive extends LeafNode {

        private LinearModel leafModel;
        private double fmseMean  = 0.0;
        private double fmseModel = 0.0;

        @Override
        public void collectLeaves(Instance inst, List<Node> result) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'collectLeaves'");
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
                if (fmseMean < fmseModel) return getMean();
                return leafModel.predict(inst);
        }
        
    }

    public class AdaSplitNode extends SplitNode {

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
        public void collectLeaves(Instance inst, List<Node> result) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'collectLeaves'");
        }

        @Override
        public double predict(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
                // TODO Auto-generated method stub
                throw new UnsupportedOperationException("Unimplemented method 'predict'");
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

    public static class TEBSTSplitter {}

    public static class NominalSplitter {}

    //endregion === CLASSES ===
    
}
