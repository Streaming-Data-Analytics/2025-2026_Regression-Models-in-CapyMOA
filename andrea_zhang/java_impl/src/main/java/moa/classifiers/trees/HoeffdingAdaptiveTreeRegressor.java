package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.MultiChoiceOption;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;

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
    
    public enum LeafPrediction { MEAN, MODEL, ADAPTIVE }

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
    public void resetLearningImpl() {}

    @Override
    public boolean isRandomizable() {
        return true; // MOA will manage classifierRandom and randomSeed
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

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
    public double[] getVotesForInstance(Instance inst) {
        return new double[]{};
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{};
    }

    private void estimateModelSize() {}

    protected LeafNode newLeaf(Node parent, int depth) {
        return new AdaLeafNode(parent, depth);
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
    }

    public abstract class LeafNode extends Node {}

    public abstract class SplitNode extends Node {}

    public class AdaLeafNode extends LeafNode {
        public AdaLeafNode(Node parent, int depth) {}

        @Override
        public void learn(
                Instance inst, 
                HoeffdingAdaptiveTreeRegressor tree, 
                Node parent, 
                int parentBranch
        ){}

    }

    //endregion === CLASSES ===
    
}
