package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.FlagOption;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;

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
    
    //endregion === OPTIONS ===

    //region === CLASSES ===

    //endregion === CLASSES ===

    //region === METHODS ===

    public static double hoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt((range * range * Math.log(1.0 / confidence)) / (2.0 * n));
    }

    //endregion === METHODS ===
    
}
