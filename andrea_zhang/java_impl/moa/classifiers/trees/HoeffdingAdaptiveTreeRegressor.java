package moa.classifiers.trees;

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
}
