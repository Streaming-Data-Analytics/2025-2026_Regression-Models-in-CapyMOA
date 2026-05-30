package moa.classifiers.trees.hatr;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Attribute observer for nominal (categorical) features.
 * Generates binary splits: value == X vs value != X (by nominal index).
 */
public class HANominalObserver implements HAAttributeObserver {
    private final Map<Integer, VarStats> distPerVal = new HashMap<>();

    @Override
    public void observe(double attVal, double targetVal, double weight) {
        int valIdx = (int) attVal;
        distPerVal.computeIfAbsent(valIdx, k -> new VarStats()).update(targetVal, weight);
    }

    @Override
    public SplitCandidate bestSplitCandidate(int attIndex, VarStats preSplit, int minSamples) {
        SplitCandidate best = new SplitCandidate();
        for (Map.Entry<Integer, VarStats> e : distPerVal.entrySet()) {
            VarStats left = e.getValue().copy();
            VarStats right = preSplit.subtract(left);
            if (left.getN() < minSamples || right.getN() < minSamples) continue;
            double n = preSplit.getN();
            double vr = preSplit.get() - (left.getN()/n)*left.get() - (right.getN()/n)*right.get();
            if (vr > best.merit) {
                best = new SplitCandidate(vr, attIndex, e.getKey(), Arrays.asList(left, right));
            }
        }
        return best;
    }

    @Override
    public HAAttributeObserver createNew() { return new HANominalObserver(); }
}
