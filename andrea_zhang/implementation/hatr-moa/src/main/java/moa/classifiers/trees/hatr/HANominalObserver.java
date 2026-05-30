package moa.classifiers.trees.hatr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Attribute observer for nominal (categorical) features.
 *
 * Mirrors River's NominalSplitterReg: when a feature has > 2 distinct values,
 * a multiway candidate (one child per value, sorted by category index) is
 * evaluated first. Binary candidates (value == X vs rest) are also evaluated
 * and replace the multiway candidate only if they achieve strictly higher merit.
 * This matches River's default behaviour (binary_split=False).
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
        SplitCandidate best = new SplitCandidate(); // null, merit = -inf

        // Multiway candidate: one child per distinct value (mirrors River when n_values > 2).
        // Entries sorted by category index to match River's sorted(statistics.keys()) order.
        if (distPerVal.size() > 2) {
            List<Map.Entry<Integer, VarStats>> sorted = new ArrayList<>(distPerVal.entrySet());
            sorted.sort(Map.Entry.comparingByKey());

            double vr = preSplit.get();
            double n  = preSplit.getN();
            boolean valid = true;
            int[] nomVals      = new int[sorted.size()];
            List<VarStats> stats = new ArrayList<>(sorted.size());

            for (int i = 0; i < sorted.size(); i++) {
                VarStats s = sorted.get(i).getValue();
                if (s.getN() < minSamples) { valid = false; break; }
                vr -= (s.getN() / n) * s.get();
                nomVals[i] = sorted.get(i).getKey();
                stats.add(s.copy());
            }
            if (valid) best = new SplitCandidate(vr, attIndex, nomVals, stats);
        }

        // Binary candidates: value == X vs rest.
        // Replace multiway only if strictly better merit (mirrors River's NominalSplitterReg).
        for (Map.Entry<Integer, VarStats> e : distPerVal.entrySet()) {
            VarStats left  = e.getValue().copy();
            VarStats right = preSplit.subtract(left);
            if (left.getN() < minSamples || right.getN() < minSamples) continue;
            double n  = preSplit.getN();
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
