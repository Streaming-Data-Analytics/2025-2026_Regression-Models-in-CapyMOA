package moa.classifiers.trees.hatr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;


public class HAQOObserver implements HAAttributeObserver {

    private final double radius;
    private final boolean allowMultiwaySplits;
    private final FeatureQuantizer quantizer;

    public HAQOObserver(double radius, boolean allowMultiwaySplits) {
        this.radius = radius;
        this.allowMultiwaySplits = allowMultiwaySplits;
        this.quantizer = new FeatureQuantizer(radius);
    }

    @Override
    public void observe(double attVal, double targetVal, double weight) {
        if (!Double.isNaN(attVal)) quantizer.update(attVal, targetVal, weight);
    }

    @Override
    public SplitCandidate bestSplitCandidate(int attIndex, VarStats preSplit, int minSamples, boolean binarySplit) {
        if (quantizer.size() <= 1) return new SplitCandidate();

        SplitCandidate best = new SplitCandidate();

        // Numeric multiway candidate: one branch per slot
        if (allowMultiwaySplits && !binarySplit) {
            int[] slotIds = quantizer.sortedKeys();
            List<VarStats> slotStats = quantizer.slotStats(slotIds);
            double n = preSplit.getN();
            double vr = preSplit.get();
            boolean allSufficient = true;
            for (VarStats s : slotStats) {
                if (s.getN() < minSamples) { allSufficient = false; }
                vr -= (s.getN() / n) * s.get();
            }
            if (allSufficient) {
                best = new SplitCandidate(vr, attIndex, radius, slotIds, slotStats);
            }
        }

        // Binary candidates: threshold at midpoint between consecutive slot means
        VarStats acc = new VarStats();
        Double prevX = null;
        for (Map.Entry<Integer, QOSlot> e : quantizer.sortedEntries()) {
            double x = e.getValue().xMean();
            acc.addInPlace(e.getValue().yStats);

            if (prevX == null) {
                prevX = x;
                continue;
            }

            VarStats right = preSplit.subtract(acc);
            if (acc.getN() >= minSamples && right.getN() >= minSamples) {
                double n = preSplit.getN();
                double vr = preSplit.get()
                    - (acc.getN() / n) * acc.get()
                    - (right.getN() / n) * right.get();
                if (vr > best.merit) {
                    double threshold = (prevX + x) / 2.0;
                    best = new SplitCandidate(vr, attIndex, threshold,
                        Arrays.asList(acc.copy(), right));
                }
            }

            prevX = x;
        }

        return best;
    }

    @Override
    public HAAttributeObserver createNew() {
        return new HAQOObserver(radius, allowMultiwaySplits);
    }

    // Inner classes --------------------------------------------------------

    /** One bucket in the quantization hash: tracks x-mean and target variance. */
    static class QOSlot {
        private double xSum;
        private double xCount;
        final VarStats yStats;

        QOSlot(double x, double y, double w) {
            xSum = x * w;
            xCount = w;
            yStats = new VarStats();
            yStats.update(y, w);
        }

        void update(double x, double y, double w) {
            xSum += x * w;
            xCount += w;
            yStats.update(y, w);
        }

        double xMean() { return xCount > 0 ? xSum / xCount : 0.0; }
    }

    /** Hash-based dynamic quantization: maps floor(x/radius) → QOSlot. */
    static class FeatureQuantizer {
        private final double radius;
        private final HashMap<Integer, QOSlot> hash = new HashMap<>();

        FeatureQuantizer(double radius) { this.radius = radius; }

        int size() { return hash.size(); }

        void update(double x, double y, double w) {
            int idx = (int) Math.floor(x / radius);
            QOSlot slot = hash.get(idx);
            if (slot == null) hash.put(idx, new QOSlot(x, y, w));
            else slot.update(x, y, w);
        }

        /** Returns entries sorted by slot key (ascending). */
        Iterable<Map.Entry<Integer, QOSlot>> sortedEntries() {
            return new TreeMap<>(hash).entrySet();
        }

        int[] sortedKeys() {
            int[] keys = new int[hash.size()];
            int i = 0;
            for (int k : new TreeMap<>(hash).keySet()) keys[i++] = k;
            return keys;
        }

        /** Returns a copy of yStats for each slot in sorted key order. */
        List<VarStats> slotStats(int[] sortedKeyArray) {
            List<VarStats> out = new ArrayList<>(sortedKeyArray.length);
            for (int k : sortedKeyArray) out.add(hash.get(k).yStats.copy());
            return out;
        }
    }
}
