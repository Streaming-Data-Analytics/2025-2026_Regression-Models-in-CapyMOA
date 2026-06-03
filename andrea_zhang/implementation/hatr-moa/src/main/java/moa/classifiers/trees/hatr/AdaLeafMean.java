package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.Random;

/** Adaptive leaf predicting with the running target mean. */
public class AdaLeafMean extends AdaLeafNode {

    public AdaLeafMean(VarStats stats, int depth, HAAttributeObserver obs, ADWINDetector driftDet, Random rng) {
        super(stats, depth, obs, driftDet, rng);
    }

    @Override public void updateStats(double y, double w) { stats.update(y, w); }
    @Override public double getPrediction(Instance inst) { return stats.getMean(); }
    @Override public double getTotalWeight() { return stats.getN(); }
    @Override public int calculatePromise() { return -depth; }
}
