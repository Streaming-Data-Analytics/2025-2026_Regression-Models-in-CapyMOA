package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.Random;

/** Adaptive leaf predicting with an online linear model (River-equivalent LinearRegression). */
public class AdaLeafModel extends AdaLeafNode {
    public HAPerceptron model;

    public AdaLeafModel(VarStats stats, int depth, HAAttributeObserver obs, ADWINDetector driftDet, Random rng, HAPerceptron model) {
        super(stats, depth, obs, driftDet, rng);
        this.model = model;
    }

    /** Train the linear model with the bootstrapped weight (before the split attempt). */
    @Override
    protected void trainLeafModel(Instance inst, double y, double w) {
        model.train(inst, y, w);
    }

    @Override public void updateStats(double y, double w) { stats.update(y, w); }
    @Override public double getPrediction(Instance inst) { return model.predict(inst); }
    @Override public double getTotalWeight() { return stats.getN(); }
    @Override public int calculatePromise() { return -depth; }
}
