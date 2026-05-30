package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor;
import java.util.Random;

/**
 * Adaptive leaf choosing between mean and linear-model predictors via EWMA error
 * tracking (faded mean squared error). Equivalent to River's AdaLeafRegAdaptive.
 */
public class AdaLeafAdaptive extends AdaLeafNode {
    public HAPerceptron model;
    public double fmseMean = 0.0;
    public double fmseModel = 0.0;

    public AdaLeafAdaptive(VarStats stats, int depth, HAAttributeObserver obs, ADWINDetector driftDet, Random rng, HAPerceptron model) {
        super(stats, depth, obs, driftDet, rng);
        this.model = model;
    }

    /** Faded MSE update — uses the pre-update target mean and model prediction. */
    @Override
    protected void updateModelSelector(Instance inst, double y, HoeffdingAdaptiveTreeRegressor tree) {
        double predMean  = stats.getMean();
        double predModel = model.predict(inst);
        double decay = tree.modelSelectorDecay;
        fmseMean = decay * fmseMean + (y - predMean) * (y - predMean);
        fmseModel = decay * fmseModel + (y - predModel) * (y - predModel);
    }

    @Override
    protected void trainLeafModel(Instance inst, double y, double w) {
        model.train(inst, y, w);
    }

    @Override
    public void updateStats(double y, double w) { stats.update(y, w); }

    @Override
    public double getPrediction(Instance inst) {
        // River: act as regression tree iff fmse_mean < fmse_model (strict).
        return (fmseMean < fmseModel) ? stats.getMean() : model.predict(inst);
    }

    @Override public double getTotalWeight() { return stats.getN(); }
    @Override public int calculatePromise() { return -depth; }
}
