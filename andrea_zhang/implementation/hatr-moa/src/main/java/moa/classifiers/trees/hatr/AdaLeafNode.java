package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor;

import java.util.Random;

/**
 * Abstract adaptive leaf node for HATR.
 * Adds ADWIN drift tracking and bootstrap sampling to the base leaf learning.
 * Subclasses implement the specific prediction strategy (mean / model / adaptive).
 *
 * Learning order matches River exactly:
 *   1. prediction BEFORE update (for drift error)
 *   2. bootstrap sampling (Poisson(1)), effective weight w
 *   3. drift detector update with |y - yPred|
 *   4. model-selector update (adaptive fmse), using stats/model BEFORE update
 *   5. update target stats + attribute observers (with bootstrapped w)
 *   6. train leaf model (with bootstrapped w)
 *   7. split attempt after the grace period
 */
public abstract class AdaLeafNode extends HALeafNode {
    public ADWINDetector driftDetector;
    public final Random rng;
    public VarStats errorTracker;

    protected AdaLeafNode(VarStats stats, int depth, HAAttributeObserver defaultObserver, ADWINDetector driftDetector, Random rng) {
        super(stats, depth, defaultObserver);
        this.driftDetector = driftDetector;
        this.rng = rng;
        this.errorTracker = new VarStats();
    }

    public void adaLearnOne(Instance inst, HoeffdingAdaptiveTreeRegressor tree, HABranchNode parent, int parentBranch) {
        double y = inst.classValue();
        double w = inst.weight();

        double yPred = getPrediction(inst);

        // Bootstrap sampling: multiply weight by Poisson(1)
        if (tree.bootstrapSampling) {
            int k = poissonSample(rng);
            if (k > 0) w *= k;
        }

        // Drift tracking on the raw absolute error.
        double err = Math.abs(y - yPred);
        double oldMeanErr = errorTracker.getMean();
        driftDetector.update(err);
        errorTracker.update(err, 1.0);
        if (driftDetector.isDrift() && errorTracker.getMean() < oldMeanErr) {
            errorTracker = new VarStats(); // error is improving, ignore
        }

        // Model-selector statistics must use the pre-update target stats / model.
        updateModelSelector(inst, y, tree);

        // Update target stats and attribute observers (base leaf logic).
        updateStats(y, w);
        if (isActive()) updateObserversFromInst(inst, y, w, tree);

        // Train the leaf model (if any) with the bootstrapped weight, before split.
        trainLeafModel(inst, y, w);

        // Split attempt after the grace period.
        double weightSeen = getTotalWeight();
        if (weightSeen - lastSplitAttemptAt >= tree.gracePeriod) {
            if (depth >= tree.maxDepth) {
                deactivate();
                tree.nActiveLeaves--; tree.nInactiveLeaves++;
            } else if (isActive() && tree.growthAllowed) {
                tree.attemptToSplit(this, parent, parentBranch, tree.driftDetectorProto.createNew());
                lastSplitAttemptAt = weightSeen;
            }
        }
    }

    /** Hook: update the adaptive model selector (fmse). Uses pre-update stats/model. */
    protected void updateModelSelector(Instance inst, double y, HoeffdingAdaptiveTreeRegressor tree) { }

    /** Hook: train the leaf prediction model with the (bootstrapped) weight. */
    protected void trainLeafModel(Instance inst, double y, double w) { }

    private void updateObserversFromInst(Instance inst, double y, double w, HoeffdingAdaptiveTreeRegressor tree) {
        int classIdx = inst.classIndex();
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i == classIdx || disabledAtts.contains(i)) continue;
            if (Double.isNaN(inst.value(i))) continue;
            HAAttributeObserver obs = observers.get(i);
            if (obs == null) {
                boolean nom = inst.attribute(i).isNominal();
                obs = nom ? new HANominalObserver() : defaultObserver.createNew();
                observers.put(i, obs);
            }
            obs.observe(inst.value(i), y, w);
        }
    }

    private static int poissonSample(Random rng) {
        double L = Math.exp(-1.0), p = 1.0; int k = 0;
        do { k++; p *= rng.nextDouble(); } while (p > L);
        return k - 1;
    }
}
