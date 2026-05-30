package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;

/** Adaptive numeric binary branch: x[attIndex] <= threshold -> left, else right. */
public class AdaNumBinaryBranch extends AdaBranchNode {
    public final double threshold;

    public AdaNumBinaryBranch(VarStats stats, int attIndex, double threshold, int depth, HANode left, HANode right, ADWINDetector driftDet) {
        super(stats, depth, attIndex, driftDet, left, right);
        this.threshold = threshold;
    }

    @Override
    public int branchNo(Instance inst) {
        double val = inst.value(attIndex);
        if (Double.isNaN(val)) throw new IllegalArgumentException("Missing att " + attIndex);
        return val <= threshold ? 0 : 1;
    }

    @Override
    public String toString() { return "att[" + attIndex + "] <= " + threshold; }
}
