package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;

/** Adaptive nominal binary branch: left if x[attIndex] == nomVal, else right. */
public class AdaNomBinaryBranch extends AdaBranchNode {
    public final int nomValIndex;

    public AdaNomBinaryBranch(VarStats stats, int attIndex, int nomValIndex, int depth, HANode left, HANode right, ADWINDetector driftDet) {
        super(stats, depth, attIndex, driftDet, left, right);
        this.nomValIndex = nomValIndex;
    }

    @Override
    public int branchNo(Instance inst) {
        return ((int) inst.value(attIndex) == nomValIndex) ? 0 : 1;
    }

    @Override
    public String toString() { return "att[" + attIndex + "] == " + nomValIndex; }
}
