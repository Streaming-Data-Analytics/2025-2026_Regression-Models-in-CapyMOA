package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.HashMap;
import java.util.Map;

public class AdaNumMultiwayBranch extends AdaBranchNode {

    public final double radius;
    private final Map<Integer, Integer> slotToChild;

    /**
     * @param slotIds  QO slot keys (floor(x/radius)) in child order;
     *                 slotIds[i] routes to children[i].
     */
    public AdaNumMultiwayBranch(VarStats stats, int attIndex, double radius,
            int[] slotIds, int depth, ADWINDetector driftDet, HANode... children) {
        super(stats, depth, attIndex, driftDet, children);
        this.radius = radius;
        slotToChild = new HashMap<>();
        for (int i = 0; i < slotIds.length; i++) slotToChild.put(slotIds[i], i);
    }

    @Override
    public int branchNo(Instance inst) {
        int slot = (int) Math.floor(inst.value(attIndex) / radius);
        Integer idx = slotToChild.get(slot);
        return (idx != null) ? idx : mostCommonChildIndex();
    }

    @Override
    public String toString() {
        return "att[" + attIndex + "] QO-slot (radius=" + radius + ") -> " + slotToChild.keySet();
    }
}
