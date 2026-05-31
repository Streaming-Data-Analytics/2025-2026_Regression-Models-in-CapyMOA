package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.HashMap;
import java.util.Map;

/**
 * Adaptive nominal multiway branch node for HATR.
 *
 * Routes each instance to the child corresponding to its category index.
 * For unseen category values, falls back to the most-traversed child
 * (mirrors River's NominalMultiwayBranch.most_common_path behaviour).
 *
 * Equivalent to River's AdaNomMultiwayBranchReg + NominalMultiwayBranch.
 */
public class AdaNomMultiwayBranch extends AdaBranchNode {

    /** Maps category index → child position in children list. */
    private final Map<Integer, Integer> categoryToChild;

    /**
     * @param nomValOrder  category indices in child order: nomValOrder[i] is the
     *                     category routed to children[i]. Mirrors River's feature_values.
     */
    public AdaNomMultiwayBranch(VarStats stats, int attIndex, int[] nomValOrder,
                                int depth, ADWINDetector driftDet, HANode... children) {
        super(stats, depth, attIndex, driftDet, children);
        this.categoryToChild = new HashMap<>();
        for (int i = 0; i < nomValOrder.length; i++) {
            categoryToChild.put(nomValOrder[i], i);
        }
    }

    @Override
    public int branchNo(Instance inst) {
        Integer childIdx = categoryToChild.get((int) inst.value(attIndex));
        return (childIdx != null) ? childIdx : mostCommonChildIndex();
    }

    /** Returns true if this category index already has a mapped child. */
    public boolean hasCategory(int catIdx) {
        return categoryToChild.containsKey(catIdx);
    }

    /**
     * Adds a new child leaf for a previously unseen category value.
     * Mirrors River's NominalMultiwayBranch.add_child().
     * @return the index of the new child in children list.
     */
    public int addNewChild(HANode leaf, int catIdx) {
        int idx = children.size();
        categoryToChild.put(catIdx, idx);
        children.add(leaf);
        return idx;
    }

    /** Unlimited branches — mirrors River's NominalMultiwayBranch.max_branches() = -1. */
    @Override
    public int maxBranches() { return -1; }

    @Override
    public String toString() {
        return "att[" + attIndex + "] in " + categoryToChild.keySet();
    }
}
