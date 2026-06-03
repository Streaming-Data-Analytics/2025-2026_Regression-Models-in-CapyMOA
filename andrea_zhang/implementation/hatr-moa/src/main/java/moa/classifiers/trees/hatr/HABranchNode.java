package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.*;

/** Base abstract branch (split) node for HATR. */
public abstract class HABranchNode implements HANode {
    public VarStats stats;
    public int depth;
    public List<HANode> children;
    public final int attIndex;

    protected HABranchNode(VarStats stats, int depth, int attIndex, HANode... ch) {
        this.stats = (stats != null) ? stats : new VarStats();
        this.depth = depth;
        this.attIndex = attIndex;
        this.children = new ArrayList<>(Arrays.asList(ch));
    }

    /** Return child index for the given instance. Throws if feature is missing. */
    public abstract int branchNo(Instance inst);

    public HANode next(Instance inst) {
        int idx = branchNo(inst);
        return (idx >= 0 && idx < children.size()) ? children.get(idx) : null;
    }

/** Traverse to a leaf following the instance. */
    public HALeafNode traverseToLeaf(Instance inst) {
        HANode cur = this;
        while (cur instanceof HABranchNode) {
            HABranchNode b = (HABranchNode) cur;
            HANode next;
            try { next = b.next(inst); } catch (Exception e) { next = null; }
            cur = (next != null) ? next : b.mostCommonChild();
        }
        return (cur instanceof HALeafNode) ? (HALeafNode) cur : null;
    }

    public HANode mostCommonChild() {
        return children.stream()
            .filter(Objects::nonNull)
            .max(Comparator.comparingDouble(HANode::getTotalWeight))
            .orElse(null);
    }

    public int mostCommonChildIndex() {
        int best = 0; double bw = -1;
        for (int i = 0; i < children.size(); i++) {
            if (children.get(i) != null && children.get(i).getTotalWeight() > bw) {
                bw = children.get(i).getTotalWeight(); best = i;
            }
        }
        return best;
    }

    public void addChild(HANode child) { children.add(child); }

    public void killChildren(Object tree) { /* overridden by AdaBranchNode */ }

    // HANode interface 

    @Override
    public double getTotalWeight() {
        double t = 0; for (HANode c : children) if (c != null) t += c.getTotalWeight(); return t;
    }

    @Override
    public int getHeight() {
        int h = 0; for (HANode c : children) if (c != null) h = Math.max(h, c.getHeight()); return 1 + h;
    }

    @Override
    public int getNNodes() {
        int n = 1; for (HANode c : children) if (c != null) n += c.getNNodes(); return n;
    }

    @Override
    public int getNLeaves() {
        int n = 0; for (HANode c : children) if (c != null) n += c.getNLeaves(); return n;
    }

    @Override
    public List<HALeafNode> iterLeaves() {
        List<HALeafNode> leaves = new ArrayList<>();
        for (HANode c : children) if (c != null) leaves.addAll(c.iterLeaves());
        return leaves;
    }

    @Override
    public double getPrediction(Instance inst) {
        HALeafNode leaf = traverseToLeaf(inst);
        return (leaf != null) ? leaf.getPrediction(inst) : 0.0;
    }
}
