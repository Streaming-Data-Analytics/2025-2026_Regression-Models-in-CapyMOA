package moa.classifiers.trees.hatr;

import java.util.Arrays;

/**
 * Extended Binary Search Tree (E-BST) attribute observer for numeric features.
 * Each BST node's estimator accumulates stats for all elements with value <= node.key.
 * In-order traversal + accumulator finds the best split threshold.
 */
public class HAEBSTObserver implements HAAttributeObserver {
    private EBSTNode root;

    @Override
    public void observe(double attVal, double targetVal, double weight) {
        if (Double.isNaN(attVal)) return;
        if (root == null) root = new EBSTNode(attVal, targetVal, weight);
        else root.insert(attVal, targetVal, weight);
    }

    @Override
    public SplitCandidate bestSplitCandidate(int attIndex, VarStats preSplit, int minSamples, boolean binarySplit) {
        if (root == null) return new SplitCandidate();
        VarStats[] aux = {new VarStats()};
        return findBest(root, new SplitCandidate(), attIndex, preSplit, minSamples, aux);
    }

    private SplitCandidate findBest(EBSTNode node, SplitCandidate best, int attIdx, VarStats preSplit, int minSamples, VarStats[] aux) {
        if (node.left != null) best = findBest(node.left, best, attIdx, preSplit, minSamples, aux);

        VarStats left = node.est.add(aux[0]);
        VarStats right = preSplit.subtract(left);

        {
            double n = preSplit.getN();
            double vr = (left.getN() >= minSamples && right.getN() >= minSamples)
                      ? preSplit.get() - (left.getN()/n)*left.get() - (right.getN()/n)*right.get()
                      : 0.0;
            if (vr > best.merit) {
                best = new SplitCandidate(vr, attIdx, node.key, Arrays.asList(left, right));
            }
        }

        if (node.right != null) {
            aux[0].addInPlace(node.est);
            SplitCandidate rc = findBest(node.right, best, attIdx, preSplit, minSamples, aux);
            if (rc.merit > best.merit) best = rc;
            aux[0].subtractInPlace(node.est);
        }
        return best;
    }

    public void pruneBadSplits(VarStats preSplit, double lastRatio, double lastVr, double lastE, int minSamples) {
        if (root == null || lastVr <= 0) return;
        VarStats[] aux = {new VarStats()};
        pruneNode(root, null, true, preSplit, lastRatio, lastVr, lastE, minSamples, aux);
    }

    private boolean pruneNode(EBSTNode cur, EBSTNode parent, boolean isLeft, VarStats preSplit, double lastRatio, double lastVr, double lastE, int minSamples, VarStats[] aux) {
        boolean bad = false;
        if (cur.left != null) bad = pruneNode(cur.left, cur, true, preSplit, lastRatio, lastVr, lastE, minSamples, aux);
        else bad = true;

        if (bad) {
            if (cur.right != null) {
                aux[0].addInPlace(cur.est);
                bad = pruneNode(cur.right, cur, false, preSplit, lastRatio, lastVr, lastE, minSamples, aux);
                aux[0].subtractInPlace(cur.est);
            } else bad = true;
        }

        if (bad) {
            VarStats l = cur.est.add(aux[0]), r = preSplit.subtract(l);
            double n = preSplit.getN();
            double vr = (l.getN() >= minSamples && r.getN() >= minSamples) ? preSplit.get() - (l.getN()/n)*l.get() - (r.getN()/n)*r.get() : 0;
            if (vr / lastVr < lastRatio - 2 * lastE) {
                cur.left = cur.right = null;
                if (parent == null) root = null;
                else if (isLeft) parent.left = null;
                else parent.right = null;
                return true;
            }
        }
        return false;
    }

    @Override
    public HAAttributeObserver createNew() { return new HAEBSTObserver(); }

    // Inner BST node 

    private static class EBSTNode {
        double key;
        VarStats est;
        EBSTNode left, right;

        EBSTNode(double key, double target, double w) {
            this.key = key; est = new VarStats(); est.update(target, w);
        }

        void insert(double val, double target, double w) {
            EBSTNode cur = this, par = null; boolean goRight = false;
            while (cur != null) {
                par = cur;
                if (val == cur.key) { cur.est.update(target, w); return; }
                else if (val < cur.key) { cur.est.update(target, w); cur = cur.left; goRight = false; }
                else { cur = cur.right; goRight = true; }
            }
            EBSTNode n = new EBSTNode(val, target, w);
            if (goRight) par.right = n; else par.left = n;
        }
    }
}
