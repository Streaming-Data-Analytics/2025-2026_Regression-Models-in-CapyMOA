package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor;
import moa.classifiers.trees.hatr.util.NormalDist;

import java.util.*;

/**
 * Abstract adaptive branch node for HATR.
 * Monitors prediction error with ADWIN; on drift grows an alternate subtree in the
 * background and swaps it in via z-test once both trees are mature enough to compare.
 */
public abstract class AdaBranchNode extends HABranchNode {
    public ADWINDetector driftDetector;
    public HANode alternateTree;
    public VarStats errorTracker;

    protected AdaBranchNode(VarStats stats, int depth, int attIndex, ADWINDetector driftDetector, HANode... children) {
        super(stats, depth, attIndex, children);
        this.driftDetector = driftDetector;
        this.alternateTree = null;
        this.errorTracker = new VarStats();
    }

    /** Adaptive learning step: drift detection + alternate tree management + child dispatch. */
    public void adaLearnOne(Instance inst, HoeffdingAdaptiveTreeRegressor tree, HABranchNode parent, int parentBranch) {
        double y = inst.classValue();

        // Prediction from main tree path (not alternate)
        HALeafNode mainLeaf = traverseToLeaf(inst);
        double yPred = (mainLeaf != null) ? mainLeaf.getPrediction(inst) : 0.0;

        stats.update(y, inst.weight());

        double err = Math.abs(y - yPred);
        double oldMean = errorTracker.getMean();
        driftDetector.update(err);
        errorTracker.update(err, 1.0);
        boolean driftOccurred = driftDetector.isDrift();

        // Error is decreasing, so ignore the detection
        if (driftOccurred && errorTracker.getMean() < oldMean) {
            errorTracker = new VarStats();
            driftOccurred = false;
        }

        // Create alternate tree on drift
        if (driftOccurred && alternateTree == null) {
            errorTracker = new VarStats();
            alternateTree = tree.newLeaf(null, depth);   // same depth as this branch
            tree.nAlternateTrees++;
        }
        // Compare alternate vs. main via z-test
        else if (alternateTree != null) {
            VarStats altErr = getErrorTracker(alternateTree);
            if (altErr != null) {
                double altN = altErr.getN(), curN = errorTracker.getN();
                if (altN > tree.driftWindowThreshold && curN > tree.driftWindowThreshold) {
                    double altMu = altErr.getMean(), curMu = errorTracker.getMean();
                    double altV = altErr.get(), curV = errorTracker.get();
                    double denom = Math.sqrt(altV / altN + curV / curN);
                    double z = (denom > 1e-12) ? (altMu - curMu) / denom : 0.0;
                    double p = 2.0 * NormalDist.cdf(-Math.abs(z));

                    if (p <= tree.switchSignificance) {
                        if (altMu < curMu) {
                            // Alternate is better: swap.
                            tree.nActiveLeaves -= this.iterLeaves().size();
                            tree.nActiveLeaves += alternateTree.iterLeaves().size();
                            killChildren(tree);
                            if (parent != null) {
                                parent.children.set(parentBranch, alternateTree);
                                tree.nSwitchAlternateTrees++;
                                return; // Non-root: replaced in parent, stop here
                            } else {
                                tree.root = alternateTree;
                                tree.nSwitchAlternateTrees++;
                                // Root switch: fall through so alternateTree is trained on this sample
                            }
                        } else {
                            // Current is better: prune alternate
                            killNode(alternateTree, tree);
                            alternateTree = null;
                            tree.nPrunedAlternateTrees++;
                        }
                    }
                }
            }
        }

        // Forward to alternate tree (let it grow)
        if (alternateTree != null) {
            dispatch(alternateTree, inst, tree, parent, parentBranch);
        }

        // Forward to the appropriate child.
        // For nominal multiway branches, unseen categories get a new child leaf
        HANode child = null;
        int childBranch = -1;

        if (this instanceof AdaNomMultiwayBranch) {
            AdaNomMultiwayBranch nomBranch = (AdaNomMultiwayBranch) this;
            double featureVal = inst.value(attIndex);
            // Missing feature (NaN): fall through
            // to mostCommonChildIndex() rather than creating a child for category 0.
            if (!Double.isNaN(featureVal)) {
                int catIdx = (int) featureVal;
                if (!nomBranch.hasCategory(catIdx)) {
                    HALeafNode newLeaf = tree.newLeaf(null, depth + 1);
                    childBranch = nomBranch.addNewChild(newLeaf, catIdx);
                    tree.nActiveLeaves++;
                    child = newLeaf;
                }
            }
        }

        if (child == null) {
            try { child = next(inst); } catch (Exception e) { child = null; }
            if (child != null) {
                try { childBranch = branchNo(inst); } catch (Exception e) { childBranch = 0; }
            } else {
                childBranch = mostCommonChildIndex();
                child = children.get(childBranch);
            }
        }

        dispatch(child, inst, tree, this, childBranch);
    }

    private static void dispatch(HANode node, Instance inst, HoeffdingAdaptiveTreeRegressor tree, HABranchNode parent, int parentBranch) {
        if (node instanceof AdaLeafNode) {
            ((AdaLeafNode) node).adaLearnOne(inst, tree, parent, parentBranch);
        } else if (node instanceof AdaBranchNode) {
            ((AdaBranchNode) node).adaLearnOne(inst, tree, parent, parentBranch);
        }
    }

    private static VarStats getErrorTracker(HANode node) {
        if (node instanceof AdaLeafNode) return ((AdaLeafNode) node).errorTracker;
        if (node instanceof AdaBranchNode) return ((AdaBranchNode) node).errorTracker;
        return null;
    }

    private static void killNode(HANode node, HoeffdingAdaptiveTreeRegressor tree) {
        if (node instanceof AdaBranchNode) ((AdaBranchNode) node).killChildren(tree);
    }

    /**
     * Recursively clean up children: decrement leaf counters, remove alternate trees.
     */
    @Override
    public void killChildren(Object treeObj) {
        HoeffdingAdaptiveTreeRegressor tree = (HoeffdingAdaptiveTreeRegressor) treeObj;
        for (HANode child : children) {
            if (child instanceof AdaBranchNode) {
                AdaBranchNode ab = (AdaBranchNode) child;
                if (ab.alternateTree != null) {
                    killNode(ab.alternateTree, tree);
                    tree.nPrunedAlternateTrees++;
                    ab.alternateTree = null;
                }
                ab.killChildren(tree);
            } else if (child instanceof HALeafNode) {
                HALeafNode hl = (HALeafNode) child;
                if (hl.isActive()) tree.nActiveLeaves--; else tree.nInactiveLeaves--;
            }
        }
    }

    /**
     * Collect leaves from both the main tree path and any alternate trees along the way.
     * Used by HATR's predict_one to average predictions.
     */
    public List<HALeafNode> traverseWithAlternate(Instance inst) {
        List<HALeafNode> found = new ArrayList<>();
        HANode cur = this;
        while (cur instanceof HABranchNode) {
            HABranchNode b = (HABranchNode) cur;
            // Collect from alternate tree at this branch
            if (b instanceof AdaBranchNode) {
                HANode alt = ((AdaBranchNode) b).alternateTree;
                if (alt != null) {
                    if (alt instanceof AdaBranchNode) found.addAll(((AdaBranchNode) alt).traverseWithAlternate(inst));
                    else if (alt instanceof HALeafNode) found.add((HALeafNode) alt);
                }
            }
            HANode next;
            try { next = b.next(inst); } catch (Exception e) { next = null; }
            if (next == null) next = b.mostCommonChild();
            cur = next;
        }
        if (cur instanceof HALeafNode) found.add((HALeafNode) cur);
        return found;
    }

    /**
     * Override iterLeaves to also include alternate tree leaves.
     */
    @Override
    public List<HALeafNode> iterLeaves() {
        List<HALeafNode> leaves = new ArrayList<>();
        for (HANode child : children) {
            if (child != null) leaves.addAll(child.iterLeaves());
            if (child instanceof AdaBranchNode) {
                HANode alt = ((AdaBranchNode) child).alternateTree;
                if (alt != null) leaves.addAll(alt.iterLeaves());
            }
        }
        return leaves;
    }
}
