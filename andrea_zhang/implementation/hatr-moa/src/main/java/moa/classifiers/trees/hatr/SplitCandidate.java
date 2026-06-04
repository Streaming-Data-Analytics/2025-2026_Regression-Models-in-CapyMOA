package moa.classifiers.trees.hatr;

import java.util.List;

/**
 * A split candidate produced by an attribute observer.
 * Stores merit, the split attribute (by index), threshold/value, and post-split statistics.
 */
public class SplitCandidate implements Comparable<SplitCandidate> {
    public final double merit;
    public final int attIndex;            // -1 = null (no-split / pre-pruning)
    public final double numericThreshold; // numeric binary splits
    public final int nominalValue;        // nominal binary split: the split value index
    public final int[] nominalValues;     // nominal multiway split: category indices in child order
    public final double qoRadius;         // QO numeric multiway: quantization radius
    public final int[] qoSlotIds;         // QO numeric multiway: slot keys in child order
    public final boolean isNumeric;
    public final List<VarStats> childrenStats;

    /**
     * Null / "no valid split" candidate. Merit is -inf to match River's
     * BranchFactory default, so it always sorts below any real candidate and is
     * never selected as best/second-best unless it is the only option.
     */
    public SplitCandidate() {
        this.merit = Double.NEGATIVE_INFINITY; this.attIndex = -1;
        this.numericThreshold = Double.NaN; this.nominalValue = -1; this.nominalValues = null;
        this.qoRadius = Double.NaN; this.qoSlotIds = null;
        this.isNumeric = true; this.childrenStats = null;
    }

    /** Numeric binary split. */
    public SplitCandidate(double merit, int attIndex, double threshold, List<VarStats> children) {
        this.merit = merit; this.attIndex = attIndex;
        this.numericThreshold = threshold; this.nominalValue = -1; this.nominalValues = null;
        this.qoRadius = Double.NaN; this.qoSlotIds = null;
        this.isNumeric = true; this.childrenStats = children;
    }

    /** Nominal binary split: left if value == nominalValue, else right. */
    public SplitCandidate(double merit, int attIndex, int nominalValue, List<VarStats> children) {
        this.merit = merit; this.attIndex = attIndex;
        this.numericThreshold = Double.NaN; this.nominalValue = nominalValue; this.nominalValues = null;
        this.qoRadius = Double.NaN; this.qoSlotIds = null;
        this.isNumeric = false; this.childrenStats = children;
    }

    /** Nominal multiway split: one child per distinct value, ordered by nominalValues[i]. */
    public SplitCandidate(double merit, int attIndex, int[] nominalValues, List<VarStats> children) {
        this.merit = merit; this.attIndex = attIndex;
        this.numericThreshold = Double.NaN; this.nominalValue = -1; this.nominalValues = nominalValues;
        this.qoRadius = Double.NaN; this.qoSlotIds = null;
        this.isNumeric = false; this.childrenStats = children;
    }

    /** Numeric multiway split (QO): one child per slot, routed by floor(x/radius). */
    public SplitCandidate(double merit, int attIndex, double qoRadius, int[] slotIds, List<VarStats> children) {
        this.merit = merit; this.attIndex = attIndex;
        this.numericThreshold = Double.NaN; this.nominalValue = -1; this.nominalValues = null;
        this.qoRadius = qoRadius; this.qoSlotIds = slotIds;
        this.isNumeric = true; this.childrenStats = children;
    }

    public boolean isNullSplit()       { return attIndex == -1; }
    public boolean isNumericMultiway() { return isNumeric && qoSlotIds != null; }
    public boolean isNominalMultiway() { return !isNumeric && nominalValues != null; }
    public boolean isMultiway()        { return isNominalMultiway() || isNumericMultiway(); }

    @Override
    public int compareTo(SplitCandidate o) { return Double.compare(this.merit, o.merit); }
}
