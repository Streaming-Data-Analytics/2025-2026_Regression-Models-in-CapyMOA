package moa.classifiers.trees.hatr;

import java.util.List;

/**
 * A split candidate produced by an attribute observer.
 * Stores merit, the split attribute (by index), threshold/value, and post-split statistics.
 * Equivalent to River's BranchFactory; adapted to use int attIndex for MOA's Instance API.
 */
public class SplitCandidate implements Comparable<SplitCandidate> {
    public final double merit;
    public final int attIndex;           // -1 = null (no-split / pre-pruning)
    public final double numericThreshold; // for numeric binary splits
    public final int nominalValue;        // for nominal binary splits (index in attribute's value list)
    public final boolean isNumeric;
    public final List<VarStats> childrenStats; // post-split distributions

    /**
     * Null / "no valid split" candidate. Merit is -inf to match River's
     * BranchFactory default, so it always sorts below any real candidate and is
     * never selected as best/second-best unless it is the only option.
     */
    public SplitCandidate() {
        this.merit = Double.NEGATIVE_INFINITY; this.attIndex = -1;
        this.numericThreshold = Double.NaN; this.nominalValue = -1;
        this.isNumeric = true; this.childrenStats = null;
    }

    /** Numeric binary split. */
    public SplitCandidate(double merit, int attIndex, double threshold, List<VarStats> children) {
        this.merit = merit; this.attIndex = attIndex;
        this.numericThreshold = threshold; this.nominalValue = -1;
        this.isNumeric = true; this.childrenStats = children;
    }

    /** Nominal binary split. */
    public SplitCandidate(double merit, int attIndex, int nominalValue, List<VarStats> children) {
        this.merit = merit; this.attIndex = attIndex;
        this.numericThreshold = Double.NaN; this.nominalValue = nominalValue;
        this.isNumeric = false; this.childrenStats = children;
    }

    public boolean isNullSplit() { return attIndex == -1; }

    @Override
    public int compareTo(SplitCandidate o) { return Double.compare(this.merit, o.merit); }
}
