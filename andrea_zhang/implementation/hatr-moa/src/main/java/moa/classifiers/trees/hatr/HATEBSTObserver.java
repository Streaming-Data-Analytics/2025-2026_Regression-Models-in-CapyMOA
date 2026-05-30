package moa.classifiers.trees.hatr;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Truncated E-BST: rounds feature values before inserting (reduces memory).
 *
 * The rounding replicates Python's built-in {@code round(x, digits)} used by
 * River's {@code TEBSTSplitter}: the exact double value is rounded to {@code digits}
 * decimal places with round-half-to-even (banker's rounding). Using
 * {@code new BigDecimal(double)} (the exact binary value, not the shortest string)
 * with {@link RoundingMode#HALF_EVEN} matches CPython's float rounding.
 */
public class HATEBSTObserver extends HAEBSTObserver {
    private final int digits;

    public HATEBSTObserver() { this(1); }

    public HATEBSTObserver(int digits) {
        super();
        this.digits = digits;
    }

    @Override
    public void observe(double attVal, double targetVal, double weight) {
        if (Double.isNaN(attVal) || Double.isInfinite(attVal)) {
            if (Double.isNaN(attVal)) return;
        }
        super.observe(roundHalfEven(attVal, digits), targetVal, weight);
    }

    /** Equivalent to Python's round(x, digits). */
    static double roundHalfEven(double x, int digits) {
        if (Double.isNaN(x) || Double.isInfinite(x)) return x;
        return new BigDecimal(x).setScale(digits, RoundingMode.HALF_EVEN).doubleValue();
    }

    @Override
    public HAAttributeObserver createNew() { return new HATEBSTObserver(digits); }
}
