package moa.classifiers.trees.hatr;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Truncated E-BST: rounds feature values before inserting (reduces memory).
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
        super.observe(roundHalfEven(attVal, digits), targetVal, weight);
    }

    static double roundHalfEven(double x, int digits) {
        if (Double.isNaN(x) || Double.isInfinite(x)) return x;
        return new BigDecimal(x).setScale(digits, RoundingMode.HALF_EVEN).doubleValue();
    }

    @Override
    public HAAttributeObserver createNew() { return new HATEBSTObserver(digits); }
}
