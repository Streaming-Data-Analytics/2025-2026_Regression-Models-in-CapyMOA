package moa.classifiers.trees.hatr.util;

import org.apache.commons.math3.special.Erf;

/**
 * Standard-normal CDF for the z-test in AdaBranchNode.
 *
 * Matches Python's {@code statistics.NormalDist().cdf(x)} which computes
 * {@code 0.5 * erfc(-x / sqrt(2))}.
 */
public class NormalDist {
    private static final double SQRT2 = Math.sqrt(2.0);

    public static double cdf(double x) {
        return 0.5 * Erf.erfc(-x / SQRT2);
    }
}
