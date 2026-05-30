package moa.classifiers.trees.hatr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ADWIN2 drift detector faithfully ported from River's Cython implementation
 * (river/drift/adwin_c.pyx).
 *
 * Key properties that match River exactly:
 *  - Per-bucket variance tracking (Babcock et al. 2003)
 *  - Same epsilon formula: sqrt(2 * m_recip * variance_in_window * delta') + 2/3 * m_recip * delta'
 *    where delta' = log(2 * log(width) / delta)
 *  - Same scan order: oldest row → newest row, oldest slot → newest slot within row
 *  - Same compress direction: merges the two OLDEST slots in a row when overflow
 *  - delete_element removes ONE oldest bucket at a time; while-loop keeps shrinking until no drift
 *
 * This is the formulation from:
 *   Babcock et al., "Maintaining Variance and k-Medians over Data Stream Windows", PODS 2003.
 * (River's version, NOT the simpler Bifet & Gavalda 2007 formula.)
 */
public class ADWINDetector {
    private final double delta;
    private final int clock;
    private final int maxBuckets;
    private final int minWindowLength;
    private final int gracePeriod;

    // Row list: index 0 = newest row (size-1 buckets), index n-1 = oldest row
    private final List<ADWINBucket> rows = new ArrayList<>();

    // Global window statistics
    private double width;       // number of elements in window
    private double total;       // sum of all window elements
    private double variance;    // sum of (x - running_mean)^2 across window (Welford)
    private int tick;
    private int nDetections;
    private boolean drift;

    public ADWINDetector() { this(0.002, 32, 5, 5, 10); }

    public ADWINDetector(double delta, int clock, int maxBuckets, int minWindowLength, int gracePeriod) {
        this.delta = delta; this.clock = clock; this.maxBuckets = maxBuckets;
        this.minWindowLength = minWindowLength; this.gracePeriod = gracePeriod;
        reset();
    }

    private void reset() {
        rows.clear();
        rows.add(new ADWINBucket(maxBuckets));
        width = 0; total = 0; variance = 0; tick = 0; nDetections = 0; drift = false;
    }

    public void update(double x) {
        if (drift) reset();
        insertElement(x);
        tick++;
        drift = (tick % clock == 0 && width > gracePeriod) && detectChange();
    }

    // Insert 

    private void insertElement(double x) {
        width++;
        // River's exact variance update (matches adwin_c.pyx _insert_element):
        //   incremental = (old_width) * (x - old_mean)^2 / new_width
        // width is already incremented; total is not yet updated → old_mean = total/(width-1)
        if (width > 1) {
            double oldMean = total / (width - 1);
            variance += (width - 1) * (x - oldMean) * (x - oldMean) / width;
        }
        total += x;
        rows.get(0).insertData(x, 0.0);
        compressBuckets();
    }

    private void compressBuckets() {
        for (int i = 0; i < rows.size(); i++) {
            ADWINBucket bucket = rows.get(i);
            if (bucket.currentIdx == maxBuckets + 1) {
                if (i + 1 >= rows.size()) rows.add(new ADWINBucket(maxBuckets));
                ADWINBucket next = rows.get(i + 1);
                double n = bucketSize(i);               // size of each slot at row i = 2^i
                double mu1 = bucket.getTotalAt(0) / n;
                double mu2 = bucket.getTotalAt(1) / n;
                double total12 = bucket.getTotalAt(0) + bucket.getTotalAt(1);
                double var12 = bucket.getVarianceAt(0) + bucket.getVarianceAt(1) + n * n * (mu1 - mu2) * (mu1 - mu2) / (2 * n);
                next.insertData(total12, var12);
                bucket.compress(2);
                if (next.currentIdx <= maxBuckets) break;
            } else {
                break;
            }
        }
    }

    // Delete (oldest element)

    private double deleteElement() {
        // Find oldest non-empty row
        int lastIdx = rows.size() - 1;
        ADWINBucket bucket = rows.get(lastIdx);
        double n = bucketSize(lastIdx);
        double u = bucket.getTotalAt(0);
        double v = bucket.getVarianceAt(0);
        double mu = u / n;

        // Remove from global window stats
        width -= n;
        total -= u;
        double muWindow = (width > 0) ? total / width : 0.0;
        double incVariance = v + n * width * (mu - muWindow) * (mu - muWindow) / (n + width);
        variance -= incVariance;
        variance = Math.max(0.0, variance);

        bucket.remove();
        if (bucket.currentIdx == 0) {
            rows.remove(lastIdx);
        }
        return n;
    }

    // Drift detection

    private boolean detectChange() {
        boolean changeDetected = false;
        boolean reduceWidth = true;

        while (reduceWidth) {
            reduceWidth = false;
            boolean exitFlag = false;

            double n0 = 0, n1 = width;
            double u0 = 0, u1 = total;
            double v0 = 0, v1 = variance;

            // Scan from oldest row to newest row
            for (int idx = rows.size() - 1; idx >= 0; idx--) {
                if (exitFlag) break;
                ADWINBucket bucket = rows.get(idx);
                double n2 = bucketSize(idx);

                for (int k = 0; k < bucket.currentIdx; k++) {
                    double u2 = bucket.getTotalAt(k);
                    double mu2 = u2 / n2;

                    if (n0 > 0) {
                        double mu0 = u0 / n0;
                        v0 += bucket.getVarianceAt(k) + n0 * n2 * (mu0 - mu2) * (mu0 - mu2) / (n0 + n2);
                    }
                    if (n1 > 0) {
                        double mu1 = u1 / n1;
                        v1 -= bucket.getVarianceAt(k) + n1 * n2 * (mu1 - mu2) * (mu1 - mu2) / (n1 + n2);
                    }

                    n0 += n2; n1 -= n2;
                    u0 += u2; u1 -= u2;

                    if (idx == 0 && k == bucket.currentIdx - 1) {
                        exitFlag = true;
                        break;
                    }

                    if (n1 >= minWindowLength && n0 >= minWindowLength) {
                        double deltaMean = u0 / n0 - u1 / n1;
                        if (evaluateCut(n0, n1, deltaMean)) {
                            changeDetected = true;
                            reduceWidth = true;
                            if (width > 0) {
                                n0 -= deleteElement();
                                exitFlag = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (changeDetected) nDetections++;
        return changeDetected;
    }

    private boolean evaluateCut(double n0, double n1, double deltaMean) {
        if (width <= 0 || width <= 1) return false;
        double deltaPrime = Math.log(2.0 * Math.log(width) / delta);
        if (deltaPrime <= 0) return false;
        double mRecip = (1.0 / (n0 - minWindowLength + 1)) + (1.0 / (n1 - minWindowLength + 1));
        double varianceInWindow = variance / width;
        double epsilon = Math.sqrt(2.0 * mRecip * varianceInWindow * deltaPrime) + 2.0 / 3.0 * mRecip * deltaPrime;
        return Math.abs(deltaMean) > epsilon;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private static double bucketSize(int rowIdx) { return Math.pow(2, rowIdx); }

    public boolean isDrift() { return drift; }
    public double getWidth() { return width; }
    public double getMean() { return width > 0 ? total / width : 0.0; }
    public int getNDetections() { return nDetections; }

    public ADWINDetector createNew() {
        return new ADWINDetector(delta, clock, maxBuckets, minWindowLength, gracePeriod);
    }

    // ── Inner bucket class (matches River's Bucket exactly) ───────────────────

    private static class ADWINBucket {
        final int maxSize;
        double[] totals;
        double[] variances;
        int currentIdx;

        ADWINBucket(int maxSize) {
            this.maxSize = maxSize;
            totals = new double[maxSize + 1];
            variances = new double[maxSize + 1];
            currentIdx = 0;
        }

        void insertData(double total, double variance) {
            totals[currentIdx] = total;
            variances[currentIdx] = variance;
            currentIdx++;
        }

        void remove() { compress(1); }

        /** Remove first n elements (oldest slots) by shifting left. */
        void compress(int n) {
            System.arraycopy(totals, n, totals, 0, totals.length - n);
            System.arraycopy(variances, n, variances, 0, variances.length - n);
            Arrays.fill(totals, totals.length - n, totals.length, 0.0);
            Arrays.fill(variances, variances.length - n, variances.length, 0.0);
            currentIdx -= n;
        }

        double getTotalAt(int k) { return totals[k]; }
        double getVarianceAt(int k) { return variances[k]; }
    }
}
