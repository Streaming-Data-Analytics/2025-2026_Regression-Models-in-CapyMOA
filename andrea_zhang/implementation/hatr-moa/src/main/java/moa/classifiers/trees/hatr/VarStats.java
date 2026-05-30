package moa.classifiers.trees.hatr;

/**
 * Incremental sample variance using Welford's algorithm.
 *
 */
public class VarStats {
    double n = 0.0;      // sum of weights
    double mean = 0.0;   // running mean (river.stats.Mean)
    double S = 0.0;      // sum of squared deviations (river.stats.Var._S)

    /** Welford update — matches Mean.update + Var.update. */
    public void update(double x, double w) {
        // Mean.update: n += w; mean += (w/n)*(x - mean)
        double meanOld = mean;
        n += w;
        mean += (w / n) * (x - mean);
        // Var.update: S += w*(x - mean_old)*(x - mean_new)
        S += w * (x - meanOld) * (x - mean);
    }

    /** Sample variance S/(n-1). No clamping, matching River's Var.get(). */
    public double get() {
        if (n > 1.0) return S / (n - 1.0);
        return 0.0;
    }

    public double getMean() { return n > 0.0 ? mean : 0.0; }
    public double getN() { return n; }

    /** this + o  (river Var.__add__). */
    public VarStats add(VarStats o) {
        VarStats r = new VarStats();
        combineAdd(r, this.n, this.mean, this.S, o.n, o.mean, o.S);
        return r;
    }

    /** this - o  (river Var.__sub__). */
    public VarStats subtract(VarStats o) {
        VarStats r = new VarStats();
        combineSub(r, this.n, this.mean, this.S, o.n, o.mean, o.S);
        return r;
    }

    public void addInPlace(VarStats o) {
        combineAdd(this, this.n, this.mean, this.S, o.n, o.mean, o.S);
    }

    public void subtractInPlace(VarStats o) {
        combineSub(this, this.n, this.mean, this.S, o.n, o.mean, o.S);
    }

    // River's exact parallel combination formulas 

    // Var.__iadd__:
    //   S = S_a + S_b + (mean_a - mean_b)^2 * n_a * n_b / (n_a + n_b)
    //   Mean.__iadd__: n = n_a + n_b; mean = (n_a*mean_a + n_b*mean_b) / n
    private static void combineAdd(VarStats out, double na, double ma, double sa, double nb, double mb, double sb) {
        double newN = na + nb;
        double newS = sa + sb;
        if (newN != 0.0) {
            newS += (ma - mb) * (ma - mb) * na * nb / newN;
        }
        double newMean = newN > 0.0 ? (na * ma + nb * mb) / newN : 0.0;
        out.n = newN; out.mean = newMean; out.S = newS;
    }

    // Var.__isub__ (Mean is updated first, then S uses the NEW mean and NEW n):
    //   n = n_a - n_b; mean = (n_a*mean_a - n_b*mean_b) / n
    //   S = S_a - S_b - (mean_new - mean_b)^2 * n * n_b / (n + n_b)
    private static void combineSub(VarStats out, double na, double ma, double sa, double nb, double mb, double sb) {
        double newN = na - nb;
        double newMean;
        if (newN > 0.0) {
            newMean = (na * ma - nb * mb) / newN;
        } else {
            newN = 0.0;
            newMean = 0.0;
        }
        double newS = sa - sb;
        double denom = newN + nb;
        if (denom != 0.0) {
            newS -= (newMean - mb) * (newMean - mb) * newN * nb / denom;
        }
        out.n = newN; out.mean = newMean; out.S = newS;
    }

    public VarStats copy() {
        VarStats v = new VarStats();
        v.n = n; v.mean = mean; v.S = S;
        return v;
    }

    @Override
    public String toString() {
        return String.format("Var(mean=%.4f, var=%.4f, n=%.0f)", getMean(), get(), n);
    }
}
