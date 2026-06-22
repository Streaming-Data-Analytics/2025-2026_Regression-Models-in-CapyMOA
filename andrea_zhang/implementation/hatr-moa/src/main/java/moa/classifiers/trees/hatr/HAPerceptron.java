package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;

/**
 * Online linear regression (SGD, squared loss) for HATR leaf nodes.
 * Gradient is clipped to ±1e12 before updating weights and intercept.
 * Weights are indexed by attribute position; unseen attributes start at 0.
 */
public class HAPerceptron {
    private double[] weights;
    private double intercept = 0.0;
    private final double lr;          // weight learning rate (SGD)
    private final double interceptLr; // intercept learning rate
    private final double l2;
    private static final double CLIP_GRADIENT = 1e12;

    public HAPerceptron(int numAtts, double lr, double interceptLr, double l2) {
        this.weights = new double[numAtts];
        this.lr = lr;
        this.interceptLr = interceptLr;
        this.l2 = l2;
    }

    private HAPerceptron(double[] weights, double intercept, double lr, double interceptLr, double l2) {
        this.weights = weights.clone();
        this.intercept = intercept;
        this.lr = lr;
        this.interceptLr = interceptLr;
        this.l2 = l2;
    }

    public double predict(Instance inst) {
        double pred = 0.0;
        int nAtt = inst.numAttributes() - 1;
        for (int i = 0; i < nAtt; i++) {
            if (i < weights.length && !inst.attribute(i).isNominal() && !Double.isNaN(inst.value(i))) {
                pred += weights[i] * inst.value(i);
            }
        }
        return pred + intercept;
    }

    public void train(Instance inst, double y, double w) {
        ensureCapacity(inst.numAttributes());
        double pred = predict(inst);

        // loss_gradient = Squared.gradient(y, pred) * w, then clamped.
        // Squared.gradient(y_true, y_pred) = 2 * (y_pred - y_true).
        double lossGrad = 2.0 * (pred - y) * w;
        if (lossGrad > CLIP_GRADIENT) lossGrad = CLIP_GRADIENT;
        if (lossGrad < -CLIP_GRADIENT) lossGrad = -CLIP_GRADIENT;

        intercept -= interceptLr * lossGrad;

        // Weight update via SGD: w_i -= lr * (loss_gradient * x_i + l2 * w_i).
        int nAtt = inst.numAttributes() - 1;
        for (int i = 0; i < nAtt; i++) {
            if (!inst.attribute(i).isNominal() && !Double.isNaN(inst.value(i))) {
                weights[i] -= lr * (lossGrad * inst.value(i) + l2 * weights[i]);
            }
        }
    }

    private void ensureCapacity(int needed) {
        if (needed > weights.length) {
            double[] w2 = new double[needed];
            System.arraycopy(weights, 0, w2, 0, weights.length);
            weights = w2;
        }
    }

    public HAPerceptron copy() {
        return new HAPerceptron(weights, intercept, lr, interceptLr, l2);
    }
}
