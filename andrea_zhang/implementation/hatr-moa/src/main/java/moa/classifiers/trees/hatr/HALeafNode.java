package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor;
import java.util.*;

/**
 * Base abstract leaf node for HATR.
 * Manages attribute observers (one per observed attribute, keyed by index).
 */
public abstract class HALeafNode implements HANode {
    public VarStats stats;
    public int depth;
    public final HAAttributeObserver defaultObserver; // prototype for numeric attributes
    public Map<Integer, HAAttributeObserver> observers; // null when inactive
    public Set<Integer> disabledAtts;
    public double lastSplitAttemptAt;

    protected HALeafNode(VarStats stats, int depth, HAAttributeObserver defaultObserver) {
        this.stats = (stats != null) ? stats : new VarStats();
        this.depth = depth;
        this.defaultObserver = defaultObserver;
        this.observers = new HashMap<>();
        this.disabledAtts = new HashSet<>();
        this.lastSplitAttemptAt = getTotalWeight();
    }

    public boolean isActive() { return observers != null; }
    public void activate() { if (!isActive()) observers = new HashMap<>(); }
    public void deactivate() { observers = null; }

    @Override
    public abstract double getTotalWeight();
    public abstract void updateStats(double y, double w);
    @Override
    public abstract double getPrediction(Instance inst);
    public abstract int calculatePromise();

    /** Base learning: update stats and attribute observers. */
    public void baseLearnOne(Instance inst, HoeffdingAdaptiveTreeRegressor tree) {
        double y = inst.classValue();
        double w = inst.weight();
        updateStats(y, w);
        if (isActive()) updateObservers(inst, y, w, tree);
    }

    private void updateObservers(Instance inst, double y, double w, HoeffdingAdaptiveTreeRegressor tree) {
        int classIdx = inst.classIndex();
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i == classIdx || disabledAtts.contains(i)) continue;
            if (Double.isNaN(inst.value(i))) continue;
            HAAttributeObserver obs = observers.get(i);
            if (obs == null) {
                boolean isNominal = inst.attribute(i).isNominal();
                obs = isNominal ? new HANominalObserver() : defaultObserver.createNew();
                observers.put(i, obs);
            }
            obs.observe(inst.value(i), y, w);
        }
    }

    /** Collect split candidates from all attribute observers. */
    public List<SplitCandidate> bestSplitCandidates(HoeffdingAdaptiveTreeRegressor tree) {
        List<SplitCandidate> candidates = new ArrayList<>();
        if (tree.meritPreprune) candidates.add(new SplitCandidate()); // null split
        for (Map.Entry<Integer, HAAttributeObserver> e : observers.entrySet()) {
            candidates.add(e.getValue().bestSplitCandidate(e.getKey(), stats, tree.minSamplesSplit));
        }
        return candidates;
    }

    public void disableAttribute(int attIndex) {
        if (observers != null) observers.remove(attIndex);
        disabledAtts.add(attIndex);
    }

    // HANode interface 

    @Override public int getHeight() { return 0; }
    @Override public int getNNodes() { return 1; }
    @Override public int getNLeaves() { return 1; }
    @Override public List<HALeafNode> iterLeaves() { return Collections.singletonList(this); }
}
