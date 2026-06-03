package moa.classifiers.trees.hatr;

/**
 * Interface for attribute observers (splitters) used in HATR regression leaves.
 * Processes observations for a single attribute and proposes the best binary split.
 */
public interface HAAttributeObserver {
    void observe(double attVal, double targetVal, double weight);
    SplitCandidate bestSplitCandidate(int attIndex, VarStats preSplitDist, int minSamplesSplit);
    HAAttributeObserver createNew();
}
