package moa.classifiers.trees.hatr;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor;
import java.util.List;

/** Base interface for all HATR nodes. */
public interface HANode {
    double getTotalWeight();
    int getHeight();
    int getNNodes();
    int getNLeaves();
    List<HALeafNode> iterLeaves();
    double getPrediction(Instance inst);
}
