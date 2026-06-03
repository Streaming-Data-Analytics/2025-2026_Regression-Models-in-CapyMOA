# Implementation

This folder contains the core implementation of the **Hoeffding Adaptive Tree Regressor (HATR)**:
a native MOA port of River's `HoeffdingAdaptiveTreeRegressor`, wrapped for CapyMOA.

## Subfolders

| Folder | Description |
|---|---|
| `hatr-moa/` | Java implementation for MOA — compile with Maven |
| `hatr_capymoa/` | Python CapyMOA wrapper — uses the JAR built above |

## Setup

### 1. Create the conda environment and install Python dependencies

```bash
conda create -n sda_project python=3.11
conda activate sda_project
pip install -r ../requirements.txt
```

### 2. Register `moa.jar` with Maven

Maven needs `moa.jar` in the local repository. `capymoa` bundles it, so run this
**after** activating the environment above:

```bash
MOA_JAR=$(python -c "import capymoa, os; print(os.path.join(os.path.dirname(capymoa.__file__), '..', '..', 'moa.jar'))")
mvn install:install-file -Dfile="$MOA_JAR" \
  -DgroupId=nz.ac.waikato.cms.moa -DartifactId=moa \
  -Dversion=2024.07.0 -Dpackaging=jar
```

### 3. Build the JAR

```bash
cd hatr-moa
mvn package -DskipTests
# Produces: target/hoeffding-adaptive-tree-regressor-moa-1.0-SNAPSHOT.jar
```

## Using the wrapper

```python
import os, sys, jpype

IMPL_DIR = "/path/to/andrea_zhang/implementation"
HATR_JAR = os.path.join(IMPL_DIR, "hatr-moa", "target",
                         "hoeffding-adaptive-tree-regressor-moa-1.0-SNAPSHOT.jar")
jpype.addClassPath(HATR_JAR)   # must come before `import capymoa`
sys.path.insert(0, IMPL_DIR)

from hatr_capymoa import HoeffdingAdaptiveTreeRegressor
from capymoa.datasets import Fried
from capymoa.evaluation import prequential_evaluation

stream = Fried()
learner = HoeffdingAdaptiveTreeRegressor(stream.get_schema(), random_seed=1)
results = prequential_evaluation(stream, learner, max_instances=1000)
print(results["cumulative"].rmse())
```

## Contributing to CapyMOA upstream

1. Merge `hatr-moa/src/` into MOA upstream so the class ends up in `moa.jar`.
2. Copy `hatr_capymoa/_hatr.py` to `src/capymoa/regressor/_hatr.py`.
3. Add `from ._hatr import HoeffdingAdaptiveTreeRegressor` to `src/capymoa/regressor/__init__.py`.
