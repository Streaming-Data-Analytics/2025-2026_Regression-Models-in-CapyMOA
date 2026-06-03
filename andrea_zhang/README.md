# Andrea Zhang — Streaming Data Analytics Project

**Course:** [Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/) — Politecnico di Milano  
**Project:** Wrapping Regression Models from River to CapyMOA  
**Student:** Andrea Zhang

## Project Overview

This project extends **CapyMOA** with the **Hoeffding Adaptive Tree Regressor (HATR)**, a regression model from the **River** library. The implementation follows the full pipeline required by the assignment:

1. **Java (MOA)**: native re-implementation of River's HATR, compatible with MOA streaming interfaces.
2. **Python (CapyMOA wrapper)**: wraps the Java class as a `MOARegressor` usable in CapyMOA pipelines.
3. **Cross-framework comparison**: quality and performance benchmarks (River vs CapyMOA).
4. **CapyMOA tutorials**: extended versions of official tutorials that include HATR.

## Repository Structure

```
andrea_zhang/
├── README.md                          ← this file
│
├── implementation/                    ← Core implementation (Parts 1 + wrapper)
│   ├── README.md                      ← build instructions and usage examples
│   ├── hatr-moa/                      ← Java/MOA implementation
│   │   ├── pom.xml                    ← Maven build file
│   │   ├── src/main/java/moa/classifiers/trees/
│   │   │   ├── HoeffdingAdaptiveTreeRegressor.java   ← main learner
│   │   │   └── hatr/                  ← helper classes (nodes, observers, …)
│   │   └── target/                    ← build output (JAR after mvn package)
│   └── hatr_capymoa/                  ← Python CapyMOA wrapper package
│       ├── _hatr.py                   ← HoeffdingAdaptiveTreeRegressor wrapper
│       └── __init__.py                ← package entry point
│
├── experiments/                       ← Cross-framework benchmarks (Parts 2 + 3)
│   ├── README.md
│   ├── hatr-comparison.ipynb          ← River vs CapyMOA
│   ├── hatr-crossframework-quality.ipynb     ← MAE/RMSE comparison
│   └── hatr-crossframework-performance.ipynb ← training time & memory usage
│
└── tutorials/                         ← Extended CapyMOA tutorials (Part 4)
    ├── README.md
    ├── tutorial_01ext_hatr_evaluation.ipynb   ← Tutorial 01 + HATR
    └── tutorial_04ext_hatr_drift_streams.ipynb ← Tutorial 04 + HATR
```
