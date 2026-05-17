# Airfoil Lift & Drag Prediction System (with Streamlit Deployment)

## Introduction
Evaluating aerodynamic performance traditionally relies on computationally intensive Computational Fluid Dynamics (CFD) simulations or high-cost physical wind tunnel testing. While highly precise, these traditional approaches require extensive execution time per design iteration, slowing down the conceptual optimization phase.

This repository presents an end-to-end Machine Learning engineering framework developed to predict critical aircraft aerodynamic properties—specifically the **Coefficient of Lift ($C_L$)** and **Coefficient of Drag ($C_D$)**—instantaneously. By utilizing parameterized geometric profile features (CST coefficients) alongside environmental flow variables, this framework bypasses slow simulation cycles. 

To make the models actionable, the system is deployed via an interactive **Streamlit web application** that handles feature lookup, drives the pre-trained ensemble pipelines, and conducts real-time physical validation checks on predictions.

## Repository Structure
The project is decoupled into production-ready modular steps following the standard machine learning lifecycle:
* `data_cleaning.ipynb`: Pipeline initialization, mixed-type handling, and robust statistical outlier filtering.
* `RandomForest.ipynb`: Core implementation, cross-validation, and parallel core allocation for the Random Forest baseline.
* `Gradient Boosting.ipynb`: Comparative training structures mapping Base vs. Hyperparameter-Tuned Gradient Boosting Regressors.
* `LightGBM.ipynb`: High-efficiency Light Gradient Boosting Machine scripts capturing final performance enhancements.
* `Final_Test_app.py`: The production deployment file launching the interactive user interface.

## Web Application Features (`Final_Test_app.py`)
The user-facing Streamlit dashboard features engineering guardrails to bridge the gap between static models and real-world utility:
* **Dynamic Geometry Mapping:** Links directly with an internal lookup database (`airfoil_geometry_lookup.csv`) to automatically load complex structural CST geometric coefficients based on a user's selected airfoil profile name.
* **Operational Constraint Boundaries:** Enforces strict physical operational checks tailored to the training spectrum (e.g., Reynolds numbers between $5 \times 10^4$ and $5 \times 10^6$, and Angles of Attack between $-10^\circ$ and $15^\circ$).
* **Real-Time Physics Validation Engine:** Programmatically monitors prediction anomalies. If the model outputs extreme states—such as an absolute lift configuration $|C_L| > 2.5$ or highly skewed Lift-to-Drag ($L/D$) efficiency scales—the application flags the occurrence with contextual warnings to alert users to potential aerodynamic stall boundaries.

## Methodology

### 1. Data Engineering & Cleaning (`data_cleaning.ipynb`)
* **Scale Management:** Managed a large-scale simulation dataset encompassing over 860,000 distinct flight-state patterns.
* **Interquartile Range (IQR) Filtering:** Implemented a strict outlier elimination step applying a $1.5 \times \text{IQR}$ cutoff boundary explicitly on target variables ($C_L$ and $C_D$). This process effectively isolated and discarded unstable boundary-condition simulation entries (accounting for approx. 3.33% of raw data), ensuring clean gradient convergence during training.

### 2. Machine Learning Pipeline Architecture
The system evaluates three distinct tree-based ensemble frameworks using **K-Fold Cross-Validation** to model complex, non-linear fluid dynamics:
* **Random Forest Regressor:** Implemented with explicit multi-core constraints (`n_jobs` restrictions) to securely process large dimensional arrays without encountering local memory overheads.
* **Gradient Boosting Machine (GBM):** Built sequentially to minimize loss residual patterns, comparing a default configuration with tuned learning rate bounds.
* **LightGBM Regressor:** Deployed to maximize execution velocity across the heavy feature matrix. The optimized LightGBM model yielded a **7.66% improvement in Root Mean Squared Error (RMSE)** and a **7.87% reduction in Mean Absolute Error (MAE)** over default structural benchmarks for $C_D$ tracking.

## Dataset & Model Artifact Note
* **Portfolio Standards:** To keep the GitHub repository clean and lightweight, the raw data file (`AirfoilDatset.csv`) and the serialized inference binaries (`cl_model.pkl`, `cd_model.pkl`, `scaler.pkl`, `airfoil_label_encoder.pkl`, `feature_columns.pkl`) are untracked locally via the `.gitignore` setup.

## How to Run

### 1. Installation & Environment Setup
Clone the repository and install the production dependencies:
```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn streamlit joblib
