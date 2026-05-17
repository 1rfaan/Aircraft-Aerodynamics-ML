# Predicting Aircraft Aerodynamics Using Machine Learning

## Introduction
Computational Fluid Dynamics (CFD) is traditionally used to evaluate the aerodynamic performance of aircraft designs. While highly accurate, CFD simulations require massive computational power and hours—sometimes days—to run. This Final Year Project bridges the gap by developing a Machine Learning framework to predict critical aerodynamic properties (such as Lift and Drag coefficients) instantaneously based on geometric and environmental input variables. 

This solution provides aerospace designers with a rapid, real-time optimization tool during the conceptual design phase.

## Methodology
The machine learning architecture follows a robust data science workflow tailored for engineering optimization:

### 1. Data Preprocessing & Feature Engineering
* **Data Normalization:** Applied scaling techniques to handle engineering parameters with wide numerical distributions (e.g., Reynolds numbers, Mach numbers, and Angle of Attack).
* **Dimensionality Reduction:** Conducted multi-collinearity checks and feature importance analyses to retain only the most impactful aerodynamic indicators.

### 2. Predictive Modeling & Algorithms
Implemented, optimized, and cross-evaluated multiple machine learning models to capture complex, non-linear fluid dynamics relationships:
* **Linear Baselines:** Used for benchmarking performance thresholds.
* **Tree-Based Ensembles:** Utilized Random Forests and Gradient Boosting architectures to accurately map continuous surface changes to aerodynamic coefficients.
* **Neural Networks (Optional/If applicable):** Deployed Multilayer Perceptrons (MLPs) for high-dimensional feature interaction mapping.

### 3. Validation Strategy
* Utilized k-fold cross-validation to guarantee model stability.
* Evaluated predictions using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared ($R^2$) variance scoring.

## Dataset Note
* **Data Privacy:** In accordance with university and institutional confidentiality agreements, the raw aircraft engineering datasets used for model training are restricted and omitted from this public repository. 
* **Execution:** Code scripts are configured to accept mock geometric and environmental arrays to demonstrate architectural viability.

## How to Run
1. **Environment Setup:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
