# Fairness-Preserving-ML-Framework
A fairness-preserving ML framework integrating data bias quantification via Earth Mover’s Distance and model-level fairness evaluation. Tested on health datasets across 8 fairness metrics and 5 classifiers, it shows that data-level bias removal outperforms existing methods in eliminating fairness violations.

**Dataset**
- Heart Disease Prediction Dataset (https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)
- Adult Census Income Dataset (https://www.kaggle.com/datasets/uciml/adult-census-income)
- Sepsis Survival Minimal Clinical Records (https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records)

**Models Used**
- Logistic Regression
- k-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

**Fairness Metrics used**
- Demographic parity
- Equalised odds
- Equal opportunity
- False positive rate parity
- Treatment equality
- Discrimination score
- Disparate Impact
- Balanced Error Rate

**File Descriptor**
main.py: Python script implementing the fairness-aware ML pipeline.

**How to Run**
- Clone the repository.
- Install the required packages (e.g., pandas, scikit-learn, numpy).
- Run the Python script to explore the fairness evaluation workflow.

**Reference**
Uddin, S., Lu, H., and Hajati, F., Fairness-preserving framework for machine learning: data bias quantification, model evaluation, and robustness across multiple datasets. AI and Ethics, 2026. 6(1): p. 25.
