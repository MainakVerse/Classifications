# 🧠 Machine Learning Classification Algorithms

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.0-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

A comprehensive list of **Machine Learning Classification Algorithms** (non–neural network based) with their respective Python packages and imports.

---

## 📘 Table of Algorithms

| No. | Algorithm | Python / scikit-learn Module |
|-----|------------|------------------------------|
| 1 | Logistic Regression | `from sklearn.linear_model import LogisticRegression` |
| 2 | Linear Discriminant Analysis (LDA) | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` |
| 3 | Quadratic Discriminant Analysis (QDA) | `from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis` |
| 4 | Ridge Classifier | `from sklearn.linear_model import RidgeClassifier` |
| 5 | Perceptron | `from sklearn.linear_model import Perceptron` |
| 6 | Decision Tree Classifier | `from sklearn.tree import DecisionTreeClassifier` |
| 7 | Random Forest Classifier | `from sklearn.ensemble import RandomForestClassifier` |
| 8 | Extra Trees Classifier | `from sklearn.ensemble import ExtraTreesClassifier` |
| 9 | Gradient Boosting Classifier | `from sklearn.ensemble import GradientBoostingClassifier` |
| 10 | XGBoost | `from xgboost import XGBClassifier` *(install via `pip install xgboost`)* |
| 11 | LightGBM | `from lightgbm import LGBMClassifier` *(install via `pip install lightgbm`)* |
| 12 | CatBoost | `from catboost import CatBoostClassifier` *(install via `pip install catboost`)* |
| 13 | AdaBoost | `from sklearn.ensemble import AdaBoostClassifier` |
| 14 | Gaussian Naive Bayes | `from sklearn.naive_bayes import GaussianNB` |
| 15 | Multinomial Naive Bayes | `from sklearn.naive_bayes import MultinomialNB` |
| 16 | Bernoulli Naive Bayes | `from sklearn.naive_bayes import BernoulliNB` |
| 17 | Complement Naive Bayes | `from sklearn.naive_bayes import ComplementNB` |
| 18 | Categorical Naive Bayes | `from sklearn.naive_bayes import CategoricalNB` |
| 19 | Bayesian Network Classifier | `from pomegranate import BayesianNetwork` *(install via `pip install pomegranate`)* |
| 20 | K-Nearest Neighbors (KNN) | `from sklearn.neighbors import KNeighborsClassifier` |
| 21 | Radius Neighbors Classifier | `from sklearn.neighbors import RadiusNeighborsClassifier` |
| 22 | Support Vector Machine (SVM) | `from sklearn.svm import SVC` |
| 23 | Linear SVM | `from sklearn.svm import LinearSVC` |
| 24 | Non-linear SVM (RBF / Polynomial) | `from sklearn.svm import SVC` *(kernel='rbf' or 'poly')* |
| 25 | Nu-Support Vector Classifier | `from sklearn.svm import NuSVC` |
| 26 | Bagging Classifier | `from sklearn.ensemble import BaggingClassifier` |
| 27 | Stacking Classifier | `from sklearn.ensemble import StackingClassifier` |
| 28 | Voting Classifier | `from sklearn.ensemble import VotingClassifier` |
| 29 | Hidden Markov Model (HMM) | `from hmmlearn import hmm` *(install via `pip install hmmlearn`)* |
| 30 | Conditional Random Field (CRF) | `from sklearn_crfsuite import CRF` *(install via `pip install sklearn-crfsuite`)* |

---

## ⚙️ Installation

```bash
pip install scikit-learn xgboost lightgbm catboost pomegranate hmmlearn sklearn-crfsuite wittgenstein sklearn-genetic simpful
```

## Example Usage

```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

MADE WITH 💜 BY MAINAK
