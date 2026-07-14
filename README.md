# Customer Churn Prediction using Machine Learning with Advanced Feature Engineering

An end-to-end, production-quality data science project that predicts telecom customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Built to demonstrate the full ML lifecycle — from raw data to a saved, deployable model — with detailed comments throughout for beginners.

---

## 1. Project Overview

Customer churn (a customer leaving for a competitor) is one of the costliest problems in the telecom industry — acquiring a new customer typically costs far more than retaining an existing one. This project builds a classification model that flags customers who are likely to churn **before** they leave, so a retention team can intervene with targeted offers.

The project covers: data cleaning, exploratory data analysis, advanced feature engineering, training and comparing 7 ML algorithms, hyperparameter tuning, rigorous evaluation, feature importance analysis, and a reusable prediction function — all wrapped in modular, documented, PEP8-compliant Python code.

## 2. Objectives

- Predict whether a customer will churn (`Yes`/`No`) from demographic, account, and usage data.
- Engineer new, business-meaningful features beyond the raw columns.
- Compare multiple algorithms fairly using precision, recall, F1, and ROC AUC (not just accuracy, since the target is imbalanced).
- Tune the top candidate models with `GridSearchCV`.
- Surface **actionable business insights** a retention team could act on immediately.
- Package everything as a reusable, portfolio-ready codebase.

## 3. Dataset Description

- **Source:** Telco Customer Churn dataset (Kaggle / IBM sample dataset)
- **Rows:** 7,043 customers
- **Columns:** 21 (20 features + target)
- **Target:** `Churn` (Yes / No) — **26.5%** of customers in the dataset churned
- **Feature groups:**
  - *Demographics:* gender, SeniorCitizen, Partner, Dependents
  - *Account info:* tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
  - *Services:* PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

## 4. Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost |
| Model persistence | joblib |
| Statistics | scipy |
| Notebooks | Jupyter |

## 5. Machine Learning Models

Seven classifiers were trained and benchmarked:

Logistic Regression · Decision Tree · Random Forest · K-Nearest Neighbors · Support Vector Machine · Gradient Boosting · XGBoost

The top three (Random Forest, Decision Tree, Gradient Boosting) were additionally tuned with `GridSearchCV` (5-fold cross-validation, `roc_auc` scoring).

## 6. Installation

```bash
# 1. Clone / unzip the project, then move into it
cd Customer_Churn_Project

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 7. Usage

**Run the full pipeline** (cleaning → feature engineering → training → tuning → evaluation → save model):
```bash
python run_pipeline.py
```

**Run EDA only** (prints summary stats + saves all plots to `outputs/plots/`):
```bash
python run_eda.py
```

**Predict a single new customer** (bonus feature):
```bash
python predict.py
```

**Explore interactively** — open the notebooks in order:
```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Preprocessing.ipynb
jupyter notebook notebooks/03_Model_Training.ipynb
```

## 8. Folder Structure

```
Customer_Churn_Project/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
│
├── src/
│   ├── preprocessing.py          # DataCleaner: load, fix dtypes, handle missing, dedupe
│   ├── feature_engineering.py    # FeatureEngineer: encode, scale, transform, create features
│   ├── train_model.py            # ModelTrainer, get_baseline_models, tune_model (GridSearchCV)
│   ├── evaluation.py             # Confusion matrix, ROC/PR curves, learning curve, feature importance
│   └── utils.py                  # Logging, path constants, save/load helpers
│
├── models/
│   ├── best_model.pkl            # Overall best model (by ROC AUC)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── feature_engineer.pkl      # Fitted encoders/scalers, for reuse at inference time
│   └── feature_columns.pkl       # Exact column order expected by the model
│
├── outputs/
│   ├── plots/                    # 16 saved PNG visualizations
│   └── reports/                  # Model comparison CSVs, classification report, run metadata
│
├── run_eda.py                    # Standalone EDA script
├── run_pipeline.py               # Full end-to-end training pipeline
├── predict.py                    # Single-customer prediction demo
├── requirements.txt
└── README.md
```

## 9. Results

### Model Comparison (test set, 20% hold-out, n=1,409)

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | **0.8084** | **0.6722** | **0.5428** | **0.6006** | **0.8468** |
| Gradient Boosting (Tuned) | 0.7977 | 0.6606 | 0.4893 | 0.5622 | 0.8427 |
| Gradient Boosting | 0.8013 | 0.6621 | 0.5134 | 0.5783 | 0.8427 |
| Random Forest (Tuned) | 0.8013 | 0.6632 | 0.5107 | 0.5770 | 0.8425 |
| Decision Tree (Tuned) | 0.7921 | 0.6576 | 0.4519 | 0.5357 | 0.8299 |
| SVM | 0.7771 | 0.7273 | 0.2567 | 0.3794 | 0.8252 |
| KNN | 0.7928 | 0.6235 | 0.5535 | 0.5864 | 0.8221 |
| Random Forest | 0.7814 | 0.6071 | 0.5000 | 0.5484 | 0.8215 |
| XGBoost | 0.7708 | 0.5785 | 0.5027 | 0.5379 | 0.8195 |
| Decision Tree | 0.7239 | 0.4816 | 0.5241 | 0.5019 | 0.6596 |

**Best model: Logistic Regression** — a simple, highly interpretable linear model beat every tree-based ensemble on this dataset. That's a common (and useful!) outcome when the underlying signal is largely linear/monotonic (e.g., "shorter tenure → higher churn risk"), and it means the production model is also cheap to explain to non-technical stakeholders.

5-fold cross-validation on the training set confirmed this wasn't a lucky test split: **mean CV ROC AUC = 0.8497** (scores ranged 0.8237–0.8673).

### Best Hyperparameters Found (GridSearchCV)

| Model | Best Parameters |
|---|---|
| Random Forest | `n_estimators=300, max_depth=8, min_samples_split=5` |
| Decision Tree | `criterion=entropy, max_depth=4, min_samples_split=2` |
| Gradient Boosting | `n_estimators=100, learning_rate=0.05, max_depth=2` |

### Top 10 Most Important Features (Random Forest, used for interpretability)

1. `Contract_Month-to-month` — 0.099
2. `tenure` — 0.091
3. `TotalCharges` — 0.067
4. `TotalCharges_log` — 0.066
5. `MonthlyCharges` — 0.060
6. `TechSupport_No` — 0.058
7. `AvgMonthlySpend` — 0.055
8. `OnlineSecurity_No` — 0.052
9. `InternetService_Fiber optic` — 0.052
10. `TenureGroup_New Customer` — 0.041

*(Full top-20 chart: `outputs/plots/feature_importance.png`)*

### Generated Artifacts

- 16 visualizations in `outputs/plots/` (target distribution, contract/internet/payment vs. churn, correlation heatmap, pairplot, boxplots, confusion matrix, ROC curve, precision-recall curve, learning curve, feature importance)
- `outputs/reports/full_model_comparison.csv`, `classification_report.txt`, `feature_importance.csv`, `best_hyperparameters.json`, `run_metadata.json`
- Saved models in `models/` (loadable with `joblib.load(...)`)

## 10. Business Insights

1. **Contract type is the single biggest churn driver.** Month-to-month customers churn at ~43%, vs. under 3% for two-year contracts — pushing customers toward longer contracts (even with a small discount) is likely the highest-leverage retention lever available.
2. **Tenure and churn are strongly inversely related.** The first 12 months are the highest-risk window; a structured "first-year" onboarding/retention program would target the point of greatest loss.
3. **Electronic check payers churn the most (~45%)**, far above customers on automatic bank transfer or credit card. Incentivizing a switch to autopay (e.g., a small bill credit) is a low-cost retention lever.
4. **Automatic payment methods correlate with lower churn**, likely because auto-pay customers are more "set and forget" / less actively comparing prices.
5. **Fiber optic internet customers churn more (~42%) than DSL customers.** This may reflect pricing, competitive market pressure, or service reliability complaints and deserves a targeted satisfaction survey.
6. **Higher monthly charges increase churn risk.** Price-sensitive customers near the top of the pricing tiers are prime targets for a loyalty discount before they consider leaving.
7. **Customers without TechSupport or OnlineSecurity add-ons churn more.** Bundling a free trial of these services for new customers could reduce early churn.
8. **Long-tenure customers rarely churn** — once a customer passes ~3 years, they are unlikely to leave, so retention spend should be weighted toward newer customers.
9. **Senior citizens churn more than non-seniors** (~42% vs ~24%), suggesting a need for senior-friendly support channels or pricing.
10. **Customers without a partner or dependents churn more**, consistent with single-person households having fewer switching costs.
11. **Number of subscribed services is protective.** Customers with more bundled services are "stickier" — cross-selling additional services is a legitimate retention strategy, not just an upsell.
12. **Gender has virtually no effect on churn** — marketing/retention spend should not be segmented by gender.
13. **Paperless billing customers show slightly higher churn**, possibly correlating with more price-conscious, less loyal segments (worth a follow-up study, not a causal claim).
14. **No meaningful outliers exist in tenure, MonthlyCharges, or TotalCharges** — the data is clean and business-plausible, so no aggressive outlier removal was needed.
15. **A simple, interpretable Logistic Regression model matched or beat every tree ensemble**, meaning churn risk here is largely explainable through linear combinations of a few strong signals (contract type, tenure, charges) — a great sign for building trust with business stakeholders who need to understand *why* a customer is flagged.

## 11. Future Improvements

- **Class imbalance handling:** try `class_weight='balanced'`, SMOTE, or threshold tuning to push recall higher (currently ~54% of churners are caught — a retention team may prefer higher recall at the cost of some precision).
- **SHAP values** for per-customer, per-feature explanations (more granular than global feature importance).
- **Time-aware validation:** if historical snapshots become available, backtest with a time-based split rather than random split.
- **Deploy as an API** (e.g., FastAPI) or interactive **Streamlit dashboard** for the retention team to score customers in real time.
- **Cost-sensitive optimization:** incorporate the actual dollar cost of a false negative (lost customer) vs. false positive (wasted retention offer) into model selection, rather than optimizing ROC AUC alone.
- **A/B test** retention interventions suggested by the model's top features to measure real causal impact, since the insights above are correlational.

---

*Built as a portfolio-ready, industry-style project. Every function includes type hints, docstrings, and inline comments explaining the reasoning behind each decision.*
