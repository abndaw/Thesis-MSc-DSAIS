# Agentic AI System for Churn Prevention

MSc Dissertation : Data Science & Artificial Intelligence Strategy  
emlyon business school, 2024 cohort  
**Author:** Abdoulaye Ndaw

---

## What this is

End-to-end pipeline combining ML churn prediction, profit-driven targeting, SHAP-based offer selection, and LLM-generated personalised retention emails. Built on the IBM Telco Customer Churn dataset (7,043 customers).

**Key results on held-out test set:**
- ML targeting alone: **+46.3% profit** vs mass campaign baseline, contacting only 55.9% of customers
- ML + LLM (full agentic system): **+64.8% profit** vs baseline

## Repo structure

```
churn_prevention_system/
├── data/               # IBM Telco dataset
├── src/
│   ├── preprocess.py           # Cleaning & feature engineering
│   ├── train_models.py         # 5-fold CV benchmark
│   ├── tune_hyperparams.py     # Optuna Bayesian optimisation
│   ├── evaluate.py             # Model selection & SHAP analysis
│   ├── threshold_optimisation.py  # Profit-driven threshold search
│   └── agent.py                # Retention agent (SHAP + LLM)
├── app/app.py          # Streamlit demo
├── models/             # Saved LightGBM model
├── outputs/            # Figures, metrics, results
├── notebooks/          # EDA notebook
└── requirements.txt
```