# Sri Lankan Vehicle Price Prediction — Full Project (Two Notebooks)

**MSc in Artificial Intelligence · Local tabular regression · Explainable AI**

**Live demo (Streamlit Community Cloud):** <https://srilankavehiclepricepredictionproject-qpxf6lfrt2547rgriw7pir.streamlit.app/>

Two Jupyter notebooks and a Streamlit app. Notebook 01 cleans and explores the data; Notebook 02 trains, selects and tests six regressors (including a glass-box **Explainable Boosting Machine, EBM**) under one protocol and explains them with global and local XAI; the app serves the resulting price estimates and explanations.

**Live app (Streamlit Community Cloud):** https://srilankavehiclepricepredictionproject-qpxf6lfrt2547rgriw7pir.streamlit.app/

**Source code (GitHub):** https://github.com/VDARW-UOM/Sri_Lanka_Vehicle_Price_Prediction_Project

## Files

| File | What it contains |
|---|---|
| `01_Vehicle_Data_Cleaning_EDA_Submission.ipynb` | Original reorganized cleaning + EDA: inspect raw listings, normalize column names and values, detect/remove duplicates/anomalies, visualize cleaned dataset and export CSV. |
| `02_Vehicle_Price_Regression_Submission.ipynb` | Modelling and XAI: input representations, six candidates (baseline, Decision Tree, Random Forest, XGBoost, LightGBM, EBM), validation comparison, group-aware tuning, selection, held-out test, error analysis, explainable AI (permutation importance for both models, EBM shape functions and a local explanation), model export and conclusions. |
| `app.py` | Self-contained Streamlit app: price estimate from the selected model, EBM estimate with its local explanation, permutation importance for both models, EBM term importance and shape functions, and test metrics for all candidates. |
| `vehicle_ml_outputs/` | Cleaned CSV and everything Notebook 02 saves: both models (`trained_vehicle_price_model.pkl`, `ebm_vehicle_price_model.pkl`), validation/test metrics for all candidates, predictions, error bands, importance tables, EBM explanation outputs, figures and metadata. |
| `requirements.txt` | Python dependencies, including `interpret-core==0.7.8`. |
| `ikman-scrapes.csv` | Raw dataset: 10,388 used-car advertisements scraped from ikman.lk (vehicle attributes and asking price only; no personal data). |
| `DEMO_VIDEO_4_MINUTES.md` | 3–5 minute demo outline. |
| `Sri_Lanka_Vehicle_Price_Prediction_Report.pdf` | Final written report (PDF), including EBM results and XAI. Editable source: `Sri_Lanka_Vehicle_Price_Prediction_Report.docx`. |

## Run the project locally

Requires **Python 3.11–3.13**. Open a terminal in the project folder (the one containing `ikman-scrapes.csv`):

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

1. Open **Notebook 01** and run all cells (**Run → Run All Cells**). It produces/updates `vehicle_ml_outputs/cleaned_vehicle_data.csv` and `cleaning_report.json`.
2. Open **Notebook 02** and run all cells (a few minutes). It trains, tunes, selects and tests all candidates including EBM, runs the XAI analysis, and saves the models and results to `vehicle_ml_outputs/`.
3. Launch the app:

```bash
python -m streamlit run app.py
```

Jupyter must be started from the project folder; both notebooks read and write files relative to it.

All model training (including EBM) happens inside the two notebooks; there are no separate training scripts. Schema, split-size, and explanation-reconstruction checks run as `assert` statements inside Notebook 02.

## What we implemented — and why

- **Column and value normalization:** rename columns, normalize categorical strings, parse prices/mileage/engine capacity into numbers to make source advertisements consistent.
- **Quality control:** remove identical full records and one extreme implausible target; preserve usable listings with missing/invalid mileage/engine cells for leakage-safe imputation.
- **EDA:** compare brands, examine skewed asking prices, visualize price versus year/mileage to interpret data characteristics.
- **Group-aware splitting:** use feature-row hashes and two `GroupShuffleSplit` operations so identical input-feature vectors do not cross partitions.
- **Two input representations, one interface:** black-box models use training-only median imputation, robust scaling, infrequent-aware one-hot encoding and a log target; EBM receives the raw features (native categorical terms and missing-value handling) with price in millions of LKR. Both are wrapped to accept the same features and LKR target, so one training/selection/test loop serves all six candidates.
- **Candidates:** median baseline, Decision Tree, Random Forest, XGBoost, LightGBM (black-box ensembles) and EBM (glass-box additive model, `interactions=0`, fixed settings).
- **Optimization and selection:** 3-fold GroupKFold + RandomizedSearchCV for the two leading ensembles; the deployed model is whichever candidate (EBM included) or tuned variant has the lowest validation MAE. Test data is never used for selection. In the current run the default-settings XGBoost won.
- **Global XAI:** permutation importance (increase in validation MAE, LKR) computed the same way for the selected model and EBM, so their rankings are directly comparable; plus EBM's built-in term importance (mean absolute contribution, million LKR). **Global, not per-listing.**
- **EBM shape functions:** the learned per-feature price curves for manufacture year, engine capacity and mileage, which are the model itself rather than an approximation.
- **Local XAI:** for one listing, EBM's intercept plus per-feature contributions equals its **unclipped prediction**, in LKR. It **does not explain XGBoost**.

## Results (current run)

Both notebooks have been executed end to end (Python 3.13, scikit-learn 1.8, XGBoost 3.4, LightGBM 4.7, interpret-core 0.7.8). All artifacts in `vehicle_ml_outputs/` come from this run.

**Validation-based selection.** On the validation partition (1,274 listings) the default-settings XGBoost had the lowest MAE (~Rs 1.87M), ahead of its tuned variant (~Rs 1.90M) and tuned Random Forest (~Rs 1.94M), so **XGBoost (default settings)** is the deployed model.

**Held-out test (1,285 listings, refit on train + validation):**

| Model | MAE (LKR) | RMSE (LKR) | R² | Median APE |
|---|---|---|---|---|
| Random Forest | 1.88M | 6.69M | 0.900 | 5.4% |
| **XGBoost (selected)** | **1.97M** | **6.68M** | **0.901** | **6.1%** |
| LightGBM | 2.16M | 7.13M | 0.887 | 6.7% |
| Decision Tree | 2.20M | 7.49M | 0.875 | 6.3% |
| **EBM (glass box)** | **4.99M** | **10.51M** | **0.755** | **23.9%** |
| Median baseline | 10.53M | 22.20M | −0.095 | 43.3% |

- Random Forest has a slightly lower *test* MAE than XGBoost, but XGBoost was selected on *validation* scores before test results were viewed; the selection is not changed retroactively.
- **EBM is substantially less accurate** than the tree ensembles (test MAE ~2.5× XGBoost's; validation MAE ~Rs 4.88M, so this is not a test-set artefact). 103 of its 1,285 test predictions (8%) fall at or below zero and are clipped: on the raw price scale, large additive deductions for an old year or a small engine can outweigh the baseline for cheap vehicles. Its value is transparency: every EBM prediction decomposes exactly into a baseline plus one additive contribution per feature.
- **The explanations largely agree:** permutation importance ranks the same five features (engine capacity, manufacture year, brand, mileage, model) highest for both models, so EBM's transparent account describes much the same drivers the black-box model uses. The trade-off between accuracy and built-in interpretability is a key finding.
- If the notebooks are re-run with different package versions, the numbers and even the selected variant (default vs tuned XGBoost) may change slightly; the selection rule stays the same.

## Important limitations

- These are **advertised asking prices**, not verified vehicle transactions or professional valuations.
- The dataset was scraped from ikman.lk; it has no scrape date or advertisement IDs.
- No unique listing IDs or dates, possible reposts, categorical imbalance, and substantial errors for expensive vehicles.
- The raw dataset is supplied for academic reproducibility only; respect ikman.lk terms of use before redistributing it publicly.
- The assignment calls for a **3–5 minute front-end demo video**, not the earlier 12–14 minute presentation.
