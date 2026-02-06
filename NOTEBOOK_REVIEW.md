# Review of `AI_project.ipynb`

## Overall assessment
The notebook establishes a solid preprocessing + exploratory analysis workflow for CPU temperature prediction, but it stops before model training/evaluation and has several methodological issues that will affect model quality and reproducibility.

## What is working well
- Clear stepwise preprocessing flow (timestamp features, null checks, missing-value handling).
- Explicit attempt to avoid leakage by dropping `cpu_temp_max_c`.
- Train/test split is done before feature scaling (correct sequencing).
- EDA charts and observations are relevant to the prediction objective.

## Key issues found

### 1) `LabelEncoder` is misused for feature columns
The same `LabelEncoder` instance is reused for `server_id`, `source`, and `workload_type`, which imposes arbitrary ordinal relationships and can bias many models.

**Recommendation**
- Replace with `OneHotEncoder` (or `pd.get_dummies`) for nominal categorical variables.
- Keep encoding in a pipeline so train/test transformations remain consistent.

### 2) Temporal leakage risk from random split
A random `train_test_split` can leak future information when data is timestamped.

**Recommendation**
- Use chronological split (`shuffle=False`) or time-series validation (`TimeSeriesSplit`) if this is temporal telemetry.

### 3) Incomplete ML workflow
Notebook ends after scaling and a preview of `X_train.head()`; there is no baseline model, tuning, or evaluation.

**Recommendation**
- Add at least one baseline regressor and one non-linear model (e.g., RandomForest/XGBoost).
- Report MAE, RMSE, and R² on holdout data.

### 4) Reproducibility issues (Colab-specific paths)
Hardcoded path `/content/drive/MyDrive/Colab Notebooks/cpu_temp_dataset.csv` makes reruns difficult outside one Drive setup.

**Recommendation**
- Parameterize the dataset path.
- Add a cell that checks file existence and prints a clear error message.

### 5) Feature-engineering opportunities not yet leveraged
Only `hour` and `dayofweek` are extracted from timestamp.

**Recommendation**
- Add cyclic encodings for hour/day (`sin`/`cos`) to represent periodic behavior.
- Consider rolling statistics if the data is sequential per server.

## Suggested next-step implementation order
1. Replace categorical encoding with `OneHotEncoder` in a `ColumnTransformer` pipeline.
2. Switch to time-aware splitting strategy.
3. Train/evaluate baseline + tree model and log metrics.
4. Add feature importance/SHAP-style interpretability.
5. Externalize configuration (paths, random seed, selected features).

## Quick scorecard
- **Clarity/readability:** 8/10
- **Data preparation correctness:** 6/10
- **Modeling completeness:** 3/10
- **Reproducibility:** 4/10
- **Production readiness:** 3/10
