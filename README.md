# Porto Seguro — Safe Driver Prediction

Group project for the *Advanced Data Analytics* course.
The task is to predict whether a driver will file an insurance claim in the
following year, using the [Porto Seguro dataset](https://www.openml.org/search?type=data&status=active&id=42742)
(595,212 observations, 57 anonymised features).

The target is strongly imbalanced (3.64% positive cases), so accuracy is not a
meaningful metric. The group uses the **normalised Gini coefficient**
(`Gini = 2 × AUC − 1`), the metric of the original Kaggle competition, together
with **Average Precision**.

---

## Repository structure

```
notebooks/
  01_Leyla_Louache.ipynb        EDA, preprocessing, LR / RF / XGBoost, stacking
  02_Dangui_Klossouan.ipynb     EDA, feature engineering, LightGBM
  03_Mariam_Alawil.ipynb        EDA, CatBoost
  04_Gilles_Koudjou.ipynb       EDA, LR / SVM / XGBoost, averaging ensembles
  05_final_comparison.ipynb     Aggregates all results and compares the models
results/
  01_models.json                One file per member, written by their notebook
  02_models.json
  03_models.json
  04_models.json
reports/
  figures/                      Figures exported from the notebooks
```

Each member keeps their exploratory analysis in their own notebook. This was a
deliberate decision: merging four analyses would have produced a redundant
document and obscured each member's individual contribution.

---

## Getting the data

The dataset is **not stored in this repository** — it exceeds GitHub's file size
limit. It is downloaded on first use and cached locally by scikit-learn:

```python
from sklearn.datasets import fetch_openml

porto = fetch_openml(data_id=42742, as_frame=True, parser="auto")
df = porto.frame
```

The numeric `data_id` is pinned rather than the dataset name, since OpenML may
host several active versions under the same name. The `data/` folder is listed
in `.gitignore`.

---

## Common protocol

All notebooks follow the same conventions so that results remain comparable:

| Setting | Value |
|---|---|
| Split | 80/20, stratified |
| Random seed | 42 |
| Primary metric | Normalised Gini (`2 × AUC − 1`) |
| Secondary metric | Average Precision |

Preprocessing pipelines differ between members, since each model has different
requirements — XGBoost and LightGBM handle missing values natively, whereas
linear models need imputation, scaling and one-hot encoding.

---

## Sharing results

Every notebook ends with a cell writing its results to `results/`, using a
common schema:

```json
{
  "model": "...",
  "auc": ...,
  "gini": ...,
  "ap": ...,
  "base_rate": ...,
  "params": { ... },
  "notes": "..."
}
```

The `notes` field matters: it distinguishes otherwise identical rows (with or
without the `ps_calc` block, tuned or default parameters) and prevents an
untuned baseline from being read as a poor model.

`05_final_comparison.ipynb` loads every JSON file in the folder, aggregates them
into a single dataframe and ranks the models. Adding a new model requires no
change to the comparison code.

---

## Results

Best model of each member, evaluated on the held-out test set:

| Member | Model | Gini (with calc) | Gini (without calc) |
|---|---|---|---|
| Mariam | CatBoost | 0.2836 | **0.2856** |
| Leyla | XGBoost | 0.2825 | 0.2846 |
| Gilles | Weighted Average (XGB + SVM) | 0.2795 | 0.2834 |
| Dangui | LightGBM | 0.2740 | 0.2741 |

Two findings stand out. The four approaches fall within 0.0115 Gini of each
other despite spanning different algorithm families, which suggests that
performance is bounded by the information contained in the features rather than
by the choice of model. And removing the 20 `ps_calc_*` features — identified as
uninformative during the exploratory analyses — leaves performance unchanged
across all four approaches, so they can be dropped for parsimony.

---

## Requirements

```
python >= 3.10
pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
xgboost, lightgbm, catboost
```

```bash
pip install -r requirements.txt
```

---

## Team

Leyla Louache · Dangui Klossouan · Mariam Alawil · Gilles Armel Takoufouet Koudjou
