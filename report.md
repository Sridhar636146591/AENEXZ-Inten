# KNN Classification: Predicting Low vs. High Tips
### Dataset: seaborn `tips` (244 restaurant bills)

## 1. Problem Setup

The goal is to predict whether a customer will leave a **Low** or **High** tip,
using the available features:

| Feature | Type | Description |
|---|---|---|
| `total_bill` | numeric | Bill amount ($) |
| `size` | numeric | Party size |
| `sex` | categorical | Male / Female |
| `smoker` | categorical | Yes / No |
| `day` | categorical | Thur / Fri / Sat / Sun |
| `time` | categorical | Lunch / Dinner |

## 2. Defining "Low" vs "High" tip

The dataset has no ready-made label, so one had to be engineered. Two
reasonable definitions were tested:

1. **Tip percentage** (`tip / total_bill`) vs. its median (**primary model**)
   — this is the fairer measure of "generosity," since a $5 tip on a $15
   bill is very different from a $5 tip on a $150 bill.
2. **Raw tip amount** (`tip`) vs. its median — included as a bonus
   comparison.

In both cases: value ≥ median → **"High"**, value < median → **"Low"**
(a balanced 50/50 split by construction).

## 3. Method

- **Preprocessing**: `total_bill` and `size` were standardized (z-scored);
  `sex`, `smoker`, `day`, `time` were one-hot encoded. Scaling is essential
  for KNN since it is a distance-based algorithm — unscaled `total_bill`
  (range ~3–50) would dominate distance calculations over categorical
  dummy variables (0/1).
- **Model**: `KNeighborsClassifier` (scikit-learn) inside a `Pipeline`.
- **Hyperparameter tuning**: 5-fold cross-validated grid search over
  `n_neighbors` (3–25, odd values), `weights` (uniform/distance), and
  `metric` (euclidean/manhattan).
- **Split**: 80% train / 20% test, stratified by class.
- **Leakage control**: `tip` and `tip_pct` were excluded from the feature
  matrix `X` (they were only used to build the label).

## 4. Results

### Primary model — target = tip percentage

| Metric | Value |
|---|---|
| Best params | `n_neighbors=17, weights=uniform, metric=manhattan` |
| Cross-val accuracy | **0.574** |
| Test accuracy | **0.408** |
| Test ROC-AUC | 0.396 |

Confusion matrix (test set, rows = actual, cols = predicted):

| | Predicted Low | Predicted High |
|---|---|---|
| **Actual Low** | 12 | 12 |
| **Actual High** | 17 | 8 |

### Bonus model — target = raw tip dollar amount

| Metric | Value |
|---|---|
| Best params | `n_neighbors=5, weights=uniform, metric=euclidean` |
| Cross-val accuracy | 0.718 |
| Test accuracy | **0.755** |

## 5. Interpretation — why such different accuracy?

This gap is the most important finding of the analysis, not a bug:

- **Raw tip amount** is strongly correlated with `total_bill` (larger
  bills mechanically produce larger dollar tips even at a fixed tipping
  rate), so a model predicting "high $ tip" is largely just relearning
  "high bill → high tip." That's why it scores ~75%.
- **Tip percentage**, on the other hand, is much closer to the *actual
  human decision* of how generous someone feels, and it turns out to be
  only weakly related to bill size, party size, day, time, sex, or
  smoking status in this dataset. A model near/below 50% accuracy
  confirms tipping *rate* is close to random noise with respect to these
  features — a genuinely useful and realistic conclusion, not a failure
  of the model. (K was tuned, multiple distance metrics were tried, and
  cross-validation was used, so this isn't an artifact of poor tuning.)

**Practical takeaway**: if the business question is "will this table's
dollar tip be above or below average," bill size alone gets you most of
the way there. If the question is "will this customer be a generous
tipper (%)," these features don't tell you much — you'd need other
signals (e.g. service quality, past visit history) not present in this
dataset.

## 6. Files delivered

- `knn_tips_classifier.py` — full, runnable script (data load → target
  engineering → preprocessing → grid search → evaluation → plots →
  example prediction → bonus comparison)
- `knn_tips_results.png` — accuracy-vs-K plot, confusion matrix, and ROC
  curve for the primary (tip %) model
- `report.md` — this report
