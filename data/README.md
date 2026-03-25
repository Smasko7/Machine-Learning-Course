# Datasets

The notebooks in this course use external datasets that are **not committed to the repository** (to keep the repo lightweight and avoid licensing issues). Download each dataset and place it in this `data/` folder, or upload it directly to Google Colab.

---

## Exercise 1 — Diabetes Prediction

**Dataset:** Pima Indians Diabetes Dataset
**Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
**File name:** `diabetes.csv`
**Rows / Columns:** 768 × 9

---

## Exercise 2 — Water Potability

**Dataset:** Water Potability Dataset
**Source:** [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
**File name:** `water_potability.csv`
**Rows / Columns:** 3,276 × 10

---

## Exercise 3 — USD Index Time Series

**Dataset:** US Dollar Index (DX-Y.NYB) — downloaded live via `yfinance`
**No manual download required.** The notebook fetches the data automatically:

```python
import yfinance as yf
df = yf.download("DX-Y.NYB", period="5y")
```

---

## Exercise 4 — Wine Quality

**Dataset:** Wine Quality Dataset (with missing values)
**Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) (or Kaggle)
**File name:** `wine-missing.csv`

---

## Exercise 5 — MNIST Digit Classification

**Dataset:** MNIST — downloaded automatically via Keras
**No manual download required.** The notebook fetches the data automatically:

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---

## Exercise 7 — Disaster Tweet Classification

**Dataset:** NLP with Disaster Tweets
**Source:** [Kaggle Competition](https://www.kaggle.com/c/nlp-getting-started)
**File name:** `train.csv`
**Rows / Columns:** 7,613 × 5

---

## Exercise 8 — Dog Breed Classification

**Dataset:** Custom dog breed image dataset (folder structure per breed)
**File structure:**
```
data/
  dogs/
    Beagle/
      Beagle_1.jpg
      ...
    Labrador/
      ...
```
Contact your course instructor for access to this dataset.

---

## Exercise 10 — Spotify Music Clustering

**Dataset:** Spotify Tracks Dataset
**Source:** [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
**File name:** `dataset.csv`
**Rows / Columns:** 114,000 × ~20
