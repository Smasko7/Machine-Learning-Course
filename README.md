# Machine Learning Course — MSc AI

A collection of hands-on exercises covering core machine learning topics, implemented in Python using scikit-learn, TensorFlow, and related libraries. Each notebook corresponds to a graded exercise from the MSc Artificial Intelligence curriculum.

## Course Structure

| Exercise | Topic | Dataset | Key Concepts |
|---|---|---|---|
| [Exercise 1](notebooks/Exercise_1.ipynb) | Logistic & Linear Regression | Diabetes (Kaggle) | Classification, feature scaling, evaluation metrics |
| [Exercise 2](notebooks/Exercise_2.ipynb) | Decision Trees & Random Forests | Water Potability (Kaggle) | Ensemble methods, grid search, overfitting |
| [Exercise 3](notebooks/Exercise_3.ipynb) | Time Series Forecasting | USD Index (yfinance) | Stationarity, lag features, log returns |
| [Exercise 4](notebooks/Exercise_4.ipynb) | Wine Quality Classification | Wine Quality (Kaggle) | GridSearchCV, class imbalance, F1-score |
| [Exercise 5](notebooks/Exercise_5.ipynb) | Image Classification with KNN | MNIST (Keras) | KNN, feature scaling, confusion matrix |
| [Exercise 7](notebooks/Exercise_7.ipynb) | NLP — Disaster Tweet Classification | Kaggle NLP Competition | TF-IDF, SVM, text preprocessing, PCA |
| [Exercise 8](notebooks/Exercise_8.ipynb) | Dog Breed Classification | Custom dog breed images | MLP, CNN, dropout, early stopping |
| [Exercise 10](notebooks/Exercise_10.ipynb) | Music Clustering & Recommendation | Spotify Tracks (Kaggle) | K-Means, HDBSCAN, cosine similarity |

> **Note:** Exercises 6 and 9 are not included in this repository.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Machine-Learning-Course.git
cd Machine-Learning-Course
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the datasets

Each exercise uses a different dataset. See [`data/README.md`](data/README.md) for download links and placement instructions.

### 5. Launch Jupyter

```bash
jupyter notebook notebooks/
```

Or open any notebook directly in [Google Colab](https://colab.research.google.com/).

---

## Project Structure

```
Machine-Learning-Course/
├── notebooks/               # Jupyter notebooks — one per exercise
│   ├── Exercise_1.ipynb
│   ├── Exercise_2.ipynb
│   ├── Exercise_3.ipynb
│   ├── Exercise_4.ipynb
│   ├── Exercise_5.ipynb
│   ├── Exercise_7.ipynb
│   ├── Exercise_8.ipynb
│   └── Exercise_10.ipynb
├── src/                     # Reusable Python utilities
│   ├── __init__.py
│   ├── preprocessing.py     # Scaling, imputation, text & image cleaning
│   ├── evaluation.py        # Metrics printing, confusion matrix, feature importance plots
│   └── timeseries.py        # Lag features, log returns, stationarity helpers
├── data/                    # Dataset download instructions
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Reusable Modules (`src/`)

Common patterns extracted from the notebooks into importable utilities:

- **`src.preprocessing`** — feature scaling with train/test fit-transform, mean imputation, regex-based text cleaning pipeline (Exercise 7), image loading and resizing (Exercises 5 & 8)
- **`src.evaluation`** — `print_metrics()` for unified classification report, confusion matrix plotter, feature-importance bar chart, Keras learning-curve plotter
- **`src.timeseries`** — `create_lag_features()` for sliding-window feature matrices, log-return transformation and its inverse, ADF stationarity test wrapper

Example import in a notebook:

```python
from src.preprocessing import scale_features, clean_text
from src.evaluation import print_metrics, plot_confusion_matrix
from src.timeseries import create_lag_features, to_log_returns
```
