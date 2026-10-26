# Naive Bayes from Scratch (Iris Dataset)

A Gaussian Naive Bayes classifier implemented from scratch with NumPy and
evaluated on the Iris dataset (150 records, 3 species, 4 features).

## Project structure

```
naive-bayes-from-scratch/
├── README.md          This file
├── naive_bayes.py     Full implementation (load, split, train, predict, evaluate, plot)
├── dataset.csv        Iris dataset with a header row (source: UCI Machine Learning Repository)
├── report.pdf         Final written report
└── results/
    ├── confusion_matrix_heatmap.png
    └── actual_vs_predicted_bar_chart.png
```

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python naive_bayes.py
```

The script prints the metrics and saves the graphs into `results/`.

## Method

1. Load the dataset and check for missing values (none found; 50 records per species).
2. Stratified 80/20 train/test split (`random_state=42`): 120 training, 30 testing records.
3. For each species compute the prior probability, and the mean and variance of each feature.
4. Classify each test flower by maximising
   `log P(class) + sum(log Gaussian(feature | class))`.
5. Evaluate with accuracy, macro precision/recall/F1 and a confusion matrix.

## Results (test set, 30 flowers)

| Metric | Value |
|---|---|
| Accuracy | 0.9667 |
| Macro Precision | 0.9697 |
| Macro Recall | 0.9667 |
| Macro F1-score | 0.9666 |

Confusion matrix (rows = actual, columns = predicted: setosa, versicolor, virginica):

```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

Only one *Iris-versicolor* flower was misclassified (as *Iris-virginica*).

## Dataset citation

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
Dua, D. and Graff, C. UCI Machine Learning Repository: Iris Data Set.
https://archive.ics.uci.edu/ml/datasets/iris
