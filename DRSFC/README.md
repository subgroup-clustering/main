# DRSFC (Doubly Regressing Subgroup Fair Clustering)

This repository contains the official implementation of **DRSFC**, the algorithm proposed in the paper:

**"Doubly Regressing Subgroup Fair Clustering"**

---

## Installation

```bash
git clone https://github.com/your-repo/DRSFC.git
cd DRSFC
pip install -r requirements.txt
```

## Environments

- Python >= 3.9
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- torch >= 2.0.0
- tqdm >= 4.65.0
- optuna >= 3.3.0 (for hyperparameter tuning)

---

## Usage

Run the DRSFC algorithm via the command-line interface.

### Example Commands

```bash
# Subgroup fair clustering on adult dataset with q=2 sensitive attributes
python -m src.main --data_name adult(2) --K 5 --lambda_fair 10

# Subgroup fair clustering with custom hyperparameters
python -m src.main --data_name adult(4) --K 5 --lambda_fair 100 --lr 0.01 --max_iter 300

# Baseline K-Means (unfair) for comparison
python -m src.main --data_name adult(2) --K 5 --unfair

# Use GPU acceleration
python -m src.main --data_name adult(2) --K 5 --use_cuda
```

### Arguments

| Flag                | Type    | Default   | Description                                       |
|---------------------|---------|-----------|---------------------------------------------------|
| --seed              | int     | 42        | Random seed                                       |
| --data_name         | str     | adult(2)  | Dataset name (subfolder in data/)                 |
| --data_dir          | str     | None      | Override data directory path                      |
| --K                 | int     | 5         | Number of clusters                                |
| --lambda_fair       | float   | 1.0       | Fairness regularization weight                    |
| --gamma             | float   | 0.01      | Minimum subgroup proportion threshold             |
| --max_order         | int     | 2         | Maximum interaction order for subgroup construction |
| --max_iter          | int     | 200       | Maximum training iterations                       |
| --lr                | float   | 0.01      | Learning rate                                     |
| --n_disc_steps      | int     | 5         | Discriminator steps per iteration                 |
| --unfair            | flag    | off       | Use baseline K-Means (no fairness)                |
| --use_cuda          | flag    | off       | Use GPU if available                              |
| --verbose           | flag    | off       | Verbose output                                    |

---

## Hyperparameter Tuning

Use tune.py for Bayesian optimization with Optuna:

```bash
# Tune for single K
python tune.py --data_name adult(2) --K 5 --objective subgroup --n_trials 100

# Tune for K range
python tune.py --data_name adult(2) --K_min 2 --K_max 10 --n_trials 50
```

---

## Datasets

Datasets should be preprocessed as .npy files:

```
data/
    adult(1)/
        X.npy       # Features (n, d)
        y.npy       # Labels (n,)
        S.npy       # Sensitive attrs (n, q)
    adult(2)/
    adult(4)/
    dutch(2)/
    communities(18)/
```

---

## Key Metrics

| Metric              | Description                                               |
|---------------------|-----------------------------------------------------------|
| Cost                | Clustering cost (SSE / n)                                 |
| SubgroupBalance     | min_k min_{j,j'} n_j^k / n_{j'}^k                         |
| MarginalBalance     | Balance per individual attribute                          |
| Delta               | Max deviation from global proportions                     |
| MP(l)               | l-th order Marginal Parity                                |
| SP                  | Subgroup Parity                                           |
| WMP                 | Wasserstein Marginal Parity                               |

---

## License

This project is licensed under the MIT License.

---

## Citation

```bibtex
@inproceedings{drsfc2026,
  title={Doubly Regressing Subgroup Fair Clustering},
  author={...},
  booktitle={ICML},
  year={2026}
}
```
