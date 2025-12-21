# DRSFC (Doubly Regressing Subgroup Fair Clustering)

A subgroup-fair clustering method that learns **soft assignments** $A$ while hiding sensitive subgroup information via adversarial training.

This README documents the training objective and the implementation options:

- `--assignment_type`: how to compute $A$
- `--center_update`: how to update centers $\mu$

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Recommended stability on K=10 in our runs
python -m src.main --data_name "adult(1)" --K 10 --lambda_fair 10 --lr 0.001 \
    --center_update mstep --assignment_type distance --epochs 200

# Generalizable (test-time inference for A) but can be less stable than distance
python -m src.main --data_name "adult(1)" --K 10 --lambda_fair 10 --lr 0.001 \
    --center_update mstep --assignment_type nn_dist --epochs 200
```

### Center initialization options

Initial centers $\mu$ are set either by scikit-learn KMeans seeding or by directly providing centers.

- `--center_init {k-means++,random}`: seeding strategy used when KMeans is run (default: `k-means++`).
- `--init_centers_path <path.npy>`: overrides `--center_init` and loads initial centers from a numpy file of shape $(K,d)$.

---

## Objective

Let $X=\{x_i\}_{i=1}^n$, $x_i\in\mathbb{R}^d$, $K$ clusters, and $A\in\Delta^{K-1}$ soft assignments where $A_{ik}\ge 0$ and $\sum_k A_{ik}=1$.

### Clustering loss

$$
L_{\text{cluster}}(A,\mu) \,=\, \frac{1}{n}\sum_{i=1}^n\sum_{k=1}^K A_{ik}\,\lVert x_i-\mu_k\rVert_2^2.
$$

### Fairness (adversarial) loss

The code builds a subgroup indicator matrix $C\in\{-1,+1\}^{n\times M}$ from sensitive attributes $S$ (with interactions up to `--max_order`).

For a weight vector $v\in\mathbb{R}^M$ and discriminator $g$, the fairness loss is:

$$
L_{\text{fair}}(A; g, v) \,=\, \frac{1}{nK}\sum_{i=1}^n\sum_{k=1}^K\big( (Cv)_i - g(A_{ik})\big)^2.
$$

### Minimax training

DRSFC alternates:

1) **Inner loop** (adversary update): update $(g,v)$ to minimize $L_{\text{fair}}$ with $A$ fixed.

2) **Outer step** (clustering update): update parameters of $A$ (and optionally $\mu$) to minimize:

$$
L_{\text{total}} \,=\, L_{\text{cluster}}(A,\mu) - \lambda\,L_{\text{fair}}(A; g, v),\quad \lambda=\texttt{--lambda_fair}.
$$

---

## Assignment types ($A$) and center updates ($\mu$)

Below are the **exact formulas** corresponding to the current implementation in `src/DRSFC/DRSFC.py`.

### Common notation

- Centers: $\mu = \{\mu_k\}_{k=1}^K$, $\mu_k\in\mathbb{R}^d$
- Temperature: $\tau>0$ (implemented as $\tau = \tau_{\min}+\exp(\log\tau)$ with $\tau_{\min}=0.1$)
- Squared distances: $D_{ik}=\lVert x_i-\mu_k\rVert_2^2$

### `--assignment_type nn`

Assignment is a neural net that does **not** see centers:

$$
A_{i:} = \text{softmax}\Big(\frac{f_\theta(x_i)}{\tau}\Big).
$$

### `--assignment_type nn_dist`

Assignment is a neural net that sees both $x$ and distances to centers.
Let $r_i\in\mathbb{R}^K$ be the (non-squared) distances:

$$
(r_i)_k = \sqrt{\lVert x_i-\mu_k\rVert_2^2 + \varepsilon},\; \varepsilon\approx 10^{-8}.
$$

Then:

$$
A_{i:} = \text{softmax}\Big(\frac{f_\theta([x_i, r_i])}{\tau}\Big).
$$

### `--assignment_type distance`

No neural net. Assignments are purely distance-based:

$$
A_{i:} = \text{softmax}\Big(-\frac{D_{i:}}{\tau}\Big).
$$

---

### `--center_update sgd`

Centers are treated as parameters and updated by gradient descent together with the main parameters:

$$
\mu \leftarrow \mu - \eta\,\nabla_\mu\big(L_{\text{cluster}}(A,\mu) - \lambda L_{\text{fair}}(A;g,v)\big),\quad \eta=\texttt{--lr}.
$$

### `--center_update mstep`

After the outer step, centers are updated by the closed form weighted mean (M-step):

$$
\mu_k \leftarrow \frac{\sum_{i=1}^n A_{ik}\,x_i}{\sum_{i=1}^n A_{ik} + \epsilon},\quad \epsilon\approx 10^{-8}.
$$

---

## 조합별 정리 (assignment_type × center_update)

아래 6개 조합은 “$A$를 어떻게 계산하는가”와 “$\mu$를 어떻게 업데이트하는가”를 합친 형태입니다.

### 1) `nn` + `sgd`

$$
A_{i:}=\text{softmax}\Big(\frac{f_\theta(x_i)}{\tau}\Big),\qquad
\mu \leftarrow \mu - \eta\,\nabla_\mu\big(L_{\text{cluster}}(A,\mu) - \lambda L_{\text{fair}}(A;g,v)\big).
$$

### 2) `nn` + `mstep`

$$
A_{i:}=\text{softmax}\Big(\frac{f_\theta(x_i)}{\tau}\Big),\qquad
\mu_k \leftarrow \frac{\sum_i A_{ik}x_i}{\sum_i A_{ik}+\epsilon}.
$$

### 3) `nn_dist` + `sgd`

$$
(r_i)_k = \sqrt{\lVert x_i-\mu_k\rVert_2^2+\varepsilon},\qquad
A_{i:}=\text{softmax}\Big(\frac{f_\theta([x_i,r_i])}{\tau}\Big),
$$

$$
\mu \leftarrow \mu - \eta\,\nabla_\mu\big(L_{\text{cluster}}(A,\mu) - \lambda L_{\text{fair}}(A;g,v)\big).
$$

### 4) `nn_dist` + `mstep`

$$
(r_i)_k = \sqrt{\lVert x_i-\mu_k\rVert_2^2+\varepsilon},\qquad
A_{i:}=\text{softmax}\Big(\frac{f_\theta([x_i,r_i])}{\tau}\Big),\qquad
\mu_k \leftarrow \frac{\sum_i A_{ik}x_i}{\sum_i A_{ik}+\epsilon}.
$$

### 5) `distance` + `sgd`

$$
A_{i:}=\text{softmax}\Big(-\frac{D_{i:}}{\tau}\Big),\qquad
\mu \leftarrow \mu - \eta\,\nabla_\mu\big(L_{\text{cluster}}(A,\mu) - \lambda L_{\text{fair}}(A;g,v)\big).
$$

### 6) `distance` + `mstep`

$$
A_{i:}=\text{softmax}\Big(-\frac{D_{i:}}{\tau}\Big),\qquad
\mu_k \leftarrow \frac{\sum_i A_{ik}x_i}{\sum_i A_{ik}+\epsilon}.
$$

---

## Outputs

Runs create a folder like `results/<data_name>_K<K>/` containing (at least):

- `cluster_ids.npy`, `soft_assignments.npy`, `centers.npy`
- `X.npy`, `y.npy`, `S.npy` (inputs)
- `metrics.json`, `config.json`

## Example: comparing methods

```bash
# SGD + NN
python -m src.main --data_name "adult(1)" --K 10 --lr 0.01 --lambda_fair 10 \
    --center_update sgd --assignment_type nn

# M-step + NN
python -m src.main --data_name "adult(1)" --K 10 --lr 0.001 --lambda_fair 10 \
    --center_update mstep --assignment_type nn

# M-step + Distance (most stable)
python -m src.main --data_name "adult(1)" --K 10 --lr 0.01 --lambda_fair 10 \
    --center_update mstep --assignment_type distance

# M-step + NN_dist (generalizable)
python -m src.main --data_name "adult(1)" --K 10 --lr 0.001 --lambda_fair 10 \
    --center_update mstep --assignment_type nn_dist
```
