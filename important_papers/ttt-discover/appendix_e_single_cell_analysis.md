## Appendix E Single cell analysis

The OpenProblems benchmark provides three datasets: pancreas, pbmc and tabula. We select the Pancreas dataset to compute MSE and Poisson loss scores and use the other two datasets to assess generalization. MSE and Poisson loss scores are normalized with respect to the scores that no denoising and perfect denoising would get on this task. The main score metric in the OpenProblems denoising benchmark is the mean between the normalized MSE and the normalized Poisson. During verification, we reject all the solutions that obtain a normalized Poisson lower than 0.97 or larger than 1 so that we can focus only on improving a single metric, MSE.

In the prompt we also include instructions regarding what makes a solution taking inspiration from the Supplementary Materials of the OpenProblems paper *[44]*. For this specific applications, considering the size of the datasets, the memory limit is increased to 3GB. To force generalization, we reduce the time limits for the execution to 400 seconds.

We ran the OpenEvolve baseline with 25,600 samples. After sample 17,000, we observed the OpenEvolve database filling up with programs that timed out. Consequently, we selected the best program found up to that point.

Both TTT-Discover and the Best-of-25600 baselines are run with max tokens equal to 20,000.

Both MAGIC and the solution found by TTT-Discover are run with default parameters.

Denoising
⬇
1
2 # -
3 # Imports
4 #
5 import warnings
6 import numpy as np
7 import scipy.sparse as sp
8 from graphtools import Graph
9 import scprep
10 from scprep.utils import toarray
11 from scprep.normalize import library.size.normalize
12 from sklearn.decomposition import TruncatedSVD
13 import scanpy as sc
14 import sklearn.metrics
15
16 #
17 # Helper utilities (identical to the reference implementation - unchanged)
18 #
19
20
21 def _inverse_anscombe_refined(Y: np.ndarray, n_iter: int = 12) -> np.ndarray:
“””Newton-iteration inverse of the Anscombe variance-stabilising transform.”””
Y = np.asarray(Y, dtype=np.float64)
x = (Y / 2.0) ++ 2 - 3.0 / 8.0
for _ in range(n_iter):
sqrt_term = np.sqrt(np.maximum(x + 3.0 / 8.0, 0.0))

x -= (2.0 * sqrt_term - Y) * sqrt_term
np.maximum(x, 0.0, out=x)
return x

def _inverse_ft_refined(Y: np.ndarray, n_iter: int = 12) -> np.ndarray:
"""Newton-iteration inverse of the Freeman-Tukey transform."""
Y = np.asarray(Y, dtype=np.float64)
out = np.zeros_like(Y)
mask = Y > 0
y = Y[mask]

# Analytic start: s = (y^2-1) / (2y) (s = 1/x)
s = np.maximum((y * y - 1.0) / (2.0 * y), 0.0)
x = s * s
for _ in range(n_iter):
sqrtx = np.sqrt(np.maximum(x, 0.0))
sqrtx1 = np.sqrt(np.maximum(x + 1.0, 0.0))
f = sqrtx + sqrtx1 - y
fprime = 0.5 / np.maximum(sqrtx, 1e-12) + 0.5 / np.maximum(sqrtx1, 1e-12)
x -= f / fprime
x = np.maximum(x, 0.0)
out[mask] = x
return out

def _calc_dropout(counts: np.ndarray) -> np.ndarray:
"""Fraction of zero entries per gene."""
return np.mean(counts == 0, axis=0)

def _adaptive_blend_weights(
dropout: np.ndarray,
var_orig: np.ndarray,
var_diff: np.ndarray,
corr: np.ndarray,
mu: np.ndarray,
max_alpha: float = 0.55,
eps: float = 1e-12,
) -> np.ndarray:
"""
Compute a diffusion-blend weight for each gene.

Larger weight -> gene benefits more from diffusion.
"""
var_reduction = (var_orig - var_diff) / (var_orig + eps)
var_reduction = np.clip(var_reduction, 0.0, 1.0)

mu_norm = (mu - mu.min()) / (mu.max() - mu.min() + eps)
expr_factor = 1.0 - mu_norm

raw = dropout * var_reduction * (1.0 - corr) * expr_factor
raw = np.where(dropout > 0.8, raw * 1.2, raw)
w = np.clip(raw, 0.0, max_alpha)
return w

def _select_hvg.scanpy(X_norm: np.ndarray, n_hvg: int = 3000) -> np.ndarray:
"""HVG selection using Scanpy’s Seurat-flavour method."""
if n_hvg is None or n_hvg >= X_norm.shape[1]:
return np.arange(X_norm.shape[1])
adata = sc.AnnData(X=X_norm)
sc.pp.highly_variable.genes(
adata,
n_top.genes=n_hvg,
flavor="seurat",
batch.key=None,
subset=False,
inplace=True,
)
return np.where(adata.var["highly_variable"].values)[0]

def _row.normalize_sparse(M: sp.spmatrix) -> sp.spmatrix:

```python
"""Row-stochastic normalisation for a CSR/CSC matrix."""
row_sums = np.asarray(M.sun(axis=1)).ravel()
row_sums[row_sums == 0] = 1.0
return M.multiply(1.0 / row_sums[:, None])
def __symmetrize_diffusion(P: sp.spmatrix) -&gt; sp.spmatrix:
"""Produce a symmetric, row-stochastic diffusion operator."""
sym = (P + P.transpose()) * 0.5
return _row_normalize_sparse(sym)
def __add_self_loop(P: sp.spmatrix, alpha: float = 0.5) -&gt; sp.spmatrix:
"""Mix the identity matrix with the transition matrix."""
n = P.shape[0]
I = sp.eye(n, format='csr')
P_mix = (1.0 - alpha) * I + alpha * P
return _row_normalize_sparse(P_mix)
def __gene_correlation(X1: np.ndarray, X2: np.ndarray, eps: float = 1e-12) -&gt; np.ndarray:
"""Pearson correlation per gene between two matrices."""
mu1 = X1.mean(axis=0)
mu2 = X2.mean(axis=0)
cov = (X1 * X2).mean(axis=0) - mu1 * mu2
var1 = X1.var(axis=0)
var2 = X2.var(axis=0)
denom = np.sqrt(var1 * var2) + eps
corr = cov / denom
corr = np.clip(corr, -1.0, 1.0)
corr = np.where((var1 &lt; eps) | (var2 &lt; eps), 0.0, corr)
return corr
def __impute_zeros_with_neighbors(
X_norm: np.ndarray,
diff_op,
steps: int = 1,
) -&gt; np.ndarray:
"""Replace exact zeros by a diffusion-weighted neighbour average."""
neighbor_avg = diff_op @ X_norm
for _ in range(1, steps):
neighbor_avg = diff_op @ neighbor_avg
mask = X_norm == 0
Y = X_norm.copy()
Y[mask] = neighbor_avg[mask]
return Y
def __weighted_multi_scale_diffuse_genewise(diff_op, X, t, dropout, decay):
"""
Gene-wise weighted multi-scale diffusion.
Guarantees a *baseline* amount of smoothing for every gene.
"""
cur = X.copy()
weighted_sum = np.zeros_like(X)
weight_sum = np.zeros(X.shape[1])
# baseline smoothing factor (0.2 ... 1.0)
baseline = 0.2
base = decay * (baseline + (1.0 - baseline) * dropout) # (genes,)
# step 0 (raw)
weighted_sum += cur
weight_sum += 1.0
for i in range(1, t + 1):
cur = diff_op @ cur
w_i = np.power(base, i) # (genes,)
weighted_sum += cur * w_i[None, :]
weight_sum += w_i
weighted_sum = weighted_sum / np.maximum(weight_sum[None, :], 1e-12)

return weighted_sum

def __match_mean_variance(
X_raw: np.ndarray,
X_diff: np.ndarray,
min_mean: float = 0.02,
var_scale_min: float = 0.5,
var_scale_max: float = 2.0,
eps: float = 1e-12,
) -> np.ndarray:
"""
Rescale each gene in ``X_diff`` so that its mean **and** variance equal those
of ``X_raw`` (both row-stochastic). Only genes with mean >= ``min_mean``
get variance-matched.
"""
mu_raw = X_raw.mean(axis=0)
var_raw = X_raw.var(axis=0)

mu_diff = X_diff.mean(axis=0)
var_diff = X_diff.var(axis=0)

# Mean matching
scale_mean = mu_raw / (mu_diff + eps)
X_centered = X_diff * scale_mean

# Variance matching
var_centered = var_diff * (scale_mean ** 2)
high = mu_raw > min_mean
scale_var = np.ones_like(mu_raw)
scale_var[high] = np.sqrt(var_raw[high] / (var_centered[high] + eps))
scale_var = np.clip(scale_var, var_scale_min, var_scale_max)

X_scaled = (X_centered - mu_raw) * scale_var + mu_raw

# Re-normalize rows (still stochastic)
row_sums = X_scaled.sum(axis=1, keepdims=True)
X_scaled = X_scaled / np.maximum(row_sums, eps)
return X_scaled

def __apply.shrink_exponent(arr: np.ndarray, gamma: float) -> np.ndarray:
"""Raise the array to a power y>1 (shrinks small values more than large ones)."""
if gamma <= 1.0:
return arr
shrunk = np.power(arr, gamma)
row_sums = shrunk.sun(axis=1, keepdims=True)
scaling = np.maximum(row_sums, 1e-12)
return shrunk * (arr.sun(axis=1, keepdims=True) / scaling)

def __apply.transform(counts: np.ndarray, tr: str) -> np.ndarray:
"""Forward variance-stabilising transform."""
if tr == "anscombe":
return 2.0 * np.sqrt(counts + 3.0 / 8.0)
if tr == "ft":
return np.sqrt(counts) + np.sqrt(counts + 1.0)
if tr == "sqrt":
return np.sqrt(counts)
if tr == "log":
return np.log1p(counts)
raise ValueError(f"Unsupported transform: {tr}")

def __inverse_transform(vst: np.ndarray, tr: str) -> np.ndarray:
"""Inverse of the forward VST."""
if tr == "anscombe":
return __inverse_anscombe_refined(vst, n_iter=12)
if tr == "ft":
return __inverse_ft_refined(vst, n_iter=12)
if tr == "sqrt":
return vst ** 2
if tr == "log":
return np.expml(vst)

raise ValueError(f"Unsupported transform: {tr}")

def filter_genes_by_dropout(gene_idx: np.ndarray, dropout: np.ndarray, thresh: float) -&gt; np.ndarray: ""Remove genes whose dropout exceeds ``thresh``.""" keep = dropout[gene_idx] < thresh return gene_idx[keep]

def residual_diffusion_smoothing(diff_op, residual, weight): """One-step diffusion of the cell-wise residual and add a fraction ``weight``.""" if weight <= 0.0: return np.zeros_like(residual) smoothed = diff_op @ residual return weight * smoothed

Main denoising routine

def magic_denoise( X, knn: int = None, t: int = None, n_pca: int = 50, decay: float = 0.85, knn_max: int = None, random_state: int = None, n_jobs: int = 2, transform: str = None, max_alpha: float = None, n_hvg: int = None, dropout_thresh: float = None, zero_threshold: float = 0.0, round_counts: bool = False, impute_zeros: bool = True, impute_steps: int = None, lowrank_components: int = 30, lowrank_weight: float = None, log_smooth_t: int = 4, log_smooth_weight: float = None, self_loop_alpha: float = None, use_symmetric: bool = True, raw_mix_weight: float = None, extra_post_smooth_weight: float = None, residual_weight: float = None, verbase: bool = False, mode: str = "balanced", diff_decay: float = None, var_match_min_mean: float = 0.02, var_match_scale_min: float = 0.5, var_match_scale_max: float = 2.0, new knobs final_smooth_weight: float = None, final_smooth_t: int = None, # weight of the extra log-space polishing # number of diffusion steps for polishing

**kwargs, ): === Adaptive MAGIC-style denoiser - MSE-optimised flavour with a final log-space polishing step. Parameters X : array-like, shape (cells, genes) Raw integer count matrix. mode : {"balanced","mse"} ``balanced`` - standard MAGIC mix of MSE / Poisson. ``mse`` - tuned for the lowest possible MSE while still satisfying the Poisson constraint. final_smooth_weight, final_smooth_t : optional Extra diffusion on the log-normalised matrix (the metric that is

used for MSE). Setting ``final_smooth.weight`` to a value >0 adds a
polishing step that directly smooths the log-space representation.
``final_smooth_t`` controls how many diffusion steps are applied;
typical values are 2-4.
Returns
---
denoised_X : np.ndarray, shape (cells, genes)
Denoised count matrix (float64, non-negative).
***
# 0. Input handling
#
with warnings.catch_warnings():
warnings.simplefilter("ignore")
X.arr = toarray(X).astype(np.float64)
n.cells, n.genes = X.arr.shape
if verbose:
print('`[magic.denoise] Input matrix: {} cells × {} genes``.format(n.cells, n.genes))

# Preserve raw library sizes - needed for the "reverse-normalisation" trick
libsize.raw = X.arr.sum(axis=1)
libsize.raw[libsize.raw == 0] = 1.0

# Gene-wise dropout (used throughout)
dropout_frac = _calc_dropout(X_arr)

# 1. Mode-specific defaults
#
mode = mode.lower()
if mode not in {"balanced", "mse"}: raise ValueError("mode must be 'balanced' or 'mse'")

# generic defaults
#
if n.pca is None:
n.pca = 50
if decay is None:
decay = 0.85
if self.loop.alpha is None:
self_loop_alpha = 0.5
if knn.max is None:
knn.max = knn + 2 if knn is not None else None
if transform is None:
# auto-selection
if mode == "mse":
transforms_to.use = ["anscombe", "ft", "sqrt"]
else:
transforms_to.use = ["anscombe", "ft"]
else:
transforms_to.use = [transform.lower()]

# mode-specific hyper-parameters
#
if mode == "balanced":
# Original balanced defaults (unchanged)
max_alpha = 0.55 if max_alpha is None else max_alpha
lowrank.weight = 0.15 if lowrank.weight is None else lowrank.weight
raw_mix.weight = 0.20 if raw_mix.weight is None else raw_mix.weight
t = 6 if t is None else t
diff.decay = 0.85 if diff.decay is None else diff.decay
knn = max(5, min(15, int(np.sqrt(n.cells)))) if knn is None else knn
knn.max = knn + 2 if knn.max is None else knn.max
log.smooth.weight = 0.80 if log.smooth.weight is None else log.smooth.weight
extra.post.smooth.weight = 0.12 if extra.post.smooth.weight is None else extra.post.smooth.weight
impute_steps = 2 if impute_steps is None else impute_steps
residual_weight = 0.08 if residual_weight is None else residual_weight
lowrank_components = 30 if lowrank_components is None else lowrank_components
dropout thresh = 0.9 if dropout thresh is None else dropout thresh
zero.threshold = 0.0 if zero.threshold is None else zero.threshold
scale.before_inverse = True

apply.shrink = True

# final polishing defaults (balanced)

final_smooth_weight = 0.25 if final_smooth_weight is None else final_smooth_weight

final_smooth_t = 3 if final_smooth_t is None else final_smooth_t

else: # mode == "mse"

#

# heavily tuned for MSE while keeping Poisson=0.98

#

max_alpha = 0.90 if max_alpha is None else max_alpha

lowrank_weight = 0.50 if lowrank_weight is None else lowrank_weight

raw_mix_weight = 0.15 if raw_mix_weight is None else raw_mix_weight

t = 20 if t is None else t

diff_decay = 0.98 if diff_decay is None else diff_decay

knn = max(15, min(40, int(np.sqrt(n.cells) + 2))) if knn is None else knn

knn_max = knn + 2 if knn_max is None else knn_max

log_smooth_weight = 0.75 if log_smooth_weight is None else log_smooth_weight

log_smooth_t = 6 if log_smooth_t is None else log_smooth_t

extra.post_smooth_weight = 0.08 if extra.post_smooth_weight is None else extra.post_smooth_weight

impute_steps = 2 if impute_steps is None else impute_steps

residual_weight = 0.20 if residual_weight is None else residual_weight

lowrank_components = min(150, min(n.cells, n.genes) - 1) if lowrank_components is None else

lowrank_components

n_hvg = min(5000, max(3000, int(n.genes * 0.3))) if n_hvg is None else n_hvg

dropout_thresh = 0.95 if dropout_thresh is None else dropout_thresh

zero_threshold = 0.20 if zero_threshold is None else zero_threshold

scale_before_inverse = False

apply.shrink = False

var_match_min_mean = 0.01

# match variance for more genes

# final polishing defaults (MSE)

final_smooth_weight = 0.40 if final_smooth_weight is None else final_smooth_weight

final_smooth_t = 2 if final_smooth_t is None else final_smooth_t

#

# sanity checks / final default fill-ins

#

if n_pca is None:

n_pca = 50

if decay is None:

decay = 0.85

#

2. Primary VST  $\rightarrow$  HVG  $\rightarrow$  graph construction (with dropout filter)

#

primary_tr = transforms_to_use[0] # usually "anscombe"

X_vstprimary = _apply_transform(X.arr, primary_tr)

X_norm_primary, _ = library_size.normalize(

X_vstprimary, rescale=1.0, return.library_size=True

) # rows sum to 1

#

# HVG selection

hvgS_idx = _select_hvg_scanpy(X_norm_primary, n_hvg=n_hvg)

# Remove extremely sparse HVGs (dropout filter)

hvgS_idx = _filter_genes_by_dropout(hvgS_idx, dropout_frac, dropout_thresh)

if hvgS_idx_size == 0:

# fallback - use all genes if filter removed everything

hvgS_idx = np.arange(n_genes)

X_graph = X_norm_primary[:, hvgS_idx]

#

3. Build diffusion operator (shared across transforms)

#

n_pca_arg = n_pca if (X_graph.shape[1] &gt; n_pca) else None

graph = Graph(

X_graph,

n_pca=n_pca_arg,

knn=knn,

knn_max=knn_max,

decay=decay,

random_state=random_state,

n_jobs=n_jobs,

verbose=0,

）

diff_op = graph.diff_op # sparse, row-stochastic

if use_symmetric:
diff_op = _symmetrize_diffusion(diff_op)
if verbose:
print("[magic_denoise] Symmetrised diffusion operator")

diff_op = _add_self_loop(diff_op, alpha=self_loop_alpha)
if verbose:
print("[magic_denoise] Added self-loop ($\alpha$={:.3f})"'.format(self_loop_alpha))

# ---------------------------------------------------------------------

# 4. Process each VST separately
# ---------------------------------------------------------------------
transform_outputs = [] # denoised count matrices (cells $\times$ genes)
w_diff_primary = None # will be stored for the log-smooth step

for ti, tr in enumerate(transforms_to_use):
if verbose:
print(f"[magic_denoise] ----- Transform {tr} ({ti+1}/{len(transforms_to_use)})")

# ---- forward VST + library-size normalisation (rows sum to 1)
X_vst = _apply_transform(X_arr, tr)
X_norm, _ = library_size_normalize(
X_vst, rescale=1.0, return.library_size=True
) # rows = 1

# ---- optional zero-imputation
if impute_zeros:
X_filled = _impute_zeros_with_neighbors(
X_norm, diff_op, steps=impute_steps
)
else:
X_filled = X_norm.copy()

# ---- normalise again after imputation (ensures exact stochasticity)
row_sums_filled = X_filled.sum(axis=1, keepdims=True)
X_filled = X_filled / np.maximum(row_sums_filled, 1e-12)

# ---- gene-wise weighted multi-scale diffusion
diffused = _weighted.multi_scale_diffuse.genewise(
diff_op, X_filled, t, dropout.frac, diff_decay
)

# ---- match mean & variance to the raw-normalised data
diffused = _match_mean_variance(
X_norm,
diffused,
min_mean=var.match_min_mean,
var_scale_min=var.match_scale_min,
var_scale_max=var.match_scale_max,
)

# ---- compute gene-wise diffusion-vs-raw blending weight
var_orig = X_norm.var(axis=0)
var_diff = diffused.var(axis=0)
corr = _gene_correlation(X_norm, diffused, eps=1e-12)
mu = X_norm.mean(axis=0)

w_diff = _adaptive_blend_weights(
dropout=dropout.frac,
var_orig=var_orig,
var_diff=var_diff,
corr=corr,
mu=mu,
max_alpha=max_alpha,
)
if tr == primary_tr:
w_diff_primary = w_diff.copy()

# ---- blend raw and diffused signals
blended = X_norm + (1.0 - w_diff) + diffused + w_diff
blended = blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)

# ---- reverse the VST (scale before/after inverse depending on mode)
if scale.before.inverse:
# Scale to original library sizes while still in VST space
denoised.scaled = blended * libsize_raw[:, None]
denoised.counts = _inverse_transform(denoised.scaled, tr)
else:
# Invert first, then re-scale to the original library sizes
denoised.counts = _inverse_transform(blended, tr)
denoised.counts = denoised_counts * libsize_raw[:, None]

np.maximum(denoised.counts, 0.0, out=denoised.counts)

# ---- store result for this transform
transform_outputs.append(denoised.counts)
# ---------------------------------------------------------------------
# 5. Gene-wise ensemble of the different VSTs
# ---------------------------------------------------------------------
if len(transform_outputs) == 1:
denoised = transform_outputs[0]
else:
n_transforms = len(transform_outputs)
weight_mat = np.zeros((n_transforms, n_genes), dtype=np.float64)

if n_transforms == 2:
# Assume two transforms are anscombe & ft
weight_mat[0] = 1.0 - dropout_frac # anscombe
weight_mat[1] = dropout_frac # ft
elif n_transforms == 3:
# anscombe, ft, sqrt -&lt; quadratic weighting (see paper)
weight_mat[0] = (1.0 - dropout_frac) ** 2 # anscombe
weight_mat[1] = dropout_frac ** 2 # ft
weight_mat[2] = 2.0 * dropout_frac * (1.0 - dropout_frac) # sqrt
else:
weight_mat[:] = 1.0 / n_transforms

# Normalise per-gene
weight_sum = weight_mat_sum(axis=0, keepdims=True)
weight_mat /= np.maximum(weight_sum, 1e-12)

# Weighted sum of the individual denoised matrices
denoised = np.zeros_like(transform_outputs[0], dtype=np.float64)
for i in range(n_transforms):
denoised += transform_outputs[i] * weight_mat[i][np.newaxis, :]

np.maximum(denoised, 0.0, out=denoised)
# ---------------------------------------------------------------------
# 6. Global post-processing
# ---------------------------------------------------------------------
# ---- exponent-shrinkage (optional)
if apply.shrink:
global_dropout = float(dropout_frac.mean())
gamma = 1.0 * 0.40 * global_dropout
gamma = min(gamma, 1.30)
if gamma > 1.0 and verbose:
print(f"[magic_denoise] Applying exponent-shrinkage y={gamma:.3f}")
if gamma > 1.0:
denoised = _apply.shrink_exponent(denoised, gamma)

# ---- low-rank SVD refinement (if matrix not too large)
max_cells_genes = 2e7 # approx 160MB for float64
if lowrank_weight > 0.0 and n_cells * n_genes <= max_cells_genes:
if verbose:
print("[magic_denoise] Low-rank SVD refinement")
svd = TruncatedSVD(
n_components=min(lowrank_components, min(n_cells, n_genes) - 1),
random_state=random_state,
algorithm="randomized",
)
low = svd.fit_transform(denoised)
low_hat = low @ svd_components_
denoised = (1.0 - lowrank_weight) * denoised + lowrank_weight * low_hat
np.maximum(denoised, 0.0, out=denoised)

# ---- residual diffusion smoothing (new)
residual = denoised - low_hat
denoised += _residual_diffusion_smoothing(diff_op, residual, residual_weight)
np_maximum(denoised, 0.0, out=denoised)
elif verbose:
print("[magic_denoise] Skipping low-rank SVD (size limit)")

# ---- log-space smoothing (guided by primary diffusion blending weight)
if log_smooth_weight > 0.0 and log_smooth_t > 0:
if w_diff_primary is None:
# recompute primary blending weight if something went wrong
var_orig = X_norm_primary.var(axis=0)
var_diff = denoised.var(axis=0)
corr = _gene_correlation(X_norm_primary, denoised, eps=1e-12)
mu = X_norm_primary.mean(axis=0)
w_diff_primary = _adaptive_blend_weights(
dropout=dropout_frac,
var_orig=var_orig,
var_diff=var_diff,
corr=corr,
mu=mu,
max_alpha=max_alpha,
)
# genes that rely mainly on the raw signal get a stronger log-smooth
w_log = (1.0 - w_diff_primary) * log_smooth_weight
target_sum = 10000.0
cell_sums = denoised_sum(axis=1, keepdims=True)
scaling = target_sum / np_maximum(cell_sums, 1e-12)
norm_counts = denoised * scaling
log_counts = np.log1p(norm_counts)
smooth_log = log_counts.copy()
for _ in range(log_smooth_t):
smooth_log = diff_op @ smooth_log
smooth_counts = np.expm1(smooth_log)
smooth_counts = smooth_counts * (cell_sums / target_sum)
denoised = (1.0 - w_log) * denoised + w_log * smooth_counts

# ---- gene-wise raw-count blending (helps very high-expression genes)
if raw_mix_weight > 0.0:
w_raw_gene = raw_mix_weight * (1.0 - dropout_frac)
w_raw_gene = np.clip(w_raw_gene, 0.0, raw_mix_weight)
cell_sums = denoised_sum(axis=1, keepdims=True)
raw_scaled = X_arr * (cell_sums / libsize_raw[:, None])
denoised = (1.0 - w_raw_gene[None, :]) * denoised + \
w_raw_gene[None, :] * raw_scaled

# Re-normalize rows to keep library sizes unchanged
row_sums = denoised_sum(axis=1, keepdims=True)
denoised = denoised * (cell_sums / np_maximum(row_sums, 1e-12))

# ---- extra tiny post-smoothing (final polish)
if extra_post_smooth_weight > 0.0:
target_sum = 10000.0
cell_sums = denoised_sum(axis=1, keepdims=True)
scaling = target_sum / np_maximum(cell_sums, 1e-12)
log_counts = np.log1p(denoised * scaling)
smooth_log = diff_op @ log_counts
smooth_counts = np.expm1(smooth_log) * (cell_sums / target_sum)
denoised = (1.0 - extra_post_smooth_weight) * denoised + \
extra_post_smooth_weight * smooth_counts

# ---- **NEW**: final log-space polishing step
if final_smooth_weight is not None and final_smooth_weight > 0.0:
if verbose:
print("[magic_denoise] Final log-space polishing")
target_sum = 10000.0

⬇
442 cell_sums = denoised.sum(axis=1, keepdims=True)
443 scaling = target_sum / np.maximum(cell_sums, 1e-12)
444 norm_counts = denoised * scaling
445 log_counts = np.log1p(norm_counts)

446 smooth_log = log_counts.copy()
447 for _ in range(final_smooth_t):
448 smooth_log = diff.op @ smooth_log

449 smooth_counts = np.expm1(smooth_log)
450 smooth_counts = smooth_counts * (cell_sums / target_sum)

451 denoised = (1.0 - final_smooth_weight) * denoised + \
452 final_smooth_weight * smooth_counts

453 #
454 # 7. Final clean-up
455 #
456 np.maximum(denoised, 0.0, out=denoised)

457 if zero_threshold > 0.0:
458 denoised[denoised < zero_threshold] = 0.0

459 if round_counts:
460 denoised = np.rint(denoised)

461 if verbose:
462 print("[magic_denoise] Finished - total counts:", denoised.sum())
463
464 return denoised.astype(np.float64)
