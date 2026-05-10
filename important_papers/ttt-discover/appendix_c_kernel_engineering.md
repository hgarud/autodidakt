## Appendix C Kernel engineering

For trimul, we provide the a matrix multiplication kernel that triton provides in README, mostly for syntax purposes For MLA-Decode, we first put a softmax kernel in a preliminary prompt to let the base model generate a correct but unoptimized MLA-Decode kernel, and then use that as the initial state with the earlier softmax example removed.

C.1 Kernel evaluation details

Setup of verifier for training. We follow the exact same practice for evaluating kernel correctness and runtime as the original GPUMode competitions. Specifically, the verifier used in our training jobs uses the same code as the official GPU Mode Competition Github repository, with minor adjustment to integrate into our training codebase. The verification process includes a correctness check that compare output values between the custom kernel and a pytorch reference program under a designated precision, followed by runtime benchmarking of the custom kernel across multiple iterations. All the details in our verification procedure follow the official competition exactly, including the test cases used for correctness check and benchmarking, hyper-parameters such as matching precision and iterations used for timing, etc. We run our verifier on H100s for TriMul, and H200s for MLA-Decode, both from the Modal cloud platform.

Setup of environments for final report. For final report, we submit to the official TriMul A100/H100 leaderboard and report the runtime shown. For TriMul B200/MI300X and MLA-Decode MI300X tasks, due to an infra problem on GPU Mode’s server, we could not submit to the official leaderboard. For these tasks, we work with the GPU Mode team closely to set up our local environment, which replicates the official environment and gets GPU Mode team’s review and confirmation.

Selection protocol for best kernels. For TriMul H100 task, we select 20 kernels with the best verifier score throughout training. For other tasks, since our verifier hardware in training is different from the target hardware, we select 20 kernels with the best training scores plus 20 random correct kernels every 10 steps of training. Finally, we used our verifier with the target hardware to verify each selected kernels for three times, and submit the kernel with the smallest average runtime for final report.

### C.2 Analysis of best generated kernels

TriMul H100 kernels. The below code shows the best TriMul kernels discovered by TTT for H100 GPU. At the high level, the kernel correctly identifies a major bottleneck of the problem, which is the heavy memory I/O incurred by a series of elementwise operations, and then focuses on fusing them with Triton. Specifically, the kernel fuses: (i) operations in the input LayerNorm, (ii) elementwise activation and multiplication for input gating, and (iii) operations in the output Layernorm and output gating. As for the compute-heavy operation, which is an $O(N^{3})$ matmul, the kernel converts its inputs to fp16 and delegate the computation to cuBLAS to effectively leverage the TensorCores on H100 GPU.

Compared with kernels generated early in training, the final kernel achieves a big improvement by (i) fusing more operations together, and (ii) deeper optimization of the memory access pattern inside fused kernels. For example, a kernel generated early fuses LayerNorm operations, but does not fuse the input gating process. A kernel generated in the middle of training fuses the same operations as the final kernel, but has less efficient memory access pattern in the fused kernel for output LayerNorm, gating, and output projection.

Compared with the best human leaderboard kernel, the TTT discovered kernel adopts a similar fusion strategy for the input LayerNorm and input gating. Different from human kernel, the TTT kernel does not perform as much auto-tuning of block size, which could be a limitation. However, the TTT kernel fuses the output LayerNorm and gating with output projection whereas the human kernel does not, which could explain the moderate advantage of the former.

TriMul H100

Outgoing TriMul (AlphaFold-3) - Triton accelerated forward pass.

The implementation follows the reference ``TriMul`` module but fuses the expensive kernels:

1. Row-wise LayerNorm over the last dimension (FP16 output, FP32 reduction).
2. Fused projection, gating and optional scalar mask:
- left_proj, right_proj = x_norm @ W_proj
- left_gate, right_gate, out_gate = sigmoid(x_norm @ W_gate)
- left = left_proj + left_gate + mask
- right = right_proj + right_gate + mask
3. Pairwise multiplication across the sequence dimension (batched GEMM on fp16 tensors).
4. Fused hidden-dim LayerNorm -> out-gate multiplication -> final linear projection (all in one kernel, FP16 matmul with FP32 accumulation).

The output tensor has shape ``[B, N, N, dim]`` and dtype ``float32``.

```
from typing import Tuple, Dict
import torch
import triton
import triton.language as tl
```

```
# 1) Row-wise LayerNorm (FP16 output, FP32 accumulator)
# ---
@triton.jit
def ..row.ln_fp16_kernel(
X.ptr, Y.ptr, # (M, C) input / output
w.ptr, b.ptr, # LN weight & bias (fp32)
M, C: tl.constexpr, # rows, columns (C is compile-time constant)
eps,
BLOCK_M: tl.constexpr,
BLOCK_C: tl.constexpr,
):
pid = tl.program_id(0)
row.start = pid * BLOCK_M
rows = row.start + tl.arange(0, BLOCK_M)
row_mask = rows < M

# --- mean / var (fp32) ---
sum.val = tl.zeros([BLOCK_M], dtype=tl.float32)
sumsq.val = tl.zeros([BLOCK_M], dtype=tl.float32)

for c in range(0, C, BLOCK_C):
cur_c = c + tl.arange(0, BLOCK_C)
col_mask = cur_c < C
x = tl.load(
X.ptr + rows[:, None] * C + cur_c[None, :],
mask=row_mask[:, None] & col_mask[None, :],
other=0.0,
).to(tl.float32) # (BLOCK_M, BLOCK_C)

sum.val += tl.sum(x, axis=1)
sumsq.val += tl.sum(x * x, axis=1)

mean = sum.val / C
var = sumsq.val / C - mean + mean
inv_std = 1.0 / tl.sqrt(var + eps)

# --- normalize + affine (fp16) ---
for c in range(0, C, BLOCK_C):
cur_c = c + tl.arange(0, BLOCK_C)
col_mask = cur_c < C
x = tl.load(
X.ptr + rows[:, None] * C + cur_c[None, :],
mask=row_mask[:, None] & col_mask[None, :],
other=0.0,
).to(tl.float32)

y = (x - mean[:, None]) * inv_std[:, None]w = tl.load(w.ptr + cur.c, mask=col_mask, other=0.0)b = tl.load(b.ptr + cur.c, mask=col_mask, other=0.0)y = y * w[None, :] + b[None, :]tl.store(Y.ptr + rows[:, None] * C + cur.c[None, :], y.to(tl.float16), mask=row_mask[:, None] & col_mask[None, :],

def _row.layersnorm.fp16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5, ) -> torch.Tensor: """Row-wise LayerNorm over the last dim -> FP16 output.""B, N, _, C = x.shapeM = B * N * Nx.flat = x.view(M, C).contiguous()y.flat = torch.empty((M, C), dtype=torch.float16, device=x.device)BLOCK_M = 128BLOCK_C = 128grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),) _row.ln.fp16_kernel[grid](x.flat,y.flat, weight,bias,M,C,eps,BLOCK_M=BLOCK_M,BLOCK_C=BLOCK_C, num_warps=8,) return y.flat.view(B, N, N, C)

# 2) Fused projection + gating + optional mask

@triton.jit
def _proj_gate_mask_kernel(x.ptr, # (M, C) fp16 mask.ptr, # (M,) fp16 (if MASKED==1) left.proj.w.ptr, # (C, H) fp16 left.gate.w.ptr, # (C, H) fp16 right.proj.w.ptr, # (C, H) fp16 right.gate.w.ptr, # (C, H) fp16 out_gate.w.ptr, # (C, H) fp16 left.ptr, # (B, H, N, N) fp16 right.ptr, # (B, H, N, N) fp16 out_gate.ptr, # (B, N, N, H) fp16 M, N, C: tl.constexpr, H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, MASKED: tl.constexpr, ): pid_m = tl.program_id(0) # row block pid_h = tl.program_id(1) # hidden block row_start = pid_m * BLOCK_M hid_start = pid_h * BLOCK_H rows = row_start + tl.arange(0, BLOCK_M) # (BLOCK_M,) hids = hid_start + tl.arange(0, BLOCK_H) # (BLOCK_H,)

34

row_mask = rows < M
hid_mask = hid

i_idx = rem // N
k_idx = rem - i_idx * N

# layout for left/right: (B, H, N, N) -> flat index:
off = {{b_idx[:, None] * H + hids[None, :]} * N_sq} + i_idx[:, None] * N + k_idx[:, None]

tl.store(
left.ptr + off,
left_out.to(tl.float16),
mask=row_mask[:, None] & hid_mask[None, :],
)

tl.store(
right.ptr + off,
right_out.to(tl.float16),
mask=row_mask[:, None] & hid_mask[None, :],
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3) Fused hidden-dim LayerNorm -> out-gate -> final linear
@triton.jit
def _ln.gate_out_linear_fused_kernel(
hidden.ptr, # (B*H*N*N,N,) fp16 flattened
out.gate.ptr, # (B*N*N*N,H,) fp16 flattened
ln.w.ptr, ln.b.ptr, # (H,) fp32
w.out.ptr, # (H, D) fp16
out.ptr, # (B, N, N, D) fp32
B, N, H, D: tl.constexpr,
eps: tl.constexpr,
BLOCK_M: tl.constexpr,
BLOCK_H: tl.constexpr,
BLOCK_D: tl.constexpr,
);
pid = tl.program_id(0)
row.start = pid * BLOCK_M
rows = row.start + tl.arange(0, BLOCK_M) # flat index for (b,i,j)
row_mask = rows < (B * N * N)

N_sq = N * N
b_idx = rows // N_sq
rem = rows - b_idx * N_sq
i_idx = rem // N
j_idx = rem - i_idx * N

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

b_ln = tl.load(ln.b_ptr + hids, mask=hid_mask, other=0.0) # (H,)
hidden_norm = (hidden_fp32 - mean[:, None]) + inv_std[:, None]
hidden_norm = hidden_norm * w_ln[None, :] + b_ln[None, :] # (BLOCK_M, BLOCK_H)

# ----- out-gate (fp32) -----
out_gate_off = rows[:, None] * H + hids[None, :]
out_gate_file = tl.load(
out_gate_ptr + out_gate_off,
mask=row_mask[:, None] & hid_mask[None, :],
other=0.0,
).to(tl.float32) # (BLOCK_M, BLOCK_H)

gated = hidden_norm * out_gate_file # (BLOCK_M, BLOCK_H)

# ----- final linear projection (fp16 matmul, fp32 acc) -----
gated_fp16 = gated.to(tl.float16)

for d0 in range(0, D, BLOCK_D):
cols = d0 + tl.arange(0, BLOCK_D)
col_mask = cols < D

w_out = tl.load(
w_out_ptr + hids[:, None] * D + cols[None, :],
mask=hid_mask[:, None] & col_mask[None, :],
other=0.0,
) # (BLOCK_H, BLOCK_D) fp16

out = tl.dot(gated_fp16, w_out) # (BLOCK_M, BLOCK_D) fp32

tl.store(
out_ptr + rows[:, None] * D + cols[None, :],
out,
mask=row_mask[:, None] & col_mask[None, :],
)

# 4) Entrypoint

def custom_kernel(
data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]
) -> torch.Tensor:
"""
Forward pass of the outgoing TriMul operator (no gradients).

Arguments
-----
data : (input, mask, weights, config)
- input : Tensor[B, N, N, C] (float32)
- mask : Tensor[B, N, N] (bool/float) or None
- weights: dict of module parameters (float32)
- config : dict with ``dim`` (C) and ``hidden_dim`` (H) and optional ``nomask``

Returns
-----
Tensor[B, N, N, C] (float32)
"""
inp, mask, weights, cfg = data
dim = cfg["dim"] # C
hidden_dim = cfg["hidden_dim"] # H
nomask = cfg.get("nomask", True)
eps = 1e-5

device = inp.device
B, N, .., .. = inp.shape
M = B * N * N # total rows for row-wise ops

# 1) Row-wise LayerNorm (fp16 output)
#
x_norm = ..row.layersnorm.fp16(
inp,
weights["norm.weight"],
weights["norm.bias"],

```python
eps=eps,
# (B, N, N, C) fp16
# 2) Prepare projection / gate weights (C, H) fp16, column-major
left.proj.w.T = weights["left.proj.weight"].t().contiguous().to(torch.float16)
right.proj.w.T = weights["right.proj.weight"].t().contiguous().to(torch.float16)
left.gate_w.T = weights["left.gate.weight"].t().contiguous().to(torch.float16)
right.gate_w.T = weights["right.gate.weight"].t().contiguous().to(torch.float16)
out.gate_w.T = weights["out.gate.weight"].t().contiguous().to(torch.float16)
# 3) Mask handling (optional)
if not nomask and mask is not None:
mask-flat = mask.reshape(M).to(torch.float16).contiguous()
MASKED = 1
else:
mask-flat = torch.empty(0, dtype=torch.float16, device=device)
MASKED = 0
# 4) Allocate buffers for fused projection + gating
left = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)
right = torch.empty.like(left)
out.gate = torch.empty((B, N, N, hidden_dim), dtype=torch.float16, device=device)
# 5) Fused projection / gating / optional mask
BLOCK.M = 64
BLOCK.H = 64
BLOCK.K = 32
grid.proj = (triton.cdiv(M, BLOCK.M), triton.cdiv(hidden_dim, BLOCK.H))
proj.gate_mask_kernel[grid.proj]
x(norm,
mask-flat,
left.proj.w.T,
left.gate.w.T,
right.proj.w.T,
right.gate.w.T,
out.gate.w.T,
left,
right,
out.gate,
M,
N,
dim,
hidden_dim,
BLOCK.M=BLOCK.M,
BLOCK.H=BLOCK.H,
BLOCK.K=BLOCK.K,
MASKED=MASKED,
num_warps=4,
)
# 6) Pairwise multiplication (batched GEMM) - left @ right^T
left.mat = left.view(B + hidden_dim, N, N)
right.mat = right.view(B + hidden_dim, N, N).transpose(1, 2)
hidden.fp16 = torch.bmm(left.mat, right.mat)
hidden = hidden.fp16.view(B, hidden_dim, N, N)
# 7) Fused hidden-dim LayerNorm -&gt; out-gate -&gt; final linear
to.out.norm_w = weights["to.out.norm.weight"] # (H,) fp32
to.out.norm_b = weights["to.out.norm.bias"] # (H,) fp32
to.out.w.T = weights["to.out.weight"].t().contiguous().to(torch.float16) # (H, C)

```txt
out = torch.empty((B, N, N, dim), dtype=torch.float32, device=device)
BLOCK_M_OUT = 64
BLOCK_H_OUT = hidden_dim # cover the whole hidden dim in one kernel launch
BLOCK_D_OUT = 64
grid_out = (triton.cdiv(B * N * N, BLOCK_M_OUT),)
_ln_gate_out(linear_fused_kernel[grid_out](
hidden.view(-1), # flat fp16 hidden
out_gate.view(-1), # flat fp16 out-gate
to_out(norm_w,
to_out(norm_b,
to_out_w.T,
out,
B,
N,
hidden_dim,
dim,
eps,
BLOCK_M=BLOCK_M_OUT,
BLOCK_H=BLOCK_H_OUT,
BLOCK_D=BLOCK_D_OUT,
num_warps=4,
)
return out
```

# C.3 TTT MLA-Decode kernels filtered with Triton kernels

|  Method | Model | AMD MI300X - MLA Decode (↓, μs) [95% CI]  |   |   |
| --- | --- | --- | --- | --- |
|   |   |  Instance 1 | Instance 2 | Instance 3  |
|  1st human | - | 1653.8[1637.3, 1670.3] | 1688.6[1672.8, 1704.3] | 1668.7[1637.0, 1700.3]  |
|  2nd human | - | 1662.8[1648.8, 1676.8] | 1688.6[1677.6, 1699.5] | 1679.7[1653.4, 1705.9]  |
|  3rd human | - | 1723.0[1711.5, 1734.5] | 1765.8[1758.1, 1773.5] | 1718.0[1698.3, 1737.7]  |
|  4th human | - | 1768.7[1750.3, 1787.2] | 1769.9[1755.2, 1784.6] | 1767.0[1736.2, 1797.8]  |
|  5th human | - | 2038.6[2017.8, 2059.3] | 2037.3[2021.0, 2053.6] | 2041.9[1989.0, 2094.8]  |
|  Best-of-25600 | gpt-oss-120b | 2286.0[2264.2, 2307.8] | 2324.1[2306.0, 2342.1] | 2275.2[2267.3, 2283.1]  |
|  TTT-Discover | gpt-oss-120b | 1740.6[1697.9, 1783.2] | 1754.4[1736.7, 1772.2] | 1707.1[1664.5, 1749.8]  |

Table 10. Results of TTT MLA-Decode kernels filtered with Triton kernels.
