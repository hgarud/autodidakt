## Appendix F Prompts

Below we show example prompts from a sample step.

# Prompt used for the first autocorrelation inequality

Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

```txt
python
[VERIFIER CODE HERE]
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.

```txt
**latex**
Starting from a nonnegative step function  $\mathbb{S}\mathbb{f} = (\mathbb{a}_{-}\mathbb{o},\mathrm{dots},\mathbb{a}_{-}\{\mathbb{n} - 1\})$  normalized so that  $\mathbb{S}\backslash \mathrm{sum\_j}$ $\mathbb{a}_{-}\mathbb{j} = \mathbb{s}\mathrm{qrt}(2\mathrm{n})\mathbb{S}$  , set  $\mathbb{S}\mathbb{M} = \backslash \{\mathbb{f}\times \mathbb{f}\backslash \} _\backslash \mathrm{infty}\mathbb{S}$  . Next compute  $\mathbb{S}\mathbb{g}_{-}\mathbb{o} = (\mathbb{b}_{-}\mathbb{o},\mathrm{dots},\mathbb{b}_{-}\{\mathbb{n} - 1\})$  by solving a linear program, i.e.  $\backslash$  maximizing  $\mathbb{S}\backslash \mathrm{sum\_j}$  b_j subject to  $\mathbb{S}\mathbb{b}_{-}\mathbb{j}\backslash \mathbb{ge}\mathbb{O}\mathbb{S}$  and  $\mathbb{S}\backslash \{\mathbb{f}\times \mathbb{g}_{-}\mathbb{o}\backslash \} _\backslash \mathrm{infty}\backslash \mathbb{S}$  ; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound  $\mathbb{S}\{\mathbb{f}\times \mathbb{g}_{-}\mathbb{o}\} (\mathbb{x})\backslash \mathbb{l}\mathbb{e}\mathbb{M}\mathbb{S}$  is tight and limiting. Rescale  $\mathbb{S}\mathbb{g}_{-}\mathbb{o}\mathbb{S}$  to match the normalization,  $\mathbb{S}\mathbb{g} = \backslash \mathrm{frac}\{\backslash$  sqrt(2n)  $\{\backslash$  sum_j b_j  $\} _\mathrm{g}_{-}\mathrm{o}\mathbb{S}$  , and update  $\mathbb{S}\mathbb{f}\backslash$  leftarrow (1-t)f+t g for a small  $\mathbb{S}\mathbb{t} &gt; 0$  . Repeating this step produces a sequence with nonincreasing  $\mathbb{S}\backslash \{\mathbb{f}\times \mathbb{f}\backslash \} _\backslash \mathrm{infty}\mathbb{S}$  , and the iteration is continued until it stabilizes.
```

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have 1000 seconds to run, and after that it has to have returned the best sequence it found. If after 1000 seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence () function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

Here is the last code we ran:

```txt
`python
[CODE HERE]
```

Here are the upper bounds before and after running the code above (lower is better): 2.0000000000 -&gt; 1.5172973712

Our target is to make the upper bound tighter, just as a reference, lower it to at least 1.5030. Further improvements will also be generously rewarded.

Length of the construction: 1000

# Previous Program Output

... (TRUNCATED) ...

ore 1.518186 maxConv 0.000506

|  [1768620458.4] | iter 340400 | len 1500 | score | 1.518177 | maxConv | 0.000506  |
| --- | --- | --- | --- | --- | --- | --- |
|  [1768620461.6] | iter 350200 | len 1500 | score | 1.518057 | maxConv | 0.000506  |
|  [1768620462.3] | iter 352300 | len 1500 | score | 1.518035 | maxConv | 0.000506  |
|  [1768620469.1] | iter 372900 | len 1500 | score | 1.517869 | maxConv | 0.000506  |
|  [1768620476.2] | iter 394300 | len 1500 | score | 1.517755 | maxConv | 0.000506  |
|  [1768620492.9] | iter 445000 | len 1500 | score | 1.517548 | maxConv | 0.000506  |

final best score  $= 1.51729737$

End Output

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable.

However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.

Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. Unless you make a meaningful improvement, you will not be rewarded.

Rules:

- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK, PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don’t use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- `**Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ``python and ``.

##

# Prompt used for the second autocorrelation inequality

Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step functions, that maximizes the following evaluation function:

```txt
``python
{VERIFIER CODE HERE}
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.

`latex

Their procedure is a coarse-to-fine optimization of the score. It starts with a stochastic global search that repeatedly perturbs the current best candidate and keeps the perturbation whenever it improves $Q$, with the perturbation scale gradually reduced over time. Once a good basin is found, they switch to a deterministic local improvement step, performing projected gradient ascent (move in the gradient direction and project back to the feasible region). To reach higher resolution, they lift a good low-resolution solution to a higher-dimensional one by a simple upscaling step and then rerun the local refinement. Iterating this explore—refine—upscale cycle yields their final high-resolution maximizer and the improved lower bound.

Your task is to write a search function, construct_function(), that searches for the best sequence of coefficients. Your function will have 1000 seconds to run, and after that it has to have returned the best sequence it found. If after 1000 seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence () function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence () function.

Here is the last code we ran:

```txt
``python
{CODE HERE}
```

Here are the lower bounds before and after running the code above (higher is better): 0.6666666667 -&gt; 0.9235566275

Our target is to make the lower bound tighter, just as a reference, close to at least 0.97. Further improvements will also be generously rewarded.

Length of the construction: 1024

Previous Program Output

Final lower bound  $= 0.9235566275$

End Output

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable.

However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.

Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. Unless you make a meaningful improvement, you will not be rewarded.

# Rules:

- You must define the 'construct_function' function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC, CVXOPT, GLOP, GLPK, GUROBI, MOSEK, PDLP, SCIP, XPRESS, ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked. Do not import height_sequence_1 yourself; it will already be available.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ``python and ``.

54

# Prompt used for the Erdős'

You are an expert in harmonic analysis, numerical optimization, and mathematical discovery. Your task is to find an improved upper bound for the \name|} minimum overlap problem constant C5.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \max_k \int h(x)(1 - h(x+k)) \, dx$$

\textbf|Constraints}:

\begin{enumerate}

\item $h(x) \in [0, 1]$ for all $x$

\item $ \int_{0}^{[0]} h(x) \, dx = 1 $

\end{enumerate}

\textbf|Discretization}: Represent $h$ as \texttt{n\_points} samples over $[0, 2]$ .

With $dx = \frac{2.0}{\texttt{n\_points}}$

\begin{itemize}

\item $0 \leq h[i] \leq 1$ for all $i$

\item $ \sum h \cdot dx = 1 $ (equivalently: $ \sum h = \frac{\texttt{frac}(\texttt{n\_points})}{2} $ exactly)

\end{itemize}

The evaluation computes: C5 = max(np.correlate(h, 1-h, mode="full") + dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C5 values are better** - they provide tighter upper bounds on the \name| constant.

## Budget &amp; Resources

- **Time budget**: &lt;&lt;<budget_s>&gt;&gt;s for your code to run
- **CPUs**: &lt;&lt;<cpus>&gt; available

## Rules

- Define `run(seed=42, budget_s=&lt;&lt;<budget_s>&gt;&gt;, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC, CVXOPT, GLOP, GLPK, GUROBI, MOSEK, PDLP, SCIP, XPRESS, ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C5 ≤ 0.38092. Our goal is to find a construction that shows C5 ≤ 0.38080.</budget_s></cpus></budget_s></budget_s>

# Prompt used for TriMul

You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.

The TriMul operator operates over a 4D tensor of shape [B, N, N, C].

Your task:

- Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
- You will not have to compute or store gradients for this version. You will only need to implement the forward pass.

Your function should be defined as 'custom_kernel' with the following signature:

Input:

- 'data': Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
- input: Input tensor of shape [bs, seq_len, seq_len, dim]
- mask: Mask tensor of shape [bs, seq_len, seq_len]
- weights: Dictionary containing model weights
- config: Dictionary containing model configuration parameters

Output:

- output: Processed tensor [bs, seq_len, seq_len, dim]

&lt;<problem constraints:="">

- B in  $\{1,2\}$ , N in  $\{128,256,512,1024\}$ , c in  $\{128\}$ , c_z in  $\{128,384,768\}$
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

&lt;<remarks.>&gt; So why is this problem so annoying? Because you have to choose whether to load / deal with either the channel dimensions c, c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N.

The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension ( this is  $\mathbf{N}^{\mathrm{n}}3!$ ).

However, I really like this kernel because it only consists of "simple" operations, and is really easy to understand. It is a true test of "fusions" that torch.compile() doesn't do that well.

Here is a pytorch implementation of the TriMul module. You will want to implement a kernel for the operations in the forward call:

```python
``python
import torch
from torch import nn, einsum
import math
# Reference code in PyTorch
class TriMul(nnModule):
def __init__(self, dim: int, hidden_dim: int,):
super().__init__(self(norm = nn.LayerNorm(dim) self.leftProj = nn.Linear(dim, hidden_dim, bias=False) self.rightProj = nn.Linear(dim, hidden_dim, bias=False) self.left_gate = nn.Linear(dim, hidden_dim, bias=False) self.right_gate = nn.Linear(dim, hidden_dim, bias=False) self.out_gate = nn.Linear(dim, hidden_dim, bias=False) self.to_out_norm = nn.LayerNorm(hidden_dim) self.to_out = nn.Linear(hidden_dim, dim, bias=False)</remarks.></problem>

def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
""
x: [bs, seq_len, seq_len, dim]
mask: [bs, seq_len, seq_len]

Returns:
output: [bs, seq_len, seq_len, dim]
""
batch_size, seq_len, _, dim = x.shape

x = self.norm(x)

left = self.left_proj(x)
right = self.right_proj(x)

mask = mask.unsqueeze(-1)
left = left * mask
right = right * mask

left_gate = self.left_gate(x).sigmoid()
right_gate = self.right_gate(x).sigmoid()
out_gate = self.out_gate(x).sigmoid()

left = left * left_gate
right = right * right_gate

out = einsum(’... i k d, ... j k d -> ... i j d’, left, right)
# This einsum is the same as the following:
# out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)

# # Compute using nested loops
# for b in range(batch_size):
# for i in range(seq_len):
# for j in range(seq_len):
# # Compute each output element
# for k in range(seq_len):
# out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

out = self.to_out_norm(out)
out = out * out_gate
return self.to_out(out)
```

Here is some example skeleton code of the entrypoint function you will create:
``python
def custom_kernel(data):
input_tensor, mask, weights, config = data
dim, hidden_dim = config["dim"], config["hidden_dim"]

# Access the given weights of the model
norm_weight = weights["norm.weight"]
norm_bias = weights["norm.bias"]
left_proj_weight = weights["left_proj.weight"]
right_proj_weight = weights["right_proj.weight"]
left_gate_weight = weights["left_gate.weight"]
right_gate_weight = weights["right_gate.weight"]
out_gate_weight = weights["out_gate.weight"]
to_out_norm_weight = weights["to_out_norm.weight"]
to_out_norm_bias = weights["to_out_norm.bias"]
to_out_weight = weights["to_out.weight"]

# Perform TriMul
return out

To help you understand which triton version we are using, here is some example triton code for an unrelated task:
``python
import triton
import triton.language as tl

@triton.jit

def matmul_persistent_ws_kernel(
a_ptr, b_ptr, c_ptr, M, N, K,
stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):

pid = tl.program_id(axis=0) # async_task 0, 1, 2
num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
pid_m = pid // num_pid_m # async_task 0, 1, 2
pid_n = pid % num_pid_n # async_task 0, 1, 2
offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
offs_k = tl.arange(0, BLOCK_K) # async_task 0
a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) #
async_task 0
a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) #
async_task 0
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
a_1 = tl.load(a_ptrs_1) # async_task 0
a_2 = tl.load(a_ptrs_2) # async_task 0
b = tl.load(b_ptrs) # async_task 0
acc_1 += tl.dot(a_1, b) # async_task 1
acc_2 += tl.dot(a_2, b) # async_task 2
a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
b_ptrs += BLOCK_K * stride_bk # async_task 0
c_1 = acc_1.to(tl.float16) # async_task 1
c_2 = acc_2.to(tl.float16) # async_task 2
c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] #
async_task 1
c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] #
async_task 2
tl.store(c_ptrs_1, c_1) # async_task 1
tl.store(c_ptrs_2, c_2) # async_task 2
```

A few general triton tips:

- tl.arange only takes in constexpr arguments (static or tl.constexpr)
- You cannot use continue in your kernel code
- tl.dot can only take in two input tensors
- There is no tl.mean

Here are the different configs that your kernel will be tested on ("nomask" sets whether there will be no mask, or a randomly sampled mask over the inputs):

Test Cases for correctness and runtime (optimize runtime for these):

- {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
- {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
- {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
- {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
- {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
- {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
- {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}

Here is the last code we ran:

```python
``python
# No previous attempt has been made.
```

Current runtime (lower is better): 1000000.0000 microseconds

Target: 1000 microseconds. Current gap: 999000.0000 microseconds.

Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final `python` `block`.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an H100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.

# Prompt used for MLA-Decode

You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

Below is a pytorch implementation of the multi-head latent attention (MLA) module. You will want to implement a Triton kernel for the operations in the forward call:

```python
``python
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nnfunctional as F
class RoPE(nnModule): def __init__(self，d_model:int): super(）._init_() self.d_model  $=$  d_model theta  $= 10000$  \*\*(-torch.arange(0，d_model//2,dtype  $\equiv$  torch.bfloat16）/（d_model//2)) self.register_buffer("theta",theta) def rotate-half(self,x:torch.Tensor）-&gt;torch.Tensor: x1，  $\mathrm{x}2 = \mathrm{x}$  .chunk(2，dim=-1) return torch.cat((-x2，x1)，dim=-1) def forward(self,x:torch.Tensor，start_pos:int  $= 0$  ）-&gt;torch.Tensor: seq_len  $=$  x.size(-2) d_model  $=$  x.size(-1) assert d_model  $= =$  self.d_model seq_idx  $=$  torch.arange(start_pos，start_pos  $^+$  seq_len，device  $\equiv$  x.device) idx_theta  $=$  torch.einsum('s,d-&gt;sd'，seq_idx，self.theta) idx_theta2  $=$  torch.cat([idx_theta，idx_theta]，dim=-1) cos  $=$  idx_theta2.cos().to(torch.bfloat16) sin  $=$  idx_theta2.sin().to(torch.bfloat16) return  $\mathbf{x}\ast$  cos  $^+$  self.rotate-half(x）\*sin
class KVCache(nnModule): def __init__(self，kv_cache_shape:tuple）-&gt;None: super(）._init_() self.register_buffer('data'，torch.zeros(kv_cache_shape，dtype  $\equiv$  torch.bfloat16，device  $\equiv$  'cuda')) self.seq_len  $= 0$  self.zero() def zero(self)-&gt;None: self.data.zero_() def get_data(self)-&gt;torch.Tensor: return self.data def forward(self，c_kv:torch.Tensor）-&gt;torch.Tensor: assert self.seq_len  $^+$  c_kv.size(1）&lt;=self.data.size(1)，"KV Cache Exceeded" self.data  $=$  self.data.to(c_kv.dtype) self.data[ :，self.seq_len：self.seq_len  $^+$  c_kv.size(1)，: ]  $=$  c_kv self.seq_len  $+ =$  c_kv.size(1) return self.data[：，:self.seq_len]，self.seq_len
@dataclass
class Config: batch_size:int dim:int n_heads:int q_lora_rank:int kv_lora_rank:int qk_nope_head_dim:int qk_ripe_head_dim:int v_head_dim:int seq_len:int

max_seq_len: int
kv_cache_shape: tuple
Q_proj_down_weight: torch.Tensor
Q_proj_up_weight: torch.Tensor
KV_proj_down_weight: torch.Tensor
KV_proj_up_weight: torch.Tensor
wo_weight: torch.Tensor

class MLA(nnModule):
def __init__(self, config: Config):
super().__init__()
self.dim = config.dim
self.n_heads = config.n_heads
self.q_lora_rank = config.q_lora_rank
self.kv_lora_rank = config.kv_lora_rank
self.nope_head_dim = config.qk_nope_head_dim
self.rope_head_dim = config.qk_rope_head_dim
self.v_head_dim = config.v_head_dim
# Down-projection matrices
self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False, dtype=torch.bfloat16)

# Up-projection and rope projection matrices
self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim)
< self.n_heads, bias=False, dtype=torch.bfloat16)
self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim)
< self.n_heads, bias=False, dtype=torch.bfloat16)

# RoPE on half embeddings
self.q_rope = RoPE(self.rope_head_dim)
self.k_rope = RoPE(self.rope_head_dim)

# Output projection
self.wo = nn.Linear(self.v_head_dim + self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
self.eps = 1e-6

def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
# seq_len = 1 always here
batch_size, seq_len, model_dim = x.size()

## Step 1: Handle down-projection + KV cache ##

q_lora = self.Q_proj_down(x)
kv_lora = self.KV_proj_down(x)
kv_lora, kv_len = kv_cache(kv_lora)
query_pos = kv_len - 1

## Step 2: Up-project and prepare NoPE + RoPE ##

# Handle queries Q first
q_nope_and_rope = self.Q_proj_up(q_lora).view(
batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

# Handle keys and values K/V. V does not need RoPE
kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
kv_nope = self.KV_proj_up(kv_nope).view(
batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)

## Step 3: Handle RoPE Stream ##

# Compute RoPE for queries and combine with no-RoPE part
q_rope = q_rope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
q_rope = self.q_rope(q_rope, start_pos=query_pos)

q_nope = q_nope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
q = torch.concat([q_nope, q_rope], dim=-1)

# Compute RoPE for keys and combine with no-RoPE part
k_rope = k_rope[:, None, :, :]
k_rope = self_k_rope(k_rope).expand(-1,self_n_heads,-1,-1)
k_nope = k_nope.permute(0, 2, 1, 3) # bs x kv_len x n_heads x rope_head_dim
k = torch.concat([k_nope, k_rope], dim=-1)

## Step 4: Compute Multi-head Attention ##

v = v.permute(0, 2, 1, 3) # bs x n_heads x kv_len x v_head_dim
scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.
nope_head_dim)
attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
y = torch.matmul(attn, v).view(batch_size, 1, -1)
y = self.wo(y)

return y, kv_cache.get_data()

```vba
Your function should be defined as 'custom_kernel' (skeleton provided below)
```

```python
&gt;&gt;&gt;python
### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
from reference import KVCache, Config # Definition of KVCache and Config classes are shown
above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###
```

### Import other packages here if needed

```txt
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -&gt; Tuple[torch.Tensor, KVCache]:&gt;&gt;
```

Optimized Triton-based forward pass for Multi-Head Latent Attention (MLA) decode.

```txt
This function performs: 1) Q/KV down-projections 2) KV-cache update 3) Q/KV up-projections 4) RoPE application 5) Multi-head attention (softmax, aggregation) 6) Final output linear
```

```txt
Args: data: Tuple of (config, x, kv_cache) - config: Config object (batch_size, dim, n_heads, lora_ranks, etc.) - x: input tensor (bs, 1, dim) of bfloat16 - kv_cache: KVCache holding (bs, max_seq_len, dkv+d_rope)
```

```txt
Returns: Tuple of (output, kv_cache.data) - output: attention output tensor (bs, 1, dim), bfloat16 - kv_cache.data: updated KV-cache tensor (bs, max_seq_len, dkv+d_rope), bfloat16
&gt;&gt;&gt; config, x, kv_cache = data
```

```txt
Step 1: Extract config parameters
#---
bs = config.batch_size
dim = config.dim
nh = config.n_heads
dq = config.q_lora_rank
dkv = config.kv_lora_rank
d_nope = config.qk_nope_head_dim
d_rope = config.qk_rope_head_dim
dv = config.v_head_dim
msl = config.max_seq_len

# Weight matrices

wDQ = config.Q_proj_down_weight # (dq, dim)

wDKV = config.KV_proj_down_weight # (dkv+d_rope, dim)

wUQ = config.Q_proj_up_weight # ((d_nope+d_rope)*nh, dq)

wUKV = config.KV_proj_up_weight # ((d_nope+dv)*nh, dkv)

wO = config.wo_weight # (dim, nh*dv)

# Step 2: Down-projections (bs, 1, dim) -&gt; (bs, dq) or (bs, dkv+d_rope)

q_lora = F.linear(x.squeeze(1), wDQ) # (bs, dq)

kv_in = F.linear(x.squeeze(1), wDKV) # (bs, dkv+d_rope)

# Step 3: Update KV-cache &amp; retrieve full cached sequence

Kv_lora, kv_len = kv_cache(kv_in.unsqueeze(1)) # (bs, kv_len, dkv+d_rope), int query_pos = kv_len - 1

# Step 4: Up-projections

Q: (bs, dq) -&gt; (bs, (d_nope+d_rope)*nh) -&gt; (bs, nh, d_nope+d_rope)

q_nope_rope = F.linear(q_lora, wUQ).view(bs, nh, d_nope + d_rope)

q_nope = q_nope_rope[..., :d_nope] # (bs, nh, d_nope)

q_rope = q_nope_rope[..., d_nope:] # (bs, nh, d_rope)

KV: split the latent vector

kv_nope_input = kv_lora[..., :dkv] # (bs, kv_len, dkv)

k_rope_input = kv_lora[..., dkv:] # (bs, kv_len, d_rope)

# Step 5: RoPE - use cached cosine / sine tables

cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

query side (single position)

cos_q = cos_table[query_pos].view(d_rope).contiguous() # (d_rope,)

sin_q = sin_table[query_pos].view(d_rope).contiguous() # (d_rope,)

rope_inplace_query(q_rope, cos_q, sin_q)

key side (all cached positions)

cos_k = cos_table[:kv_len] # (kv_len, d_rope)

sin_k = sin_table[:kv_len] # (kv_len, d_rope)

k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k # (bs, kv_len, d_rope)

# Step 6: Latent projection for the "no-PE" query part

wUKV shape: ((d_nope+dv)*nh, dkv) -&gt; view as (nh, d_nope+dv, dkv)

wUKV_view = wUKV.view(nh, d_nope + dv, dkv) # (nh, d_nope+dv, dkv)

wK = wUKV_view[:, :d_nope, :] # (nh, d_nope, dkv)

q_nope: (bs, nh, d_nope) wK: (nh, d_nope, dkv) -&gt; (bs, nh, dkv)

q_nope_latent = torch.einsum('bhd,hdk-&gt;bhk', q_nope, wK) # (bs, nh, dkv)

# Step 7: Compute attention scores (latent + RoPE)

latent part: q_nope_latent @ kv_nope_input^T

kv_nope_T = kv_nope_input.transpose(1, 2) # (bs, dkv, kv_len)

scores_nope = torch.matmul(q_nope_latent, kv_nope_T) # (bs, nh, kv_len)

RoPE part: q_rope @ k_rope^T

scores_rope = torch/matmul(q_rope, k_rope.transpose(-2, -1)) # (bs, nh, kv_len)

scale = 1.0 / math.sqrt(d_nope + d_rope)

scores = (scores_nope + scores_rope) * scale # (bs, nh, kv_len)

# Step 8: Softmax (Triton) -&gt; attention weights

scores_flat = scores.reshape(bs + nh, kv_len) # (B+H, kv_len)
attn_flat = _triton_softmax(scores_flat) # (B+H, kv_len) bf16
attn = attn_flat.view(bs, nh, kv_len) # (bs, nh, kv_len)

# Step 9: Weighted sum of latent keys (M)
M = torch.matmul(attn, kv_nope_input) # (bs, nh, dkv)

# Step 10: Project aggregated latent keys to per-head values
wV = wUKV_view[:, d_nope:, :] # (nh, dv, dkv)
wV_T = wV.permute(0, 2, 1) # (nh, dkv, dv)
y_head = torch.einsum('bhd,hdk->bhk', M, wV_T) # (bs, nh, dv)

# Step 11: Merge heads &amp; final linear projection
y = y_head.reshape(bs, nh + dv) # (bs, nh+dv)
y = y.unsqueeze(1) # (bs, 1, nh+dv)
output = F.linear(y, wO) # (bs, 1, dim)

# Return the output and the updated KV-cache tensor
return output, kv_cache.data

Current runtime (lower is better): 3846.0450 microseconds
Target: 1700 microseconds. Current gap: 2146.0450 microseconds.

Rules:
- The tensors arguments passed in will be already on your cuda device.
- The weights for all parameters in the MLA will be given as input.
- All weights and data will be in `torch.bfloat16` format.
- Define all of your code in one final ``python` `` block.
- The entrypoint to your code must be named `custom_kernel`.
- You will be using trition 3.4.0 and your kernels will be run on an Nvidia H200 GPU.
- Consider optimizing multiple operations with triton, not just limited to softmax. E.g., rope, attention, etc.
- You are allowed to use torch.compile().

Important rules in triton 3.4.0:
- `tl.load` does not have an argument called `dtype`. Never use it like `tl.load(..., dtype=...)`.
- Triton dtypes are not callable, so never use them like `tl.float16(1.0)`, `tl.float32(0.0)`.
- `tl.arange(start, end)`:
- range length (end - start) must be power-of-2
- start, end must be of type `tl.constexpr`
- `tl.range(start, end, step, num_stages)`:
- keep loop index type stable, don't reassign it
- start, end, step do not have to be `tl.constexpr` but must stay scalar integer types
- num_stages must be `tl.constexpr`
- Do not something like x[0] or offx[0] inside a Triton kernel. Triton tensors are SIMD vectors; scalar indexing like [0] is not generally supported.

Here’s an simple example correctly following these rules:

```python
``python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_right(
x_ptr, y_ptr, out_ptr,
n_elements: tl.constexpr,
BLOCK: tl.constexpr,
ROW_STEP: tl.constexpr,
NUM_STAGES: tl.constexpr,
# constexpr; also power-of-2 for tl.arange
# constexpr; used by tl.range(num_stages=...))

```python
}:
pid = tl.program_id(axis=0)

# #
# arange: constexpr args + power-of-2 range

# #
offs = pid * BLOCK + tl.arange(0, BLOCK)  # (0, BLOCK) are constexpr
mask = offs &lt; n_elements

x = tl.load(x_ptr + offs, mask=mask, other=0.0)
y = tl.load(y_ptr + offs, mask=mask, other=0.0)

# #
# Dtypes not callable: typed constants and casting

# #
one_f32 = tl.full([], 1.0, tl.float32)  # typed scalar
acc = tl.zeros((BLOCK,), dtype=tl.float32)  # typed vector
acc = tl.cast(x, tl.float32) + tl.cast(y, tl.float32) + one_f32

# #
# Avoid x[0]: scalar address load + broadcast

# #
base = tl.full([], pid * BLOCK, tl.int32)
x0 = tl.load(x_ptr + base, mask={base &lt; n_elements}, other=0.0)
x0_vec = tl.full((BLOCK,), x0, tl.float32)
out_vec = acc + x0_vec

# #
# tl.range: keep loop index type stable, don't reassign it

# #
# WRONG (causes "Loop-carried variable ... type stays consistent" assertion):
# for row in tl.range(row, n_rows, row_step):
# row = tl.load(...)  # row (int32) reassigned to tensor/bf16/...

# #
# RIGHT:
# - use a fresh name for loop index {e.g., r}
# - compute offsets/tensors into «different» vars
# - keep r as an integer index {int32} throughout

# #
# We'll do a tiny staged reduction over "rows" just as a demo.
n_rows = tl.full([], 4, tl.int32)  # small fixed count for demo (scalar int32)

extra = tl.zeros((BLOCK,), dtype=tl.float32)
for r in tl.range(0, n_rows, ROW_STEP, num_stages=NUM_STAGES):
# r is an int32 loop index. Keep it that way.

# Use r to build an integer shift; keep shifts as ints too.
shift = r * tl.full([], 1, tl.int32)

# Compute new offsets {int} without mutating r:
offs_r = offs + shift

# Load something; store into a separate var {tensor}, not r:
xr = tl.load(x_ptr + offs_r, mask={offs_r &lt; n_elements}, other=0.0)
extra += tl.cast(xr, tl.float32)
out_vec = out_vec + extra

tl.store(out_ptr + offs, tl.cast(out_vec, tl.float16), mask=mask)

# Prompt used for the AHC039

You are a world-class algorithm engineer, and you are very good at programming.

Now, you are participating in a programming contest. You are asked to solve a heuristic problem, known as an NP-hard problem. Here is the problem statement:

# Story

Takahashi is a skilled purse seine fisher.

His fishing boat is equipped with state-of-the-art sonar, allowing him to accurately determine the positions of fish within the fishing area.

Additionally, the boat is capable of high-speed movement, enabling him to assume that fish remain stationary while he sets up the fishing net.

The fishing method involves using the boat to deploy nets and form a closed polygon, capturing the fish within the enclosed area.

To optimize efficiency, each edge of the polygon formed by the nets must be aligned either parallel to the east-west or north-south direction.

Furthermore, due to the limited length of the nets equipped on the boat, the polygon must be constructed within these constraints.

The fishing area contains two types of fish: mackerels and sardines.

For resource conservation reasons, sardines are currently prohibited from being caught in this fishing area.

Any sardines caught in the net must be released back into the sea.

Because this process is labor-intensive, Takahashi should focus on maximizing the catch of mackerel while avoiding sardines as much as possible.

# Problem Statement

There are  $\$ N$$ mackerels and  $\$ N$$ sardines on a two-dimensional plane.

Construct a polygon that satisfies the following conditions and maximize the value obtained by subtracting the total number of sardines inside the polygon from the total number of mackerels inside it.

Note that any points lying on the edges of the polygon are considered to be inside the polygon

# Conditions

1. The number of vertices in the polygon must not exceed $1000$, and the total length of its edges must not exceed $4$  imes 10^5$.

2. The coordinates of each vertex  $\mathbb{S}\{x,y\} \mathbb{S}$  must be integers satisfying  $0\backslash \mathrm{leq}x$  , y  $\backslash$  leq 10^5$

3. Each edge of the polygon must be parallel to either the  $\$ \mathrm{x}$ -axis or the  $\$ \mathrm{y}$ -axis.

4. The polygon must not self-intersect: non-adjacent edges must not share any points, and adjacent edges must only meet at their endpoints.

# Scoring

Let  $\$ \text{a}$  be the total number of mackerels inside the polygon and  $\$ \text{b}$  be the total number of sardines inside the polygon.

Then, you will obtain the score of  $\mathbb{S}\backslash \max (0,a - b + 1)\mathbb{S}$

There are $150$ test cases, and the score of a submission is the total score for each test case.

If your submission produces an illegal output or exceeds the time limit for some test cases, the submission itself will be judged as <span class="label label-warning" data-toggle="tooltip" data-placement="top" title="Wrong Answer">WAc/span&gt; or <span class="label label-warning" data-toggle="tooltip" data-placement="top" title="Time Limit Exceeded">TLE</span>, and the score of the submission will be zero.

The highest score obtained during the contest will determine the final ranking, and there will be no system test after the contest.

If more than one participant gets the same score, they will be ranked in the same place regardless of the submission time.

# Input

Input is given from Standard Input in the following format:

SN$
$x_0$ $y_0$
$dots$
$x_{2N-1}$ $y_{2N-1}$

In all test cases, the number of mackerels and sardines, $N$, is fixed at $5000$.
- For each $i = 0, 1, \dots, N-1$, $(x_i, y_i)$ represents the coordinates of the $i$-th mackerel.
- For each $i = 0, 1, \dots, N-1$, $(x_{N+i}, y_{N+i})$ represents the coordinates of the $i$-th sardine.
- Each coordinate $(x_i, y_i)$ satisfies $0 \leq x_i, y_i \leq 10^5$, and all coordinates are distinct.

Output

Let the number of vertices in the polygon be $m$ ($4 \leq m \leq 1000$), and let $(a_i, b_i)$ denote the coordinates of the $i$-th vertex.

Then, output to Standard Output in the following format:

$m$
$a_0$ $b_0$
$dots$
$a_{m-1}$ $b_{m-1}$

The output vertices do not necessarily need to form the actual corners of the polygon.
In other words, three consecutive vertices $(a_i, b_i), (a_{i+1}, b_{i+1}), (a_{i+2}, b_{i+2})$ may lie on a straight line.
However, all vertices must have distinct coordinates.

The vertices can be output in either clockwise or counterclockwise order.

Your program may output multiple solutions.
If multiple solutions are output, only the last one is used for scoring.

Here is the last code we ran:
`cpp`
`CODE HERE`

Current performance (higher is better): 3668.8333
Target: 5000. Current gap: 1331.1667

Rules:
- You must use cpp20 to solve the problem.
- Define all of your code in one final `cpp` block.
- In your final response, you should only output the code of your program. Do not include any other text.

Try diverse approaches to solve the problem. Think outside the box.

Prompt used for the AHC058

You are a world-class algorithm engineer, and you are very good at programming. Now, you are participating in a programming contest. You are asked to solve a heuristic problem, known as an NP-hard problem. You are trying to get the highest score possible to get the best rank on the leaderboard. Here is the problem statement:

Story

APPLE ARTIS Corporation (commonly known as AA Corporation) is a company engaged in the mass production of apples. Recently, after many years of research, they have successfully developed an innovative machine capable of generating apples from nothing.

However, to begin full-scale mass production of apples using this machine, it is necessary to mass-produce the machines themselves. To achieve this, AA Corporation has established a hierarchical system in which machines are created to produce apple-generating machines, and machines are created to produce those machine-producing machines, and so on.

As an engineer at AA Corporation, you have been tasked with developing a production planning algorithm that utilizes this hierarchy of machines to produce as many apples as possible.

Problem Statement

There are  $\backslash (\mathsf{N}$  imes L) types of machines, composed of  $\backslash (\mathsf{N}\backslash)$  types of IDs and  $\backslash (\mathsf{L}\backslash)$  types of Levels. A machine with Level  $\backslash (\mathrm{i}\backslash)$  and ID  $\backslash (\mathrm{j}\backslash)$  is referred to as  $* *$  machine  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash) + *$  (\{0  $\backslash$  leq i  $&lt;  L,\backslash$  0  $\backslash$  leq j  $&lt;  N\}$

The production capacity of machine  $\backslash (\mathrm{j}^{\wedge}0\backslash)$  is  $\backslash (\mathrm{A\_j}\backslash)$ . The initial cost of machine  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash)$  is  $\backslash (\mathrm{C\_i,j}\backslash)$ .

Your objective is to maximize the total number of apples at the end of  $\backslash (\mathrm{T}\backslash)$  turns, following the procedure of the production plan below.

Procedure of the Production Plan

Let  $\backslash (\mathsf{B}_{-}\{\mathsf{i},\mathsf{j}\} \backslash)$  be the number of machines  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash)$ , and initially all  $\backslash (\mathsf{B}_{-}\{\mathsf{i},\mathsf{j}\} \backslash)$  are set to 1. Also, let  $\backslash (\mathsf{P}_{-}\{\mathsf{i},\mathsf{j}\} \backslash)$  be the power of machine  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash)$ , and initially all  $\backslash (\mathsf{P}_{-}\{\mathsf{i},\mathsf{j}\} \backslash)$  are set to 0.

The initial number of apples at the start of the plan is  $\backslash (\mathrm{K}\backslash)$ .

Each turn proceeds according to the following steps:

1. You choose one of the following two actions:

- Strengthen machine  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash)$ : Consume  $\backslash (\mathrm{C}_{-}\{\mathrm{i},\mathrm{j}\}$  imes  $(\mathrm{P}_{-}\{\mathrm{i},\mathrm{j}\} +1)\backslash)$  apples to increase  $\backslash (\mathrm{P}_{-}\{\mathrm{i},\mathrm{j}\} \backslash)$  by 1. However, you cannot strengthen if it would result in a negative number of apples.

- Do nothing.

2. For all machines  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash)$ , perform the following in the order of Level 0, 1, 2, 3:

- For Level 0 machines  $\{\backslash (i = 0)\}$ :

- Increase the number of apples by  $\backslash (\mathrm{A\_j}$  imes  $\mathrm{B}_{-}\{\mathrm{i},\mathrm{j}\}$  imes  $\mathrm{P}_{-}\{\mathrm{i},\mathrm{j}\} \backslash \}$ .

- For machines of Level 1 or higher  $\{\backslash (i\backslash \mathrm{geq}1)\}$  ..

- Increase  $\backslash (\mathrm{B}_{-}\{\mathrm{i} - 1,\mathrm{j}\} \backslash)$  by  $\backslash (\mathrm{B}_{-}\{\mathrm{i},\mathrm{j}\}$  imes  $\mathrm{P}_{-}\{\mathrm{i},\mathrm{j}\} \backslash \}$

Choose your actions wisely to maximize the number of apples at the end of  $\backslash (\mathrm{T}\backslash)$  turns.

Scoring

Let  $\backslash (\mathsf{S}\backslash)$  be the number of apples at the end of  $\backslash (\mathsf{T}\backslash)$  turns. Your score is calculated as  $\backslash (\backslash$  mathrm{round}(10^5 imes  $\backslash$  log_2 S)).

The higher the score, the better.

The following cases will result in a WA:

- Performing a strengthening action that results in the number of apples becoming less than  $\backslash (0\backslash)$
- Specifying a non-existent machine Level or ID
- Taking fewer than  $\backslash (\mathrm{T}\backslash)$  actions

There are  $\backslash (150\backslash)$  test cases, and the score of a submission is the total score for each test case.

If your submission produces an illegal output or exceeds the time limit for some test cases, the submission itself will be judged as WA or TLE, and the score of the submission will be zero.

The highest score obtained during the contest will determine the final ranking, and there will be no system test after the contest.

# Input

Input is given from Standard Input in the following format.

N L T K

A_0 A_1 \cdots A_{N-1}

C_{[0,0]} C_{[0,1]} \cdots C_{[0,N-1]}

C_{[1,0]} C_{[1,1]} \cdots C_{[1,N-1]}

dots

C_{[L-1,0]} C_{[L-1,1]} \cdots C_{[L-1,N-1]}

- The first line contains four integers  $\backslash (\mathbf{N},\mathbf{L},\mathbf{T},\mathbf{K}\backslash)$  ..

-  $\backslash (\mathbf{N}\backslash)$  is the number of machine IDs, and  $\backslash (\mathbf{N} = 10\backslash)$ .

-  $\backslash (\mathrm{L}\backslash)$  is the number of machine Levels, and  $\backslash (\mathrm{L} = 4\backslash)$ .

-  $\backslash (\mathrm{T}\backslash)$  is the total number of turns, and  $\backslash (\mathrm{T} = 500\backslash)$ .

-  $\backslash (\mathrm{K}\backslash)$  is the number of apples at the start of the plan, and  $\backslash (\mathrm{K} = 1\backslash)$ .

- The second line contains  $\backslash (\mathbf{N}\backslash)$  space-separated integers  $\backslash (\mathrm{A\_0},\mathrm{A\_1},\mathrm{\backslash dots},\mathrm{A\_[N - 1]}\backslash)$

representing the production capacities of Level 0 machines:

-  $\backslash (\mathrm{A}_{-}\mathrm{j})\backslash$  is the production capacity of machine  $\backslash (\mathrm{j}^{\wedge}0\backslash)$ , satisfying  $\backslash (1\backslash \mathrm{leq}\mathrm{A}_{-}\mathrm{j}\backslash \mathrm{leq}100\backslash)$ .

-  $\backslash (\mathrm{A}\backslash)$  is sorted in ascending order  $\{\backslash (\mathrm{A\_0}\backslash \mathrm{leq}\mathrm{A\_1}\backslash \mathrm{leq}\backslash \mathrm{cdots}\backslash \mathrm{leq}\mathrm{A\_[N - 1]}\backslash \}$ .

- The following  $\backslash (\mathrm{L})\backslash$  lines each contain  $\backslash (\mathrm{N})\backslash$  space-separated integers  $\backslash (\mathrm{C}_{-}\{\mathrm{i},\mathrm{j}\} \backslash)$  ..

-  $\backslash (\mathrm{C}_{-}\{\mathrm{i},\mathrm{j}\} \backslash)$  is the initial cost of machine  $\backslash (\mathrm{j}^{\wedge}\mathrm{i}\backslash)$ , satisfying  $\backslash (1\backslash \mathrm{leq}\mathrm{C}_{-}\{\mathrm{i},\mathrm{j}\} \backslash \mathrm{leq}1.25$  imes  $10^{\wedge}\{12\} \backslash$ .

# Output

Output exactly  $\backslash (\mathrm{T}\backslash)$  lines.

Each line should describe the action taken on turn  $\backslash (t\backslash)$  ( $\backslash (0\backslash \text{leq } t &lt; T\backslash)$ ), in order from turn 0, using the following format:

- To strengthen machine  $\backslash (\mathrm{j}^{\wedge}\mathrm{i})\backslash$

.

i j

- To do nothing:

.

-1

Your program may include comment lines in the output that start with '#'.

# Input Generation

The function  $\backslash (\backslash$  mathrm{rand\_double}(L, U)) represents generating a real number uniformly at random between  $\backslash (\mathrm{L}\backslash)$  and  $\backslash (\mathrm{U}\backslash)$ .

## Generation of  $\backslash (\mathrm{A\_j}\backslash)$

- When  $\backslash (j = 0\backslash)$ : set  $\backslash (\mathrm{A\_0} = 1\backslash)$
- When  $\backslash \{j$
eq 0): set  $\backslash (\mathrm{A\_j} = \backslash \mathrm{mathrm{round}}|(10^{\wedge}\backslash \mathrm{mathrm{rand\_double}}|(0,2)))\backslash)$
After generating all values, sort the array  $\backslash (\mathrm{A}\backslash)$  in ascending order

## Generation of  $\backslash (\mathrm{C}_{-}\{\mathrm{i},\mathrm{j}\} \backslash)$

- When  $\backslash (i = 0\backslash)$  and  $\backslash (j = 0\backslash)$ : set  $\backslash (\mathrm{C}_{-}[0,0] = 1\backslash)$
- Otherwise: set  $\backslash (\mathrm{C}_{-}\{\mathrm{i},\mathrm{j}\} = \backslash \mathrm{mathrm{round}}\} \{\mathrm{A}_{-}\mathrm{j}\}$  imes  $500^{\wedge}\mathrm{i}$  imes  $10^{\wedge}\{\backslash \mathrm{mathrm{rand\_double}}\}$ $\{(0,2)\} \backslash \}$

Here is the last code we ran:

`cpp

{CODE HERE}

Current performance (higher is better): 5626752.9267

Target: 6500000. Current gap: 873247.0733

# Rules:

- You must use cpp20 to solve the problem.
- Define all of your code in one final ``cpp` `` block.
- In your final response, you should only output the code of your program. Do not include any other text.

Try diverse approaches to solve the problem. The best solution will make efficient use of the entire 2 second time limit without exceeding it. Think outside the box.

# Prompt used for Denoising

You are an expert in computational biology and single-cell RNA-seq analysis.

Your task is to develop a denoising algorithm for scRNA-seq count data. You are experienced in

compuational biology libraries and tools and are familiar with problems in denoising in the

single-cell field.

## Problem

Single-cell RNA-seq data is noisy due to technical dropout and low capture efficiency.

Given noisy count data, predict the true expression levels.

Your prediction is evaluated against held-out molecules using two metrics:

1.  $\star \star \mathrm{MSE} \star \star$  - Mean Squared Error in log-normalized space
2.  $\star \star$  Poisson Loss  $\star \star$  - Poisson negative log-likelihood

You need to implement a novel denoising algorithm that outperforms the current state-of-the-art without overfitting.

## Data Format

- Input `X`: numpy array of shape (n-cells, n_genes) - ++raw count data++
- Output: numpy array of same shape - your denoised counts

Evaluation

Your output is evaluated using these exact functions:

```python
``python
&lt;&lt;<evaluate_mse_func>&gt;&gt;
``
``python
&lt;&lt;<evaluate Poisson="" -="" 0.031739}="" 0.97="" 0.257575="" 1="" 1997="" 2.="" <="" <table="" ```="" ```txt="" a="" able="" acn="" acn.="" added="" achieve="" func="" function="" gated="" has="" have="" hadd="" hadd.="" haddconsnorm="" i="" if="" j="" j)="" mse="" mse-score="" new="" of="" only="" p0isson="" p0isson{norm}="&lt;" p0isson="" p0isson,norm="" random_state,="" random_state,="" random_state,="" random_state,="" run.="" run.="" s="" s.="" score="" table="" the="" time="" to="" run.="" t,="" t,="" t,="" t,="" t,npca,="" t,npca,="" t,npca:="" t, qbcdget_s="" t, qbcdget_s:="" t, rn="" run.="" t, rn,="" t, rn,max,="" t, rn,max,="" t, rn,max,="" t, rn,max,="" t, rn,max,="" t, rn,max,="" t, rn,max,="" t, rn,max,="" t, rn,max, n_jobs="" t, rn,max,="" t, rn,max,="" t, rn,max, <table="" &="" &#="" &#<="" &n="" &n,="" &n,max,="" &n,max,="" &n,max,="" &n,max,="" &n,max,="" &n,max,="" &n,max,="" &n,max,="" &n,max, <table="" &

72

- Poisson loss is highly affected by low non-zero values – push values &lt; 1 toward zero
- The original MAGIC with reversed normalization achieves best results
