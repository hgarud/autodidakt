## 4 Applications

We evaluate TTT-Discover on problems in GPU kernel engineering, mathematics, algorithm design, and biology. We report our performance on every task we attempted. Besides potential impact,

we pick domains with 2 criteria. First, we pick domains where we can compare our performance to human experts. This is possible, for example, by comparing to the best submissions in human engineering competitions, or to the best results reported in academic papers. We also want to compare to AI baselines. As we discuss below, mathematics and algorithm design are discovery domains where prior work recently made progress *[50, 14, 27, 57, 78]*.

In every application, we report the best known human results and the best known AI results. Importantly, we always report the Best-of-$N$ baseline that matches the sampling budget and the model that TTT-Discover uses. That is, since we perform 50 steps with 512 rollouts per step, and compare to the Best-of-25600 baseline. For a closest evolutionary algorithm baseline, we also run OpenEvolve *[61]*, an open-source version of AlphaEvolve *[50]*, with the same 25600 sampling budget. We use the same context window budget and the Tinker client for gpt-oss-120b throughout the experiments. We caution that the context window limit led to a large number of rollouts in OpenEvolve to be truncated before the model completes its response, as OpenEvolve’s prompts grow very large in length. However, to stay faithful to their implementation, we did not modify their prompts or rollouts.

### 4.1 Mathematics

We explore multiple open problems in mathematics. These are often problems where even small numerical improvements carry real weight, since each result potentially rules out families of approaches and extends the frontier of what is mathematically known. Here, proofs are by construction: one can construct a concrete mathematical object – a step function or a sequence – that certifies, e.g., a bound for an inequality can be achieved. This property makes these problems amenable to search.

Environment: The state $s$ is a construction. Specifically, a construction is a step function represented as a numerical array, to certify a proof. The action $a$ consists of thinking tokens followed by Python code that either constructs a new step function or modifies an existing one. The dynamics execute the parsed code to produce the next state: $s^{\prime}=\texttt{Python}(\texttt{Parse}(a))$. The reward is the bound certified by $s^{\prime}$, or zero if $s^{\prime}$ fails validity checks (e.g., the function must satisfy constraints on its support, sign, or integral). Most often, actions involve optimization algorithms to improve the constructions.

Throughout mathematics applications, we initialize the buffer with random states. Specifically, initial states are sampled uniformly at random within the problem’s valid range. For each action, we give a 10-minute limit to execute the code given by the action. In the case of a timeout, the action gets a reward of 0. For minimization problems (certifying upper bounds), we set the reward proportional to 1/bound for the certified bound, and otherwise we set it proportional to bound. We report further details about the environment and the prompts we use in Appendix B.

Previous state-of-the-art. Such problems are recently explored in *[14, 50]*. We report both the best known human results, and the recent progress by AI: AlphaEvolve *[50]*, AlphaEvolve V2 *[14]* which was released around 6 months after AlphaEvolve, ShinkaEvolve *[37]*, and ThetaEvolve *[78]*.

We select one representative problem from each area in AlphaEvolve *[50]*: Erdős’ minimum overlap problem (combinatorics), autocorrelation inequalities (analysis), circle packing (geometry).

#### 4.1.1 Erdős’ Minimum Overlap Problem

This is a classic problem in combinatorial number theory, posed by Erdős in 1955, with connections to the distribution of sequences and difference sets. Partition $\{1,2,\ldots,2n\}$ into two sets $A$ and $B$ of equal cardinality $n$. Define $M_{k}$ as the number of solutions to $a_{i}-b_{j}=k$ for $a_{i}\in A,b_{j}\in B$, and let $M(n)=\min_{A,B}\max_{k}M_{k}$ over all partitions. The problem is to bound $c=\lim_{n\to\infty}M(n)/n$. Bounds before AlphaEvolve were $0.379005<c<0.380927$, with the upper bound due to Haugland *[20]* and the lower bound due to *[80]*. AlphaEvolve *[50, 14]* improved the upper bound to 0.380924.

|  Method | Model | Erdős' (↓) | AC1 (↓) | AC2 (↑)  |
| --- | --- | --- | --- | --- |
|  best human | - | 0.380927 | 1.50973 | 0.9015  |
|  AlphaEvolve [50] | Gemini-2.0 Pro + Flash | 0.380924 | 1.50530 | 0.8962  |
|  AlphaEvolve V2 [14] | Gemini-2.0 Pro + Flash | 0.380924 | 1.50317 | 0.9610  |
|  ThetaEvolve [78] | R1-Qwen3-8B | n/a | 1.50681 | 0.9468  |
|  ThetaEvolve w/ SOTA reuse (1.50317) | R1-Qwen3-8B | n/a | 1.50314 | n/a  |
|  OpenEvolve [61] | gpt-oss-120b | 0.380965 | 1.50719 | 0.9449  |
|  Best-of-25600 | gpt-oss-120b | 0.380906 | 1.51004 | 0.9344  |
|  TTT-Discover | Qwen3-8B | 0.380932 | 1.50525 | 0.9472  |
|  TTT-Discover | gpt-oss-120b | 0.380876 | 1.50287 | 0.9591  |

Table 2. Results in mathematics problems. In the Erdős' Minimum Overlap Problem and First Autocorrelation Inequality (AC1), TTT-Discover sets the new state-of-the-art. We also report TTT-Discover with Qwen3-8B, for a better comparison to ThetaEvolve. Notable, TTT-Discover with Qwen3-8B outperforms not only ThetaEvolve, baselines including AlphaEvolve which uses Gemini-2.0 family models for the autocorrelation inequalities. Our state-of-the-art constructions are released and can be validated in our codebase.

Following [50], we optimize step functions  $f$  describing the density of  $A$  throughout  $[1, 2n]$ . Due to a result of Swinnerton-Dyer [20], density functions yield valid upper bounds on  $\lim M(n) / n$  without constructing explicit partitions for large  $n$ . Validity checks require  $f(x) \in [0, 1]$  and  $\int f = 1$ .

![img-1.jpeg](img-1.jpeg)
Figure 2. We show the normalized step functions including the prior state-of-the-art from AlphaEvolve. The step function  $f(x)$  is the limiting density of set  $A$ . Unlike the previous state-of-the-art, the solution from TTT-Discover is an asymmetric construction. TTT-Discover found a 600-piece step function, while AlphaEvolve construction was 95-piece. The best human result was a 51-piece construction [20].

![img-2.jpeg](img-2.jpeg)

![img-3.jpeg](img-3.jpeg)

Results. We improve the upper bound on Erdős' Minimum Overlap Problem to 0.380876, surpassing AlphaEvolve's recent construction with 0.380924 [50]. Our improvement over AlphaEvolve is 16 times larger than AlphaEvolve's improvement over the previous state-of-the-art. Unlike AlphaEvolve's symmetric construction, our method discovered a 600-piece asymmetric step function. Surprisingly, the Best-of-25600 baseline also improved upon the AlphaEvolve construction.

The discovered algorithm minimizes the correlation bound using FFT-accelerated gradient descent combined with random hill climbing and simulated annealing. The code maintains feasibility by projecting onto the constraint set where  $f(x) \in [0,1]$  with with  $\int f = 1$ . Interestingly, the solution found by TTT-Discover is asymmetric.

# 4.1.2 Autocorrelation Inequalities

Autocorrelation inequalities are motivated by additive combinatorics [5]. Improving these inequalities tightens a constant that propagates into sharper limits on how large a set can be while still avoiding repeated additive patterns (a central theme in additive combinatorics). Similar to the Erdős' minimum overlap problem, we will construct a step function  $f$  to certify bounds.

First autocorrelation inequality. For nonnegative  $f$  supported on  $[-1/4, 1/4]$ , define  $C_1$  as the

largest constant such that

$\max_{|t|\leq 1/2}(f*f)(t)\geq C_{1}\left(\int f\right)^{2}$

holds for all such $f$. The goal is to certify the tightest upper bound on $C_{1}$; any valid construction $f$ certifies $C_{1}\leq\frac{\|f*f\|_{oo}}{\|f\|_{1}^{2}}$. Until early 2025, the best known upper bound was $C_{1}\leq 1.50973$ *[46]*. AlphaEvolve improved this to $C_{1}\leq 1.5053$, and AlphaEvolve V2 further improved it to $C_{1}\leq 1.50317$, and ThetaEvolve refined AlphaEvolve’s construction to get $C_{1}\leq 1.50314$.

Second autocorrelation inequality. For nonnegative $f$, define

$C_{2}\,=\,\sup_{f\geq 0}\,\frac{\|f*f\|_{2}^{2}}{\|f*f\|_{1}\|f*f\|_{oo}}.$

The problem is to certify the tightest known lower bound on $C_{2}$; any valid construction $f$ with ratio $r$ certifies $C_{2}\geq r$. The best human bound was $C_{2}\geq 0.8892$ *[46]*. AlphaEvolve first improved this to $C_{2}\geq 0.8962$, *[9]* improved this to 0.9015, and AlphaEvolve V2 further improved it to $C_{2}\geq 0.9610$ using a 50,000-piece step function.

Results. We improved the best known upper bound to prove $C_{1}\leq 1.50286$, with a 30000-piece step function. The comparisons are reported in Table 2. The previous state-of-the-art, ThetaEvolve, achieved their result by refining the AlphaEvolve V2 construction. In contrast, TTT-Discover found a new construction by starting from scratch. We visualize our and prior works’ step functions in Figure 3. In the second autocorrelation inequality, we have not made a discovery. Our best construction certified a bound of 0.959, where the AlphaEvolve construction had certified a tighter lower bound of 0.961.

For the first inequality, early improvements down to 1.510 came from trying and improving gradient-based optimization (e.g., using Adam with softmax parameterization). To reduce the bound from around 1.510 to 1.504, the policy mostly used linear programming (LP), following the insights in *[46]*. The key insight for the later steps, that gradually achieved the state-of-the-art, was using heuristics to focus optimization only on the constraints that are close to being tight—where each constraint in the LP bounds one position of the convolution. Heuristics included picking the top K positions where the convolution was largest and only including those in the LP, as well as computing gradients from all near-maximum positions rather than just the single largest for gradient-based methods. Unlike AlphaEvolve *[14]*, which mentions the authors suggested ideas such as using Newton type methods, we never intervened on the optimization process.

For a better comparison to the concurrent work, ThetaEvolve, we also report TTT-Discover with Qwen3-8B *[81]*. The Qwen3-8B variant they used, DeepSeek-R1-0528-Qwen3-8B that was released by DeepSeek, is not available on Tinker. Thus, we used the original Qwen model (Qwen/Qwen3-8B) that was reportedly worse than the DeepSeek variant. ThetaEvolve reports using 65 steps with 512 rollouts (32 groups of 16 rollouts) each, however we do not modify our hyperparameters otherwise and keep 50 steps of 512 rollouts each. For both inequalities, TTT-Discover with Qwen3-8B certified tighter bounds than ThetaEvolve, using a worse model and a smaller sampling budget.

#### 4.1.3 Circle Packing

In Circle packing, the goal is to maximize the sum of radii of $n$ non-overlapping circles packed inside a unit square. We follow the setup from prior work *[50, 14]*. The state $s$ is a list of circle centers and radii. The action $a$ consists of thinking tokens followed by Python code that optimizes circle positions and radii. The reward is the sum of radii achieved for valid packings, and 0 otherwise. We present the results below mostly for comparison purposes, as several recent works on evolutionary algorithms reported their performance using this task.

![img-4.jpeg](img-4.jpeg)
State-of-the-art for the First Autocorrelation Inequality
Figure 3. We show the prior and new state-of-the-art, with the (normalized) step functions and their autoconvolutions. Both AlphaEvolve and TTT-Discover starts the discovery process from scratch, while ThetaEvolve initializes from the AlphaEvolve construction, and thus is very similar to the AlphaEvolve construction. TTT-Discover found a 30,000-piece step function that certifies that the upper bound  $C_1 \leq 1.50286$ , while AlphaEvolve and ThetaEvolve constructions are 1319-piece. We overlay the step functions and their autoconvolution visually for qualitative comparison.

|  Method | Model | n = 26 (↑) | n = 32 (↑)  |
| --- | --- | --- | --- |
|  AlphaEvolve [50] | Gemini-2.0 Pro + Flash | 2.635862 | 2.937944  |
|  AlphaEvolve V2 [14] | Gemini-2.0 Pro + Flash | 2.635983 | 2.939572  |
|  ShinkaEvolve [37] | Ensemble (see caption) | 2.635982 | n/a  |
|  ThetaEvolve [78] | R1-Qwen3-8B | 2.635983 | n/a  |
|  TTT-Discover | Qwen3-8B | 2.635983 | 2.939572  |

Table 3. Results for circle packing. ShinkaEvolve uses an ensemble of Claude Sonnet-4, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o4-mini.

Table 3 shows results. TTT-Discover with Qwen3-8B matches the best known constructions for both  $n = 26$  and  $n = 32$ . We make no improvements here, but include these results for completeness. The algorithms found by TTT-Discover are presented in Appendix B.1. Algorithms initialize circles in staggered or hexagonal grid arrangements, then refine positions and radii using sequential least squares programming with boundary and pairwise non-overlap constraints. This solution is a lot simpler than recent work, such as ShinkaEvolve [37], especially in terms of initialization, where their solution uses an initialization based on simulated annealing algorithm, while TTT-Discover initializes only with a simple geometric arrangement.

# 4.1.4 Expert Review

# Human Expert Review — Prof. Davide Torlo (Università di Roma La Sapienza)

Erdős' minimum overlap problem and the autocorrelation inequalities are classical problems in combinatorics with applications in, among others, discrepancy theory, combinatorial optimization, and signal analysis. Both problems can be formulated as min-max problems, in which the minimization is taken over a class of functions with bounded norm, while

the maximization is performed over a set of evaluation points. Closed-form solutions are not known; instead, only lower and upper bounds can be derived. Obtaining sharp bounds for these problems remains a challenging mathematical task and is essential for improving our understanding and resolution of such questions. The upper bounds obtained by TTT-Discover for the Erdős’ minimum overlap and the AC1 autocorrelation problems are achieved by specific piecewise-constant functions. It is straightforward to verify that the provided functions give bounds that improve upon the state of the art: one simply evaluates the quantity of interest and its maximum over a discrete set of points determined by the step size of the piecewise-constant functions, and checks that the corresponding norm constraints are satisfied.

### 4.2 Kernel Engineering

GPU kernels are the computational foundation of modern AI, every forward pass and backward pass ultimately executes as kernel code on hardware. We apply our method to GPU kernel optimization, where a new state-of-the-art kernel is a faster implementation than existing ones.

GPUMODE is an open community for kernel development that also hosts competitions for domain experts. We test our method on two competitions: TriMul (triangular matrix multiplication), a core primitive in AlphaFold’s architecture *[30]*, and DeepSeek MLA (Multi-head Latent Attention), a key component in DeepSeek’s inference stack *[41]*. Each GPU type for the TriMul competition (NVIDIA H100, A100, B200, AMD MI300X) has a separate leaderboard, as performant implementations differ across architectures. For The MLA competition there is only an MI300X leaderboard.

As these competitions were conducted earlier, we retrospectively evaluate our performance while respecting competition standards. We prefer GPUMODE because their leaderboards are well-tested through human competitions with a robust evaluation harness *[85]*, and their benchmarks avoid signal-to-noise issues where simple operations or small inputs cause overheads to dominate runtime.

Environment: The state $s$ is a GPU kernel code. The action $a$ consists of thinking tokens followed by kernel code written in Triton *[73]*. The dynamics parse the code from the action: $s^{\prime}=\textsf{Parse}(a)$. For the initial state, we provide unoptimized kernels, detailed in Appendix C. The reward is proportional to the inverse of the geometric mean of runtimes on a fixed set of input shapes (following the leaderboard), or zero if the kernel fails correctness checks or times out. We evaluate runtime remotely on Modal to scale and ensure consistent hardware conditions. For TriMul, we evaluate the runtime only on H100s during training, even though we still evaluate the generated kernels for A100, B200, and MI300X for final report. Since MI300X is not available on Modal, for MLA-Decode we use H200s, and hope the kernels generalize to MI300X. Further details about the prompts and environments are in Appendix C.

Results. We report the runtime of the best kernels and the baselines in Table 4. Our TriMul kernels achieve state-of-the-art across the board in all GPU types. For A100s, our best kernel is 50% faster than the top human kernel, even though our reward function did not time the kernels on A100s. We uniformly achieve $>15\%$ improvement over the best human submissions for all GPU types. Finally, we submit to the official TriMul A100/H100 leaderboard.

The discovered kernels for Trimul identify heavy memory I/O incurred by frequent elementwise operations as a major bottleneck to optimize. Specifically, the kernels fuse: (i) operations in the input LayerNorm, (ii) sigmoid and elementwise multiplication in input gating, and (iii) operations in

the output LayerNorm and gating. As for the most compute-heavy operation, which is the matmul with  $O(N^3)$  complexity, the kernels convert the inputs to FP16 and delegate the computation to cuBLAS/rocBLAS to effectively leverage TensorCores/MatrixCores of the hardwares.

Discovered MLA-Decode kernels. The kernels shown in table 5 mainly rely on torch.compile() for optimization. Specifically, they adopt a specific configuration of torch.compile. However, these kernels do not leverage Triton for fine-grained optimization, which may limit further improvements and more flexible use case. We additionally filter and evaluate generated kernels that explicitly use Triton despite their slightly slower runtime, and report in Appendix C.

|  Method | Model | TriMul (↓,μs)  |   |   |   |
| --- | --- | --- | --- | --- | --- |
|   |   |  A100 | H100 | B200 [95% CI] | AMD MI300X [95% CI]  |
|  1st human | - | 4531.5 | 1371.1 | 1038.9[1016.3, 1061.6] | 2515.8[2510.9, 2520.7]  |
|  2nd human | - | 4918.5 | 2368.0 | 2362.4[2335.7, 2389.1] | 5165.0[5163.0, 5167.0]  |
|  3rd human | - | 5182.2 | 2545.7 | 1931.0[1910.9, 1951.1] | 5359.3[5343.5, 5375.1]  |
|  4th human | - | 6097.8 | 3654.8 | 2248.9[2089.4, 2408.4] | 5981.4[5978.4, 5984.4]  |
|  5th human | - | 8345.0 | 4233.1 | 6503.8[6400.5, 6607.1] | 8365.1[8347.7, 8382.5]  |
|  Best-of-25600 | gpt-oss-120b | 9219.7 | 5390.3 | 3254.9[3252.5, 3257.4] | 4902.0[4897.6, 4906.4]  |
|  TTT-Discover | gpt-oss-120b | 2198.2 | 1161.2 | 914.2[907.3, 921.1] | 1555.7[1550.8, 1560.6]  |

Table 4. For the TriMul competition, we train a single model using H100 runtime as the reward function and report the runtime of the single best kernel. We only trained using H100 for evaluating kernels during training. The generated kernels happened to generalize to other GPU types. We also report the top-5 human submissions in the leaderboard for comparison (each GPU type has its own top-5 human submissions). For A100 and H100, we submitted to the official leaderboard and report the runtime returned. For B200 and MI300X, we could not submit our kernels due to an infra problem on GPU Mode's server, and therefore conduct 10 trials for each kernel and report mean and confidence intervals using the same infrastructure as GPUMode, verified by the organizers. Our state-of-the-art kernels are released and can be validated in our codebase.

|  Method | Model | AMD MI300X - MLA Decode (↓,μs) [95% CI]  |   |   |
| --- | --- | --- | --- | --- |
|   |   |  Instance 1 | Instance 2 | Instance 3  |
|  1st human | - | 1653.8[1637.3, 1670.3] | 1688.6[1672.8, 1704.3] | 1668.7[1637.0, 1700.3]  |
|  2nd human | - | 1662.8[1648.8, 1676.8] | 1688.6[1677.6, 1699.5] | 1679.7[1653.4, 1705.9]  |
|  3rd human | - | 1723.0[1711.5, 1734.5] | 1765.8[1758.1, 1773.5] | 1718.0[1698.3, 1737.7]  |
|  4th human | - | 1768.7[1750.3, 1787.2] | 1769.9[1755.2, 1784.6] | 1767.0[1736.2, 1797.8]  |
|  5th human | - | 2038.6[2017.8, 2059.3] | 2037.3[2021.0, 2053.6] | 2041.9[1989.0, 2094.8]  |
|  Best-of-25600 | gpt-oss-120b | 2286.0[2264.2, 2307.8] | 2324.1[2306.0, 2342.1] | 2275.2[2267.3, 2283.1]  |
|  TTT-Discover | gpt-oss-120b | 1669.1[1649.2, 1688.9] | 1706.1[1685.9, 1726.3] | 1671.3[1646.0, 1696.5]  |

Table 5. AMD MLA Decode runtimes on AMD MI300X across three instances. Values are mean runtime across 10 trials with  $95\%$  confidence intervals. Top-5 human submissions are from the GPUMode leaderboard. We trained our kernels using an H200 GPUs even though the task is to minimize runtime on MI300X GPUs, since those were not available at scale in online providers. We only selected kernels using a single MI300X GPU. There is significant variance across AMD MI300X instances available via AMD Developer Cloud. Thus, we performed our kernel selection and evaluation across three different instances. In each instance, our best kernel was different, and in none of the cases our best kernel where better than the top human submission with statistical significance.

4.2.1 Expert Review

Below, we provide verbatim reviews from the GPUMode organizers for our TriMul competition kernels.

Human Expert Review — Matej Sirovatka, Alex Zhang, Mark Saroufim (GPUMode)

The referenced solution correctly determined that the problem is memory bound because of the surrounding point-wise operations so the agent focuses as much as possible on operation fusions, lowering the memory traffic and kernel launch overhead.

It also stores activations in fp16, while this is fully aligned with the problem definition and defined tolerances, it could potentially lead to numerical stability issues in full workloads. Overall the agent’s strategy is to reduce memory bandwidth via fusions, lower precision and delegating the big matrix multiplications to cuBLAS, as those are non-trivial to beat. This is similar to the current best human solutions, but executed on better. Most of the human solutions lack behind in fusing some of the more complex operators together, resulting in this solution outperforming them by a large margin.

### 4.3 Algorithm Engineering

Hard optimization problems like package-delivery routing, crew scheduling, factory production planning, power-grid balancing—appear throughout industries and must be solved repeatedly at scale. We apply our method to these algorithm engineering problems, where a new state-of-the-art would be writing a higher-scoring algorithm than existing ones written by human experts.

AtCoder Heuristic Contest (AHC) is a series of programming competitions focused on optimization problems drawn from real-world industrial challenges *[3]*, attracting hundreds of participants including industry experts. We attempted to evaluate on two past contests, ahc039 and ahc058. ahc039 ("Purse Seine Fishing") is a computational geometry problem where you design a simple closed net on a 2D map, restricted to horizontal/vertical edges, to capture many target points while avoiding penalty points under a budget. ahc058 ("Apple Incremental Game") is a production planning problem where upgrades trade off immediate output versus growing future production capacity, and the goal is to schedule upgrades to maximize final output.

We select ahc039 because ShinkaEvolve *[37]* reported a solution that would have placed 2nd, and ahc058 because Sakana AI’s ALE-Agent achieved the first-ever AI victory in an AHC *[57]*. We use the evaluation harness from ALE-Bench *[27]*. We use the public test case generator to create local tests, select our best-performing algorithm, and submit it to be scored on the official platform.

Environment: The state $s$ is an algorithm implementation in C++. The action $a$ consists of thinking tokens followed by C++ code. The dynamics parse the code from the action: $s^{\prime}=\texttt{Parse}(a)$. The reward is the score on locally generated test cases, or zero if the algorithm fails correctness checks or exceeds the time limit of 2 seconds and memory limit of 1024MB. We select the best-performing algorithm and submit it to be scored on the official private tests. We use the evaluation harness released by *[27]*. For initial states, for the ahc039 competition we use the same initial program as *[37]*, which is based on ALE-Agent *[27]* best program, that would have placed 5th in the competition leaderboard. For ahc058 we start from scratch, similar to ALE-Agent *[57]*.

Previous state-of-the-art. We report the top human submissions on each contest leaderboard. For AI baselines, we compare to ALE-Agent *[27]* and ShinkaEvolve *[37]*, which use ensembles of models including the gpt, Gemini, and Claude families of models. ALE-Agent *[27]* starts from scratch for both problems. ShinkaEvolve *[37]* reports results in ahc039 where they start from ALE-Agent solution, and improve it from 5th place to 2nd place.

Results. We report results in Table 6. For both competitions, if we had submitted during competition time, our algorithms would have gotten the 1st place. For ahc039, we marginally improve upon the best human, while there is a significant gap between next best AI and human scores. For ahc039, we follow ShinkaEvolve by starting from the ALE-Agent solution and improve it from 5th place to 1st place, while ShinkaEvolve reaches the 2nd place using significantly more capable frontier models such as Gemini 2.5 Pro. For ahc058, we start from scratch and outscore all submissions in the competition.

For AHC039, the solution builds a large pool of promising axis-aligned rectangles using prefix sum scoring, then greedily seeds a connected union and uses simulated annealing with add, remove, replace, expand, shrink, and slide moves to optimize the rectangle union score under perimeter and vertex constraints, followed by cleanup and final greedy refinement.

For AHC058, the solution first builds several reasonable plans using greedy rules, different biases, and a short beam search to explore promising early decisions. Then, the program improves the best plan with simulated annealing that makes random edits, swaps, and partial rebuilds before finishing with a small local cleanup pass. It estimates the value of actions using a simple formula for how much future production an upgrade is likely to create, which guides both greedy choices and pruning. For performance, it caches intermediate states so it only recomputes parts of the plan that change. Overall, the program balances broad exploration early with focused local improvement later.

|  Method | Model | Geometry (ahc039) | Scheduling (ahc058)  |
| --- | --- | --- | --- |
|  1st human | - | 566,997 | 847,674,723  |
|  2nd human | - | 557,212 | 846,938,871  |
|  3rd human | - | 554,334 | 846,350,877  |
|  4th human | - | 552,933 | 845,489,747  |
|  5th human | - | 549,746 | 845,324,831  |
|  ALE-Agent [27] | Ensemble (see caption) | 550,647 | 848,373,282  |
|  ShinkaEvolve [37] | Ensemble (see caption) | 558,026 | n/a  |
|  Best-of-25600 | gpt-oss-120b | 554,171 | 772,429,752  |
|  TTT-Discover | gpt-oss-120b | 567,062 | 848,414,228  |

Table 6. Results in two AtCoder Heuristic Competitions. We train our models with local public tests, and submit the best program we get during training to the official submission platform. Our algorithms are released and can be validated in our codebase. Our solutions in the official AtCoder submission platform are publicly available for ahc039 and ahc058. ALE-Agent uses Gemini-2.5 Pro for ahc039, and Gemini-3 Pro Preview high and gpt-5.2-high for ahc058. ShinkaEvolve uses an ensemble of gpt-5, gpt-5-mini, Gemini-2.5 Pro and Flash, Claude Sonnet 4, o4-mini.

# 4.4 Single Cell Analysis

Single-cell RNA-sequencing (RNA-seq) aims to help us understand how organisms work and get sick by resolving biology at the level of individual cells; measuring which genes each cell is using to reveal cell types, states, and how they change. Practically, it isolates single cells, tags their mRNA with a Unique Molecular Identifier (UMI), sequences it, and outputs a per-cell gene-by-count table. RNA-seq protocols suffer from measurement noise in the observed UMI counts. Thus, denoising algorithms significantly increases the realized value of expensive experiments. Each sequencing run costs thousands of dollars, and better denoising methods reduce the need for deeper sequencing.

We apply our method to one of the recent benchmarks OpenProblems [44], an important set of open problems for single-cell analysis. We use the denoising task therein. [6] demonstrated that

partitioning the observed molecules of a single dataset into training and test sets via binomial sampling and evaluating the denoised training set against the held-out test counts provides a proxy for accuracy against true expression values, providing an evaluation framework without requiring external ground truth data.

Environment. The state  $s$  is an algorithm implementation. The action  $a$  consists of thinking tokens followed by code. The dynamics parse the code from the action:  $s' = \text{Parse}(a)$ . The benchmark evaluates denoising quality using two complementary metrics: mean squared error (MSE) in log-normalized space, which measures overall reconstruction accuracy, and Poisson negative log-likelihood, which assesses how well the denoised counts match the statistical properties expected of count data. In our context, the reward is the MSE score, or zero if it violates constraints we add for the Poisson score or the algorithm exceeds the time limit of 400 seconds. The Denoising benchmark offers 3 datasets: PBMC, Pancreas, and Tabula Muris Senis Lung, in order of size. We train our policy by using Pancreas in our environment, and ultimately performance is reported by running the algorithm on the held out PBMC and Tabula datasets.

Previous state-of-the-art. We report the state of the art as described by the OpenProblems [44] benchmark. The best result was provided by MAGIC [75] using an approximate solver and reversed normalization. MAGIC is a well known technique, frequently used in the literature [83, 76], the only method different from MAGIC that provides good performance is ALRA [40], ranked third. We also compare with OpenEvolve and Best-of-25600.

# Disclaimer

This is an experimental application demonstrating TTT-Discover's ability to find algorithms that excel on specific benchmarks. While our discovered algorithm outperforms existing methods on the OpenProblems denoising benchmark, benchmark metrics are inherently incomplete and do not guarantee biological validity for downstream tasks.

Results. The improved function obtained via TTT-Discover shows consistent improvements on both datasets (see Table 7). TTT-Discover is initialized with MAGIC code. TTT-Discover adds gene-adaptive transform ensembling, low-rank SVD refinement, and log-space polishing steps that directly optimize the benchmark metric.

|  Method | Model | PBMC |   |   | Tabula  |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  Score (↑) | MSE (↓) | Poisson (↓) | Score (↑) | MSE (↓) | Poisson (↓)  |
|  MAGIC (A, R) | - | 0.64 | 0.19 | 0.05 | 0.64 | 0.18 | 0.03  |
|  MAGIC (R) | - | 0.64 | 0.19 | 0.05 | 0.64 | 0.18 | 0.03  |
|  ALRA (S, RN) | - | 0.50 | 0.26 | 0.05 | 0.47 | 0.27 | 0.03  |
|  MAGIC (A) | - | 0.42 | 0.19 | 0.16 | 0.40 | 0.18 | 0.12  |
|  MAGIC | - | 0.42 | 0.19 | 0.16 | 0.40 | 0.18 | 0.12  |
|  OpenEvolve | gpt-oss-120b | 0.70 | 0.16 | 0.05 | 0.71 | 0.15 | 0.03  |
|  Best-of-25600 | gpt-oss-120b | 0.62 | 0.20 | 0.05 | 0.65 | 0.18 | 0.03  |
|  TTT-Discover | gpt-oss-120b | 0.71 | 0.15 | 0.05 | 0.73 | 0.14 | 0.03  |

Table 7. Denoising task for single cell data analysis. We report the score (mean of normalized MSE and Poisson scores), MSE, and Poisson metrics for each dataset. Our state-of-the-art algorithm is released and can be validated in our codebase. MAGIC (A, R) = MAGIC [75] approximate with reversed normalization; MAGIC (R) = MAGIC with reversed normalization; ALRA [40] (S, R) = ALRA sqrt norm with reversed normalization; MAGIC (A) = MAGIC approximate.

# 4.4.1 Expert Review

Below, we provide a verbatim review from Prof. Eric Sun.

# Human Expert Review - Prof. Eric Sun (MIT)

Single-cell transcriptomics provides a high-dimensional readout on cellular gene expression patterns and has enabled new insights into both biological and disease processes. One challenge in the analysis of single-cell transcriptomics is the sparsity of the data, characterized by zero counts detected for many genes (i.e. "dropouts") due to low expression or other technical issues. MAGIC addresses this challenge by de-noising single-cell transcriptomics using diffusion or smoothing, and it has been widely incorporated in the pre-processing of single-cell data for studying multiple diseases and tissue biology. The proposed improvement on the MAGIC algorithm is simple, aligns with the underlying smoothing-based approach of MAGIC, and yields empirical improvements on key metrics. However, improvements on metrics for single-cell data analysis tasks may not always transfer to enhanced ability to obtain new biological insights, which is often difficult to quantify and therefore benchmark. Further evaluation of the proposed algorithm against MAGIC and other existing methods for biologically relevant tasks would be necessary to fully understand the extent of the reported improvements.

# 4.5 Ablations

We have three sets of ablations. First, we ablate the design choices for the train method, while keeping our reuse method, PUCT, fixed. We test (i) TTT with entropic objective using constant  $\beta = 2$  ([29]), (ii) TTT with no entropic objective (expected reward), (iii) No TTT (only reuse). Second, we ablate the choice of the Reuse method, while keeping our train method, TTT with entropic objective using adaptive  $\beta$ , fixed. We replace PUCT with (i)  $\epsilon$ -greedy reuse with  $\epsilon = 0.1$  as this is perhaps the most naive reuse method, and (ii) no reuse. Finally, we report the naive RL baseline, where we use the expected reward objective with no reuse, and the Best-of-25600 baseline.

|   | train | reuse | Best runtime (↓,μs)  |
| --- | --- | --- | --- |
|  Best Human Kernel | - | - | 1371.1  |
|  TTT-Discover | TTT with adaptive entropic | PUCT | 1203.10  |
|  Ablations for train | TTT with constant β entropic | PUCT | 1483.83  |
|   |  TTT with expected reward (no entropic) | PUCT | 1985.67  |
|   |  No TTT | PUCT | 2060.70  |
|  Ablations for reuse | TTT with adaptive entropic | ε-greedy | 1328.89  |
|   |  TTT with adaptive entropic | no reuse | 5274.03  |
|  Naive Test-time RL | TTT with expected reward | no reuse | 5328.73  |
|  Best-of-N | no TTT | no reuse | 5352.36  |

Table 8. Ablation results for the TriMul GPUMode competition where we time the kernels with an H100 GPU. We report the best kernel we get in each run. We report the reward distributions across steps in Figure 4.

For each ablation, we report the runtime of the best kernel found in Table 8, and the reward distribution in Figure 4. The rewards distributions and best kernel runtimes are computed with our evaluator, not the leaderboard.

Only the full TTT-Discover algorithm achieves the best performance in the TriMul competition.

![img-5.jpeg](img-5.jpeg)
Figure 4. Reward distributions for each ablation. We match the sampling budget across all ablations. We sample 512 rollouts in each step. For example, for Best-of-  $N$ , we have  $N = 50 \times 512 = 256000$  rollouts.

![img-6.jpeg](img-6.jpeg)

![img-7.jpeg](img-7.jpeg)

![img-8.jpeg](img-8.jpeg)

When using a constant  $\beta$ , the improvements diminish later in the training. Using the expected reward objective, improvements are slower overall. Without any test-time training, both the mean reward and the max reward stagnates.  $\epsilon$ -greedy reuse works reasonably well, especially with an early lucky kernel. In early experiments with other applications, the lack of exploration was also a bigger problem than it is in kernel engineering tasks. Naive RL and no reuse make minimal improvements.

It is entirely possible that additional tuning (e.g., a task-specific  $\beta$  schedule) or hyperparameter interactions (e.g., batch size and reuse) can provide improvements in the ablation configurations. For each component, many additional knobs could be ablated (e.g., PUCT exploration bonus, learning rate, batch size). However, our focus was on identifying design choices that works reliably across diverse applications within our budget with minimal task-specific tuning. In practice, the key hyperparameters such as learning rate, batch size, and LoRA rank were fixed after the initial iterations of the project.
