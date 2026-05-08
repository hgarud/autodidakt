# Learning to Discover at Test Time

Mert Yuksekgonul\*, Daniel Koceja\*, Xinhao Li\*, Federico Bianchi\*

Jed McCaleb $^{3}$ , Xiaolong Wang $^{4}$ , Jan Kautz $^{2}$ , Yejin Choi $^{2}$ , James Zou $^{\dagger 1,5}$ , Carlos Guestrin $^{\dagger 1}$ , Yu Sun $^{*1,2}$

$^{1}$  Stanford University  $^{2}$  NVIDIA  $^{3}$  Astera Institute  $^{4}$  UC San Diego  $^{5}$  Together AI

# Abstract

How can we use AI to discover a new state of the art for a scientific problem? Prior work in test-time scaling, such as AlphaEvolve, performs search by prompting a frozen LLM. We perform reinforcement learning at test time, so the LLM can continue to train, but now with experience specific to the test problem. This form of continual learning is quite special, because its goal is to produce one great solution rather than many good ones on average, and to solve this very problem rather than generalize to other problems. Therefore, our learning objective and search subroutine are designed to prioritize the most promising solutions. We call this method Test-Time Training to Discover (TTT-Discover). Following prior work, we focus on problems with continuous rewards.

We report results for every problem we attempted, across mathematics, GPU kernel engineering, algorithm design, and biology. TTT-Discover sets the new state of the art in almost all of them: (i) Erdős' minimum overlap problem and an autocorrelation inequality; (ii) a GPUMode kernel competition (up to  $2 \times$  faster than prior art); (iii) past AtCoder algorithm competitions; and (iv) denoising problem in single-cell analysis. Our solutions are reviewed by experts or the organizers.

All our results are achieved with an open model, OpenAI gpt-oss-120b, and can be reproduced with our publicly available code, in contrast to previous best results that required closed frontier models. Our test-time training runs are performed using Tinker, an API by Thinking Machines, with a cost of only a few hundred dollars per problem.

|   | Mathematics Erdős' Min. Overlap (↓) | Kernel Eng. (TriMul) A100 (↓) | H100 (↓) | Algorithms (AtCoder) Heuristic Contest 39 (↑) | Biology Denoising (↑)  |
| --- | --- | --- | --- | --- | --- |
|  Best Human | 0.380927 [20] | 4531 μs | 1371 μs | 566,997 [56] | 0.64  |
|  Prev. Best AI | 0.380924 [50] | N/A | N/A | 558,026 [37] | N/A  |
|  TTT-Discover | 0.380876 | 2198 μs | 1161 μs | 567,062 | 0.71  |

![img-0.jpeg](img-0.jpeg)
Figure 1. TTT-Discover continues to train an LLM on a single problem at test time.  $\pi_{\theta_i}$  denotes the policy with the updates weights at test-time training step  $i$ . We plot the reward distribution at step 0, 9, 24, and 49 (final), recorded while test-time training for the GPUMode TriMul competition. We generate 512 solutions at each step. As training progresses, the LLM generates better solutions that ultimately surpass the prior art (best human). For comparison, we plot the reward distribution of best-of-  $N$  with the same total sampling budget.
