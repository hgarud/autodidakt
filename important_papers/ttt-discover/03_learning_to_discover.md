## 3 Learning to Discover at Test Time

So far, the policy’s experience with the test problem can only improve the next prompt ($d,s_{i},c_{i}$), but not the policy $\pi_{\theta}$ itself, since $\theta$ remains frozen. We use this experience to improve the policy in an online fashion, by training $\pi_{\theta}$ on its own search attempts accumulated in the buffer $\mathcal{H}_{i}$.

Algorithm 1 outlines the general form of our method, where the two key subroutines to instantiate are $\texttt{reuse}$ and $\texttt{train}$.

### 3.1 Naive RL at Test Time

Algorithm 1 falls under the formulation of reinforcement learning (RL). A natural baseline is to use a standard RL algorithm:

$\texttt{train}:\quad\theta_{i+1}=\theta_{i}+\eta\nabla_{\theta}\mathbb{E}_{a\sim\pi_{\theta_{i}}(\cdot|s)}\big{[}R(s,a)\big{]},\qquad\texttt{reuse}(\mathcal{H}_{i})=\delta_{\texttt{<}\texttt{empty}>\text{,}}$

i.e., optimize for expected reward with no reuse, where $\delta_{\texttt{<}\texttt{empty}>\text{ is a delta distribution with mass only on the initial state <}\texttt{empty}>\text{.}$ We will use $\theta_{i}$ to denote the model weights for rollout $i$. We can straightforwardly apply popular RL algorithms, such as PPO or GRPO *[59, 16]*, only in the environment defined by the single problem.

Algorithm 1 Test-Time Training to Discover (TTT-Discover)
1: Input: problem description  $d$  and policy  $\pi_{\theta_0}$  with initial weights  $\theta_0$ .
2:  $R, T = \text{get\_env}(d)$ $\triangleright d$  induces the reward and transition functions of the environment (§2.1)
3:  $\mathcal{H}_0 = \{(\langle \text{empty} \rangle, R(\langle \text{empty} \rangle), \{\}\}$ $\triangleright$  Initialize buffer with the empty solution (§2.2)
4: for  $i = 0, 1, \ldots, N - 1$  do
5:  $s_i, c_i \sim \text{reuse}(\mathcal{H}_i)$ $\triangleright$  Sample initial state and context with a reuse heuristic
6:  $a_i \sim \pi_{\theta_i}(\cdot | d, s_i, c_i)$ $\triangleright$  Sample action from policy
7:  $s_i' = T(a_i)$ $\triangleright$  Transition to next state
8:  $r_i = R(s_i')$ $\triangleright$  Evaluate reward of next state
9:  $\mathcal{H}_{i+1} = \mathcal{H}_i \cup \{(s_i, a_i, s_i', r_i)\}$ $\triangleright$  Add current attempt to buffer
10:  $\theta_{i+1} = \text{train}(\theta_i, (d, s_i, c_i, a_i, r_i))$ $\triangleright$  Improve the model weights with train
11: end for
12: return  $s_i^*$ , where  $i^* = \arg \max_{i=0,1,\ldots,N-1} r_i$ $\triangleright$  Return the state with the highest reward

However, these algorithms are designed with the standard RL problem in mind. Discovery problems have important distinctions from standard RL problems.

In standard RL problems, the goal is to find a policy that maximizes the expected reward. This policy is to be deployed repeatedly in the same environment. The primary artifact is the policy.

In discovery problems, the goal is to find a single state that improves upon the state-of-the-art. We do not care about the average performance. There is no separate deployment phase and thus the policy need not maintain robust performance in many states it may encounter starting from the same initial state distribution. In fact, a policy can have very low expected reward, so long as it reaches a new state-of-the-art once.

Due to these differences, the naive RL instantiation has important shortcomings.

Objective function. Naive RL optimizes average performance, and is indifferent to the state of the art. In discovery, however, success is determined by the maximum, and whether it improves upon the state of the art. Consider a kernel engineering problem where the state-of-the-art runtime is  $2000\mu s$ . Achieving  $1900\mu s$  would require substantial optimization and perhaps a breakthrough. Yet, without complicated reward shaping, both would receive nearly the same reward.

Short effective horizon. Starting each attempt from scratch limits how far the policy can reach in an attempt. Reusing a previous solution effectively adds extra timesteps to an attempt, extending the horizon. As a result, more complex solutions can emerge during training. In standard RL, a fixed initial state distribution makes sense as the policy must perform robustly from states it will encounter at deployment. Discovery has no such deployment phase.

Exploration. Exploration requires care at two levels. Optimizing for expected reward, the policy can collapse to safe, high-reward actions rather than risky ones that might achieve discovery. At the reuse level, naive prioritization can over-exploit a few promising states at the expense of diversity.

# 3.2 TTT-Discover

To address these shortcomings, we introduce two simple components.

Entropic objective. We define the entropic objective that favors the maximum reward actions:

$$
J _ {\beta} (\theta) = \mathbb {E} _ {s \sim \text {r e u s e} (\mathcal {H})} \left[ \log \mathbb {E} _ {a \sim \pi_ {\theta} (\cdot | s)} \left[ e ^ {\beta (s) R (s, a)} \right] \right],
$$

$\nabla_{\theta}J_{\beta}(\theta)=\mathbb{E}_{s\sim\textsf{reuse}(\mathcal{H})\atop a\sim\pi_{\theta}(\cdot|s)}\bigg{[}w_{\beta(s)}(a)\nabla_{\theta}\log\pi_{\theta}(a\mid s)\bigg{]},\qquad w_{\beta(s)}(a)=\frac{e^{\beta(s)R(s,a)}}{\mathbb{E}_{\pi_{\theta}(\cdot|s)}[e^{\beta(s)R(s,a)}]},$

where we also shape advantages with a KL penalty: $A(a;s)=w_{\beta(s)}(a)-1-\lambda\log\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{0}}(a|s)}$ *[59, 87, 72]*, and $-1$ is the baseline since $\mathbb{E}[w_{\beta(s)}]=1$. Concurrent work *[29]* also explored the entropic objective $J_{\beta}$ to maximize the pass@k performance for (training-time) RL with binary reward problems.

As $\beta\to\infty$, the entropic objective tends to the max, which is intuitively what we want. However, too large $\beta$ early in training causes instabilities, while too small later makes advantages vanish as even smaller improvements become harder. Empirically, we found that setting a constant $\beta$ that works well across different tasks is challenging. Therefore, different than *[29]*, we set $\beta(s)$ adaptively per initial state by constraining the KL divergence of the induced policy; see Appendix A.1 for details.

PUCT. We select initial states using a PUCT-inspired rule *[55, 63, 65, 64]*. Each state $s$ is scored by $Q(s)+c\cdot P(s)\cdot\sqrt{1+T}/(1+n(s))$, where $Q(s)$ is the maximum reward among states generated when the initial state was $s$ (or $R(s)$ if $s$ has not yet been selected). $P(s)$ is proportional to $s$’s rank in the buffer sorted by reward, $n(s)$ counts how many times $s$ or its descendants have been expanded, and $T$ is the total number of expansions, and $c$ is the exploration coefficient.

Rather than the mean (as in prior work), we use the maximum reward of children in $Q(s)$: we care about the best outcome starting from a state, not the average. The prior $P(s)$ captures the intuition that high-reward states are more likely to yield high-reward children—e.g., a fast kernel is more likely to seed a faster kernel than a slow one—while the exploration bonus prevents over-exploitation by keeping under-visited states as candidates. See Appendix A.2 for implementation details.

Test-time Training to Discover. With these building blocks, we can introduce our method, TTT-Discover. We combine $J_{\beta(s)}$ as our (test-time) training objective and PUCT as our reuse routine:

$\texttt{train:}\quad\theta_{i+1}=\theta_{i}+\eta\nabla_{\theta}J_{\beta(s_{i})}(\theta_{i}),\qquad\textsf{reuse:}\quad s_{i}\sim\text{PUCT}(\mathcal{H}_{i}).$

### 3.3 Implementation Details

We run TTT-Discover with gpt-oss-120b *[1]* on Tinker *[36]* for 50 training steps. We use LoRA *[23]* with rank 32. At each step, we generate a batch of 512 rollouts, with 8 groups of 64 rollouts each. Each group of rollouts is generated using the same context and initial state selected from the reuse buffer. We use the entropic objective, and apply importance sampling ratio correction to the gradients due to the sampler/learner mismatch in the RL infrastructure *[82]*. We do not take any off-policy steps, i.e., take 1 gradient step on the entire batch.

We set the reasoning effort to high. The context window of gpt-oss-120b is limited to 32,768 tokens on Tinker. Thus, each rollout stops when the context window is exhausted or the LM produces the end of sequence token. In most domains, we limit the total length of the prompt and the thinking tokens to 26000 tokens, so as to leave enough tokens to generate the final response, e.g., to allow generating longer algorithm code. We enforce this by token forcing the model to generate its final response. All hyperparameters reported in Table 9, and are fixed unless otherwise stated. Assuming an average prompt length of 3000 tokens and 16000 sampling tokens on average, a training run with 50 steps and 512 rollouts costs around $500 on Tinker.
