# 2 Preliminaries

All methods in this paper, including the baselines, share a common goal: Given a scientific problem at test time, the goal is to discover a new state-of-the-art solution with an LLM policy  $\pi_{\theta}$ , whose weights  $\theta$  have already been trained (at training time). To formalize this goal, we first introduce how each scientific problem defines an environment, i.e., a Markov Decision Process (§2.1), which can then be used for search (§2.2) and learning (§3).

# 2.1 Discovery Problem

Our definition of the environment follows prior work in test-time scaling, such as AlphaEvolve [50]: A scientific problem comes in the form of a text description  $d$ , which we always feed as context to the policy. We define a state  $s$  as a candidate solution, such as a kernel implementation of the PyTorch code in  $d$ . In our applications, the problem description also induces a continuous reward function  $R(s) \in \mathbb{R}$ , such as the inverse runtime of the kernel.

We denote  $s_{\mathrm{sota}}$  as the best-known solution among all existing candidates, and  $r_{\mathrm{sota}} = R(s_{\mathrm{sota}})$  as the best-known reward. And in case there is no existing solution,  $s_{\mathrm{sota}}$  can be the empty string <empty>. For example,  $s_{\mathrm{sota}}$  can be the kernel currently at the top of the leaderboard. These notations allow us to formalize the notion of a discovery:

Definition (Discovery). A discovery is an event where a state  $s$  is found such that  $R(s) &gt; r_{\text{sota}}$ . The larger the difference, the more significant the discovery.

Under this formalism, we define a discovery problem as finding such a state  $s$  with large  $R(s) - r_{\mathrm{sota}}$  within the environment defined by the scientific problem.

To produce a better solution, both search and learning methods use the LLM policy to generate an action  $a \sim \pi_{\theta}(\cdot \mid d,s)$ , where the choice of the initial solution  $s$  (e.g.,  $= s_{\mathrm{sota}}$ ) is an important part of the method's design. Similar to the reward function, the transition function  $(s,a) \to s'$  of the environment is also induced by the problem description. Here, we consider only a single timestep since state reuse, which we will introduce soon, effectively subsumes multiple timesteps.

In all our applications, a valid action contains a piece of code and optionally some thinking tokens. For coding problems (e.g., kernel engineering), the environment produces  $s'$  by simply parsing the code out of  $a$ . For problems in mathematics, the environment also needs to execute the code in  $a$  after it is parsed. Table 1 provides an overview of the environments for all our applications.</empty>

2.2 Search Methods

The simplest search method, known as Best-of-$N$, samples i.i.d. rollouts from $\pi_{\theta}$:

$\textbf{Best-of-}N:\quad s=s_{\text{sota}}\text{ or <}\texttt{empty}>\quad a_{i}\sim\pi_{\theta}(\cdot\mid d,s),$

where the subscript, $i=1,\ldots,N$, denotes the index of the rollout. By using $i$ instead of $t$ for the index, we indicate that the rollouts here are independent. One reasonable choice of the initial state $s$ is $s_{\text{sota}}$, assuming that a previous solution exists. But $s_{\text{sota}}$ might be too strong a prior towards exploitation. For example, conditioning on $s_{\text{sota}}$ might prevent the policy from exploring very different, but more promising directions that would ultimately produce better solutions under a large compute budget. To address this concern, we usually set $s=\texttt{<}\texttt{empty}>\text{, the empty (or trivial) solution.}$

On the other hand, the policy might also explore a promising direction using $s=\texttt{<}\texttt{empty}>\text{, but fail to fully exploit it.}$ One technique to address this opposite concern is *state reuse*, which warm starts the policy with some of the previous solutions. Specifically, it maintains a buffer $\mathcal{H}_{i}$ of the previous solutions, and samples the initial solution $s_{i}$ from $\mathcal{H}_{i}$ using a search heuristic, $\texttt{reuse}$, which favors high-reward solutions but still assigns nontrivial likelihood to low-reward ones:

$\textbf{State reuse:}\quad s_{i}\sim\texttt{reuse}(\mathcal{H}_{i}),\ \ a_{i}\sim\pi_{\theta}(\cdot\mid d,s_{i}),\ \ \mathcal{H}_{i+1}=\mathcal{H}_{i}\cup\{(s_{i}^{\prime},r_{i})\}.$

When we reuse a previous solution $s_{i}^{\prime}$, we have effectively added an extra timestep to its trajectory.

Prior work, such as AlphaEvolve *[50]*, also reuses the actions, which can contain thinking tokens and intermediate results (e.g., code for math problems) that are not part of the states. As a consequence, the $\texttt{reuse}$ heuristic also needs to convert the information from previous actions into natural language context $c_{i}$ that can be ingested by the LLM policy:

$\textbf{State-action reuse:}\quad s_{i},c_{i}\sim\texttt{reuse}(\mathcal{H}_{i}),\ \ a_{i}\sim\pi_{\theta}(\cdot\mid d,s_{i},c_{i}),\ \ \mathcal{H}_{i+1}=\mathcal{H}_{i}\cup\{(s_{i},a_{i},s_{i}^{\prime},r_{i})\}.$

Prior work *[50, 37, 84, 42]* refers to state-action reuse as *evolutionary search*, because the $\texttt{reuse}$ heuristic usually involves sophisticated designs motivated by evolution, including hand-crafted operations for mutation and cross-over, and domain-specific measurements of fitness and diversity.
