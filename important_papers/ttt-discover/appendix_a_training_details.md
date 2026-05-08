# A Training details

Our hyperparameters are fixed throughout almost all experiment. For almost all applications we used a KL penalty coefficient of 0.1. For algorithm engineering, we used a KL coefficient of 0.01. We present details on our objective function and the reuse algorithm below.

A.1 Entropic utility objective

We define the entropic utility objective explored also in the concurrent work *[29]*:

$J_{\beta}(\theta;s)\;:=\;\log\mathbb{E}_{\tau\sim\pi_{\theta}(\cdot|s)}\big{[}e^{\beta r(\tau;s)}\big{]}.$

The gradient of this objective yields

$\nabla_{\theta}J_{\beta}(\theta;s)=\mathbb{E}_{\tau\sim\pi_{\theta}(\cdot|s)}\big{[}\nabla_{\theta}\log\pi_{\theta}(\tau\mid s)\;w_{\beta}(\tau\mid s)\big{]},\quad w_{\beta}(\tau\mid s)\;=\;\frac{e^{\beta r(\tau;s)}}{\mathbb{E}_{\pi_{\theta}}[e^{\beta r(\tau;s)}]},\quad A_{\beta}(\tau\mid s)\;=\;w_{\beta}(\tau\mid s)-1,$

since $\mathbb{E}_{\pi_{\theta}}[w_{\beta}(\tau\mid s)]=1$, we get $A_{\beta}$ as the mean baselined advantage. The remaining question is how to set $\beta$. *[29]* recommends value $\beta=2$, yet we found it tricky to set it. Later in the training, improvements become harder, and unless $\beta$ is adjusted carefully advantages can become very small. Early in the training, a large $\beta$ can cause instabilities.

Adaptive $\beta$. Define the auxiliary tilted distribution induced by the entropic weights,

$q_{\beta}(\tau\mid s)\;=\;\frac{\pi_{\theta}(\tau\mid s)\text{exp}(\beta r(\tau;s))}{\mathbb{E}_{\pi_{\theta}}[\text{exp}(\beta r(\tau;s))]},\qquad w_{\beta}(\tau\mid s)\;=\;\frac{q_{\beta}(\tau\mid s)}{\pi_{\theta}(\tau\mid s)}.$

Then $w_{\beta}$ is exactly the density ratio that appears in the entropic policy-gradient update, so $\beta$ controls the effective step size induced by this reweighting. We choose $\beta(s)$ by enforcing a KL budget on the auxiliary distribution,

$\text{KL}\big{(}q_{\beta(s)}(\cdot\mid s)\|\pi_{\theta}(\cdot\mid s)\big{)}\;=\;\gamma,$

analogous to Relative Entropy Policy Search, where the temperature is set by an exponential tilt under a relative-entropy constraint *[51]*. In words, $\beta(s)$ is increased only until the KL budget is exhausted, ensuring the induced reweighting, and hence the update, does not move too far from $\pi_{\theta}(\cdot\mid s)$. We fix $\gamma=\ln 2$ throughout our experiments.

Batch estimator. Given $N$ rollouts from the same $s$ with rewards $\{r_{n}\}_{n=1}^{N}$, the empirical sampling distribution is uniform on the batch, $u(n)=1/N$. The induced reweighting on the batch is

$q_{\beta}(n)=\frac{e^{\beta r_{n}}}{\sum_{m=1}^{N}e^{\beta r_{m}}},$

and we set $\beta(s)$ by solving the weight-concentration constraint

$\text{KL}\left(q_{\beta}\|u\right)=\sum_{n=1}^{N}q_{\beta}(n)\log\big{(}Nq_{\beta}(n)\big{)}=\gamma$

via simple bisection search over $\beta\geq 0$. With $\hat{\beta}(s)$, we compute LOO entropic advantages using $r_{\max}=\max_{n}r_{n}$, and an $\epsilon$ in the denominator for numerical stability:

$\hat{Z}_{-n}=\frac{1}{N-1}\sum_{m\neq n}\text{exp}(\hat{\beta}(s)(r_{m}-r_{\max})),\qquad A_{n}=\frac{\text{exp}(\hat{\beta}(s)(r_{n}-r_{\max}))}{\hat{Z}_{-n}+\varepsilon}-1.$

#### Discussion.

States where improvements are consistently small (e.g. high-value / near-goal states) tend to make the batch weights $q_{\beta}(n)$ less peaky for a given $\beta$, so the constraint typically permits a larger $\beta(s)$. In contrast, states that occasionally yield a few very large improvements (often earlier in training or low-value states with large headroom) make $q_{\beta}$ concentrate quickly as $\beta$ grows; the same KL budget then forces a smaller $\beta(s)$, preventing the update from being dominated by a handful of outlier trajectories while still preferring better-than-average rollouts. Finally, this estimator is invariant to shifting or scaling the reward by a constant, i.e., $r(\tau)$ and $r^{\prime}(\tau)=wr(\tau)+b$ yield the same advantage for $w\in\mathbb{R}^{+}$ and $b\in\mathbb{R}$.

## Appendix

A.2 PUCT Prioritization

We maintain an archive $\mathcal{H}_{t}$ of previously discovered states $s$ with reward $R(s)\in\mathbb{R}$. To choose the next start state, we score each $s\in\mathcal{H}_{t}$ by a PUCT-inspired rule, analogous to applying PUCT at a virtual root whose actions correspond to selecting a start state from the archive *[55, 63, 65, 64]*:

$\mathrm{score}(s)=Q(s)+c\cdot\mathrm{scale}\cdot P(s)\frac{\sqrt{1+T}}{1+n(s)},$

where $n(s)$ is a visitation count, $T$ is the number of expanded parents so far, $c>0$ is an exploration coefficient, and $\mathrm{scale}=R_{\max}-R_{\min}$ is the reward range over the archive. The prior $P(s)$ is a linear rank distribution:

$P(s)=\frac{|\mathcal{H}_{t}|-\mathrm{rank}(s)}{\sum_{s^{\prime}\in\mathcal{H}_{t}}(|\mathcal{H}_{t}|-\mathrm{rank}(s^{\prime}))},$

where $\mathrm{rank}(s)\in\{0,\ldots,|\mathcal{H}_{t}|-1\}$ orders states by descending reward (rank $0$ is the best state). The term $Q(s)$ uses the best one-step reachable reward $m(s)$:

\[ Q(s)=\begin{cases}m(s)&n(s)>0\\
R(s)&n(s)=0\end{cases}. \]

After expanding parent $p$ and observing its best child reward $y=\max_{s^{\prime}\in\mathrm{Child}(p)}R(s^{\prime})$, we update:

$m(p)$ $\leftarrow\max(m(p),y)$ (direct parent only)
$n(a)$ $\leftarrow n(a)+1\quad\forall a\in\{p\}\cup\mathrm{Anc}(p)$ (backprop visitation)
$T$ $\leftarrow T+1.$

For the archive update, we keep the top-2 children per expanded parent (largest $R$) before inserting, then enforce a global size constraint by retaining the top-1000 states in $\mathcal{H}_{t}$ by $R$, while always keeping the initial seed states.

#### Comparison to AlphaZero PUCT.

AlphaZero’s PUCT operates over a tree of state-action edges, selecting actions via $a=\arg\max_{a}[Q(s,a)+c\cdot P(s,a)\cdot\sqrt{\sum_{b}N(s,b)}/(1+N(s,a))]$, where $Q(s,a)$ is the mean value of simulations through edge $(s,a)$, $P(s,a)$ is a learned policy prior, and $N(s,a)$ counts visits to that edge *[65, 64]*. Our formulation differs in four ways: (i) $Q(s)$ tracks the maximum child reward rather than the mean, favoring optimistic expansion; (ii) $P(s)$ is a rank-based prior over archived states rather than a learned action distribution; (iii) visitation counts backpropagate to all ancestors, so expanding any descendant reduces the exploration bonus for the entire lineage; and (iv) we block the full lineage (ancestors and descendants) from the current batch to encourage diversity, whereas AlphaZero uses virtual loss as a temporary penalty.
