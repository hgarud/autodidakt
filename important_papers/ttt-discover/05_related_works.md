# 5 Related Works

In this section, we first provide a broad overview of continual learning and test-time training, using some of the exposition in [71]. Then towards the end of §5.2, we discuss the most relevant work on test-time training: MiGrATe [52] and ThetaEvolve [78]. Finally, we discuss two pieces of work with tangential formulations: RL on a single training problem that is not the test problem [79] (§5.3), and RL on the entire test set [88] (§5.4).

# 5.1 Continual Learning

Most of today's AI systems remain static after deployment, even though the world keeps changing. The high-level goal of continual learning is to enable AI systems to keep changing with the world, similar to how humans improve throughout their lives [19, 11].

Conventionally, continual learning as a research field has focused on learning from a distribution that gradually changes over time [43, 74, 17]. For example, one could update a chatbot model every hour using new knowledge from the Internet, while typical use cases of the model may require knowledge from both the past and the present [60, 31, 77]. More formally, at each timestep, we sample new training and test data from the current distribution, update our model using the new training data, and then evaluate it on all the test data up to the current timestep. Under this setting, most algorithms focus on not forgetting the past when learning from the present [58, 39, 33, 15].

5.2 Test-Time Training

The algorithmic framework of test-time training has the same high-level goal as continual learning, but it focuses on two aspects where human learning stands out from the forms of continual learning in the conventional literature.

First, each person has a unique brain that learns within the context of their individual life. This personalized form of continual learning is quite different from, for example, the chatbot model that is fine-tuned hourly using the latest information available worldwide. While such a model does change over time, it is still the same at any given moment for every user and every problem instance.

Second, most human learning happens without a boundary between training and testing. Consider your commute to work this morning. It is both "testing" because you did care about getting to work this very morning, and "training" because you were also gaining experience for future commutes. But in machine learning, the train-test split has always been a fundamental concept.

The concept of test-time training is introduced to realize these two special aspects of human learning. Training typically involves formulating a learning problem (such as empirical risk minimization) and then solving it. Following *[67]*, test-time training is defined as any kind of training that formulates a potentially different learning problem based on each individual test instance.

This concept has a rich history in AI. A well-known example in NLP is dynamic evaluation, pioneered by Mikolov et al. *[47]* and extended by Krause et al. *[35]*. In computer vision, early examples have also emerged in applications such as face detection *[28]*, video segmentation *[49]*, super-resolution *[62]*, and 3D reconstruction *[45]*. Next, we discuss three popular forms of test-time training today, with an emphasis on their connections to each other and to historical examples.

#### 5.2.1 TTT on Nearest Neighbors: Larger Effective Capacity

One simple form of test-time training was called locally weighted regression in the 1970s *[66, 10]*, local learning in the 1990s *[8]*, and KNN-SVM in the 2000s *[86]*: Given a test instance, find its nearest neighbors in the training set, and then train (or fine-tune) the model on these neighbors before making a prediction. This procedure can significantly increase the effective capacity of the model; for example, it allows a linear model to fit a highly nonlinear ground truth *[66]*.

This simple form captures one of the key intuitions of test-time training. In the conventional view of machine learning, a model, once trained, no longer changes at test time. As a consequence, it must prepare to be good at all possible inputs in the future. This task can be very hard, because being good at all possible futures limits the model’s capacity to be good at any particular one. But only one future is actually going to happen. So why not train our model once this future happens?

Recently, *[18]* extended this idea to modern language models and observed a similar benefit of larger effective model capacity after test-time training, and *[25]* further improved these results through better strategies for neighbor selection. In addition, *[26]* showed that test-time training on neighbors from the training set is also effective with RL for reasoning tasks, and *[4]* developed the same idea for visual-motor tasks.

#### 5.2.2 TTT for Novel Instances: Better Generalization

As models become larger today, their competence is often limited not by their capacity, but by the amount of available training data, especially when they need to generalize to novel test instances that are “out-of-distribution”. In this case, it is even harder to prepare for all possible test instances in the future, especially the novel ones, with a static model. But once a specific test instance is given, we can use it to generate relevant data, which we can then use for training *[68]*. In other words, the “neighbors” for TTT do not have to come from the training set; they can also be generated on-the-fly.

Since the test instance is unlabeled, one way to make it useful for training is through self-supervision, which generates new pairs of inputs and labels for an auxiliary task such as masked reconstruction (e.g., BERT *[12]* and MAE*[21]*). While the auxiliary task is different from the main prediction task, improving performance in one can help the other through their shared representations. This form of TTT can significantly improve generalization under distribution shifts *[68, 13]*.

Recently, TTT has been an important part of AlphaProof *[24]*, which achieved IMO silver-medal standard in 2024. Given each test problem, their system first generates a targeted curriculum of easier problems by prompting a language model, and then performs reinforcement learning on the generated data. Another recent work, Akyurek et al. *[2]*, found TTT effective for few-shot reasoning tasks such as ARC-AGI. Their system generates augmentations of the few-shot demonstrations in the test problem then performs supervised learning. In *[38]*, authors perform policy gradients at test time using the policy itself as an evaluator of solutions, similar to using LMs as a judge. Further, they optimize token representations with policy gradients, as opposed to optimizing the policy.

Three closest and concurrent works perform test-time training: MiGrATe *[52]*, ThetaEvolve *[78]*, and EvoTune *[69]*. All three combine per-instance RL updates with various replay/reuse mechanisms, and typically use PPO/GRPO/DPO-style updates for LMs *[59, 16, 53]*. Relative to *[52, 69]*, our contribution is to tailor both the learning objective and the reuse rule to the discovery goal, rather than largely standard RL or evolutionary baselines; also test in more realistic discovery tasks with human expert baselines. Compared to ThetaEvolve, TTT-Discover using the same model and compute budget still produces significant improvements (Table 2), which we attribute to our entropic objective and PUCT-based reuse instead of more complicated and brittle heuristics in evolutionary algorithms.

In an earlier work *[7]*, the authors train a neural policy with policy gradients to directly output solutions to combinatorial problems like TSP, using (negative) tour length as reward, and they study both training across many instances and per-instance learning at test time.

### 5.3 RL on One Example

One Example RL *[79]* is relevant as they also train on a single problem. To be specific, they train on one example from a dataset, such as the MATH training set. They show that a policy trained with on one such problem with RL generalizes to other problems in the same dataset. In contrast, TTT-Discover trains on the test problem itself, where the goal is not to generalize but to solve this specific problem.

### 5.4 RL on the Test Set

TTRL *[88]* trains on an entire test set of problems using majority voting as pseudo-labels for reward estimation. In contrast, TTT-Discover trains on a single test problem with a continuous verifiable reward, where the goal is not to improve average performance across a set of problems but to find one exceptional solution.
