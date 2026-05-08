1 Introduction

To solve hard problems, humans often need to try, fail, stumble upon partial successes, and then learn from their experiences. Consider your first really hard programming assignment. You read the textbook and trained yourself on the book exercises, but this assignment just asked for so much beyond the basics in the book. You tried to guess the solution, but these attempts merely produced small signs of life. So you had to take a deep breath and learn from your failed attempts, which made your future attempts more intelligent. Finally, after hours of trying and learning, you understood the new ideas behind the assignment. And indeed, the next attempt worked!

In this example, the assignment was hard because it required new ideas beyond your training data (the text and exercises in the book). Now consider using AI to solve scientific discovery problems. This goal is even harder: By definition, discovery problems require ideas not only beyond the model’s training data but also all existing knowledge of humanity. And out-of-distribution generalization is no easier for AI than for humans *[48, 22, 54, 34]*.

To offset this hardness, prior work has focused on test-time search in the solution space by prompting a frozen LLM to make many attempts, similar to how we tried to guess the solution to the assignment. In particular, evolutionary search methods, such as AlphaEvolve, store past attempts in a buffer and use them to generate new prompts via hand-crafted and domain-specific heuristics *[50, 37, 57, 84]*. While these prompts can help the LLM improve previous solutions, the LLM itself cannot improve, similar to a student who can never internalize the new ideas behind the assignment.

The most direct way for the LLM to improve is through learning. And indeed, while both learning and search scale well with compute *[70]*, learning has often superseded search in the history of AI for hard problems such as Go and protein folding *[65, 30]*. We believe that this observation from history is still relevant today, as we scale compute at test time. So we continue to train the LLM, while it attempts to solve this very test problem. And these attempts, in turn, provide the most valuable training data: Recall that the test problem was hard because it was out-of-distribution. Now we have a data distribution specific to this problem.

At a high level, we simply perform Reinforcement Learning (RL) in an environment defined by the single test problem, so any technique in standard RL could be applied. However, our goal has two critical differences from that of standard RL. First, our policy only needs to solve this single problem rather than generalize to other problems. Second, we only need a single best solution, and the policy is merely a means towards this end. In contrast, the policy is the end in standard RL, whose goal is to maximize the average reward across all attempts. While the first difference is a recurring theme in the field of test-time training *[68]*, the second is unique to discovery problems.

To take advantage of these differences, our learning objective and search subroutine strongly favor the most promising solutions. We call this method Test-Time Training to Discover (TTT-Discover). We focus on problems with continuous rewards, in mathematics (§4.1), GPU kernel engineering (§4.2), algorithm design (§4.3), and biology (§4.4). We report results for every problem we attempted, and TTT-Discover sets the new state of the art in almost all of them, using only an open model.

There are three pieces of concurrent work that share our high-level idea: EvoTune (Surina et al.) *[69]*, MiGrATe (Phan et al.) *[52]*, and recently ThetaEvolve (Wang et al.) *[78]*, which is especially relevant. Compared to ThetaEvolve, TTT-Discover using the same model and compute budget still produces significant improvements (Table 2), due to its special learning objective and search subroutine.

|  Problem | State s | Action a | Transition | Reward R(s)  |
| --- | --- | --- | --- | --- |
|  Erdős Minimum Overlap | Step function certificate | Thinking tokens and code | s' = Python(Parse(a)) | 1/Upper bound  |
|  Autocorr. Inequality (1st) |   |   |   | 1/Upper bound  |
|  Autocorr. Inequality (2nd) |   |   |   | Lower bound  |
|  Kernel Engineering | Kernel code | Thinking tokens and code | s' = Parse(a) | 1/Runtime  |
|  Algorithm Competition | Algorithm code |   |   | Test score  |
|  Single Cell Analysis | Analysis code | and code |   | 1/MSE  |

Table 1. Overview of the science and engineering problems in our paper, and the environments they induce (§2.1). Note that the reward is 0 if  $s$  fails validity checks.
