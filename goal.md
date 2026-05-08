We are building a system that can discover new state-of-the-art solutions to scientific problems.

High level description of the system:
1. a long lived cli coding agent session such as Claude Code, Codex CLI, Gemini CLI, etc. as the control plane for the entire system.
2. the cli coding agent will be given a scientific problem as input query `Q`.
3. the first task of this cli coding agent will be to gather all the necessary and relevant context `C` from the local directory and the internet using tools available to it.
4. the necessary and relevant context `C` can include:
    - code in the local directory
    - past attempts and their results stored in the local directory
    - research papers, blog posts, etc. from the internet
5. the cli coding agent will then generate a neatly formatted single markdown file `F` that contains all the context `C` and the query `Q` in a way that is easy for an LLM to understand.
6. this file `F` will be given to a separate LLM `M` that is specialized in solving scientific problems.
7. the LLM `M` will be hosted using vllm either locally or on a remote server and will be made available through vllm's chat completions API.
8. the following steps will be run in a loop:
    - the LLM `M` will generate `G` separate implementation plans for proposed solutions to the same input in parallel.
    - the proposed implementation plans will be provided back to the coding agent to implement the plans.
    - the implemented solutions will be evaluated by a separate subroutine `E`. different problems will have different evaluation subroutines.
    - the evaluation subroutine `E` will generate a reward `R_i` for each implemented proposed solution `G_i`.
    - each proposed solution `G_i` and the associated reward `R_i` will be stored in a local csv file buffer `B` that can be referenced by the coding agent as part of the context `C`.
    - the reward `R_i` will be used to update the policy of the LLM `M` using a reinforcement learning algorithm described in the paper "Learning to Discover at Test Time" that is provided as a reference in the `important_papers/ttt-discover` directory. the algorithm is described in the `important_papers/ttt-discover/03_learning_to_discover.md` file as `Algorithm 1`.
9. the loop will continue for a fixed number of iterations as described in the paper.
10. the final proposed solution with the highest reward will be returned to the user.


we strictly want this system to be part of the coding agent where the coding agent acts as the orchestrator. we don't want to biuld a separate orchestration system where the coding agent is a subroutine of that system. this can be achieved by making the solver `M`, the evaluator `E`, and the trainer as separate python subroutines that are called by the coding agent acting as the orchestrator for the entire process including gathering context and implementing the parallel proposed solutions.

important points to note:
- the context gathering and building step will be handled by the coding agent (Claude Code, Codex CLI, Gemini CLI, etc.). we don't want to build a separate context gathering and building subroutine.
- we want to use the training algorithm and setup using Tinker API as described in the paper and the paper's official code implementation in `discover` directory.
- we want to implement this system as a general purpose system that can be used to solve any scientific problem. however, we plan to test if this sytem works well against the erdos math problem solving setup described in the paper. so plann the implementation to be flexible enough to support this.