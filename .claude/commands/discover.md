---
description: Run an advanced scientific discovery workflow on the current 
working directory's problem and user provided high level task goal.
---

# Inputs

```
/discover <high_level_task_goal>
```

- `<high_level_task_goal>`: the high level task goal provided by the user.
    - You should rewrite it for accuracy, completeness, clarity, and unambiguity.
    - You should ask clarifying questions to the user to ensure that the high
    level task goal is clear and well defined.

# Steps

## 1. Clarify, resolve and validate
- Read `$AUTODISCOVER_SERVER_URL`. If unset, stop with: `"AUTODISCOVER_SERVER_URL
not set; export it before running /discover."`
- `curl -fsS "$AUTODISCOVER_SERVER_URL/status"` — if non-200, stop and surface the response.
- Ask clarifying questions to the user to ensure that the high level task goal is clear and well defined.

## 2. Gather necessary and sufficient context for the expert
- Gather all the necessary and relevant context `C` from the local directory and the internet using tools available to you.
   - Code in the local directory
   - Detailed and concise description of the problem
   - Detailed and concise description of the baseline starting point
   - Past attempts and their results stored in the local directory
- The context will be provided to an expert LLM that has no context about the code and the problem at hand.
   - Therefore, the context should only contain the necessary and sufficient information so as to not cause context bloat.
   - The expert will provide specific ideas and implementation plans for you to implement.
   - Do not provide unncessary context to the expert such as information about subagents or per-plan scratch dirs or any other details that are not relevant to solving the problem at hand.

## 2.a. Seed the archive on cold start
- If `./results.csv` already exists, skip this step — it carries the seed (and any prior runs' history).
- Otherwise, write a single seed row to `./results.csv` describing the current baseline as it lives in the working directory. This row becomes the root of the solution tree the expert uses to sample past plans.
- File schema (header row + one data row, comma-separated, RFC 4180 quoting for multi-line `plan_text`):
   - `plan_id,iter_idx,parent_id,reward,plan_text`
   - `seed,-1,,<baseline reward if known else 0.0>,<your baseline description>`
- The `plan_text` should describe the current baseline implementation as if it were a plan the expert had previously written — concrete enough that the next iteration can be framed as "build on this." Treat it like the first entry in the discovery log.

## 3. Provide necessary and sufficient context to the expert
- Post the high level task goal **and the current `./results.csv`** to the `$AUTODISCOVER_SERVER_URL/rollout/begin` API endpoint through an HTTP POST request. The server merges the uploaded archive into its in-memory tree, runs PUCT to pick a parent plan, and returns the canonical archive after sampling G new children.
   - The request body should contain:
      - `iter_idx`: the iteration index.
      - `context`: all the necessary and sufficient context for the expert.
      - `archive_csv`: the contents of `./results.csv` (must include the seed row from step 2.a on the very first call).
   ```
   curl -sfS -X POST "$AUTODISCOVER_SERVER_URL/rollout/begin" \
        -H 'content-type: application/json' \
        -d "$(jq -n --argjson iter_idx <iter_idx> --arg context "$(cat <context_file>)" --arg archive_csv "$(cat ./results.csv)" '{iter_idx: $iter_idx, context: $context, archive_csv: $archive_csv}')" | jq .
   ```
- Parse the response.
   - The response will contain:
      - `plans`: a list of dictionaries, each containing:
         - `plan_id`: a unique identifier for the plan.
         - `plan_text`: the plan text that will contain precise instructions for you to implement.
         - `iter_idx`: the iteration index.
      - `parent_plan_id`: the plan_id of the PUCT-selected parent the new plans are conditioned on (`"seed"` on the very first call).
      - `archive_csv`: the canonical archive after the merge + new-row insertion.
   - **Overwrite `./results.csv` with the returned `archive_csv` immediately** — this keeps the orchestrator's local mirror in lockstep with the server and lets the next iteration (or a fresh server) replay the tree.
- The expert will provide multiple plans for you to implement in parallel.
- Do not save the plans to any local directory. Directly pass them out to subagents.

## 4. Implement the plans in parallel
- Pick a `RUN_ID` for this iteration (e.g., `iter<iter_idx>-<short_random>`).
- **Pre-clone the workspace, one scratch dir per plan.** In a single Bash call, create an APFS clone of cwd at `/tmp/discover/$RUN_ID/<plan_id>` for every plan returned by `/rollout/begin`:
   ```
   mkdir -p "/tmp/discover/$RUN_ID" && \
   for plan_id in <plan_ids>; do
     cp -cR "$PWD" "/tmp/discover/$RUN_ID/$plan_id" &
   done; wait
   ```
- Fan out subagents to implement the plans in parallel.
   - In a SINGLE message, emit one subagent tool call per plan. **Do not pass `isolation: "worktree"`** — filesystem isolation is now provided by the per-plan clone created above.
   - Each subagent's prompt must be self-contained and contain:
      - The plan text.
      - The assigned `WORKDIR=/tmp/discover/$RUN_ID/<plan_id>`. The subagent must `cd "$WORKDIR"` before any edits or commands and keep **all** edits, intermediate files, and evaluator runs inside `$WORKDIR`. It must not touch the original repo path or anything outside `$WORKDIR`.
      - Instructions to implement the plan with minimal and focused code changes.
      - Instructions to investigate and execute the code to collect and return the results to you as a JSON object.

## 5. Collect and post the results to the expert
- Collect the results from the subagents and post them to the `$AUTODISCOVER_SERVER_URL/rollout/reward` API endpoint through an HTTP POST request.
   - The request body should contains:
      - `plan_id`: the plan id.
      - `results`: the results from the subagents.
   ```
   curl -sfS -X POST "$AUTODISCOVER_SERVER_URL/rollout/reward" \
        -H 'content-type: application/json' \
        -d '{"plan_id": <plan_id>, "results": <results>}' | jq .
   ```
- Do **not** write `./results.csv` here — the canonical archive is updated server-side and pulled back into `./results.csv` at the start of the next iteration's `/rollout/begin` (step 3).
- **Tear down the per-plan scratch dirs** for this iteration:
   ```
   rm -rf "/tmp/discover/$RUN_ID"
   ```
   One `rm` deletes all G clones.
