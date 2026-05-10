"""AutoDiscoverConfig: knobs the orchestrator passes to the trainer server."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoDiscoverConfig:
    # --- trainer server (FastAPI) ---
    server_host: str = "127.0.0.1"
    server_port: int = 0           # 0 = OS picks a free port

    # --- model / LoRA ---
    model_name: str = "openai/gpt-oss-120b"
    renderer_name: str = "gpt_oss_high_reasoning"
    lora_rank: int = 32

    # --- training hyperparameters (paper defaults) ---
    group_size: int = 64           # G — plans per group / per /rollout/begin
    groups_per_batch: int = 8      # P — groups per training batch
    num_epochs: int = 50           # batches before the loop self-terminates
    learning_rate: float = 4e-5
    save_every: int = 2
    temperature: float = 1.0
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 26000
    adv_estimator: str = "entropic_adaptive_beta"   # also: "mean_baseline", "entropic"
    adv_estimator_beta: float = 2.0
    loss_fn: str = "importance_sampling"            # also: "ppo"
    num_substeps: int = 1

    # --- telemetry ---
    wandb_project: str | None = "tinker-cookbook"
