"""Optional Weights & Biases logging for tuning scripts."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_ENTITY = "teresa_augustin-harvard-university"
DEFAULT_PROJECT = "evoCAST_spacer"


def init_wandb_run(
    *,
    use_wandb: bool,
    name: str,
    config: dict[str, Any],
    entity: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    job_id: str | None = None,
) -> Any:
    """
    Start a wandb run if use_wandb is True. Returns run object or None.
    Set WANDB_API_KEY in the environment for online logging.
    """
    if not use_wandb:
        return None
    import wandb  # type: ignore[import-untyped]

    ent = entity or os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY)
    proj = project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
    tag_list = list(tags or [])
    if job_id:
        tag_list.append(f"job_{job_id}")
    return wandb.init(
        entity=ent,
        project=proj,
        name=name,
        config=config,
        tags=tag_list or None,
    )


def finish_run(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        try:
            import wandb  # type: ignore[import-untyped]

            wandb.finish()
        except Exception:
            pass
