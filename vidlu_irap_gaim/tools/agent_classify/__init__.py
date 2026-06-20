"""Agent-based iRAP attribute classification (prepare task list/prompt, evaluate predictions).

CLI: ``python -m vidlu_irap_gaim.tools.agent_classify {prepare,evaluate} ...`` (see __main__).
Library entry points are re-exported here.
"""

from .prepare import prepare_agent_tasks
from .evaluate import evaluate_agent_predictions, AgentEvaluationResult
from ._common import agent_artifact_paths, load_split_and_attrs

__all__ = [
    "prepare_agent_tasks",
    "evaluate_agent_predictions",
    "AgentEvaluationResult",
    "agent_artifact_paths",
    "load_split_and_attrs",
]
