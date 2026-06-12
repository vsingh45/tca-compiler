from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScenarioTurn:
    turn_id: int
    topic: str
    query: str
    required_topics: tuple[str, ...]


def default_workload() -> list[ScenarioTurn]:
    """
    Deterministic multi-topic workload for memory experiments.

    Each turn references one or more prior topics so retrieval quality can be scored
    without using a live model.
    """

    return [
        ScenarioTurn(1, "profile", "My preferred name is Vivek and I like concise answers.", ("profile",)),
        ScenarioTurn(2, "billing", "My invoice for March is overdue and I need billing help.", ("billing",)),
        ScenarioTurn(3, "auth", "I cannot reset my password because the link expired.", ("auth",)),
        ScenarioTurn(4, "project", "The research project is about warm memory for agents.", ("project",)),
        ScenarioTurn(5, "billing", "Where should I download that billing invoice again?", ("billing",)),
        ScenarioTurn(6, "auth", "What did I say about the password reset problem?", ("auth",)),
        ScenarioTurn(7, "profile", "Remember that I want concise responses in this project.", ("profile", "project")),
        ScenarioTurn(8, "project", "Summarize the warm memory project idea and my response style.", ("project", "profile")),
        ScenarioTurn(9, "travel", "I am flying to Chicago next Friday for the demo.", ("travel",)),
        ScenarioTurn(10, "travel", "What city did I mention for the demo trip?", ("travel",)),
        ScenarioTurn(11, "billing", "Combine the invoice issue with my trip timing.", ("billing", "travel")),
        ScenarioTurn(12, "project", "Recap the project, billing issue, and password problem.", ("project", "billing", "auth")),
    ]
