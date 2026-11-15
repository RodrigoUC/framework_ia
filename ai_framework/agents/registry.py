"""Specialist definitions, capability discovery and construction."""

from dataclasses import dataclass

from ai_framework.agents.classification_agent import ClassificationAgent
from ai_framework.agents.clustering_agent import ClusteringAgent
from ai_framework.agents.eda_agent import EDAAgent
from ai_framework.agents.regression_agent import RegressionAgent


SPECIALISTS = {
    "eda": (EDAAgent, frozenset({"profile"})),
    "clustering": (ClusteringAgent, frozenset({"cluster"})),
    "classification": (ClassificationAgent, frozenset({"classify"})),
    "regression": (RegressionAgent, frozenset({"regress"})),
}


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    specialty: str
    capabilities: frozenset[str]
    active: bool = True


class AgentRegistry:
    """Keep definitions independent of the dataset and run state."""

    def __init__(self) -> None:
        self._definitions: dict[str, AgentDefinition] = {}

    @classmethod
    def with_defaults(cls) -> "AgentRegistry":
        registry = cls()
        for specialty, (_, capabilities) in SPECIALISTS.items():
            registry.register(AgentDefinition(specialty, specialty, specialty, capabilities))
        return registry

    def register(self, definition: AgentDefinition) -> None:
        specialist = SPECIALISTS.get(definition.specialty)
        if specialist is None or not definition.capabilities or not definition.capabilities <= specialist[1]:
            raise ValueError("Invalid capabilities for this specialty.")
        if not definition.agent_id.strip() or not definition.name.strip():
            raise ValueError("An agent ID and name are required.")
        if definition.agent_id in {*self._definitions, "coordinator"} or any(
            agent.name.strip().casefold() == definition.name.strip().casefold()
            for agent in self._definitions.values()
        ):
            raise ValueError("The agent ID or name is already registered.")
        self._definitions[definition.agent_id] = definition

    def list_agents(self) -> tuple[AgentDefinition, ...]:
        return tuple(self._definitions.values())

    def select(self, capability: str, preferred_id: str | None = None) -> AgentDefinition:
        eligible = [
            agent for agent in self.list_agents()
            if agent.active and capability in agent.capabilities
        ]
        if preferred_id is not None:
            for agent in eligible:
                if agent.agent_id == preferred_id:
                    return agent
            raise ValueError(f"Agent {preferred_id!r} cannot perform {capability!r}.")
        if not eligible:
            raise ValueError(f"No active agent can perform {capability!r}.")
        return next((agent for agent in eligible if agent.agent_id == agent.specialty), eligible[0])


class AgentFactory:
    """Create a fresh specialist from a registered definition for each run."""

    def create(self, definition: AgentDefinition, blackboard, llm=None, **options):
        agent_class, _ = SPECIALISTS[definition.specialty]
        agent = agent_class(blackboard, llm, **options)
        if definition.agent_id != definition.specialty or definition.name != definition.specialty:
            agent.registered_name = definition.name
        agent.name = definition.agent_id
        return agent
