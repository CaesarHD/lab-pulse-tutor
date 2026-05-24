from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
@CrewBase
class LabPulseTutor():
    """LabPulseTutor crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def lab_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['lab_analyst'], # type: ignore[index]
            verbose=True
        )

    @task
    def format_segments_task(self) -> Task:
        return Task(
            config=self.tasks_config['format_segments_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LabPulseTutor crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
