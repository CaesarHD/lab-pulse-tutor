from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from lab_pulse_tutor.tools.pdf_segmenter_tool import PDFSegmentTool

groq_llm = LLM(model="groq/llama-3.1-8b-instant", max_tokens=800)

@CrewBase
class LabPulseTutor():
    """LabPulseTutor crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def question_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['question_generator'], # type: ignore[index]
            verbose=True,
            llm=groq_llm
        )

    @agent
    def student_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config['student_simulator'], # type: ignore[index]
            verbose=True,
            llm=groq_llm
        )

    @agent
    def tutor_corrector(self) -> Agent:
        return Agent(
            config=self.agents_config['tutor_corrector'], # type: ignore[index]
            verbose=True,
            llm=groq_llm
        )

    @task
    def generate_questions_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_questions_task'], # type: ignore[index]
        )

    @task
    def answer_questions_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_questions_task'], # type: ignore[index]
        )

    @task
    def correct_answers_task(self) -> Task:
        return Task(
            config=self.tasks_config['correct_answers_task'], # type: ignore[index]
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
