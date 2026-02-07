from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from one import run_scouting_report

tactical_md = TextFileKnowledgeSource(
    file_paths=[
        "tactical_planner/knowledge/game_agenda.md",
        "tactical_planner/knowledge/maps.md",
        "tactical_planner/knowledge/planning.md",
        "tactical_planner/knowledge/sample_matches.md"
    ],
    vector_store=None,  # Pure reference (your requirement)
    chunk_overlap=0     # No splitting
)

scout_report = run_scouting_report("NRG")

@CrewBase
class TacticalPlanner():
    """TacticalPlanner crew"""

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def studying_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['studying_agent'], # type: ignore[index]
            knowledge=[tactical_md]
            verbose=True
        )

    @agent
    def planning_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['planning_agent'], # type: ignore[index]
            verbose=True
        )
    
    @agent
    def counter_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['counter_planner'], # type: ignore[index]
            verbose=True
        )

#########################################################################


    @task
    def study_game(self) -> Task:
        return Task(
            config=self.tasks_config['study_game'], # type: ignore[index]
        )

    @task
    def analyze_vul(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_vul'], # type: ignore[index]
            inputs = {"scout_report": scout_report},
            context= [self.study_game],
            output_file='report.md'
        )
        
    @task
    def plan_attack(self) -> Task:
        return Task(
            config=self.tasks_config['plan_attack'], # type: ignore[index]
            output_file='report.md'
        )

#########################################################################

    @crew
    def crew(self) -> Crew:
        """Creates the TacticalPlanner crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
