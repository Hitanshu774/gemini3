#!/usr/bin/env python
import sys
import warnings
import gradio as gr
from datetime import datetime

from crew import TacticalPlanner
from one import run_scouting_report
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run_full_pipeline(team_name):
    """
    Generate scouting report + run TacticalPlanner crew.
    """
    team_name = team_name.strip()
    print(f"🔍 Processing: {team_name}")
    
    scout_report = run_scouting_report(team_name)
    
    inputs = {
        'topic': 'Valorant',
        'current_year': str(datetime.now().year),
        'scout_report': scout_report
    }
    
    try:
        result = TacticalPlanner().crew().kickoff(inputs=inputs)
        return f"**SCOUTING + TACTICAL ANALYSIS: {team_name.upper()}**\n\n**Scouting Report:**\n{scout_report}\n\n**TacticalPlanner Output:**\n{result}"
    except Exception as e:
        return f"Error: {e}"

gr.Interface(
    fn=run_full_pipeline,
    inputs=gr.Textbox(
        label="Team Name",
        placeholder="100 Thieves, NRG, LOUD, Cloud9, MIBR, G2, FURIA, Evil Geniuses, Sentinels, 2GAME eSports, KRÜ Esports, Leviatán Esports"
    ),
    outputs=gr.Markdown(label="Complete Report"),
    title="Valorant Scouting + Tactical Crew",
    submit_btn="Generate Report"
).launch(inbrowser=True)



