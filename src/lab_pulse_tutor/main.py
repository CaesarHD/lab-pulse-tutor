#!/usr/bin/env python
import os
import sys
import warnings

from datetime import datetime

from lab_pulse_tutor.crew import LabPulseTutor

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the multi-agent system on a lab PDF.
    """
    pdf_path = "knowledge/lab8.pdf"
    
    print(f"Segmenting PDF at {pdf_path}...")
    from lab_pulse_tutor.tools.pdf_segmenter_tool import PDFSegmentTool
    tool = PDFSegmentTool()
    segmented_data = tool._run(pdf_path=pdf_path, k=2)
    
    formatted_content = ""
    for segment in segmented_data[:1]:
        formatted_content += f"\n\n### {segment['title']}...\n"
        for p in segment['paragraphs'][:1]: # Only ONE paragraph to guarantee success
            formatted_content += f"{p}\n\n"
            
    print("PDF segmented successfully. Kicking off the Crew...")
    
    inputs = {
        "segmented_content": formatted_content
    }
    
    print("Starting LabPulseTutor Crew...")
    result = LabPulseTutor().crew().kickoff(inputs=inputs)
    
    os.makedirs("output", exist_ok=True)
    output_path = "output/tutor_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(result))
    print(f"Final report saved to {output_path}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        LabPulseTutor().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        LabPulseTutor().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        LabPulseTutor().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = LabPulseTutor().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")

if __name__ == "__main__":
    run()