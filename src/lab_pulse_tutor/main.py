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
    Segment the PDF and save a formatted report.
    """
    from lab_pulse_tutor.tools.pdf_segmenter_tool import PDFSegmentTool

    segments = PDFSegmentTool()._run("knowledge/lab8.pdf", k=3)

    os.makedirs("output", exist_ok=True)

    lines = ["# Lab 8 - Segmented Report", "", f"**Source:** knowledge/lab8.pdf", f"**Clusters:** {len(segments)}", "", "---", ""]
    for i, seg in enumerate(segments, 1):
        lines.append(f"## Cluster {i}: {seg['title']}")
        lines.append("")
        for p in seg["paragraphs"]:
            lines.append(p)
            lines.append("")
        lines.append("---")
        lines.append("")

    output_path = "output/lab8_report.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")


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