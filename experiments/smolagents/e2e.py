import json
import os
from pathlib import Path
from time import time

from dotenv import load_dotenv
from llm import TimedOpenAIModel
from memory_profiler import profile
from psutil import Process
from tools import get_tables_representation, list_files

from smolagents import CodeAgent, PythonInterpreterTool


def get_prompt(question: str, dataset_name: str, dataset_path: str):
    return f"""
You are a careful and rigorous data scientist working on a structured data analysis task.

## Objective
Answer the following question using the dataset provided.

Question:
{question}

## Dataset
Name: {dataset_name}
Location: {dataset_path}

The dataset consists of one or more CSV tables stored in the directory above.

## Instructions

You may:
- Inspect available files
- Load tables
- Examine schema and sample rows
- Write and execute Python code

You are allowed to import the following in your code:
- Standard Python libraries
- Pandas
- Numpy
- SciPy

## Example Workflow

1. List available tables.
2. Inspect table schemas and sample rows.
3. Identify relevant tables.
4. Write precise Python code to compute the answer.
5. Double-check your reasoning before finalizing.

Finish by clearly stating the final answer to the question.""".strip()


@profile
def main(dataset_name: str, question_number: int, model_name: str):
    p = Process(os.getpid())
    baseline_rss = p.memory_info().rss / 1024 / 1024
    print(f"[BASELINE] RSS before run: {baseline_rss:.2f} MB")
    print(
        f"Running agent on dataset: {dataset_name}, question number: {question_number}"
    )
    model = TimedOpenAIModel(model_id=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    agent = CodeAgent(
        model=model,
        tools=[PythonInterpreterTool(), list_files, get_tables_representation],
        max_steps=20,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
    )
    model.reset_timer()

    dataset_path = Path(__file__).parent / ".." / ".." / "data_src" / dataset_name / "dataset"
    benchmark_path = (
        Path(__file__).parent
        / ".."
        / ".."
        / "data_src"
        / dataset_name
        / f"{dataset_name}_tabular.json"
    )
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)[question_number - 1]

    prompt = get_prompt(
        benchmark["query"],
        dataset_name,
        str(dataset_path.resolve()),
    )

    output_data = {
        "question_id": benchmark["id"],
        "question": benchmark["query"],
        "ground_truth_answer": benchmark["answer"],
    }

    start = time()
    result = agent.run(prompt)
    end = time()

    output_data["agent_answer"] = result
    output_data["total_input_tokens"] = agent.monitor.total_input_token_count
    output_data["total_output_tokens"] = agent.monitor.total_output_token_count
    output_data["total_wall_time"] = end - start
    output_data["total_llm_inference_time"] = model.total_llm_time
    os.makedirs(f"output_files/{dataset_name}-{model_name}", exist_ok=True)
    print(output_data)
    with open(
        f"output_files/{dataset_name}-{model_name}/result_{dataset_name}_{question_number}.json", "w"
    ) as f:
        json.dump(output_data, f, indent=4, default=str)
    print("=" * 50)


if __name__ == "__main__":
    # model_name = "gpt-4.1-mini-2025-04-14"
    model_name = "o3-2025-04-16"
    dataset_name = "environment"
    question_numbers = []
    load_dotenv(".env")
    if len(question_numbers) == 0:
        question_numbers = list(range(100))
    for question_number in question_numbers:
        main(dataset_name, question_number, model_name)
