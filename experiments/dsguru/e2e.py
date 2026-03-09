import json
import os
from time import time

from baseline_system import BaselineLLMSystem
from psutil import Process

dataset_name = "environment"
question_numbers = []  # Leave blank to process all questions, or specify a list of question indices to process specific questions (e.g., [0, 2, 5])
# model_name = "gpt-4.1-mini-2025-04-14"
model_name = "o3-2025-04-16"

current_dir = os.path.dirname(os.path.abspath("__file__"))
dataset_dir = os.path.join(current_dir, f"../../data_src/{dataset_name}/dataset")
benchmark_path = os.path.join(
    current_dir, f"../../data_src/{dataset_name}/{dataset_name}_tabular.json"
)
with open(benchmark_path, "r") as f:
    benchmark = json.load(f)

questions: list[dict[str, str]] = []
if len(question_numbers) == 0:
    question_numbers = list(range(len(benchmark)))
filtered_benchmark = [benchmark[i] for i in question_numbers]
for row in filtered_benchmark:
    questions.append(
        {
            "id": row["id"],
            "query": row["query"],
            "dataset_directory": dataset_dir,
        }
    )

for idx, question in enumerate(questions):
    print(
        f"Question ID: {question['id']}; Query: {question['query']}; Answer: {filtered_benchmark[idx]['answer']}"
    )

number_sampled_rows = 10
if dataset_name == "biomedical":
    number_sampled_rows = 5


class BaselineLLMSystemGPTo3FewShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model=model_name,
            name="BaselineLLMSystemGPTo3FewShot",
            variance="few_shot",
            verbose=verbose,
            supply_data_snippet=True,
            number_sampled_rows=number_sampled_rows,
            *args,
            **kwargs,
        )


out_dir = os.path.join(current_dir, "test_results")
os.makedirs(out_dir, exist_ok=True)
baseline_llm = BaselineLLMSystemGPTo3FewShot(verbose=True, output_dir=out_dir)
start = time()
p = Process(os.getpid())
baseline_rss = p.memory_info().rss / 1024 / 1024
print(f"[BASELINE] RSS before run_few_shot: {baseline_rss:.2f} MB")
baseline_llm.process_dataset(questions[0]["dataset_directory"])
end = time()
print(f"Dataset processing time: {end - start} seconds")
for question in questions:
    print(f"Processing question: {question['id']}")
    output = baseline_llm.serve_query(question["query"], question["id"])
    output["total_computation_time"] += end - start
    print(f"Output: {output}")
