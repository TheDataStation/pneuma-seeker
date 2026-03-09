import time
from smolagents import OpenAIModel


class TimedOpenAIModel(OpenAIModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_llm_time = 0.0

    def generate(self, *args, **kwargs):
        start = time.time()
        response = super().generate(*args, **kwargs)
        end = time.time()

        self.total_llm_time += end - start
        return response

    def reset_timer(self):
        self.total_llm_time = 0.0
