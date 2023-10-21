from langsmith import Client

import os
from dotenv import load_dotenv

load_dotenv()
client = Client()

project_runs = client.list_runs(project_name="default", 
                                  filter='and(gt(start_time, "2023-07-15T12:34:56Z"), or(neq(error, null), and(eq(feedback_key, "Correctness"), eq(feedback_score, 0.0))))'
)

for run in project_runs:
    print(run)
