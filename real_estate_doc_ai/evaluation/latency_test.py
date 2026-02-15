import requests
import time
import numpy as np
import json

API_URL = "http://127.0.0.1:8000/query"
EVAL_PATH = "evaluation/test_questions.json"

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    evaluation_set = json.load(f)

latencies = []

print("Running latency test...\n")

for item in evaluation_set:
    question = item["question"]

    start = time.time()
    response = requests.post(API_URL, json={"question": question})
    end = time.time()

    latency = end - start
    latencies.append(latency)

    print(f"{question[:60]}... -> {latency:.4f}s")

avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)

print("\n----- Results -----")
print(f"Total Queries: {len(latencies)}")
print(f"Average Latency: {avg_latency:.4f}s")
print(f"P95 Latency: {p95_latency:.4f}s")
