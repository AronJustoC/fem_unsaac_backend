import requests
import time
import json
import os
from datetime import datetime

def run_benchmarks():
    url = "http://localhost:8000/api/visualization/modal-results"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "structure_data.json")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        payload = json.load(f)

    results = []
    print(f"{'Mode Index':<12} | {'Time (s)':<10} | {'Status'}")
    print("-" * 35)

    for i in range(10):
        query_params = {"mode_index": i}
        start_time = time.perf_counter()
        try:
            response = requests.post(url, json=payload, params=query_params)
            duration = time.perf_counter() - start_time
            results.append({"mode_index": i, "duration": duration, "status": response.status_code})
            print(f"{i:<12} | {duration:<10.4f} | {response.status_code}")
        except Exception as e:
            duration = time.perf_counter() - start_time
            results.append({"mode_index": i, "duration": duration, "status": f"Error: {str(e)}"})
            print(f"{i:<12} | {duration:<10.4f} | Error")

    notepad_dir = os.path.join(base_dir, ".sisyphus", "notepads", "backend-perf-fix")
    if not os.path.exists(notepad_dir):
        os.makedirs(notepad_dir)
        
    benchmark_file = os.path.join(notepad_dir, "benchmarks.md")

    with open(benchmark_file, "a") as f:
        f.write(f"\n### Benchmark Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("| Mode Index | Duration (s) | Status |\n")
        f.write("|------------|--------------|--------|\n")
        for res in results:
            f.write(f"| {res['mode_index']} | {res['duration']:.4f} | {res['status']} |\n")

if __name__ == "__main__":
    run_benchmarks()
