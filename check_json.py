import json
with open("eval_pipeline_ensemble.json") as f:
    data = json.load(f)
print("Top-level keys:", list(data.keys()))
if "results" in data:
    print("N results:", len(data["results"]))
    print("Sample result keys:", list(data["results"][0].keys()))
    print("Sample:", json.dumps(data["results"][0], indent=2)[:500])
