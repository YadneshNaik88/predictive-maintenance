import json

with open("outputs/results_final_v1.json") as f:
    d = json.load(f)

for sc in d["scenarios"]:
    print(f"=== {sc} ===")
    for pol, p in d["scenarios"][sc]["policies"].items():
        print(f"{pol:<28} | cost={p['mean_cost']:>12.2f} | fails={p['mean_failures']:>6.2f} | pm={p['mean_pm_actions']:>6.2f} | down={p['mean_downtime_cycles']:>8.2f}")
