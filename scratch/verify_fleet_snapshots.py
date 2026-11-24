"""
verify_fleet_snapshots.py - Verify Fleet Overview snapshot distribution and consistency.
"""
import pandas as pd
import json

oof_df = pd.read_csv("outputs/oof_predictions_L30.csv")
with open("outputs/decision_log.jsonl", "r") as f:
    decision_logs = [json.loads(line) for line in f]

dec_map = {}
for d in decision_logs:
    eng_str = d.get("engine", "")
    c_val = d.get("cycle")
    try:
        u_id = int(eng_str.replace("E-", ""))
        if c_val is not None:
            dec_map[(u_id, int(c_val))] = d
    except (ValueError, TypeError):
        pass

print("==================================================================================================")
print("FLEET OVERVIEW SNAPSHOT VERIFICATION (app/app.py)")
print("==================================================================================================")
print(f"{'Snapshot':<12} | {'Min Cycle':<10} | {'Max Cycle':<10} | {'Critical':<10} | {'Planned':<10} | {'OK':<10} | {'Total':<8}")
print("--------------------------------------------------------------------------------------------------")

for snap in [20, 30, 40, 50, 60, 80, 100]:
    records = []
    for u in range(1, 101):
        sub = oof_df[oof_df["unit_number"] == u]
        max_cycles = len(sub)
        idx = max(0, min(max_cycles - 1, int(round(max_cycles * (snap / 100.0))) - 1))
        row = sub.iloc[idx]
        pred_rul = float(row["pred_rul"])
        cur_age = int(row["cycle"])
        d_info = dec_map.get((u, cur_age))

        if pred_rul <= 20.0:
            if d_info and d_info.get("escalated", False):
                status = "ESCALATED"
            else:
                status = "CRITICAL"
        elif pred_rul <= 45.0:
            if d_info and d_info.get("escalated", False):
                status = "ESCALATED"
            else:
                status = "PLANNED"
        else:
            status = "OK"

        records.append({"cycle": cur_age, "status": status, "pred_rul": pred_rul})

    df = pd.DataFrame(records)
    n_crit = int(((df["status"] == "CRITICAL") | (df["status"] == "ESCALATED")).sum())
    n_plan = int((df["status"] == "PLANNED").sum())
    n_ok = int((df["status"] == "OK").sum())
    min_c, max_c = int(df["cycle"].min()), int(df["cycle"].max())
    print(f"{snap:>3}% life    | {min_c:<10} | {max_c:<10} | {n_crit:<10} | {n_plan:<10} | {n_ok:<10} | {n_crit+n_plan+n_ok:<8}")
print("==================================================================================================")
