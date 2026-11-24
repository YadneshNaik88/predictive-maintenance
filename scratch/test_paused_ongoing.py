import numpy as np

def test_paused_ongoing():
    # Schedule: cap = [0, 1, 0, 1, 1]
    # Job of duration 2 starts at t=1 (index 1)
    # At t=1: cap=1, job starts, works cycle 1 (remaining=1)
    # At t=2: cap=0, 0 techs, job is paused! (remaining=1)
    # At t=3: cap=1, job resumes, works cycle 2 (remaining=0 -> done!)
    # Total busy = 2. Total available = 1 + 0 + 1 = 2. Conservation holds!
    cap = [0, 1, 0, 1, 1]
    job = {"id": 0, "remaining": 2, "state": "queued"}
    
    busy_cycles = 0
    available_cycles = 0
    
    for t in range(1, 4):
        cap_t = cap[t]
        available_cycles += cap_t
        
        # Ongoing jobs
        ongoing = [job] if job["state"] == "in_maint" else []
        free_techs = max(0, cap_t - len(ongoing))
        
        if job["state"] == "queued" and free_techs > 0:
            job["state"] = "in_maint"
            free_techs -= 1
            
        # Jobs that can be worked this cycle: min(in_maint, cap_t)
        in_maint = [job] if job["state"] == "in_maint" else []
        to_work = in_maint[:cap_t]
        
        busy_cycles += len(to_work)
        assert len(to_work) <= cap_t
        
        for j in to_work:
            j["remaining"] -= 1
            if j["remaining"] == 0:
                j["state"] = "completed"
                
    print("Job final state:", job["state"])
    print(f"Busy: {busy_cycles}, Available: {available_cycles}")
    assert job["state"] == "completed"
    assert busy_cycles <= available_cycles
    print("[OK] Paused ongoing test passed!")

if __name__ == "__main__":
    test_paused_ongoing()

