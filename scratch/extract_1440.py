import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step == 1440:
            tc = d['tool_calls'][0]
            content = tc['args']['Content']
            with open(r'c:\Users\naiky\OneDrive\Desktop\Projects\predictive-maintenance\scratch\agents_step1440.py', 'w', encoding='utf-8') as out:
                out.write(content)
            print("Wrote scratch/agents_step1440.py, length:", len(content))
