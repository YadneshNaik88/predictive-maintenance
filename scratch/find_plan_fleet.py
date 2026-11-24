import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'def plan_fleet' in line:
            d = json.loads(line)
            step = d.get('step_index')
            print(f"plan_fleet in step {step}")
            for tc in d.get('tool_calls', []):
                if 'agents.py' in tc['args'].get('Path', ''):
                    print(tc['args'].get('Target', '')[:200])
                    print("--- REPLACEMENT ---")
                    print(tc['args'].get('Replacement', '')[:200])
