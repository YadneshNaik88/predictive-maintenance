import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if 'tool_calls' in d:
            for tc in d['tool_calls']:
                if tc['name'] == 'edit_file':
                    p = tc['args'].get('Path', '')
                    if 'simulator.py' in p or 'agents.py' in p:
                        step = d.get('step_index')
                        print(f"Step {step}: {p}")
                        print("  Target:", repr(tc['args'].get('Target', '')[:100]))
                        print("  Replacement:", repr(tc['args'].get('Replacement', '')[:100]))
