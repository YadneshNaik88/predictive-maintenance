import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index', 0)
        tool_calls = d.get('tool_calls') or []
        for tc in tool_calls:
            if tc.get('name') == 'write_file':
                p = tc.get('args', {}).get('Path', '')
                if 'agents.py' in p or 'simulator.py' in p:
                    print(f"Step {idx}: write_file on {p}")
