import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index', 0)
        if idx >= 2800:
            tool_calls = d.get('tool_calls') or []
            for tc in tool_calls:
                name = tc.get('name')
                if name in ('edit_file', 'write_file'):
                    p = tc.get('args', {}).get('Path', '')
                    if 'agents.py' in p or 'simulator.py' in p:
                        print(f"Step {idx}: {name} on {p}")
                        if name == 'edit_file':
                            target = tc.get('args', {}).get('Target', '')
                            repl = tc.get('args', {}).get('Replacement', '')
                            print("TARGET:\n", target[:150])
                            print("REPL:\n", repl[:150])
                            print("-" * 50)
