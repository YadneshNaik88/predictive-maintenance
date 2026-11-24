import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        for tc in d.get('tool_calls', []):
            if tc['name'] in ('edit_file', 'write_file'):
                p = tc['args'].get('Path', '')
                if 'agents.py' in p:
                    print(f"Step {step}: {tc['name']}")
                    if tc['name'] == 'edit_file':
                        print("TARGET:\n", repr(tc['args'].get('Target', '')[:100]))
                        print("REPLACEMENT:\n", repr(tc['args'].get('Replacement', '')[:100]))
