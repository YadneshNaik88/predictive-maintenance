import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step and 1323 <= step <= 2800:
            for tc in d.get('tool_calls', []):
                if tc['name'] == 'edit_file' and 'agents.py' in tc['args'].get('Path', ''):
                    print(f"=== Edit at step {step} ===")
                    print("TARGET:\n", tc['args'].get('Target', ''))
                    print("REPLACEMENT:\n", tc['args'].get('Replacement', ''))
