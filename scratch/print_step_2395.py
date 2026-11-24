import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step in (2393, 2395, 2397):
            print(f"=== Step {step} ===")
            for tc in d.get('tool_calls', []):
                print(tc['args'].get('Path'))
                print("TARGET:\n", tc['args'].get('Target'))
                print("REPLACEMENT:\n", tc['args'].get('Replacement'))
