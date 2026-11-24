import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step in (2959, 3007):
            print(f"=== Step {step} ===")
            for tc in d.get('tool_calls', []):
                print("File:", tc['args'].get('Path'))
                print("Target:\n", tc['args'].get('Target'))
                print("Replacement:\n", tc['args'].get('Replacement'))
