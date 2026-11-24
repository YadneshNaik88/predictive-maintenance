import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index', 0)
        if idx in (2955, 3177):
            for tc in d.get('tool_calls', []) or []:
                if tc.get('name') == 'edit_file':
                    print(f"=== STEP {idx} ===")
                    print("TARGET:\n", tc.get('args', {}).get('Target'))
                    print("REPLACEMENT:\n", tc.get('args', {}).get('Replacement'))
