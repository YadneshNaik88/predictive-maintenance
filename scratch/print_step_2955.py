import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step == 2955:
            tc = d['tool_calls'][0]
            print("STEP 2955 TARGET:\n", tc['args'].get('Target', ''))
            print("STEP 2955 REPLACEMENT:\n", tc['args'].get('Replacement', ''))
