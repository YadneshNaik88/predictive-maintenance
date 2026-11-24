import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'is_agentic' in line:
            d = json.loads(line)
            step = d.get('step_index')
            print(f"is_agentic in step {step}")
            if 'tool_calls' in d:
                for tc in d['tool_calls']:
                    print(tc['name'], tc['args'].get('Path'))
