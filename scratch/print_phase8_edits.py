import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step and 2880 <= step <= 3000:
            for tc in d.get('tool_calls', []):
                p = tc['args'].get('Path', '')
                if 'agents.py' in p:
                    print(f"=== Step {step} ===")
                    print("TARGET:\n", tc['args'].get('Target', ''))
                    print("REPLACEMENT:\n", tc['args'].get('Replacement', ''))
                    print("=" * 60)
