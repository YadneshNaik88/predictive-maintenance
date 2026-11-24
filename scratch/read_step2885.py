import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index', 0)
        if idx in (2885, 2887, 2889, 2891, 2893):
            print(f"Step {idx}: {d.get('type')}")
            # print output content
            content = d.get('content', '')
            print(f"Content length: {len(content)}")
