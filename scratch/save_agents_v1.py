import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
full_text = []
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index', 0)
        if idx in (2886, 2888, 2890, 2892, 2894):
            content = d.get('content', '')
            full_text.append(content)

combined = "\n".join(full_text)
print("Total chars:", len(combined))
with open("scratch/agents_v1.py", "w", encoding="utf-8") as out:
    out.write(combined)
print("Saved scratch/agents_v1.py")
