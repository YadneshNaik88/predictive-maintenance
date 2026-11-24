import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        for tc in d.get('tool_calls', []):
            if tc['name'] == 'write_file' and 'agents.py' in tc['args'].get('Path', ''):
                content = tc['args']['Content']
                if 'class CoordinatorAgent' in content:
                    print(f"Step {step}: Found write_file for agents.py! Length: {len(content)}")
                    with open(f"scratch/agents_from_step_{step}.py", "w", encoding="utf-8") as out:
                        out.write(content)
