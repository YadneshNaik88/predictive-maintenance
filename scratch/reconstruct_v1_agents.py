import json

transcript_path = r'C:\Users\naiky\.gemini\antigravity\brain\405b46b3-5b5b-4eef-8bc2-a9412d22deac\.system_generated\logs\transcript_full.jsonl'

# Start with Step 1322 for agents.py
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step == 1322:
            tc = d['tool_calls'][0]
            agents_text = tc['args']['Content']
            break

# Apply all edits up to step 2800 (end of Phase 7, start of Phase 8)
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        step = d.get('step_index')
        if step and 1323 <= step <= 2800:
            for tc in d.get('tool_calls', []):
                if tc['name'] == 'edit_file' and 'agents.py' in tc['args'].get('Path', ''):
                    tgt = tc['args']['Target']
                    rep = tc['args']['Replacement']
                    assert tgt in agents_text, f"Target not in agents_text at step {step}"
                    agents_text = agents_text.replace(tgt, rep)

with open(r'c:\Users\naiky\OneDrive\Desktop\Projects\predictive-maintenance\scratch\agents_v1_canonical.py', 'w', encoding='utf-8') as f:
    f.write(agents_text)
print("Successfully reconstructed scratch/agents_v1_canonical.py! Length:", len(agents_text))
