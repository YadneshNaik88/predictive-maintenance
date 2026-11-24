import re

with open("scratch/agents_v1.py", "r", encoding="utf-8") as f:
    text = f.read()

clean_lines = []
for line in text.splitlines():
    # skip tool header lines
    if line.startswith("Created At:") or line.startswith("Completed At:") or line.startswith("Showing lines"):
        continue
    # match line number prefix like "1: " or "123: "
    m = re.match(r"^\d+:\s?(.*)$", line)
    if m:
        clean_lines.append(m.group(1))
    elif not line.strip():
        clean_lines.append("")

clean_code = "\n".join(clean_lines)
with open("scratch/clean_agents_v1.py", "w", encoding="utf-8") as out:
    out.write(clean_code)

print("Saved clean code, lines:", len(clean_lines))
