import difflib

with open(r'c:\Users\naiky\OneDrive\Desktop\Projects\predictive-maintenance\scratch\agents_step1440.py', 'r', encoding='utf-8') as f:
    orig = f.readlines()

with open(r'c:\Users\naiky\OneDrive\Desktop\Projects\predictive-maintenance\src\agents.py', 'r', encoding='utf-8') as f:
    curr = f.readlines()

diff = list(difflib.unified_diff(orig, curr, fromfile='orig_step1440', tofile='curr_agents'))
print(''.join(diff))
