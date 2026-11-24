import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from src.simulator import get_tuning_eval_pools

df = pd.read_csv('outputs/oof_predictions_L30.csv')
tuning_units, _ = get_tuning_eval_pools(seed=42)
df_tune = df[df['unit_number'].isin(tuning_units)].copy()
df_tune['residual'] = df_tune['true_rul_raw'] - df_tune['pred_rul']

print(f"Total tuning rows: {len(df_tune)}")
df_tune['bin'] = np.clip((df_tune['pred_rul'] // 10).astype(int), 0, 12)
bin_counts = df_tune['bin'].value_counts().sort_index()
print("Bin counts:")
for b, c in bin_counts.items():
    sub = df_tune[df_tune['bin'] == b]['residual']
    print(f"  Bin {b:2d} [{b*10:3d}-{(b+1)*10:3d}): {c:4d} samples, mean res={sub.mean():+.2f}, std={sub.std():.2f}")
