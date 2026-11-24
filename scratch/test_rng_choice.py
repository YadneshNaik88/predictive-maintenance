import numpy as np

rng1 = np.random.default_rng(42)
a = rng1.choice([0, 1, 2, 3], size=1020, p=[0.1106, 0.3594, 0.3894, 0.1406])

rng2 = np.random.default_rng(42)
b = rng2.choice([0, 1, 2, 3], size=1200, p=[0.1106, 0.3594, 0.3894, 0.1406])

print(f"Equal up to 1020: {np.array_equal(a, b[:1020])}")

