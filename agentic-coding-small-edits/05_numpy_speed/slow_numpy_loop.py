import numpy as np


np.random.seed(7)
signal = np.random.normal(loc=0, scale=1, size=100000)
threshold = 1.5

values_above_threshold = []
for value in signal:
    if value > threshold:
        values_above_threshold.append(value)

values_above_threshold = np.array(values_above_threshold)

print("Number of values above threshold:", len(values_above_threshold))
print("Mean of values above threshold:", values_above_threshold.mean())
