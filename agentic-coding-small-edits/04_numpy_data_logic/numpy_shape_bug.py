import numpy as np


# Each row is intended to be one trial.
# Each column is intended to be one timepoint.
trial_data = [
    [0.2, 0.4, 0.5, 0.8],
    [0.1, 0.3, 0.4, 0.6],
    [0.5, 0.7, 0.8, 1.0],
]


def baseline_subtract(data):
    baseline = data[:, :2].mean(axis=1)
    return data - baseline


corrected_data = baseline_subtract(trial_data)
trial_means = corrected_data.mean(axis=1)

print("Baseline-corrected data:")
print(corrected_data)
print("Mean response for each trial:")
print(trial_means)
