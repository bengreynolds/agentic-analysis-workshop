import numpy as np


Data = np.array([
    [1.2, 1.4, 1.5, 1.7],
    [0.9, 1.0, 1.1, 1.4],
    [2.1, 2.0, 2.4, 2.8],
])

trial_num_list = [1, 2, 3]
AvgResponse = Data.mean(axis=1)
finalOutput = []

for x1, x2 in zip(trial_num_list, AvgResponse):
    finalOutput.append({
        "TrialNumber": x1,
        "AverageResponse": x2,
    })

for ITEM in finalOutput:
    print(ITEM)
