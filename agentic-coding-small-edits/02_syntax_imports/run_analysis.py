from broken_helpers import make_trial_data, calculate_trial_mean


trial_data = make_trial_data(12, 100)
trial_means = calculate_trial_mean(trial_data)

print("Trial means:")
for trial_number, mean_value in enumerate(trial_means)
    print(trial_number, mean_value)
