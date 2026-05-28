def make_trial_data(n_trials, n_timepoints):
    data = np.random.normal(loc=0, scale=1, size=(n_trials, n_timepoints))
    data[:, 40:60] = data[:, 40:60] + 2
    return data


def calculate_trial_means(data):
    return data.mean(axis=1)
