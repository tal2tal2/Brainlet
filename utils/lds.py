import os

import numpy as np
from scipy.optimize import linear_sum_assignment


def switching_lds_step(variable, state, transition_prob=1 / 200, noise_scale=0.0001, constant_scaling=0.5,
                       constant_decay=10.):
    """
    Switching linear dynamical system step.
    """
    states_matrices = [
        np.array([[0.95, 0.05], [-0.1, 0.9]]),
        np.array([[1.05, -0.2], [0.3, 0.8]]),
        np.array([[0.85, 0.3], [-0.3, 1.0]])
    ]  # linear transformation
    state_constants = [
        np.array([2.5, -0.5]),
        np.array([-0.5, 0.5]),
        np.array([0., -0.])
    ]  # baseline offset

    if np.random.rand() < transition_prob:
        state = np.random.choice([0, 1, 2])  # randomly change state

    d_variable = constant_scaling * states_matrices[state] @ variable - constant_decay * (
            variable - state_constants[state])  # compute delta from last pos
    noise = np.random.randn(2) * noise_scale  # add noise
    return d_variable + noise, state


def continuous_dynamics(variable, state=None):
    """
    Continuous dynamics with a consistent signature.
    """
    x, y = variable
    if x < 2 and y < 0:
        d_variable, new_state = np.array([1, -1]), 0
    elif x < 2 and y >= 0:
        d_variable, new_state = np.array([-1, -1]), 1
    else:
        d_variable, new_state = np.array([-y, x + 2]), 2
    return d_variable, new_state


def generate_time_series(n_series=2, series_length=1000, dt=0.001, use_slds=True):
    """
    Generate multiple time series using either switching LDS or continuous dynamics.
    """
    series_list = []
    for _ in range(n_series):  # want n examples
        series = np.zeros((series_length, 2))  # x, y
        states = np.zeros(series_length, dtype=int)  # state
        series[0] = np.random.randn(2)  # initialize randomly
        state = np.random.choice([0, 1, 2])
        states[0] = state  # record initial state
        for t in range(1, series_length):
            dynamics = switching_lds_step if use_slds else continuous_dynamics
            f, state = dynamics(series[t - 1], state)  # why do we calculate only the delta
            series[t] = series[t - 1] + dt * f
            states[t] = state
        series_list.append((series, states))
    return series_list


def simulate_series(initial_var, steps, dt, use_slds=False):
    """
    Simulate a ground-truth time series using the specified dynamics.
    """
    dynamics = switching_lds_step if use_slds else continuous_dynamics
    series = np.zeros((steps + 1, 2))
    states_series = np.zeros(steps + 1, dtype=int)
    series[0] = initial_var
    state = np.random.choice([0, 1, 2])
    states_series[0] = state
    for t in range(1, steps + 1):
        f, state = dynamics(series[t - 1], state)
        series[t] = series[t - 1] + dt * f
        states_series[t] = state
    return series, states_series


def match_expert_to_state(pred_state, gt_states, num_classes=3):
    pred = np.array(pred_state).astype(int).flatten()
    gt = np.array(gt_states).astype(int).flatten()
    assert pred.shape == gt.shape, f"pred_state and gt_states must align, pred.shape={pred.shape}, gt.shape={gt.shape}"

    # Step 1: Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for p, g in zip(pred, gt):
        confusion[p, g] += 1

    # Step 2: Solve optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(-confusion)  # maximize matches

    # Step 3: Build mapping
    mapping = dict(zip(row_ind, col_ind))

    # Step 4: Apply mapping
    remapped_pred_state = np.array([mapping[p] for p in pred])
    return remapped_pred_state


def save_series(path: str, n_series=1000, series_length=10000):
    os.makedirs(path, exist_ok=True)
    series_list = generate_time_series(n_series=n_series, series_length=series_length)

    # Convert list of tuples into two arrays
    all_series = np.stack([s for s, _ in series_list])  # shape: (N, T, 2)
    all_states = np.stack([z for _, z in series_list])  # shape: (N, T)

    np.save(os.path.join(path, "series.npy"), all_series)  # shape (N, T, 2)
    np.save(os.path.join(path, "states.npy"), all_states)  # shape (N, T)
    print(f"Saved {n_series} series to {path}")


def load_subset(path, start=0, end=-1):
    series = np.load(os.path.join(path, "series.npy"), mmap_mode='r')[start:end]  # doesn't read entire file
    states = np.load(os.path.join(path, "states.npy"), mmap_mode='r')[start:end]
    return list(zip(series, states))


if __name__ == "__main__":
    save_series("../data/fake_dataset", n_series=100, series_length=10000)
