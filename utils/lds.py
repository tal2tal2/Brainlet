import numpy as np


def switching_lds_step(variable, state, transition_prob=1 / 200, noise_scale=0.0001):
    """
    Switching linear dynamical system step.
    """
    # is there a reason for these constants?
    states_matrices = [
        np.array([[0.95, 0.05], [-0.1, 0.9]]),
        np.array([[1.05, -0.2], [0.3, 0.8]]),
        np.array([[0.85, 0.3], [-0.3, 1.0]])
    ]  # linear transformation
    state_constants = [
        np.array([0.1, -0.1]),
        np.array([-0.2, 0.2]),
        np.array([0.3, -0.3])
    ]  # baseline offset

    if np.random.rand() < transition_prob:
        state = np.random.choice([0, 1, 2])  # randomly change state

    d_variable = states_matrices[state] @ variable - variable + state_constants[state]  # compute delta from last pos
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
