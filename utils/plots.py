import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from utils.lds import continuous_dynamics, match_expert_to_state


def plot_colored_lines(series, states, cmap_name, label):
    """Helper function to plot lines colored by state."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(states), vmax=max(states))
    for i in range(len(series) - 1):
        plt.plot(series[i:i + 2, 0], series[i:i + 2, 1], color=cmap(norm(states[i])), alpha=0.7)
    plt.plot([], [], color=cmap(0.5), label=label)  # For legend


def time_series_prediction(gt_series, gt_states, pred_series_direct, pred_series_derivative, pred_state, name='plot',
                           save_predictions: bool = False):
    gt_series, gt_states = gt_series[:len(pred_series_direct), :], gt_states[:len(pred_series_direct)]
    plt.figure(figsize=(8, 6))
    plot_colored_lines(gt_series, gt_states, 'Dark2', 'Ground Truth Series')
    plot_colored_lines(pred_series_direct, pred_state, 'Pastel1', 'Predicted Series')
    plot_colored_lines(pred_series_derivative, pred_state, 'Pastel2', 'Predicted Series Derivative')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Time Series Prediction: Ground Truth vs Predicted')
    plt.legend()
    sns.despine()
    if save_predictions:
        plt.savefig('./results/Figures/Predicted_vs_ground_truth_dynamics_' + name + '.pdf')
    plt.show()


def evolution_with_background(generator, gt_series, gt_states, pred_series_direct, pred_series_derivative, pred_state,
                              name='plot', save_predictions: bool = False):
    time_axis = np.arange(generator.series_length - generator.input_len - generator.target_len + 1) * generator.dt
    gt_series, gt_states = gt_series[:len(pred_series_direct), :], gt_states[:len(pred_series_direct)]
    offset = 1.5  # shift for y variable
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Add background colors based on the state at each time step
    unique_states = np.unique(gt_states)
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_states)))  # pastel colors
    state_colors = {state: color for state, color in zip(unique_states, colors)}
    pred_state = match_expert_to_state(pred_state, gt_states, num_classes=len(unique_states))

    for i in range(len(time_axis) - 1):
        axs[0].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[gt_states[i].item()], alpha=0.3)
        axs[1].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[pred_state[i].item()], alpha=0.3)

    # Plot x and y (with y shifted) for ground truth and predictions
    plt.plot(time_axis, gt_series[:, 0], 'b-', label='Ground Truth x')
    plt.plot(time_axis[:], pred_series_direct[:, 0], 'r--', label='Predicted x')
    plt.plot(time_axis[:], pred_series_derivative[:, 0], 'g--', label='Predicted x Derivative')

    plt.plot(time_axis, gt_series[:, 1] + offset, 'b-', alpha=0.7, label='Ground Truth y (shifted)')
    plt.plot(time_axis[:], pred_series_direct[:, 1] + offset, 'r--', alpha=0.7, label='Predicted y (shifted)')
    plt.plot(time_axis[:], pred_series_derivative[:, 1] + offset, 'g--', alpha=0.7,
             label='Predicted y Derivative (shifted)')

    plt.xlabel('Time')
    plt.ylabel('x and y (shifted)')
    plt.title('Evolution of x and y Over Time with State Background')
    plt.legend()
    # plt.grid()
    sns.despine()
    if save_predictions:
        plt.savefig('./results/Figures/Variables_Temporal_evolution_' + name + '.pdf')

    plt.tight_layout()
    plt.show()


def dynamics_expert_assignment(model, name='plot', save_predictions: bool = False, idx: int = 0):
    # Create a grid over state space
    x_vals = np.linspace(-5, 5, 30)
    y_vals = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Compute ground truth dynamics (derivative) for each grid point
    U_gt = []
    V_gt = []
    for s in grid_points:
        f, _ = continuous_dynamics(s)
        U_gt.append(f[0])
        V_gt.append(f[1])
    U_gt = np.array(U_gt).reshape(X.shape)
    V_gt = np.array(V_gt).reshape(X.shape)

    L_in = model.generator.input_len
    # from (N,2) → (N, L_in, 2)
    hist = np.repeat(grid_points[:, None, :], L_in, axis=1)
    hist_tensor = torch.tensor(hist, dtype=torch.float32).to(model.device)

    # Model predictions on grid (using the first two output values as derivative)
    model.eval()
    with torch.no_grad():
        # grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        pred, gating = model(hist_tensor)
        pred = pred.cpu().numpy()[:, idx, :]
        pred_deriv = pred[:, 0:2]  # predicted derivative
        U_pred = pred_deriv[:, 0].reshape(X.shape)
        V_pred = pred_deriv[:, 1].reshape(X.shape)
        gate1 = gating.cpu().numpy()[:, idx, :]
        expert_idx = gate1.argmax(axis=1).reshape(X.shape)
        expert_value = gate1.max(axis=1).reshape(X.shape)

    # For reference, the ground truth "next state" would be x + dt * f(x)
    U_gt_next = U_gt  # derivative remains the same for reference
    V_gt_next = V_gt

    # Plot vector fields and expert assignments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: Overlaid vector fields (ground truth in blue, predicted in red)
    ax1.quiver(X, Y, U_gt, V_gt, color='blue', label='Ground Truth Derivative')
    ax1.quiver(X, Y, U_pred, V_pred, color='red', label='Predicted Derivative')
    ax1.set_title('Dynamics: Ground Truth vs Predicted Derivative')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.legend()

    # Right subplot: Scatter of expert assignments with vector field overlay
    cmap = plt.get_cmap('viridis', 3)  # 3 experts
    sc = ax2.scatter(X, Y, c=expert_idx, cmap=cmap, s=100 * expert_value, alpha=0.5, marker='s',
                     label='Expert Assignment')
    ax2.quiver(X, Y, U_pred, V_pred, color='red', label='Predicted Derivative')
    ax2.quiver(X, Y, U_gt, V_gt, color='blue', label='Ground Truth Derivative')
    ax2.set_title('Expert Assignments & Dynamics')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    fig.colorbar(sc, ax=ax2, ticks=[0, 1, 2], label='Expert Index')
    sns.despine()
    if save_predictions:
        plt.savefig('./results/Figures/Vector_field_' + name + '.pdf')

    plt.tight_layout()
    plt.show()


def states_plot(gt_states, dt, pred_state, gt_series, pred_series_direct):
    # Plot states (ground truth vs. predicted)
    time_axis = np.arange(len(gt_states) + 1) * dt
    offset = 0  # 1.5  # shift for y variable
    fig, axs = plt.subplots(4, 1, figsize=(20, 8),
                            gridspec_kw={'height_ratios': [1, 1, 0.5, 1]})
    unique_states = np.unique(gt_states)
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(unique_states)))  # pastel colors
    state_colors = {state: color for state, color in zip(unique_states, colors)}

    remapped_pred_state = match_expert_to_state(pred_state, gt_states)

    # Create a 2D array for the background using pcolormesh
    background = np.zeros((1, len(time_axis) - 1))  # Note: len-1
    for i, state in enumerate(gt_states[:-1]):  # Note: gt_states[:-1]
        background[0, i] = state
    axs[0].imshow(background, aspect='auto', cmap='viridis', zorder=0,
                  alpha=0.6, extent=[0, len(time_axis) - 1, -1, 3])
    axs[0].plot(gt_series[:, 0], 'b-', label='Ground Truth x', zorder=10)
    axs[0].plot(gt_series[:, 1] + offset, 'b-',
                label='Ground Truth y (shifted)')
    axs[0].set_title('Ground Truth')
    axs[0].set_xticks([])
    plt.xlabel('Time')

    # Create a 2D array for the background using pcolormesh
    background = np.zeros((1, len(time_axis) - 1))  # Note: len-1
    for i, state in enumerate(remapped_pred_state[:-1]):  # Note: gt_states[:-1]
        background[0, i] = state
    axs[1].imshow(background, aspect='auto', cmap='viridis', zorder=0,
                  alpha=0.6, extent=[0, len(time_axis) - 1, -1, 3])
    axs[1].plot(pred_series_direct[:, 0], 'b-', label='Ground Truth x',
                zorder=10)
    axs[1].plot(pred_series_direct[:, 1] + offset, 'b-',
                label='Ground Truth y (shifted)')
    axs[1].set_title('Predictions')
    axs[1].set_xticks([])

    window_size = 100
    state_correctness = (remapped_pred_state == gt_states).astype(int)
    # moving_avg_correctness = np.convolve(state_correctness, np.ones(window_size)/window_size, mode='valid')
    # axs[2].plot(state_correctness)
    axs[2].fill_between(np.arange(len(state_correctness)), state_correctness,
                        step='mid', alpha=0.5)
    axs[2].set_title('State Tracking Accuracy (Moving Avg)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Moving Avg Accuracy')
    axs[2].set_xlim(0, len(time_axis) - 1)
    axs[2].set_ylim(0, 1.1)

    # 1. Plot x and y (ground truth vs. predicted)
    axs[3].plot(gt_series[:, 0], label='Ground Truth x')
    axs[3].plot(pred_series_direct[:, 0], label='Predicted x')
    axs[3].plot(gt_series[:, 1], label='Ground Truth y')
    axs[3].plot(pred_series_direct[:, 1], label='Predicted y')
    axs[3].set_title(
        f'Time Series Prediction')
    axs[3].set_xlim(0, len(time_axis) - 1)
    plt.legend()

    sns.despine()
    plt.show()


def plot_predictive_error_vs_delta_time(predictions, observations, input_len=20, max_delta=3, title=None):
    """
    Plot predictive error vs delta time for multi-step predictions.

    Args:
        predictions: (N, K, D) array — model predictions for each t (N samples), for next K steps, each of D dims
        observations: (N + input_len + max_delta, D) array — full observation stream to align with any delta
        input_len: number of steps used as input (to align the prediction start index)
        max_delta: how far to look into past/future to compute prediction error
        title: optional title for the plot
    """
    N, K, D = predictions.shape
    M = observations.shape[0]
    deltas = np.arange(-max_delta, max_delta)
    errors = np.full((K, len(deltas)), np.nan, dtype=float)

    for i, delta in enumerate(deltas):
        for k in range(K):  # prediction for t+1, t+2, ..., t+K
            start = input_len + k + delta
            end = start + N
            if start < 0 or end > M:
                continue  # skip out-of-bounds shifts

            obs_seg = observations[start:end]  # (N, D)
            pred_seg = predictions[:, k, :]  # (N, D)
            errors[k, i] = np.abs(pred_seg - obs_seg).mean()

    plt.figure(figsize=(8, 5))
    for k in range(K):
        plt.plot(
            deltas,
            errors[k],
            marker='o',
            label=f'Pred step {k + 1}'
        )
    plt.axvline(0, color='gray', linestyle='--', label='Aligned (δ=0)')
    plt.xlabel("Delta time (steps)")
    plt.ylabel("Mean absolute error")
    plt.title(title or "Predictive error vs. delta time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
