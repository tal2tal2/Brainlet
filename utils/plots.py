import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from utils.lds import continuous_dynamics


def plot_colored_lines(series, states, cmap_name, label):
    """Helper function to plot lines colored by state."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(states), vmax=max(states))
    for i in range(len(series) - 1):
        plt.plot(series[i:i + 2, 0], series[i:i + 2, 1], color=cmap(norm(states[i])), alpha=0.7)
    plt.plot([], [], color=cmap(0.5), label=label)  # For legend


def time_series_prediction(gt_series, gt_states, pred_series_direct, pred_series_derivative, pred_state, name='plot',
                           save_predictions: bool = False):
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
    time_axis = np.arange(generator.series_length) * generator.dt
    offset = 1.5  # shift for y variable
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Add background colors based on the state at each time step
    unique_states = np.unique(gt_states)
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_states)))  # pastel colors
    state_colors = {state: color for state, color in zip(unique_states, colors)}

    for i in range(len(time_axis) - 1):
        pred_num = (pred_state[i].item() + 1) % len(unique_states)
        axs[0].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[gt_states[i].item()], alpha=0.3)
        axs[1].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[pred_num], alpha=0.3)

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


def dynamics_expert_assignment(model, name='plot', save_predictions: bool = False):
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

    # Model predictions on grid (using the first two output values as derivative)
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        pred, gating = model(grid_tensor)
        pred = pred.numpy()
        pred_deriv = pred[:, 0:2]  # predicted derivative
        U_pred = pred_deriv[:, 0].reshape(X.shape)
        V_pred = pred_deriv[:, 1].reshape(X.shape)
        expert_idx = torch.argmax(gating, dim=-1).numpy().reshape(X.shape)
        expert_value = torch.max(gating, dim=-1)[0].numpy().reshape(X.shape)

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
