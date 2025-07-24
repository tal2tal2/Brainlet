# %%
import os
import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# !pip install --force-reinstall numpy==1.24
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
general questions - 
why do we assume no actual way to model what the next state will be?
"""

# %%
# ------------------------------
# Dynamics Functions
# ------------------------------

name = 'simple_SLDS'
use_slds = False
dt = 0.001  # delta time

os.makedirs('./results/Figures', exist_ok=True)


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


# ------------------------------
# Time Series Generation & Dataset
# ------------------------------

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


class TimeSeriesDataset(Dataset):
    """
    Dataset class that creates samples from the generated time series.
    Each sample consists of an input x_t and a target which is the concatenation
    of the derivative f_t and the next state x_{t+1}.
    """

    def __init__(self, series_list, dynamics, dt=0.001):
        self.samples = []
        self.dt = dt
        for series, states in series_list:
            for t in range(len(series) - 1):
                x_t = series[t]
                state_t = states[t]
                x_next = series[t + 1]
                f_t, _ = dynamics(x_t, state_t)  # why do we do this? this seems cyclic
                # Target: [derivative, next state]
                target = np.concatenate([f_t, x_next])  # why do we seperate the derivative from the next step?
                self.samples.append((x_t, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, target = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# ------------------------------
# Mixture of Experts Model & Regularization
# ------------------------------

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=2, output_dim=4, num_experts=3):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim)
                # nn.Tanh(),
                # nn.Linear(10, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Sequential(  # how much should we believe each expert
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        gating_weights = self.gating_network(x)  # Shape: [batch, num_experts]
        # Stack expert outputs: shape becomes [batch, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Combine experts' outputs weighted by the gating network
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(1), dim=-1)
        return final_output, gating_weights


def gating_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.1):
    """
    Regularization term to encourage peaky (low entropy) and diverse (high entropy) gating.
    """
    sample_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
    p_mean = gating_weights.mean(dim=0)
    avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
    return lambda_peaky * sample_entropy - lambda_diverse * avg_entropy


# %%
# ------------------------------
# Training Setup
# ------------------------------

# Generate series and create dataset (using continuous dynamics here)
series_list = generate_time_series(n_series=20, series_length=10000, dt=0.001, use_slds=False)
dataset = TimeSeriesDataset(series_list, continuous_dynamics, dt=0.001)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
model = MixtureOfExperts(input_dim=2, output_dim=4, num_experts=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

losses = []
num_epochs = 20
lambda_phys = 0.1

for epoch in range(num_epochs):
    total_loss = 0.0
    for x, target in dataloader:
        optimizer.zero_grad()
        output, gating_weights = model(x)
        loss_mse = criterion(output, target)
        loss_reg = gating_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.1)

        f_pred = output[:, :2]  # First two elements are the predicted derivative
        x_next_pred = output[:, 2:]  # Last two elements are the predicted next state
        physics_loss = criterion(x_next_pred, x + dt * f_pred)

        loss = loss_mse + loss_reg + lambda_phys * physics_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# %% Plot basic metrics

# Plot Loss Evolution
plt.figure(figsize=(8, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training')
plt.legend()
# plt.grid(True)
sns.despine()
plt.savefig('./results/Figures/loss_' + name + '.pdf')
plt.show()

# Plot Gating Distributions
samples = torch.concat([x for x, target in dataloader])
with torch.no_grad():
    _, gating_weights = model(samples)
gating_weights_np = gating_weights.detach().numpy()

plt.figure(figsize=(8, 6))
for i in range(gating_weights_np.shape[1]):
    sns.kdeplot(gating_weights_np[:, i], label=str(i))

plt.xlabel('Gating Weight')
plt.ylabel('Density')
plt.title('Gating Network Output Distribution')
plt.legend(title='Category')
# plt.grid(True)
plt.xlim(0, 1)  # Bound the x-axis between 0 and 1
sns.despine()
plt.savefig('./results/Figures/Gating_weights_distributions_' + name + '.pdf')
plt.show()


# %%
# ------------------------------
# Helper Functions for Testing / Prediction
# ------------------------------

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


def predict_time_series(model, time_series, steps, dt):
    """
    Compute one-step predictions for a given time series.
    The model outputs a 4D vector: first two entries are the predicted derivative,
    and the last two entries are the predicted next state.

    Two predictions are computed:
    - One based on integrating the predicted derivative.
    - One directly using the last two elements of the model output.
    """
    # steps = len(time_series) - 1
    pred_series_derivative = np.zeros((steps, 2))
    pred_series_direct = np.zeros((steps, 2))
    pred_state = np.zeros(steps, dtype=int)

    for t in range(steps):
        x_tensor = torch.tensor(time_series[t], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output, gating = model(x_tensor)
        output = output.squeeze(0).numpy()  # shape: (4,)

        # Extract predicted derivative and direct next state prediction
        pred_deriv = output[:2]
        pred_next_state = output[2:]
        pred_series_derivative[t] = time_series[t] + dt * pred_deriv
        pred_series_direct[t] = pred_next_state

        # Determine expert index (predicted state)
        gating = gating.squeeze(0).numpy()
        pred_state[t] = np.argmax(gating)

    return pred_series_derivative, pred_series_direct, pred_state


# %%
# ------------------------------
# Testing: Time Series Prediction
# ------------------------------

# Define simulation parameters
steps = 15000
dt = 0.001
initial_var = np.random.randn(2)
initial_state = initial_var.copy()  # using same value for prediction

# Ground truth simulation using continuous dynamics
gt_series, gt_states = simulate_series(initial_var, steps=steps, dt=dt)

# Model prediction rollout
pred_series_derivative, pred_series_direct, pred_state = predict_time_series(model, gt_series, steps=steps, dt=dt)


# %%

def plot_colored_lines(series, states, cmap_name, label):
    """Helper function to plot lines colored by state."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(states), vmax=max(states))
    for i in range(len(series) - 1):
        plt.plot(series[i:i + 2, 0], series[i:i + 2, 1], color=cmap(norm(states[i])), alpha=0.7)
    plt.plot([], [], color=cmap(0.5), label=label)  # For legend


# Model prediction rollout

plt.figure(figsize=(8, 6))
plot_colored_lines(gt_series, gt_states, 'Dark2', 'Ground Truth Series')
plot_colored_lines(pred_series_direct, pred_state, 'Pastel1', 'Predicted Series')
plot_colored_lines(pred_series_derivative, pred_state, 'Pastel2', 'Predicted Series Derivative')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Time Series Prediction: Ground Truth vs Predicted')
plt.legend()
# plt.grid()
sns.despine()
plt.savefig('./results/Figures/Predicted_vs_ground_truth_dynamics_' + name + '.pdf')
plt.show()

# %%
# ------------------------------
# Plot 2: Evolution of x and y Over Time (with y shifted)
# ------------------------------

time_axis = np.arange(steps + 1) * dt
offset = 1.5  # shift for y variable

# Create a new figure with a subplot (using subplot index 2 as provided)
# plt.figure(figsize=(10, 8))
# ax = plt.subplot(2, 1, 2)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

# Add background colors based on the state at each time step
unique_states = np.unique(gt_states)
colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_states)))  # pastel colors
state_colors = {state: color for state, color in zip(unique_states, colors)}

for i in range(len(time_axis) - 1):
    axs[0].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[gt_states[i]], alpha=0.3)
    axs[1].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[pred_state[i]], alpha=0.3)

# Plot x and y (with y shifted) for ground truth and predictions
plt.plot(time_axis, gt_series[:, 0], 'b-', label='Ground Truth x')
plt.plot(time_axis[:-1], pred_series_direct[:, 0], 'r--', label='Predicted x')
plt.plot(time_axis[:-1], pred_series_derivative[:, 0], 'g--', label='Predicted x Derivative')

plt.plot(time_axis, gt_series[:, 1] + offset, 'b-', alpha=0.7, label='Ground Truth y (shifted)')
plt.plot(time_axis[:-1], pred_series_direct[:, 1] + offset, 'r--', alpha=0.7, label='Predicted y (shifted)')
plt.plot(time_axis[:-1], pred_series_derivative[:, 1] + offset, 'g--', alpha=0.7,
         label='Predicted y Derivative (shifted)')

plt.xlabel('Time')
plt.ylabel('x and y (shifted)')
plt.title('Evolution of x and y Over Time with State Background')
plt.legend()
# plt.grid()
sns.despine()
plt.savefig('./results/Figures/Variables_Temporal_evolution_' + name + '.pdf')

plt.tight_layout()
plt.show()

# %%
# ------------------------------
# Vector Field Plot: Dynamics & Expert Assignment
# ------------------------------

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
sc = ax2.scatter(X, Y, c=expert_idx, cmap=cmap, s=100 * expert_value, alpha=0.5, marker='s', label='Expert Assignment')
ax2.quiver(X, Y, U_pred, V_pred, color='red', label='Predicted Derivative')
ax2.quiver(X, Y, U_gt, V_gt, color='blue', label='Ground Truth Derivative')
ax2.set_title('Expert Assignments & Dynamics')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim([-5, 5])
ax2.set_ylim([-5, 5])
fig.colorbar(sc, ax=ax2, ticks=[0, 1, 2], label='Expert Index')
sns.despine()
plt.savefig('./results/Figures/Vector_field_' + name + '.pdf')

plt.tight_layout()
plt.show()
