
#%%
import numpy as np
# !pip install --force-reinstall numpy==1.24
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import matplotlib.colors as mcolors
import random
import os
from scipy.optimize import linear_sum_assignment
import itertools
import pandas as pd

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%%
# ------------------------------
# Helper Functions for Testing / Prediction
# ------------------------------

def simulate_series(initial_var, steps, dt, use_slds=True, constant_decay=10., constant_scaling=0.1):
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
        if use_slds:
            f, state = dynamics(series[t - 1], state, constant_decay = constant_decay, constant_scaling = constant_scaling)
        else: 
            f, state = dynamics(series[t - 1], state)
        series[t] = series[t - 1] + dt * f
        states_series[t] = state
    return series, states_series


def predict_time_series(model, time_series, steps, dt, window):
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

    for t in range(window,steps):
        x_tensor = torch.tensor(time_series[t-window:t].flatten(), dtype=torch.float32).unsqueeze(0)
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

def match_expert_to_state(pred_state, gt_states, num_classes = 3):
    # Step 1: Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(pred_state)):
        confusion[pred_state[i], gt_states[i]] += 1

    # Step 2: Solve optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(-confusion)  # maximize matches

    # Step 3: Build mapping
    mapping = dict(zip(row_ind, col_ind))

    # Step 4: Apply mapping
    remapped_pred_state = np.array([mapping[p] for p in pred_state])
    return remapped_pred_state

# Manual loss functions conputation
def compute_MSE_loss(pred_series, gt_series):
    assert pred_series.shape == gt_series.shape, "Shape mismatch between predicted and ground truth series"

    mse_losses = np.mean((pred_series - gt_series)**2)
    return mse_losses

def compute_physics_loss(pred_series_direct, pred_series_derivative):
    return compute_MSE_loss(pred_series_direct, pred_series_derivative)

def compute_states_mov_avg_corr(remapped_pred_state, gt_states):
    state_correctness = (remapped_pred_state == gt_states[:-1]).astype(int)  # 1 if correct, 0 if wrong

    window_size = 100  # you can tune this
    return np.convolve(state_correctness, np.ones(window_size)/window_size, mode='valid')

def compute_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.3):
    sample_entropy = -np.sum(gating_weights * np.log(gating_weights + 1e-8), axis=-1).mean()
    p_mean = gating_weights.mean(axis=0)
    avg_entropy = -np.sum(p_mean * np.log(p_mean + 1e-8))
    return lambda_peaky * sample_entropy - lambda_diverse * avg_entropy

# Plot functions
def plot_mean_MSE(meaned_results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(meaned_results_df['MSE_loss'].values, label="Mean MSE Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Mean MSE Loss")
    plt.title("Mean MSE Loss over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

def plot_mean_phys_loss(meaned_results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(meaned_results_df['physics_loss'].values, label="Mean physics loss")
    plt.xlabel("Iteration")
    plt.ylabel("Mean physics loss")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.show()
    return

def plot_mean_gating_reg(meaned_results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(meaned_results_df['gating_reg_loss'].values, label="Mean gating regularization")
    plt.xlabel("Iteration")
    plt.ylabel("Mean gating regularization")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.show()
    return

def plot_mean_reg_loss(meaned_results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(meaned_results_df['reg_loss'].values, label="Mean regularization loss")
    plt.xlabel("Iteration")
    plt.ylabel("Mean regularization loss")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.show()
    return

def plot_states_pred_acc(meaned_results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(meaned_results_df['moving_avg_accuracy_mean'].values, label="State accuracy mean")
    plt.xlabel("Iteration")
    plt.ylabel("Mean state accuracy")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.show()
    return


# ------------------------------
# Dynamics Functions
# ------------------------------

name='slds'
use_slds = True
dt = 0.001



def switching_lds_step(variable, state, transition_prob=1 / 200, noise_scale=0.0001, constant_scaling = 0.5, constant_decay = 10.):
    """
    Switching linear dynamical system step.
    """
    states_matrices = [
        np.array([[0.95, 0.05], [-0.1, 0.9]]),
        np.array([[1.05, -0.2], [0.3, 0.8]]),
        np.array([[0.85, 0.3], [-0.3, 1.0]])
    ]
    state_constants = [
        np.array([2.5, -0.5]),
        np.array([-0.5, 0.5]),
        np.array([0., -0.])
    ]

    if np.random.rand() < transition_prob:
        state = np.random.choice([0, 1, 2])

    d_variable = constant_scaling * states_matrices[state] @ variable - constant_decay* (variable - state_constants[state])
    noise = np.random.randn(2) * noise_scale
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

def generate_time_series(n_series=2, series_length=1000, dt=0.001, use_slds=True, constant_decay = 10., constant_scaling = 0.0):
    """
    Generate multiple time series using either switching LDS or continuous dynamics.
    """
    series_list = []
    for _ in range(n_series):
        series = np.zeros((series_length, 2))
        states = np.zeros(series_length, dtype=int)
        series[0] = np.random.randn(2)
        state = np.random.choice([0, 1, 2])
        states[0] = state  # record initial state
        for t in range(1, series_length):
            dynamics = switching_lds_step if use_slds else continuous_dynamics
            if use_slds:
                f, state = dynamics(series[t - 1], state, constant_decay=constant_decay, constant_scaling=constant_scaling)    
            else:
                f, state = dynamics(series[t - 1], state)
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

    def __init__(self, series_list, dynamics, dt=0.001, window=10):
        self.samples = []
        self.dt = dt
        for series, states in series_list:
            for t in range(len(series) - 1):
                input_t = series[t-window:t].flatten()
                x_t = series[t]
                state_t = states[t]
                x_next = series[t + 1]
                f_t, _ = dynamics(x_t, state_t)
                # Target: [derivative, next state]
                target = np.concatenate([f_t, x_next])
                self.samples.append((input_t, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, target = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def safe_collate(batch):
    batch = [item for item in batch if item[0].shape[0] > 0 and item[1].shape[0] > 0]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

def train_on_data(series_list, dataset, dataloader, num_epochs=10, min_delta=1e-4, patience=20, batch_size=64, window=10, lambda_phys = 0.3,lambda_peaky=0.1, lambda_diverse=0.1):    
    # Initialize model, optimizer, and loss function
    model = MixtureOfExperts(input_dim=2*window, output_dim=4, num_experts=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Contiguous slicing
    train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size//batch_size * batch_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size//batch_size * batch_size, len(dataset)//batch_size * batch_size)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, target in train_loader:
            if x.nelement() == 0:  # Skip empty batches
                continue
            optimizer.zero_grad()
            output, gating_weights = model(x)
            loss_mse = criterion(output, target)
            # loss_reg = gating_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.3)

            # f_pred - gt_f
            f_pred = output[:, :2]  # First two elements are the predicted derivative
            x_next_pred = output[:, 2:]  # Last two elements are the predicted next state    
            physics_loss = lambda_phys * criterion(x_next_pred, x[:,-2:] + dt * f_pred)
        
            sample_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
            p_mean = gating_weights.mean(dim=0)
            avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
            sample_loss = lambda_peaky * sample_entropy 
            diverse_loss = - lambda_diverse * avg_entropy

            loss = loss_mse + sample_loss + diverse_loss + physics_loss    
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        # -------- Validation Step --------
        model.eval()
        total_val_loss, total_mse_loss, total_physics_loss, total_sample_loss, total_diverse_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, target in val_loader:
                output, gating_weights = model(x)
                loss_mse = criterion(output, target)
                # loss_reg = gating_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.3)

                f_pred = output[:, :2]
                x_next_pred = output[:, 2:]
                physics_loss = lambda_phys * criterion(x_next_pred, x[:,-2:] + dt * f_pred)
        
                sample_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
                p_mean = gating_weights.mean(dim=0)
                avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
                sample_loss = lambda_peaky * sample_entropy 
                diverse_loss = - lambda_diverse * avg_entropy

                val_loss = loss_mse + sample_loss + diverse_loss + physics_loss
                total_val_loss += val_loss.item()
                total_mse_loss += loss_mse.item()
                total_physics_loss += lambda_phys * physics_loss.item()
                total_sample_loss += sample_loss.item()
                total_diverse_loss += diverse_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_mse_loss = total_mse_loss / len(val_loader)
        avg_diverse_loss = total_diverse_loss / len(val_loader)
        avg_sample_loss = total_sample_loss / len(val_loader)
        avg_physics_loss = total_physics_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Physics Loss: {avg_physics_loss:.4f}, Sample Loss: {avg_sample_loss:.4f}, Diverse Loss: {avg_diverse_loss:.4f}")

        # -------- Early Stopping Check --------
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return losses, x, model, gating_weights

def test_on_data(steps = 5000, dt = 0.001, constant_decay=10., constant_scaling=0.0, window=10):
    # initial_var = np.random.randn(1)
    initial_var = series_list[0][0][0]
    initial_state = initial_var.copy()  # using same value for prediction

    # Ground truth simulation using continuous dynamics
    gt_series, gt_states = simulate_series(initial_var, steps=steps, dt=dt, use_slds=use_slds, constant_decay=10., constant_scaling=0.0)

    # gt_series, gt_states = series_list[2]

    pred_series_derivative, pred_series_direct, pred_state = predict_time_series(model, gt_series, steps=steps, dt=dt, window=window)
    
    # Model prediction rollout
    return pred_series_derivative, pred_series_direct, pred_state, gt_series, gt_states
    


#% Check ground truth dynamics
steps = 15000
dt = 0.001
initial_var = np.random.randn(2)
initial_state = initial_var.copy()  # using same value for prediction

# Ground truth simulation using continuous dynamics
gt_series, gt_states = simulate_series(initial_var, steps=steps, dt=dt, use_slds=use_slds)

if False:
    time_axis = np.arange(steps+1) * dt
    offset = 0#1.5  # shift for y variable
    fig, axs = plt.subplots(2, 1, figsize=(20,8), gridspec_kw={'height_ratios': [1, 2]})
    unique_states = np.unique(gt_states)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))  # pastel colors
    state_colors = {state: color for state, color in zip(unique_states, colors)}

    for i in range(len(time_axis) - 1):
        axs[0].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[gt_states[i]], alpha=0.3)
    plt.plot(time_axis, gt_series[:, 0], 'b-', label='Ground Truth x')
    plt.plot(time_axis, gt_series[:, 1] + offset, 'b-', alpha=0.7, label='Ground Truth y (shifted)')

    plt.xlabel('Time')
    plt.ylabel('x and y (shifted)')
    sns.despine()
    plt.show()


#%
# ------------------------------
# Mixture of Experts Model & Regularization
# ------------------------------

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=2, output_dim=4, num_experts=3):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # enabled for capturing non-linear behaviour
                # nn.Tanh(),
                # nn.Linear(10, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    # Prior gating strategy
    def forward(self, x):
        gating_weights = self.gating_network(x)  # Shape: [batch, num_experts]
        # Stack expert outputs: shape becomes [batch, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Combine experts' outputs weighted by the gating network
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(1), dim=-1)
        return final_output, gating_weights

    # # Posterior gating strategy 
    # def forward(self, x):
    #     # Stack expert outputs: shape becomes [batch, output_dim, num_experts]
    #     expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
    #     # Weight the experts' output via gating network
    #     gating_weights = self.gating_network(expert_outputs)  # Shape: [batch, num_experts]
    #     final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(1), dim=-1)
    #     return final_output, gating_weights


def gating_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.3):
    """
    Regularization term to encourage peaky (low entropy) and diverse (high entropy) gating.
    """
    sample_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
    p_mean = gating_weights.mean(dim=0)
    avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
    return lambda_peaky * sample_entropy - lambda_diverse * avg_entropy


#%% 
# ------------------------------
# Fast simulation - "main"
# ------------------------------

# Simulation configuration
repetitions = 1 # default is 10
batch_size = 256
num_epochs = 30
lambda_phys = 0.3       
lambda_peaky= 0.1
lambda_diverse= 0.3

constant_decay = [10]#np.linspace(5, 20, 3)
constant_scaling = [0.1]#np.linspace(0,0.1, 3)
# Initialize simulation
product_constants = itertools.product(constant_decay, constant_scaling)
decay_list, scaling_list = zip(*product_constants)
constants = {
    "constant_decay": list(decay_list),
    "constant_scaling": list(scaling_list)
}

pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't wrap lines
pd.set_option('display.max_colwidth', None)  # Show full content in each cell

const_df = pd.DataFrame(constants)

state_accuracies = []

all_rep_results = []
# for index, row in const_df.iterrows():
for const_scaling in constant_scaling:
    all_results_meaned = []

    # const_decay = row['constant_decay']
    # const_scaling = row['constant_scaling']
    const_decay = 10.

    for rep in range(repetitions):        

        print(f"Starting repetition number {rep + 1}...")
        print(f"const_decay: {const_decay}")
        print(f"const_scaling: {const_scaling}")

        # Generate series and create dataset 
        series_list = generate_time_series(n_series=50, series_length=5000, dt=0.001, use_slds=True, constant_decay = const_decay, constant_scaling = const_scaling)
        dataset = TimeSeriesDataset(series_list, continuous_dynamics, dt=0.001, window=10)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"train iteration number {rep}...")
        losses, x, model, gating_weights = train_on_data(series_list, dataset, dataloader, num_epochs=num_epochs, patience=5, batch_size=batch_size, lambda_phys = lambda_phys, lambda_peaky=lambda_peaky, lambda_diverse=lambda_diverse)
        print("done")

        # The testing data is generated inside test_on_data
        pred_series_derivative, pred_series_direct, pred_state, gt_series, gt_states = test_on_data(steps = 25000, constant_decay = const_decay, constant_scaling = const_scaling, window=10)
        remapped_pred_state = match_expert_to_state(pred_state, gt_states)
        

        # Compute losses
        MSE_loss = compute_MSE_loss(pred_series_direct, gt_series[1:])
        # MSE_losses.append(MSE_loss)
        physics_loss = compute_physics_loss(pred_series_direct, pred_series_derivative)
        # physics_losses.append(physics_loss)
            
        sample_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
        p_mean = gating_weights.mean(dim=0)
        avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
        sample_loss = lambda_peaky * sample_entropy 
        diverse_loss = - lambda_diverse * avg_entropy
        
        gating_reg_loss = compute_regularization(gating_weights.detach().cpu().numpy())
        # gating_reg_losses.append(gating_reg_loss) 
        reg_loss = MSE_loss + gating_reg_loss + lambda_phys * physics_loss + sample_loss + diverse_loss
        # reg_losses.append(MSE_loss + gating_reg_loss + lambda_phys * physics_loss) 

        results_for_run = {
            "rep": rep,
            "const_decay": const_decay,
            "const_scaling": const_scaling,
            "MSE_loss": MSE_loss,
            "physics_loss": physics_loss,
            "sample_loss": sample_loss,
            "diverse_loss": diverse_loss,
            "reg_loss": reg_loss,
            "state_accuracy": np.mean(remapped_pred_state == gt_states[:-1]),
            # "moving_avg_accuracy": compute_states_mov_avg_corr(remapped_pred_state, gt_states),
            "moving_avg_accuracy_mean": np.mean(compute_states_mov_avg_corr(remapped_pred_state, gt_states))
        }
        all_rep_results.append(results_for_run)

        print(f"End of repetition {rep + 1}\n")

        # Plot states (ground truth vs. predicted)
        time_axis = np.arange(len(gt_states)+1) * dt
        offset = 0#1.5  # shift for y variable
        fig, axs = plt.subplots(4, 1, figsize=(20,8), gridspec_kw={'height_ratios': [1, 1, 0.5, 1]})
        unique_states = np.unique(gt_states)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))  # pastel colors
        state_colors = {state: color for state, color in zip(unique_states, colors)}

        remapped_pred_state = match_expert_to_state(pred_state, gt_states)

        # Create a 2D array for the background using pcolormesh
        background = np.zeros((1, len(time_axis)-1))  # Note: len-1
        for i, state in enumerate(gt_states[:-1]):  # Note: gt_states[:-1]
            background[0, i] = state
        axs[0].imshow(background, aspect='auto', cmap='viridis', zorder=0, alpha=0.6, extent=[0, len(time_axis)-1, -1, 3])       
        axs[0].plot(gt_series[:, 0], 'b-', label='Ground Truth x', zorder=10)
        axs[0].plot(gt_series[:, 1] + offset, 'b-', label='Ground Truth y (shifted)')
        axs[0].set_title('Ground Truth')
        axs[0].set_xticks([])
        plt.xlabel('Time')

        # Create a 2D array for the background using pcolormesh
        background = np.zeros((1, len(time_axis)-1))  # Note: len-1
        for i, state in enumerate(remapped_pred_state[:-1]):  # Note: gt_states[:-1]
            background[0, i] = state
        axs[1].imshow(background, aspect='auto', cmap='viridis', zorder=0, alpha=0.6, extent=[0, len(time_axis)-1, -1, 3])       
        axs[1].plot(pred_series_direct[:, 0], 'b-', label='Ground Truth x', zorder=10)
        axs[1].plot(pred_series_direct[:, 1] + offset, 'b-', label='Ground Truth y (shifted)')
        axs[1].set_title('Predictions')
        axs[1].set_xticks([])

        window_size = 100
        state_correctness = (remapped_pred_state == gt_states[:-1]).astype(int)
        # moving_avg_correctness = np.convolve(state_correctness, np.ones(window_size)/window_size, mode='valid')
        # axs[2].plot(state_correctness)
        axs[2].fill_between(np.arange(len(state_correctness)), state_correctness, step='mid', alpha=0.5)
        axs[2].set_title('State Tracking Accuracy (Moving Avg)')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Moving Avg Accuracy')
        axs[2].set_xlim(0, len(time_axis)-1)
        axs[2].set_ylim(0, 1.1)

        # 1. Plot x and y (ground truth vs. predicted)
        axs[3].plot(gt_series[:, 0], label='Ground Truth x')
        axs[3].plot(pred_series_direct[:, 0], label='Predicted x')
        axs[3].plot(gt_series[:, 1], label='Ground Truth y')
        axs[3].plot(pred_series_direct[:, 1], label='Predicted y')
        axs[3].set_title(f'Time Series Prediction (decay={const_decay}, scaling={const_scaling})')
        axs[3].set_xlim(0, len(time_axis)-1)
        plt.legend()
        
        sns.despine()
        plt.show()

#%%
results_df = pd.DataFrame(all_rep_results)
results_df.to_csv("/results/simulation_metrics.csv", index=False)

# plots
grouped = results_df.groupby(['const_decay','const_scaling'])
mean_values = grouped.mean().reset_index()
filtered_decay_5 = mean_values[mean_values["const_decay"] == 5]
filtered_decay_10 = mean_values[mean_values["const_decay"] == 10]
filtered_decay_15 = mean_values[mean_values["const_decay"] == 15]
filtered_decay_20 = mean_values[mean_values["const_decay"] == 20]

fig_width = 14
fig_height = 6

os.makedirs('/results/plots' , exist_ok=True)
#%%
fig, axs = plt.subplots(2,2, figsize=(fig_width,fig_height))
axs = axs.flatten()
# df2plot = mean_values.sort_values(by='const_scaling').reset_index()
sns.barplot (data=filtered_decay_5, x="const_scaling", y="MSE_loss", label="const_decay 5", ax=axs[0])
sns.barplot (data=filtered_decay_10, x="const_scaling", y="MSE_loss", label="const_decay 10", ax=axs[1])
sns.barplot (data=filtered_decay_15, x="const_scaling", y="MSE_loss", label="const_decay 15", ax=axs[2])
sns.barplot (x="const_scaling", y="MSE_loss", label="const_decay_20", data=filtered_decay_20, ax=axs[3])

plt.suptitle("MSE loss", fontsize=14)

sns.despine()
plt.savefig('/results/plots/mean_MSE_loss.png')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(fig_width,fig_height))
axs = axs.flatten()

sns.barplot (x="const_scaling", y="physics_loss", data=filtered_decay_5, label="const_decay 5", ax=axs[0])
sns.barplot (x="const_scaling", y="physics_loss", data=filtered_decay_10, label="const_decay 10", ax=axs[1])
sns.barplot (x="const_scaling", y="physics_loss", data=filtered_decay_15, label="const_decay 15", ax=axs[2])
sns.barplot (x="const_scaling", y="physics_loss", data=filtered_decay_20, label="const_decay_20", ax=axs[3])

plt.suptitle("Physics loss", fontsize=14)

sns.despine()
plt.savefig('/results/plots/mean_physics_loss.png')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(fig_width,fig_height))
axs = axs.flatten()

sns.barplot (x="const_scaling", y="gating_reg_loss", data=filtered_decay_5, label="const_decay 5", ax=axs[0])
sns.barplot (x="const_scaling", y="gating_reg_loss", data=filtered_decay_10, label="const_decay 10", ax=axs[1])
sns.barplot (x="const_scaling", y="gating_reg_loss", data=filtered_decay_15, label="const_decay 15", ax=axs[2])
sns.barplot (x="const_scaling", y="gating_reg_loss", data=filtered_decay_20, label="const_decay_20", ax=axs[3])

plt.suptitle("Gating regularization loss", fontsize=14)

sns.despine()
plt.savefig('/results/plots/mean_gating_reg_loss.png')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(fig_width,fig_height))
axs = axs.flatten()

sns.barplot (x="const_scaling", y="state_accuracy", data=filtered_decay_5, label="const_decay 5", ax=axs[0])
sns.barplot (x="const_scaling", y="state_accuracy", data=filtered_decay_10, label="const_decay 10", ax=axs[1])
sns.barplot (x="const_scaling", y="state_accuracy", data=filtered_decay_15, label="const_decay 15", ax=axs[2])
sns.barplot (x="const_scaling", y="state_accuracy", data=filtered_decay_20, label="const_decay_20", ax=axs[3])

plt.suptitle("State accuracy", fontsize=14)

sns.despine()
plt.savefig('/results/plots/mean_state_accuracy.png')
plt.show()

# plt.figure()
# sns.barplot (x="const_scaling", y="moving_avg_accuracy", data=mean_values)
# plt.savefig('/results/plots/mean_moving_avg_accuracy.png')

fig, axs = plt.subplots(2,2, figsize=(fig_width,fig_height))
axs = axs.flatten()

sns.barplot (x="const_scaling", y="moving_avg_accuracy_mean", data=filtered_decay_5, label="const_decay 5", ax=axs[0])
sns.barplot (x="const_scaling", y="moving_avg_accuracy_mean", data=filtered_decay_10, label="const_decay 10", ax=axs[1])
sns.barplot (x="const_scaling", y="moving_avg_accuracy_mean", data=filtered_decay_15, label="const_decay 15", ax=axs[2])
sns.barplot (x="const_scaling", y="moving_avg_accuracy_mean", data=filtered_decay_20, label="const_decay_20", ax=axs[3])

plt.suptitle("Moving avrage accuracy", fontsize=14)

sns.despine()
plt.savefig('/results/plots/mean_moving_avg_accuracy_mean.png')
plt.show()

#%%

#%%
# ------------------------------
# Training Setup
# ------------------------------

# Generate series and create dataset (using continuous dynamics here)
series_list = generate_time_series(n_series=20, series_length=10000, dt=0.001, use_slds=True)
dataset = TimeSeriesDataset(series_list, continuous_dynamics, dt=0.001)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

losses, x, model, gating_weights = train_on_data(series_list, dataset, dataloader)


#%% Plot basic metrics
# Plot Loss Evolution
plt.figure(figsize=(8, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training')
plt.legend()
# plt.grid(True)
sns.despine()
os.makedirs('/results/Figures' , exist_ok=True)
plt.savefig('/results/Figures/loss_'+name+'.pdf')
plt.show()

# Plot Gating Distributions
samples = torch.concat([x for x, target in dataloader])
with torch.no_grad():
    _, gating_weights = model(samples)
gating_weights_np = gating_weights.detach().numpy()

plt.figure(figsize=(8, 6))
for i in range(gating_weights_np.shape[1]):
    sns.kdeplot(gating_weights_np[:, i], label=f'Category {i}')

plt.xlabel('Gating Weight')
plt.ylabel('Density')
plt.title('Gating Network Output Distribution')
plt.legend(title='Category')
# plt.grid(True)
plt.xlim(0, 1)  # Bound the x-axis between 0 and 1
sns.despine()
plt.savefig('/results/Figures/Gating_weights_distributions_'+name+'.pdf')
plt.show()

#%%
# -------------------------------
# Testing: Time Series Prediction
# -------------------------------
pred_series_derivative, pred_series_direct, pred_state, gt_series, gt_states = test_on_data()
steps = 5000

#%
#%%
remapped_pred_state = match_expert_to_state(pred_state, gt_states)

# Compute state prediction accuracy
state_accuracy = np.mean(remapped_pred_state == gt_states[:-1])  # (steps) vs (steps+1), drop last gt

print(f"State Prediction Accuracy: {state_accuracy:.4f}")

# ------------------------------
# 1. Compute instant correctness
# ------------------------------
state_correctness = (remapped_pred_state == gt_states[:-1]).astype(int)  # 1 if correct, 0 if wrong

# -----------------------------
# 2. Compute tracking metric: moving average
# -----------------------------
window_size = 100  # you can tune this
moving_avg_correctness = np.convolve(state_correctness, np.ones(window_size)/window_size, mode='valid')

# -----------------------------
# 3. Find breaking points
# -----------------------------
breaking_threshold = 0.5  # 50% accuracy threshold
breaking_indices = np.where(moving_avg_correctness < breaking_threshold)[0]

if len(breaking_indices) > 0:
    first_break_time = breaking_indices[0]
    print(f"Tracking breaks at time step {first_break_time} (after {first_break_time * dt:.2f} seconds)")
else:
    print("Tracking remains stable above threshold throughout the sequence.")

# -----------------------------
# 4. Plot tracking accuracy over time
# -----------------------------
time_axis_tracking = np.arange(len(moving_avg_correctness)) * dt

plt.figure(figsize=(10, 5))
plt.plot(time_axis_tracking, moving_avg_correctness, label='Tracking Accuracy (Moving Avg)', color='blue')
# plt.axhline(y=breaking_threshold, color='red', linestyle='--', label=f'Breaking Threshold ({breaking_threshold*100:.0f}%)')
# if len(breaking_indices) > 0:
#     plt.axvline(x=first_break_time * dt, color='orange', linestyle='--', label='First Break Point')

plt.text(0.02, 0.95, f"State Accuracy: {state_accuracy:.4f}\nconstant_scaling: 0.0\nconstant_decay: 10", transform=plt.gca().transAxes, 
         fontsize=12, color='black', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('Time [s]')
plt.ylabel('Tracking Accuracy')
plt.title('State Tracking Accuracy Over Time - Testing')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/results/Figures/state_tracking_accuracy_testing.pdf')
plt.show()
#%

# #%%

#%
# ------------------------------
# Plot 2: Evolution of x and y Over Time (with y shifted)
# ------------------------------

time_axis = np.arange(steps+1) * dt
offset = 1.5  # shift for y variable

# Create a new figure with a subplot (using subplot index 2 as provided)
fig, axs = plt.subplots(2, 1, figsize=(10,8), gridspec_kw={'height_ratios': [1, 2]})

# Add background colors based on the state at each time step
unique_states = np.unique(gt_states)
# colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_states)))  # pastel colors
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))  # pastel colors
state_colors = {state: color for state, color in zip(unique_states, colors)}


# Debug - check if one state is dominating
np.unique(remapped_pred_state, return_counts=True)


for i in range(len(time_axis) - 1):
    axs[0].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[gt_states[i]], alpha=0.3)
    axs[1].axvspan(time_axis[i], time_axis[i + 1], color=state_colors[(remapped_pred_state[i])], alpha=0.3)

# Plot x and y (with y shifted) for ground truth and predictions
plt.plot(time_axis, gt_series[:steps+1, 0], 'b-', label='Ground Truth x')
plt.plot(time_axis[:-1], pred_series_direct[:, 0], 'r--', label='Predicted x')
plt.plot(time_axis[:-1], pred_series_derivative[:, 0], 'g--', label='Predicted x Derivative')

plt.plot(time_axis, gt_series[:steps+1, 1] + offset, 'b-', alpha=0.7, label='Ground Truth y (shifted)')
plt.plot(time_axis[:-1], pred_series_direct[:, 1] + offset, 'r--', alpha=0.7, label='Predicted y (shifted)')
plt.plot(time_axis[:-1], pred_series_derivative[:, 1] + offset, 'g--', alpha=0.7, label='Predicted y Derivative (shifted)')

plt.xlabel('Time')
plt.ylabel('x and y (shifted)')
plt.title('Evolution of x and y Over Time with State Background')
plt.legend()
# plt.grid()
sns.despine()
plt.savefig('/results/Figures/Variables_Temporal_evolution_'+name+'.pdf')

plt.tight_layout()
plt.show()


#%%
# ------------------------------
# Vector Field Plot: Dynamics & Expert Assignment
# ------------------------------

# # Create a grid over state space
# x_vals = np.linspace(-5, 5, 30)
# y_vals = np.linspace(-5, 5, 30)
# X, Y = np.meshgrid(x_vals, y_vals)
# grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

# # Compute ground truth dynamics (derivative) for each grid point
# U_gt = []
# V_gt = []
# for s in grid_points:
#     dynamics = switching_lds_step if use_slds else continuous_dynamics
#     f, _ = dynamics(s,state=0)
#     U_gt.append(f[0])
#     V_gt.append(f[1])
# U_gt = np.array(U_gt).reshape(X.shape)
# V_gt = np.array(V_gt).reshape(X.shape)

# # Model predictions on grid (using the first two output values as derivative)
# model.eval()
# with torch.no_grad():
#     grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
#     pred, gating = model(grid_tensor)
#     pred = pred.numpy()
#     pred_deriv = pred[:, 0:2]  # predicted derivative
#     U_pred = pred_deriv[:, 0].reshape(X.shape)
#     V_pred = pred_deriv[:, 1].reshape(X.shape)
#     expert_idx = torch.argmax(gating, dim=-1).numpy().reshape(X.shape)
#     expert_value = torch.max(gating, dim=-1)[0].numpy().reshape(X.shape)

# # For reference, the ground truth "next state" would be x + dt * f(x)
# U_gt_next = U_gt  # derivative remains the same for reference
# V_gt_next = V_gt

# # Plot vector fields and expert assignments
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Left subplot: Overlaid vector fields (ground truth in blue, predicted in red)
# ax1.quiver(X, Y, U_gt, V_gt, color='blue', label='Ground Truth Derivative')
# ax1.quiver(X, Y, U_pred, V_pred, color='red', label='Predicted Derivative')
# ax1.set_title('Dynamics: Ground Truth vs Predicted Derivative')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_xlim([-5, 5])
# ax1.set_ylim([-5, 5])
# ax1.legend()

# # Right subplot: Scatter of expert assignments with vector field overlay
# cmap = plt.get_cmap('viridis', 3)  # 3 experts
# sc = ax2.scatter(X, Y, c=expert_idx, cmap=cmap, s=100 * expert_value, alpha=0.5, marker='s', label='Expert Assignment')
# ax2.quiver(X, Y, U_pred, V_pred, color='red', label='Predicted Derivative')
# ax2.quiver(X, Y, U_gt, V_gt, color='blue', label='Ground Truth Derivative')
# ax2.set_title('Expert Assignments & Dynamics')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_xlim([-5, 5])
# ax2.set_ylim([-5, 5])
# fig.colorbar(sc, ax=ax2, ticks=[0, 1, 2], label='Expert Index')
# sns.despine()
# plt.savefig('/results/Figures/Vector_field.pdf')

# plt.tight_layout()
# plt.show()






# %%
