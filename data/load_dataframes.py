# %%
# !/usr/bin/env python3
"""
Simple script to load the saved dataframes from Data/Dataframes folder
"""

from pathlib import Path

import numpy as np
import pandas as pd

PATH=r"D:\TheFolder\projects\School\Master\Stefano\data"


def load_session_dataframes(animal_id="SB026", session_date="2019-10-11"):
    """
    Load all dataframes for a specific animal and session.

    Args:
        animal_id: Animal identifier (e.g., "SB026")
        session_date: Session date (e.g., "2019-10-11")

    Returns:
        Dictionary containing all loaded dataframes with names like df_name_dataframe
    """
    dataframes_dir = Path(PATH + "/Dataframes")

    if not dataframes_dir.exists():
        print(f"Dataframes directory not found: {dataframes_dir}")
        return None

    # Define the dataframe types and their variable names
    df_types = {
        "neural_activity": "neural_activity_dataframe",
        "neuron_metadata": "neuron_metadata_dataframe",
        "behavioral_data": "behavioral_data_dataframe",
        "population_features": "population_features_dataframe",
        "spatial_relationships": "spatial_relationships_dataframe",
        "cortical_state": "cortical_state_dataframe"
    }

    loaded_dataframes = {}

    for df_type, var_name in df_types.items():
        filename = f"{animal_id}_{session_date}_{df_type}.pkl"
        filepath = dataframes_dir / filename

        if filepath.exists():
            try:
                df = pd.read_pickle(filepath)
                loaded_dataframes[var_name] = df
                print(f"✓ Loaded {filename} as {var_name}: {df.shape}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ File not found: {filename}")

    return loaded_dataframes


def list_available_sessions():
    """List all available animal-session combinations."""
    dataframes_dir = Path(PATH + "/Dataframes")

    if not dataframes_dir.exists():
        print("Dataframes directory not found")
        return

    # Get all pkl files
    pkl_files = list(dataframes_dir.glob("*.pkl"))

    if not pkl_files:
        print("No dataframe files found")
        return

    # Extract unique animal-session combinations
    sessions = set()
    for file in pkl_files:
        if "_" in file.stem:
            parts = file.stem.split("_")
            if len(parts) >= 3:
                animal_id = parts[0]
                session_date = f"{parts[1]}_{parts[2]}"
                sessions.add((animal_id, session_date))

    print("Available sessions:")
    for animal_id, session_date in sorted(sessions):
        print(f"  {animal_id} - {session_date}")


def preprocess_dataframe(neural_activity_df, window_size=10, prediction_horizon=1):
    """
    Preprocess neural activity data for time series forecasting.

    Args:
        neural_activity_df: DataFrame with neural activity (time x neurons)
        window_size: Number of time steps to use as input context
        prediction_horizon: Number of time steps to predict ahead

    Returns:
        X: Input sequences (samples x window_size x neurons)
        y: Target sequences (samples x prediction_horizon x neurons)
        time_indices: Corresponding time indices
    """
    # Convert to numpy array
    activity_data = neural_activity_df.values  # Shape: (time_steps, neurons)

    # Normalize the data (z-score per neuron)
    activity_mean = np.mean(activity_data, axis=0, keepdims=True)
    activity_std = np.std(activity_data, axis=0, keepdims=True)
    activity_std[activity_std == 0] = 1  # Avoid division by zero
    activity_normalized = (activity_data - activity_mean) / activity_std

    # Create sliding window sequences
    X, y, time_indices = [], [], []

    for t in range(window_size, len(activity_normalized) - prediction_horizon + 1):
        # Input: window_size time steps
        x_seq = activity_normalized[t - window_size:t]
        # Target: next prediction_horizon time steps
        y_seq = activity_normalized[t:t + prediction_horizon]

        X.append(x_seq)
        y.append(y_seq)
        time_indices.append(t)

    return np.array(X), np.array(y), np.array(time_indices), activity_mean, activity_std
# %%
if __name__ == "__main__":
    list_available_sessions()
    print()

    # Load data for the first session (SB026, 2019-10-11)
    print("Loading dataframes for SB026 - 2019-10-11:")
    dataframes = load_session_dataframes("SB026", "2019-10-11")
# %%
