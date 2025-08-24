"""
Data Loader for MOE Neural Data Project
Adapted from NeuPRINT data loading patterns
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOEDataLoader:
    """
    Data loader class for MOE Neural Data project.
    Loads and preprocesses neural activity, behavioral data, and metadata.
    """

    def __init__(self, data_directory: str = "Data"):
        """
        Initialize the data loader.

        Args:
            data_directory: Path to the data directory
        """
        self.data_directory = Path(data_directory)
        self.data_sessions = []

        # Check if data directory exists
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory {data_directory} not found")

        logger.info(f"Initialized data loader with directory: {self.data_directory}")

    def discover_data_structure(self) -> Dict:
        """
        Discover the structure of the data directory.

        Returns:
            Dictionary describing the data structure
        """
        structure = {
            'data_directory': str(self.data_directory),
            'subdirectories': [],
            'files': [],
            'data_types': set()
        }

        # List all subdirectories
        for item in self.data_directory.iterdir():
            if item.is_dir():
                structure['subdirectories'].append(item.name)
                # Look for data files in subdirectories
                for data_file in item.glob("*.npy"):
                    structure['files'].append(str(data_file.relative_to(self.data_directory)))
                    structure['data_types'].add(data_file.suffix)
                for data_file in item.glob("*.txt"):
                    structure['files'].append(str(data_file.relative_to(self.data_directory)))
                    structure['data_types'].add(data_file.suffix)
                for data_file in item.glob("*.csv"):
                    structure['files'].append(str(data_file.relative_to(self.data_directory)))
                    structure['data_types'].add(data_file.suffix)
            elif item.is_file():
                structure['files'].append(item.name)
                structure['data_types'].add(item.suffix)

        structure['data_types'] = list(structure['data_types'])
        return structure

    def load_neural_activity(self, file_path: str) -> np.ndarray:
        """
        Load neural activity data from .npy file.

        Args:
            file_path: Path to the neural activity file

        Returns:
            Neural activity array (time x neurons)
        """
        try:
            data = np.load(file_path)
            logger.info(f"Loaded neural activity from {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading neural activity from {file_path}: {e}")
            return None

    def load_behavioral_data(self, file_path: str) -> np.ndarray:
        """
        Load behavioral data (eye size, running speed, etc.).

        Args:
            file_path: Path to the behavioral data file

        Returns:
            Behavioral data array
        """
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path).values
            else:
                logger.warning(f"Unsupported file format for behavioral data: {file_path}")
                return None

            logger.info(f"Loaded behavioral data from {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading behavioral data from {file_path}: {e}")
            return None

    def load_metadata(self, file_path: str) -> Dict:
        """
        Load metadata files (neuron types, positions, gene counts, etc.).

        Args:
            file_path: Path to the metadata file

        Returns:
            Dictionary containing metadata
        """
        metadata = {}

        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path)
                metadata['data'] = data
                metadata['type'] = 'numpy_array'
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    metadata['data'] = [line.strip() for line in lines]
                    metadata['type'] = 'text_list'
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                metadata['data'] = data
                metadata['type'] = 'pandas_dataframe'
            else:
                logger.warning(f"Unsupported file format for metadata: {file_path}")
                return None

            logger.info(f"Loaded metadata from {file_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata from {file_path}: {e}")
            return None

    def normalize_activity(self, activity: np.ndarray, method: str = 'zscore', axis: int = 0) -> np.ndarray:
        """
        Normalize neural activity data.

        Args:
            activity: Raw neural activity array
            method: Normalization method ('zscore', 'minmax', 'none')
            axis: Axis along which to normalize (0 for time, 1 for neurons)

        Returns:
            Normalized activity array
        """
        if method == 'none':
            return activity

        if method == 'zscore':
            mean_val = np.mean(activity, axis=axis, keepdims=True)
            std_val = np.std(activity, axis=axis, keepdims=True)
            # Avoid division by zero
            std_val = np.where(std_val == 0, 1.0, std_val)
            normalized = (activity - mean_val) / std_val
        elif method == 'minmax':
            min_val = np.min(activity, axis=axis, keepdims=True)
            max_val = np.max(activity, axis=axis, keepdims=True)
            normalized = (activity - min_val) / (max_val - min_val + 1e-8)
        else:
            logger.warning(f"Unknown normalization method: {method}, returning original data")
            return activity

        # Replace NaN values with 0
        normalized = np.where(np.isnan(normalized), 0.0, normalized)

        logger.info(f"Normalized activity using {method} method")
        return normalized

    def find_neighbors(self, positions: np.ndarray, distance_threshold: float = 65.0) -> Tuple[List, List]:
        """
        Find spatial neighbors for each neuron based on positions.

        Args:
            positions: Array of neuron positions (N x 2 or N x 3)
            distance_threshold: Maximum distance for neighbors

        Returns:
            Tuple of (neighbors_list, non_neighbors_list)
        """
        from sklearn.neighbors import NearestNeighbors

        try:
            # Initialize nearest neighbors
            nbrs = NearestNeighbors(radius=distance_threshold, algorithm='auto').fit(positions)

            # Find neighbors within radius
            neighbors = nbrs.radius_neighbors(positions, return_distance=False)

            # Convert to lists and exclude self
            neighbors_list = []
            non_neighbors_list = []

            for i, neighbor_indices in enumerate(neighbors):
                # Exclude self from neighbors
                filtered_neighbors = [idx for idx in neighbor_indices if idx != i]
                neighbors_list.append(filtered_neighbors)

                # Find non-neighbors
                all_indices = set(range(len(positions)))
                all_indices.remove(i)
                non_neighbors = list(all_indices - set(filtered_neighbors))
                non_neighbors_list.append(non_neighbors)

            logger.info(f"Found neighbors for {len(positions)} neurons with threshold {distance_threshold}")
            return neighbors_list, non_neighbors_list

        except ImportError:
            logger.error("scikit-learn not available for neighbor finding")
            return [], []
        except Exception as e:
            logger.error(f"Error finding neighbors: {e}")
            return [], []

    def create_session_data(self, session_path: str) -> Dict:
        """
        Create a session data dictionary similar to NeuPRINT format.
        Adapted for Bugeon fake_dataset structure.

        Args:
            session_path: Path to session data

        Returns:
            Dictionary containing session data
        """
        session_data = {
            'session_path': session_path,
            'activity_norm': None,
            'activity_raw': None,  # Add raw activity data
            'activity_population': {},
            'frame_times': None,
            'unique_ids': None,
            'neuron_types': None,
            'neuron_pos': None,
            'activity_neighbor': None
        }

        session_dir = Path(session_path)

        # For Bugeon fake_dataset, we need to look deeper into the structure
        # Look for date subdirectories (e.g., 2019-10-16)
        date_dirs = []
        for item in session_dir.iterdir():
            if item.is_dir() and item.name.replace('-', '').replace('.', '').isdigit():
                date_dirs.append(item)

        if not date_dirs:
            # If no date directories, try to load from current level
            date_dirs = [session_dir]

        # Process each date directory
        for date_dir in date_dirs:
            logger.info(f"Processing date directory: {date_dir.name}")

            # Look for condition directories (Blank, Natural Scenes, Drifting Gratings)
            condition_dirs = []
            for item in date_dir.iterdir():
                if item.is_dir() and item.name in ['Blank', 'Natural Scenes', 'Drifting Gratings']:
                    condition_dirs.append(item)

            if not condition_dirs:
                # If no condition directories, try to load from date level
                condition_dirs = [date_dir]

            for condition_dir in condition_dirs:
                # Look for subdirectories (e.g., 01, 02)
                sub_dirs = []
                for item in condition_dir.iterdir():
                    if item.is_dir() and item.name.isdigit():
                        sub_dirs.append(item)

                if not sub_dirs:
                    # If no subdirectories, try to load from condition level
                    sub_dirs = [condition_dir]

                for sub_dir in sub_dirs:
                    logger.info(f"Processing subdirectory: {sub_dir.name}")

                    # Try to load different types of data files
                    for file_path in sub_dir.glob("*"):
                        if file_path.is_file():
                            file_name = file_path.name.lower()

                            # Neural activity
                            if 'neuralactivity' in file_name:
                                activity = self.load_neural_activity(str(file_path))
                                if activity is not None:
                                    session_data['activity_norm'] = self.normalize_activity(activity)
                                    session_data['activity_raw'] = activity  # Store raw activity
                                    logger.info(f"Loaded neural activity: {activity.shape}")

                            # Frame times
                            elif 'frame.times' in file_name:
                                frame_times = self.load_metadata(str(file_path))
                                if frame_times is not None:
                                    session_data['frame_times'] = frame_times['data']

                            # Behavioral data
                            elif any(keyword in file_name for keyword in
                                     ['eye.size', 'eye.xpos', 'eye.ypos', 'running.speed']):
                                behavioral_data = self.load_behavioral_data(str(file_path))
                                if behavioral_data is not None:
                                    # Extract a better key name from the filename
                                    if 'eye.size' in file_name:
                                        key = 'eye_size'
                                    elif 'eye.xpos' in file_name:
                                        key = 'eye_xpos'
                                    elif 'eye.ypos' in file_name:
                                        key = 'eye_ypos'
                                    elif 'running.speed' in file_name:
                                        key = 'running_speed'
                                    else:
                                        key = file_name.split('.')[-1]  # Fallback
                                    session_data['activity_population'][key] = behavioral_data

                            # Frame states
                            elif 'frame.states' in file_name:
                                frame_states = self.load_metadata(str(file_path))
                                if frame_states is not None:
                                    session_data['activity_population']['frame_states'] = frame_states['data']

                    # Load metadata from the date directory level
                    for file_path in date_dir.glob("*"):
                        if file_path.is_file():
                            file_name = file_path.name.lower()

                            # Neuron metadata
                            if 'neuron.stackposcorrected' in file_name:
                                metadata = self.load_metadata(str(file_path))
                                if metadata is not None:
                                    session_data['neuron_pos'] = metadata['data']
                                    logger.info(f"Loaded neuron positions: {metadata['data'].shape}")

                            elif 'neuron.ttype' in file_name:
                                metadata = self.load_metadata(str(file_path))
                                if metadata is not None:
                                    session_data['neuron_types'] = metadata['data']
                                    logger.info(f"Loaded neuron types: {len(metadata['data'])}")

                            elif 'neuron.uniqueid' in file_name:
                                metadata = self.load_metadata(str(file_path))
                                if metadata is not None:
                                    session_data['unique_ids'] = metadata['data']
                                    logger.info(f"Loaded unique IDs: {metadata['data'].shape}")

                    # If we found activity data, break out of the loops
                    if session_data['activity_norm'] is not None:
                        break

                if session_data['activity_norm'] is not None:
                    break

            if session_data['activity_norm'] is not None:
                break

        # Calculate population-level features if we have activity data
        if session_data['activity_norm'] is not None:
            activity = session_data['activity_norm']
            session_data['activity_population']['mean_activity'] = np.mean(activity, axis=1, keepdims=True)
            session_data['activity_population']['std_activity'] = np.std(activity, axis=1, keepdims=True)

            # Calculate neighbor activity if we have positions
            if session_data['neuron_pos'] is not None:
                neighbors, _ = self.find_neighbors(session_data['neuron_pos'])
                if neighbors:
                    num_neurons = activity.shape[1]
                    F_neighbor = np.zeros((activity.shape[0], num_neurons, 2))

                    for j in range(num_neurons):
                        if len(neighbors[j]) > 0:
                            neighbor_activity = activity[:, neighbors[j]]
                            F_neighbor[:, j, 0] = np.mean(neighbor_activity, axis=1)  # Mean
                            F_neighbor[:, j, 1] = np.std(neighbor_activity, axis=1)  # Std

                    session_data['activity_neighbor'] = F_neighbor

        return session_data

    def load_all_sessions(self) -> List[Dict]:
        """
        Load all available sessions from the data directory.
        Recursively finds all date-based sessions within each animal folder.

        Returns:
            List of session data dictionaries
        """
        sessions = []

        # Look for animal directories
        for animal_dir in self.data_directory.iterdir():
            if animal_dir.is_dir():
                logger.info(f"Processing animal directory: {animal_dir.name}")

                # Look for date subdirectories within each animal folder
                for date_dir in animal_dir.iterdir():
                    if date_dir.is_dir() and self._is_date_directory(date_dir.name):
                        logger.info(f"  Processing date directory: {date_dir.name}")

                        # Create session data for this date
                        session_data = self.create_session_data(str(date_dir))
                        if session_data['activity_norm'] is not None:  # Only add if we have activity data
                            # Add metadata about the animal and date
                            session_data['animal_id'] = animal_dir.name
                            session_data['session_date'] = date_dir.name
                            session_data['session_path'] = str(date_dir)  # Update path to point to date directory

                            sessions.append(session_data)
                            logger.info(f"  Added session: {animal_dir.name}_{date_dir.name}")

        self.data_sessions = sessions
        logger.info(f"Loaded {len(sessions)} sessions")
        return sessions

    def get_data_summary(self) -> Dict:
        """
        Get a summary of the loaded data.

        Returns:
            Dictionary containing data summary
        """
        if not self.data_sessions:
            return {
                "total_sessions": 0,
                "sessions": [],
                "message": "No sessions loaded. Check data structure and file naming."
            }

        summary = {
            "total_sessions": len(self.data_sessions),
            "sessions": []
        }

        for i, session in enumerate(self.data_sessions):
            session_summary = {
                "session_id": i,
                "path": session['session_path'],
                "activity_shape": session['activity_norm'].shape if session['activity_norm'] is not None else None,
                "population_features": list(session['activity_population'].keys()) if session[
                    'activity_population'] else [],
                "has_positions": session['neuron_pos'] is not None,
                "has_types": session['neuron_types'] is not None,
                "has_neighbors": session['activity_neighbor'] is not None
            }
            summary["sessions"].append(session_summary)

        return summary

    def organize_into_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Organize the loaded data into pandas DataFrames for easier analysis.

        Returns:
            Dictionary containing organized DataFrames:
            - 'neural_activity': Time x Neurons activity matrix
            - 'neuron_metadata': Neuron properties (positions, types, IDs)
            - 'behavioral_data': Time series behavioral measurements
            - 'population_features': Population-level features over time
            - 'spatial_relationships': Neuron spatial relationships
        """
        if not self.data_sessions:
            logger.warning("No sessions loaded. Load sessions first.")
            return {}

        # Initialize DataFrames
        neural_activity_dfs = []
        neuron_metadata_dfs = []
        behavioral_data_dfs = []
        population_features_dfs = []
        spatial_relationships_dfs = []

        for session_idx, session in enumerate(self.data_sessions):
            session_name = Path(session['session_path']).name

            # 1. Neural Activity DataFrame
            if session['activity_raw'] is not None:
                # Create time index
                time_steps = session['activity_raw'].shape[0]
                time_index = pd.RangeIndex(0, time_steps, name='time_step')

                # Create neuron columns
                neuron_cols = [f'neuron_{i:03d}' for i in range(session['activity_raw'].shape[1])]

                # Create DataFrame
                activity_df = pd.DataFrame(
                    session['activity_raw'],
                    index=time_index,
                    columns=neuron_cols
                )
                activity_df['session'] = session_name
                activity_df['session_id'] = session_idx
                activity_df.set_index(['session', 'session_id'], append=True, inplace=True)
                neural_activity_dfs.append(activity_df)

            # 2. Neuron Metadata DataFrame
            if session['neuron_pos'] is not None and session['neuron_types'] is not None:
                num_neurons = len(session['neuron_types'])
                neuron_ids = [f'neuron_{i:03d}' for i in range(num_neurons)]

                metadata_data = {
                    'neuron_id': neuron_ids,
                    'session': [session_name] * num_neurons,
                    'session_id': [session_idx] * num_neurons,
                    'neuron_type': session['neuron_types'],
                    'x_position': session['neuron_pos'][:, 0],
                    'y_position': session['neuron_pos'][:, 1]
                }

                # Add unique IDs if available
                if session['unique_ids'] is not None:
                    metadata_data['unique_id'] = session['unique_ids'].flatten()

                metadata_df = pd.DataFrame(metadata_data)
                metadata_df.set_index(['session', 'session_id', 'neuron_id'], inplace=True)
                neuron_metadata_dfs.append(metadata_df)

            # 3. Behavioral Data DataFrame
            if session['activity_population']:
                # Get time steps from neural activity
                time_steps = session['activity_raw'].shape[0] if session['activity_raw'] is not None else 1000
                time_index = pd.RangeIndex(0, time_steps, name='time_step')

                behavioral_data = {}
                for key, data in session['activity_population'].items():
                    if key not in ['mean_activity', 'std_activity', 'frame_states'] and data is not None:
                        # Resize data to match time steps if necessary
                        if len(data) != time_steps:
                            # Simple interpolation/resampling
                            if len(data) > time_steps:
                                # Downsample
                                indices = np.linspace(0, len(data) - 1, time_steps, dtype=int)
                                behavioral_data[key] = data[indices].flatten()
                            else:
                                # Upsample by repeating last value
                                resized = np.zeros((time_steps, data.shape[1] if len(data.shape) > 1 else 1))
                                resized[:len(data)] = data
                                resized[len(data):] = data[-1] if len(data) > 0 else 0
                                behavioral_data[key] = resized.flatten()
                        else:
                            behavioral_data[key] = data.flatten()

                if behavioral_data:
                    behavioral_data['session'] = [session_name] * time_steps
                    behavioral_data['session_id'] = [session_idx] * time_steps

                    behavioral_df = pd.DataFrame(behavioral_data, index=time_index)
                    behavioral_df.set_index(['session', 'session_id'], append=True, inplace=True)
                    behavioral_data_dfs.append(behavioral_df)

            # 4. Population Features DataFrame
            if session['activity_population'] and 'mean_activity' in session['activity_population']:
                time_steps = session['activity_raw'].shape[0] if session['activity_raw'] is not None else 1000
                time_index = pd.RangeIndex(0, time_steps, name='time_step')

                pop_features = {
                    'mean_activity': session['activity_population']['mean_activity'].flatten(),
                    'std_activity': session['activity_population']['std_activity'].flatten(),
                    'session': [session_name] * time_steps,
                    'session_id': [session_idx] * time_steps
                }

                # Add frame states if available
                if 'frame_states' in session['activity_population']:
                    frame_states = session['activity_population']['frame_states']
                    if frame_states is not None:
                        if len(frame_states) == time_steps:
                            pop_features['frame_states'] = frame_states.flatten()
                        else:
                            # Resize frame states
                            if len(frame_states) > time_steps:
                                indices = np.linspace(0, len(frame_states) - 1, time_steps, dtype=int)
                                pop_features['frame_states'] = frame_states[indices].flatten()
                            else:
                                resized = np.zeros(time_steps)
                                resized[:len(frame_states)] = frame_states.flatten()
                                resized[len(frame_states):] = frame_states[-1] if len(frame_states) > 0 else 0
                                pop_features['frame_states'] = resized

                pop_df = pd.DataFrame(pop_features, index=time_index)
                pop_df.set_index(['session', 'session_id'], append=True, inplace=True)
                population_features_dfs.append(pop_df)

            # 5. Spatial Relationships DataFrame
            if session['activity_neighbor'] is not None:
                num_neurons = session['activity_neighbor'].shape[1]
                time_steps = session['activity_neighbor'].shape[0]

                spatial_data = []
                for t in range(time_steps):
                    for n in range(num_neurons):
                        spatial_data.append({
                            'time_step': t,
                            'neuron_id': f'neuron_{n:03d}',
                            'session': session_name,
                            'session_id': session_idx,
                            'neighbor_mean_activity': session['activity_neighbor'][t, n, 0],
                            'neighbor_std_activity': session['activity_neighbor'][t, n, 1]
                        })

                if spatial_data:
                    spatial_df = pd.DataFrame(spatial_data)
                    spatial_df.set_index(['session', 'session_id', 'time_step', 'neuron_id'], inplace=True)
                    spatial_relationships_dfs.append(spatial_df)

        # Combine all DataFrames
        organized_data = {}

        if neural_activity_dfs:
            organized_data['neural_activity'] = pd.concat(neural_activity_dfs, axis=0)
            logger.info(f"Created neural activity DataFrame: {organized_data['neural_activity'].shape}")

        if neuron_metadata_dfs:
            organized_data['neuron_metadata'] = pd.concat(neuron_metadata_dfs, axis=0)
            logger.info(f"Created neuron metadata DataFrame: {organized_data['neuron_metadata'].shape}")

        if behavioral_data_dfs:
            organized_data['behavioral_data'] = pd.concat(behavioral_data_dfs, axis=0)
            logger.info(f"Created behavioral data DataFrame: {organized_data['behavioral_data'].shape}")

        if population_features_dfs:
            organized_data['population_features'] = pd.concat(population_features_dfs, axis=0)
            logger.info(f"Created population features DataFrame: {organized_data['population_features'].shape}")

        if spatial_relationships_dfs:
            organized_data['spatial_relationships'] = pd.concat(spatial_relationships_dfs, axis=0)
            logger.info(f"Created spatial relationships DataFrame: {organized_data['spatial_relationships'].shape}")

        return organized_data

    def get_dataframe_summary(self) -> Dict:
        """
        Get a summary of the organized DataFrames.

        Returns:
            Dictionary containing DataFrame summaries
        """
        dataframes = self.organize_into_dataframes()

        if not dataframes:
            return {"message": "No DataFrames created. Load sessions first."}

        summary = {}
        for name, df in dataframes.items():
            summary[name] = {
                'shape': df.shape,
                'index_levels': df.index.names,
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': df.dtypes.to_dict()
            }

        return summary

    def save_dataframes_to_files(self, output_dir: str = "organized_data", formats: List[str] = None):
        """
        Save the organized DataFrames to files in various formats.

        Args:
            output_dir: Directory to save the files
            formats: List of formats to save ('csv', 'parquet', 'h5', 'pickle')
        """
        if formats is None:
            formats = ['csv', 'parquet']

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get organized data
        dataframes = self.organize_into_dataframes()

        if not dataframes:
            logger.warning("No DataFrames to save. Load sessions first.")
            return

        saved_files = []

        for name, df in dataframes.items():
            logger.info(f"Saving {name} DataFrame...")

            for fmt in formats:
                try:
                    if fmt == 'csv':
                        # For CSV, we need to handle multi-index
                        filename = output_path / f"{name}.csv"
                        df.to_csv(filename)
                        saved_files.append(str(filename))

                    elif fmt == 'parquet':
                        filename = output_path / f"{name}.parquet"
                        df.to_parquet(filename, engine='pyarrow')
                        saved_files.append(str(filename))

                    elif fmt == 'h5':
                        filename = output_path / f"{name}.h5"
                        df.to_hdf(filename, key=name, mode='w')
                        saved_files.append(str(filename))

                    elif fmt == 'pickle':
                        filename = output_path / f"{name}.pkl"
                        df.to_pickle(filename)
                        saved_files.append(str(filename))

                    logger.info(f"  Saved {name}.{fmt}")

                except Exception as e:
                    logger.error(f"  Error saving {name}.{fmt}: {e}")

        # Save a summary file
        summary = self.get_dataframe_summary()
        summary_file = output_path / "dataframe_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("DataFrame Summary\n")
            f.write("================\n\n")

            for name, info in summary.items():
                f.write(f"{name.upper()}:\n")
                f.write(f"  Shape: {info['shape']}\n")
                f.write(f"  Index levels: {info['index_levels']}\n")
                f.write(f"  Memory usage: {info['memory_usage_mb']:.2f} MB\n")
                f.write(f"  Columns: {info['columns']}\n")
                f.write(f"  Dtypes: {info['dtypes']}\n\n")

        saved_files.append(str(summary_file))
        logger.info(f"Saved {len(saved_files)} files to {output_dir}")

        return saved_files

    def save_session_dataframes(self, output_dir: str = "Data/Dataframes", formats: List[str] = None):
        """
        Save DataFrames for each animal and session separately.

        Args:
            output_dir: Directory to save the files
            formats: List of formats to save ('pkl', 'csv', 'parquet', 'h5')
        """
        if formats is None:
            formats = ['pkl']  # Default to pickle format

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.data_sessions:
            logger.warning("No sessions loaded. Load sessions first.")
            return {}

        saved_files = {}

        for session_idx, session in enumerate(self.data_sessions):
            # Extract animal ID and session date from session data
            animal_id = session.get('animal_id', f'animal_{session_idx}')
            session_date = session.get('session_date', f'session_{session_idx:02d}')

            # Create session-specific dataframes
            session_dataframes = self._create_session_dataframes(session, session_idx, animal_id)

            if not session_dataframes:
                logger.warning(f"No dataframes created for session {animal_id}")
                continue

            # Save each dataframe for this session
            session_files = []
            for df_name, df in session_dataframes.items():
                if df is not None and not df.empty:
                    for fmt in formats:
                        try:
                            if fmt == 'pkl':
                                filename = output_path / f"{animal_id}_{session_date}_{df_name}.pkl"
                                df.to_pickle(filename)
                                session_files.append(str(filename))

                            elif fmt == 'csv':
                                filename = output_path / f"{animal_id}_{session_date}_{df_name}.csv"
                                df.to_csv(filename)
                                session_files.append(str(filename))

                            elif fmt == 'parquet':
                                filename = output_path / f"{animal_id}_{session_date}_{df_name}.parquet"
                                df.to_parquet(filename, engine='pyarrow')
                                session_files.append(str(filename))

                            elif fmt == 'h5':
                                filename = output_path / f"{animal_id}_{session_date}_{df_name}.h5"
                                df.to_hdf(filename, key=df_name, mode='w')
                                session_files.append(str(filename))

                            logger.info(f"  Saved {animal_id}_{session_date}_{df_name}.{fmt}")

                        except Exception as e:
                            logger.error(f"  Error saving {animal_id}_{session_date}_{df_name}.{fmt}: {e}")

            saved_files[f"{animal_id}_{session_date}"] = session_files

        # Save a summary file
        summary_file = output_path / "session_dataframe_summary.txt"
        self._save_session_summary(summary_file, saved_files)

        logger.info(f"Saved DataFrames for {len(saved_files)} sessions to {output_dir}")
        return saved_files

    def _create_session_dataframes(self, session: Dict, session_idx: int, animal_id: str) -> Dict[str, pd.DataFrame]:
        """
        Create DataFrames for a specific session.

        Args:
            session: Session data dictionary
            session_idx: Session index
            animal_id: Animal ID

        Returns:
            Dictionary of DataFrames for this session
        """
        # Get session date for indexing
        session_date = session.get('session_date', f'session_{session_idx:02d}')
        session_dataframes = {}

        # 1. Neural Activity DataFrame
        if session['activity_raw'] is not None:
            time_steps = session['activity_raw'].shape[0]
            time_index = pd.RangeIndex(0, time_steps, name='time_step')

            # Create neuron columns
            neuron_cols = [f'neuron_{i:03d}' for i in range(session['activity_raw'].shape[1])]

            # Create DataFrame
            activity_df = pd.DataFrame(
                session['activity_raw'],
                index=time_index,
                columns=neuron_cols
            )
            activity_df['session'] = animal_id
            activity_df['session_id'] = session_idx
            activity_df.set_index(['session', 'session_id'], append=True, inplace=True)
            session_dataframes['neural_activity'] = activity_df

        # 2. Neuron Metadata DataFrame
        if session['neuron_pos'] is not None and session['neuron_types'] is not None:
            num_neurons = len(session['neuron_types'])
            neuron_ids = [f'neuron_{i:03d}' for i in range(num_neurons)]

            # Extract anatomical information for this session
            layers, major_cell_types = self.extract_anatomical_info(session['neuron_types'])

            metadata_data = {
                'neuron_id': neuron_ids,
                'session': [animal_id] * num_neurons,
                'session_id': [session_idx] * num_neurons,
                'neuron_type': session['neuron_types'],  # Keep original full name
                'major_cell_type': major_cell_types,  # Add major cell type (VIP, SST, Pvalb, etc.)
                'layer': layers,  # Add layer information (L2/3, L5, etc.)
                'x_position': session['neuron_pos'][:, 0],
                'y_position': session['neuron_pos'][:, 1]
            }

            # Add unique IDs if available
            if session['unique_ids'] is not None:
                metadata_data['unique_id'] = session['unique_ids'].flatten()

            metadata_df = pd.DataFrame(metadata_data)
            metadata_df.set_index(['session', 'session_id', 'neuron_id'], inplace=True)
            session_dataframes['neuron_metadata'] = metadata_df

        # 2.5. Cortical State DataFrame
        if session['activity_population'] and 'frame_states' in session['activity_population']:
            frame_states = session['activity_population']['frame_states']
            if frame_states is not None:
                time_steps = len(frame_states)
                time_index = pd.RangeIndex(0, time_steps, name='time_step')

                # Create cortical state DataFrame
                cortical_state_data = {
                    'cortical_state': frame_states.flatten(),
                    'session': [animal_id] * time_steps,
                    'session_id': [session_idx] * time_steps
                }

                cortical_state_df = pd.DataFrame(cortical_state_data, index=time_index)
                cortical_state_df.set_index(['session', 'session_id'], append=True, inplace=True)
                session_dataframes['cortical_state'] = cortical_state_df

        # 3. Behavioral Data DataFrame
        if session['activity_population']:
            time_steps = session['activity_raw'].shape[0] if session['activity_raw'] is not None else 1000
            time_index = pd.RangeIndex(0, time_steps, name='time_step')

            behavioral_data = {}
            for key, data in session['activity_population'].items():
                if key not in ['mean_activity', 'std_activity', 'frame_states'] and data is not None:
                    # Resize data to match time steps if necessary
                    if len(data) != time_steps:
                        if len(data) > time_steps:
                            indices = np.linspace(0, len(data) - 1, time_steps, dtype=int)
                            behavioral_data[key] = data[indices].flatten()
                        else:
                            resized = np.zeros((time_steps, data.shape[1] if len(data.shape) > 1 else 1))
                            resized[:len(data)] = data
                            resized[len(data):] = data[-1] if len(data) > 0 else 0
                            behavioral_data[key] = resized.flatten()
                    else:
                        behavioral_data[key] = data.flatten()

            if behavioral_data:
                behavioral_data['session'] = [animal_id] * time_steps
                behavioral_data['session_id'] = [session_idx] * time_steps

                behavioral_df = pd.DataFrame(behavioral_data, index=time_index)
                behavioral_df.set_index(['session', 'session_id'], append=True, inplace=True)
                session_dataframes['behavioral_data'] = behavioral_df

        # 4. Population Features DataFrame
        if session['activity_population'] and 'mean_activity' in session['activity_population']:
            time_steps = session['activity_raw'].shape[0] if session['activity_raw'] is not None else 1000
            time_index = pd.RangeIndex(0, time_steps, name='time_step')

            pop_features = {
                'mean_activity': session['activity_population']['mean_activity'].flatten(),
                'std_activity': session['activity_population']['std_activity'].flatten(),
                'session': [animal_id] * time_steps,
                'session_id': [session_idx] * time_steps
            }

            # Add frame states if available
            if 'frame_states' in session['activity_population']:
                frame_states = session['activity_population']['frame_states']
                if frame_states is not None:
                    if len(frame_states) == time_steps:
                        pop_features['frame_states'] = frame_states.flatten()
                    else:
                        if len(frame_states) > time_steps:
                            indices = np.linspace(0, len(frame_states) - 1, time_steps, dtype=int)
                            pop_features['frame_states'] = frame_states[indices].flatten()
                        else:
                            resized = np.zeros(time_steps)
                            resized[:len(frame_states)] = frame_states.flatten()
                            resized[len(frame_states):] = frame_states[-1] if len(frame_states) > 0 else 0
                            pop_features['frame_states'] = resized

            pop_df = pd.DataFrame(pop_features, index=time_index)
            pop_df.set_index(['session', 'session_id'], append=True, inplace=True)
            session_dataframes['population_features'] = pop_df

        # 5. Spatial Relationships DataFrame
        if session['activity_neighbor'] is not None:
            num_neurons = session['activity_neighbor'].shape[1]
            time_steps = session['activity_neighbor'].shape[0]

            spatial_data = []
            for t in range(time_steps):
                for n in range(num_neurons):
                    spatial_data.append({
                        'time_step': t,
                        'neuron_id': f'neuron_{n:03d}',
                        'session': animal_id,
                        'session_id': session_idx,
                        'neighbor_mean_activity': session['activity_neighbor'][t, n, 0],
                        'neighbor_std_activity': session['activity_neighbor'][t, n, 1]
                    })

            if spatial_data:
                spatial_df = pd.DataFrame(spatial_data)
                spatial_df.set_index(['session', 'session_id', 'time_step', 'neuron_id'], inplace=True)
                session_dataframes['spatial_relationships'] = spatial_df

        # 6. Enhanced Metadata DataFrame (with anatomical information)
        if 'neuron_metadata' in session_dataframes:
            enhanced_metadata = self._enhance_session_metadata(session_dataframes['neuron_metadata'], session_idx,
                                                               animal_id)
            if enhanced_metadata is not None:
                session_dataframes['enhanced_metadata'] = enhanced_metadata

        return session_dataframes

    def _enhance_session_metadata(self, metadata_df: pd.DataFrame, session_idx: int, animal_id: str) -> pd.DataFrame:
        """
        Enhance session metadata with anatomical information.

        Args:
            metadata_df: Basic metadata DataFrame
            session_idx: Session index
            animal_id: Animal ID

        Returns:
            Enhanced metadata DataFrame
        """
        try:
            # Reset index to work with the data
            metadata_reset = metadata_df.reset_index()

            # Extract anatomical information
            layers, categories = self.extract_anatomical_info(metadata_reset['neuron_type'].tolist())

            # Add new columns
            enhanced_metadata = metadata_reset.copy()
            enhanced_metadata['layer'] = layers
            enhanced_metadata['cell_type_category'] = categories

            # Set index back
            enhanced_metadata.set_index(['session', 'session_id', 'neuron_id'], inplace=True)

            return enhanced_metadata

        except Exception as e:
            logger.error(f"Error enhancing metadata for {animal_id}_session_{session_idx:02d}: {e}")
            return None

    def _save_session_summary(self, summary_file: Path, saved_files: Dict):
        """
        Save a summary of all saved session DataFrames.

        Args:
            summary_file: Path to summary file
            saved_files: Dictionary of saved files by session
        """
        try:
            with open(summary_file, 'w') as f:
                f.write("Session DataFrame Summary\n")
                f.write("========================\n\n")

                for session_name, files in saved_files.items():
                    f.write(f"{session_name}:\n")
                    for file_path in files:
                        f.write(f"  {Path(file_path).name}\n")
                    f.write("\n")

                f.write(f"Total sessions: {len(saved_files)}\n")
                f.write(f"Total files: {sum(len(files) for files in saved_files.values())}\n")

        except Exception as e:
            logger.error(f"Error saving session summary: {e}")

    def extract_anatomical_info(self, cell_types: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extract anatomical layer and major cell type information from cell type names.

        Args:
            cell_types: List of cell type names

        Returns:
            Tuple of (layers, major_cell_types) lists
        """
        layers = []
        major_cell_types = []

        for cell_type in cell_types:
            if not isinstance(cell_type, str):
                cell_type = str(cell_type)

            # Initialize with default values
            layer = "Unknown"
            category = "Other"

            # Extract layer information (more specific patterns first)
            if 'L2/3' in cell_type:
                layer = "L2/3"
            elif 'L4' in cell_type:
                layer = "L4"
            elif 'L5' in cell_type:
                layer = "L5"
            elif 'L6' in cell_type:
                layer = "L6"
            elif 'L6b' in cell_type:
                layer = "L6b"
            elif 'CR' in cell_type:
                layer = "CR"  # Cajal-Retzius cells
            elif 'Meis2' in cell_type:
                layer = "Meis2"
            elif 'Astro' in cell_type:
                layer = "Astro"
            elif 'OPC' in cell_type:
                layer = "OPC"
            elif 'Oligo' in cell_type:
                layer = "Oligo"
            elif 'VLMC' in cell_type:
                layer = "VLMC"
            elif 'Peri' in cell_type:
                layer = "Peri"
            elif 'SMC' in cell_type:
                layer = "SMC"
            elif 'Endo' in cell_type:
                layer = "Endo"
            elif 'PVM' in cell_type:
                layer = "PVM"
            elif 'Microglia' in cell_type:
                layer = "Microglia"
            elif 'Zero' in cell_type:
                layer = "Zero"
            elif 'IN' in cell_type and len(cell_type) <= 3:  # Simple "IN" type
                layer = "IN"
            elif 'EC' in cell_type and len(cell_type) <= 3:  # Simple "EC" type
                layer = "EC"

            # Extract major cell type category (more specific patterns first)
            if 'Pvalb' in cell_type:
                category = "Pvalb"
            elif 'Sst' in cell_type:
                category = "Sst"
            elif 'Vip' in cell_type:
                category = "Vip"
            elif 'Lamp5' in cell_type:
                category = "Lamp5"
            elif 'Sncg' in cell_type:
                category = "Sncg"
            elif 'Serpinf1' in cell_type:
                category = "Serpinf1"
            elif 'L2/3-IT' in cell_type:
                category = "L2/3-IT"
            elif 'L4-IT' in cell_type:
                category = "L4-IT"
            elif 'L5-IT' in cell_type:
                category = "L5-IT"
            elif 'L6-IT' in cell_type:
                category = "L6-IT"
            elif 'L5-PT' in cell_type:
                category = "L5-PT"
            elif 'L5-NP' in cell_type:
                category = "L6-CT"
            elif 'L6-CT' in cell_type:
                category = "L6-CT"
            elif 'L6b' in cell_type:
                category = "L6b"
            elif 'IN' in cell_type and len(cell_type) <= 3:  # Simple "IN" type
                category = "IN"
            elif 'EC' in cell_type and len(cell_type) <= 3:  # Simple "EC" type
                category = "EC"

            layers.append(layer)
            major_cell_types.append(category)

        return layers, major_cell_types

    def enhance_neuron_metadata(self) -> pd.DataFrame:
        """
        Enhance the neuron metadata DataFrame with anatomical information.
        Note: This method is now mainly for backward compatibility since the new columns
        are already added during DataFrame creation.

        Returns:
            Enhanced DataFrame with layer and major_cell_type columns
        """
        if not self.data_sessions:
            logger.warning("No sessions loaded. Load sessions first.")
            return pd.DataFrame()

        # Get the basic neuron metadata
        basic_metadata = self.organize_into_dataframes().get('neuron_metadata')

        if basic_metadata is None or basic_metadata.empty:
            logger.warning("No neuron metadata available.")
            return pd.DataFrame()

        # Check if the columns already exist
        if 'layer' in basic_metadata.columns and 'major_cell_type' in basic_metadata.columns:
            logger.info("Anatomical information already present in metadata.")
            return basic_metadata

        # If not, enhance it (for backward compatibility)
        logger.info("Enhancing existing metadata with anatomical information...")

        # Reset index to work with the data
        metadata_reset = basic_metadata.reset_index()

        # Extract anatomical information
        layers, major_cell_types = self.extract_anatomical_info(metadata_reset['neuron_type'].tolist())

        # Add new columns
        enhanced_metadata = metadata_reset.copy()
        enhanced_metadata['layer'] = layers
        enhanced_metadata['major_cell_type'] = major_cell_types

        # Set index back - FIXED: Changed 'session_date' to 'session_id'
        enhanced_metadata.set_index(['session', 'session_id', 'neuron_id'], inplace=True)

        logger.info(f"Enhanced neuron metadata with anatomical information: {enhanced_metadata.shape}")

        return enhanced_metadata

    def get_anatomical_summary(self) -> Dict:
        """
        Get a summary of the anatomical organization of neurons.

        Returns:
            Dictionary containing anatomical summary statistics
        """
        enhanced_metadata = self.enhance_neuron_metadata()

        if enhanced_metadata.empty:
            return {"message": "No enhanced metadata available."}

        # Reset index for easier analysis
        metadata_reset = enhanced_metadata.reset_index()

        summary = {}

        # Layer distribution
        layer_counts = metadata_reset['layer'].value_counts()
        summary['layer_distribution'] = layer_counts.to_dict()

        # Cell type category distribution
        category_counts = metadata_reset['major_cell_type'].value_counts()
        summary['category_distribution'] = category_counts.to_dict()

        # Cross-tabulation of layers vs categories
        layer_category_cross = pd.crosstab(metadata_reset['layer'], metadata_reset['major_cell_type'])
        summary['layer_category_cross_tab'] = layer_category_cross.to_dict()

        # Spatial distribution by layer
        layer_spatial = metadata_reset.groupby('layer').agg({
            'x_position': ['mean', 'std', 'min', 'max'],
            'y_position': ['mean', 'std', 'min', 'max']
        }).round(2)
        summary['layer_spatial_distribution'] = layer_spatial.to_dict()

        # Spatial distribution by category
        category_spatial = metadata_reset.groupby('major_cell_type').agg({
            'x_position': ['mean', 'std', 'min', 'max'],
            'y_position': ['mean', 'std', 'min', 'max']
        }).round(2)
        summary['category_spatial_distribution'] = category_spatial.to_dict()

        return summary

    def _is_date_directory(self, dir_name: str) -> bool:
        """
        Check if a directory name represents a date (e.g., '2019-10-11').

        Args:
            dir_name: Directory name to check

        Returns:
            True if the directory name represents a date
        """
        # Check if it matches date format YYYY-MM-DD or YYYY.MM.DD
        import re
        date_pattern = r'^\d{4}[-.]\d{1,2}[-.]\d{1,2}$'
        return bool(re.match(date_pattern, dir_name))


def main():
    """
    Main function to demonstrate data loading.
    """
    # Initialize data loader
    loader = MOEDataLoader()

    # Discover data structure
    print("=== Data Structure Discovery ===")
    structure = loader.discover_data_structure()
    print(f"Data directory: {structure['data_directory']}")
    print(f"Subdirectories: {structure['subdirectories']}")
    print(f"Data types: {structure['data_types']}")
    print(f"Files found: {len(structure['files'])}")

    # Load all sessions
    print("\n=== Loading Sessions ===")
    sessions = loader.load_all_sessions()

    # Get summary
    print("\n=== Data Summary ===")
    summary = loader.get_data_summary()

    if summary.get("message"):
        print(f"   {summary['message']}")

    print(f"   Total sessions loaded: {summary['total_sessions']}")

    if summary['total_sessions'] == 0:
        print("\n   No sessions were loaded. This could be due to:")
        print("   1. Data files not found in expected locations")
        print("   2. File naming conventions not matching expectations")
        print("   3. Data structure different from expected")
        print("\n   Check the Data folder structure and file names.")
        return

    for session in summary['sessions']:
        print(f"\n   Session {session['session_id']}:")
        print(f"     Path: {session['path']}")
        print(f"     Activity shape: {session['activity_shape']}")
        print(f"     Population features: {session['population_features']}")
        print(f"     Has positions: {session['has_positions']}")
        print(f"     Has types: {session['has_types']}")
        print(f"     Has neighbors: {session['has_neighbors']}")


if __name__ == "__main__":
    main()