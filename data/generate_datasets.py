# %%
"""
Example usage of the MOE Data Loader
This script shows how to use the data loader with different data structures
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from neural_data_loader import MOEDataLoader

PATH=r"D:\TheFolder\projects\School\Master\Stefano\data"


def create_sample_data():
    """
    Create sample data structure for testing the loader.
    This simulates what your Data folder might contain.
    """
    # Create Data directory if it doesn't exist
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Create sample session 1
    session1_dir = PATH + '/session_001'
    if not os.path.exists(session1_dir):
        os.makedirs(session1_dir)

    # Sample neural activity (time x neurons)
    time_steps = 1000
    num_neurons = 50
    neural_activity = np.random.poisson(5, (time_steps, num_neurons)).astype(float)
    np.save(f'{session1_dir}/neural_activity.npy', neural_activity)

    # Sample behavioral data
    running_speed = np.random.normal(10, 3, (time_steps, 1))
    np.save(f'{session1_dir}/running_speed.npy', running_speed)

    # Sample neuron positions (2D)
    positions = np.random.uniform(0, 100, (num_neurons, 2))
    np.save(f'{session1_dir}/neuron_positions.npy', positions)

    # Sample neuron types
    neuron_types = ['IN' if i < 25 else 'EC' for i in range(num_neurons)]
    with open(f'{session1_dir}/neuron_types.txt', 'w') as f:
        for ntype in neuron_types:
            f.write(ntype + '\n')

    # Create sample session 2
    session2_dir = PATH + '/session_002'
    if not os.path.exists(session2_dir):
        os.makedirs(session2_dir)

    # Different parameters for session 2
    time_steps2 = 800
    num_neurons2 = 40

    neural_activity2 = np.random.poisson(3, (time_steps2, num_neurons2)).astype(float)
    np.save(f'{session2_dir}/neural_activity.npy', neural_activity2)

    running_speed2 = np.random.normal(8, 2, (time_steps2, 1))
    np.save(f'{session2_dir}/running_speed.npy', running_speed2)

    positions2 = np.random.uniform(0, 80, (num_neurons2, 2))
    np.save(f'{session2_dir}/neuron_positions.npy', positions2)

    neuron_types2 = ['IN' if i < 20 else 'EC' for i in range(num_neurons2)]
    with open(f'{session2_dir}/neuron_types.txt', 'w') as f:
        for ntype in neuron_types2:
            f.write(ntype + '\n')

    print("Sample data created successfully!")
    print(f"Created sessions: {session1_dir}, {session2_dir}")


def demonstrate_data_loading():
    """
    Demonstrate how to use the data loader with the sample data.
    """
    print("\n=== Demonstrating Data Loading ===")

    # Initialize the data loader
    loader = MOEDataLoader(PATH)

    # Discover the data structure
    print("\n1. Discovering data structure...")
    structure = loader.discover_data_structure()
    print(f"   Data directory: {structure['data_directory']}")
    print(f"   Subdirectories: {structure['subdirectories']}")
    print(f"   File types found: {structure['data_types']}")

    # Load all sessions
    print("\n2. Loading all sessions...")
    sessions = loader.load_all_sessions()

    # Get summary
    print("\n3. Data summary:")
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

    # Demonstrate working with individual sessions
    if sessions:
        print("\n4. Working with session data:")
        session = sessions[0]

        print(f"   Session path: {session['session_path']}")
        if session['activity_norm'] is not None:
            print(f"   Neural activity shape: {session['activity_norm'].shape}")
            print(f"   Activity range: [{session['activity_norm'].min():.3f}, {session['activity_norm'].max():.3f}]")

        if session['activity_population']:
            print(f"   Population features: {list(session['activity_population'].keys())}")
            for key, data in session['activity_population'].items():
                if data is not None:
                    print(f"     {key}: shape {data.shape}")

        if session['neuron_pos'] is not None:
            print(f"   Neuron positions: shape {session['neuron_pos'].shape}")
            print(
                f"   Position range: X[{session['neuron_pos'][:, 0].min():.1f}, {session['neuron_pos'][:, 0].max():.1f}]")
            print(
                f"                    Y[{session['neuron_pos'][:, 1].min():.1f}, {session['neuron_pos'][:, 1].max():.1f}]")

        if session['neuron_types'] is not None:
            unique_types = set(session['neuron_types'])
            print(f"   Neuron types: {unique_types}")
            for ntype in unique_types:
                count = session['neuron_types'].count(ntype)
                print(f"     {ntype}: {count} neurons")


def demonstrate_custom_usage():
    """
    Show how to use the data loader for custom analysis.
    """
    print("\n=== Custom Usage Examples ===")

    # Reuse the loader from the previous demonstration instead of creating a new one
    loader = MOEDataLoader(PATH)
    sessions = loader.load_all_sessions()

    if not sessions:
        print("No sessions loaded. Check the data structure and file naming.")
        return

    # Example 1: Calculate average firing rates across sessions
    print("\n1. Average firing rates across sessions:")
    for i, session in enumerate(sessions):
        if session['activity_raw'] is not None:  # Use raw activity for firing rates
            # Calculate mean firing rate per neuron
            mean_rates = np.mean(session['activity_raw'], axis=0)
            print(f"   Session {i}: mean rate = {np.mean(mean_rates):.3f}, std = {np.std(mean_rates):.3f}")

    # Example 2: Analyze population dynamics
    print("\n2. Population dynamics analysis:")
    for i, session in enumerate(sessions):
        if session['activity_population'] and 'mean_activity' in session['activity_population']:
            pop_activity = session['activity_population']['mean_activity']
            print(f"   Session {i}: population activity shape = {pop_activity.shape}")
            print(f"     Mean: {np.mean(pop_activity):.3f}, Std: {np.std(pop_activity):.3f}")

    # Example 3: Spatial analysis
    print("\n3. Spatial analysis:")
    for i, session in enumerate(sessions):
        if session['neuron_pos'] is not None:
            positions = session['neuron_pos']
            # Calculate density
            area = (positions[:, 0].max() - positions[:, 0].min()) * (positions[:, 1].max() - positions[:, 1].min())
            density = len(positions) / area
            print(f"   Session {i}: {len(positions)} neurons, density = {density:.3f} neurons/unit²")

    # Example 4: Show behavioral data summary
    print("\n4. Behavioral data summary:")
    for i, session in enumerate(sessions):
        if session['activity_population']:
            print(f"   Session {i}:")
            for key, data in session['activity_population'].items():
                if data is not None and key not in ['mean_activity', 'std_activity', 'frame_states']:
                    print(f"     {key}: shape {data.shape}, range [{data.min():.3f}, {data.max():.3f}]")


def demonstrate_dataframe_organization():
    """
    Demonstrate how to organize the loaded data into pandas DataFrames.
    """
    print("\n=== DataFrame Organization ===")

    # Initialize the data loader
    loader = MOEDataLoader(PATH)

    # Load all sessions first
    print("1. Loading sessions...")
    sessions = loader.load_all_sessions()

    if not sessions:
        print("No sessions loaded. Cannot create DataFrames.")
        return

    # Organize data into DataFrames
    print("2. Organizing data into DataFrames...")
    dataframes = loader.organize_into_dataframes()

    if not dataframes:
        print("No DataFrames created.")
        return

    # Get DataFrame summary
    print("3. DataFrame Summary:")
    summary = loader.get_dataframe_summary()

    for name, info in summary.items():
        print(f"\n   {name.upper()}:")
        print(f"     Shape: {info['shape']}")
        print(f"     Index levels: {info['index_levels']}")
        print(f"     Memory usage: {info['memory_usage_mb']:.2f} MB")
        print(f"     Columns: {info['columns'][:5]}{'...' if len(info['columns']) > 5 else ''}")

    # Demonstrate working with individual DataFrames
    print("\n4. Working with DataFrames:")

    # Neural Activity DataFrame
    if 'neural_activity' in dataframes:
        neural_df = dataframes['neural_activity']
        print(f"\n   Neural Activity DataFrame:")
        print(f"     Shape: {neural_df.shape}")
        print(f"     Time steps: {neural_df.index.get_level_values('time_step').nunique()}")
        print(f"     Neurons: {neural_df.shape[1]}")
        print(f"     Sessions: {neural_df.index.get_level_values('session').unique()}")

        # Show sample data
        print(f"     Sample data (first 3 time steps, first 3 neurons):")
        sample_cols = [col for col in neural_df.columns if col.startswith('neuron_')][:3]
        print(neural_df[sample_cols].head(3))

    # Neuron Metadata DataFrame
    if 'neuron_metadata' in dataframes:
        metadata_df = dataframes['neuron_metadata']
        print(f"\n   Neuron Metadata DataFrame:")
        print(f"     Shape: {metadata_df.shape}")
        print(f"     Neuron types: {metadata_df['neuron_type'].nunique()}")
        print(f"     Position range: X[{metadata_df['x_position'].min():.1f}, {metadata_df['x_position'].max():.1f}]")
        print(f"                    Y[{metadata_df['y_position'].min():.1f}, {metadata_df['y_position'].max():.1f}]")

        # Show neuron type distribution
        type_counts = metadata_df['neuron_type'].value_counts()
        print(f"     Neuron type distribution:")
        for ntype, count in type_counts.head(5).items():
            print(f"       {ntype}: {count}")

    # Behavioral Data DataFrame
    if 'behavioral_data' in dataframes:
        behavioral_df = dataframes['behavioral_data']
        print(f"\n   Behavioral Data DataFrame:")
        print(f"     Shape: {behavioral_df.shape}")
        print(f"     Features: {list(behavioral_df.columns)}")

        # Show sample behavioral data
        if 'eye_size' in behavioral_df.columns:
            print(
                f"     Eye size range: [{behavioral_df['eye_size'].min():.3f}, {behavioral_df['eye_size'].max():.3f}]")
        if 'running_speed' in behavioral_df.columns:
            print(
                f"     Running speed range: [{behavioral_df['running_speed'].min():.3f}, {behavioral_df['running_speed'].max():.3f}]")

    # Population Features DataFrame
    if 'population_features' in dataframes:
        pop_df = dataframes['population_features']
        print(f"\n   Population Features DataFrame:")
        print(f"     Shape: {pop_df.shape}")
        print(f"     Features: {list(pop_df.columns)}")

        # Show population dynamics
        if 'mean_activity' in pop_df.columns:
            print(
                f"     Mean activity range: [{pop_df['mean_activity'].min():.3f}, {pop_df['mean_activity'].max():.3f}]")

    # Demonstrate some analysis operations
    print("\n5. Analysis Examples:")

    if 'neural_activity' in dataframes and 'neuron_metadata' in dataframes:
        # Calculate mean firing rate per neuron type
        neural_df = dataframes['neural_activity']
        metadata_df = dataframes['neuron_metadata']

        # Get neuron columns
        neuron_cols = [col for col in neural_df.columns if col.startswith('neuron_')]

        # Calculate mean firing rates
        mean_rates = neural_df[neuron_cols].mean()

        # Create a DataFrame with neuron info and firing rates
        neuron_analysis = pd.DataFrame({
            'neuron_id': [col for col in neuron_cols],
            'mean_firing_rate': mean_rates.values
        })

        # Merge with metadata
        metadata_reset = metadata_df.reset_index()
        neuron_analysis = neuron_analysis.merge(
            metadata_reset[['neuron_id', 'neuron_type', 'x_position', 'y_position']],
            on='neuron_id'
        )

        print(f"     Mean firing rates by neuron type:")
        type_rates = neuron_analysis.groupby('neuron_type')['mean_firing_rate'].agg(['mean', 'std', 'count'])
        print(type_rates.head())

        # Show spatial distribution
        print(f"     Spatial distribution:")
        print(f"       Total neurons: {len(neuron_analysis)}")
        print(
            f"       Area: {neuron_analysis['x_position'].max() - neuron_analysis['x_position'].min():.1f} x {neuron_analysis['y_position'].max() - neuron_analysis['y_position'].min():.1f}")
        print(
            f"       Density: {len(neuron_analysis) / ((neuron_analysis['x_position'].max() - neuron_analysis['x_position'].min()) * (neuron_analysis['y_position'].max() - neuron_analysis['y_position'].min())):.3f} neurons/unit²")

    return dataframes


def demonstrate_session_dataframe_saving():
    """
    Demonstrate saving session DataFrames to the Data/Dataframes folder.
    """
    print("\n=== Session DataFrame Saving ===")

    # Initialize the data loader
    loader = MOEDataLoader(PATH)

    # Load all sessions first
    print("1. Loading sessions...")
    sessions = loader.load_all_sessions()

    if not sessions:
        print("No sessions loaded. Cannot save DataFrames.")
        return

    print(f"   Loaded {len(sessions)} sessions")

    # Save session DataFrames
    print("2. Saving session DataFrames...")
    saved_files = loader.save_session_dataframes(output_dir=PATH + "/Dataframes", formats=['pkl'])

    if not saved_files:
        print("No DataFrames were saved.")
        return

    # Show what was saved
    print("3. Saved DataFrames:")
    for session_name, files in saved_files.items():
        print(f"\n   {session_name}:")
        for file_path in files:
            filename = Path(file_path).name
            print(f"     {filename}")

    # Show file sizes
    print("\n4. File Sizes:")
    total_size = 0
    for session_name, files in saved_files.items():
        session_size = 0
        for file_path in files:
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            session_size += file_size
            total_size += file_size
            filename = Path(file_path).name
            print(f"     {filename}: {file_size:.2f} MB")
        print(f"   {session_name} total: {session_size:.2f} MB")

    print(f"\n   Total size: {total_size:.2f} MB")

    # Show directory structure
    print("\n5. Directory Structure:")
    dataframes_dir = Path(PATH + "/Dataframes")
    if dataframes_dir.exists():
        print(f"   Created directory: {dataframes_dir}")
        print(f"   Files in directory:")
        for file_path in dataframes_dir.glob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"     {file_path.name}: {file_size:.2f} MB")

    # Demonstrate loading saved DataFrames
    print("\n6. Loading Saved DataFrames:")
    for session_name, files in saved_files.items():
        print(f"\n   Loading {session_name}:")
        for file_path in files:
            if file_path.endswith('.pkl'):
                try:
                    df = pd.read_pickle(file_path)
                    filename = Path(file_path).name
                    print(f"     {filename}: {df.shape}")

                    # Show sample data for first file
                    if 'neural_activity' in filename:
                        print(f"       Sample data shape: {df.shape}")
                        print(f"       Index levels: {df.index.names}")
                        if not df.empty:
                            print(f"       First few values: {df.iloc[:3, :3].values.flatten()[:5]}")

                except Exception as e:
                    print(f"     Error loading {Path(file_path).name}: {e}")

    print(f"\n   Successfully saved {len(saved_files)} sessions with DataFrames")
    print(f"   Files saved to: Data/Dataframes/")

    return saved_files


def demonstrate_cell_type_analysis():
    """
    Demonstrate how to access and analyze cell type information from the organized DataFrames.
    """
    print("\n=== Cell Type Analysis ===")

    # Initialize the data loader
    loader = MOEDataLoader(PATH)

    # Load all sessions first
    print("1. Loading sessions...")
    sessions = loader.load_all_sessions()

    if not sessions:
        print("No sessions loaded. Cannot analyze cell types.")
        return

    # Organize data into DataFrames
    print("2. Organizing data into DataFrames...")
    dataframes = loader.organize_into_dataframes()

    if not dataframes:
        print("No DataFrames created.")
        return

    # Access cell type information from the neuron metadata DataFrame
    if 'neuron_metadata' in dataframes:
        metadata_df = dataframes['neuron_metadata']
        print(f"\n3. Cell Type Information from Neuron Metadata DataFrame:")
        print(f"   DataFrame shape: {metadata_df.shape}")
        print(f"   Index levels: {metadata_df.index.names}")
        print(f"   Columns: {list(metadata_df.columns)}")

        # Show the actual cell type data
        print(f"\n4. Raw Cell Type Data:")
        print(f"   First 10 rows:")
        print(metadata_df.head(10))

        # Analyze cell type distribution
        print(f"\n5. Cell Type Distribution:")
        cell_type_counts = metadata_df['neuron_type'].value_counts()
        print(f"   Total unique cell types: {len(cell_type_counts)}")
        print(f"   Cell type counts:")
        for cell_type, count in cell_type_counts.items():
            print(f"     {cell_type}: {count} neurons")

        # Show cell type statistics by position
        print(f"\n6. Cell Type Spatial Distribution:")
        cell_type_stats = metadata_df.groupby('neuron_type').agg({
            'x_position': ['mean', 'std', 'min', 'max'],
            'y_position': ['mean', 'std', 'min', 'max']
        }).round(2)
        print(cell_type_stats)

        # Show specific cell types with their properties
        print(f"\n7. Detailed Cell Type Analysis:")
        for cell_type in cell_type_counts.index[:5]:  # Show first 5 types
            type_data = metadata_df[metadata_df['neuron_type'] == cell_type]
            print(f"\n   {cell_type}:")
            print(f"     Count: {len(type_data)} neurons")
            print(f"     X position: [{type_data['x_position'].min():.1f}, {type_data['x_position'].max():.1f}]")
            print(f"     Y position: [{type_data['y_position'].min():.1f}, {type_data['y_position'].max():.1f}]")
            print(f"     Mean X: {type_data['x_position'].mean():.1f}")
            print(f"     Mean Y: {type_data['y_position'].mean():.1f}")

        # Show how to access specific cell types
        print(f"\n8. Accessing Specific Cell Types:")
        print(f"   Example: Get all EC neurons:")
        ec_neurons = metadata_df[metadata_df['neuron_type'] == 'EC']
        print(f"     EC neurons shape: {ec_neurons.shape}")
        print(f"     EC neuron IDs: {list(ec_neurons.index.get_level_values('neuron_id'))[:5]}...")

        print(f"\n   Example: Get all inhibitory neurons (IN):")
        in_neurons = metadata_df[metadata_df['neuron_type'] == 'IN']
        print(f"     IN neurons shape: {in_neurons.shape}")
        print(f"     IN neuron IDs: {list(in_neurons.index.get_level_values('neuron_id'))[:5]}...")

        # Show how to combine with neural activity data
        if 'neural_activity' in dataframes:
            print(f"\n9. Cell Type Activity Analysis:")
            neural_df = dataframes['neural_activity']

            # Get neuron columns
            neuron_cols = [col for col in neural_df.columns if col.startswith('neuron_')]

            # Calculate mean firing rates for each cell type
            print(f"   Calculating mean firing rates by cell type...")

            # Create a mapping from neuron column to cell type
            neuron_to_type = {}

            # Get unique sessions from metadata
            unique_sessions = metadata_df.index.get_level_values('session').unique()
            unique_session_ids = metadata_df.index.get_level_values('session_id').unique()

            # Use the first available session for mapping
            if len(unique_sessions) > 0 and len(unique_session_ids) > 0:
                session_name = unique_sessions[0]
                session_id = unique_session_ids[0]

                for neuron_id in metadata_df.index.get_level_values('neuron_id'):
                    try:
                        cell_type = metadata_df.loc[(session_name, session_id, neuron_id), 'neuron_type']
                        neuron_to_type[f'neuron_{neuron_id.split("_")[1]}'] = cell_type
                    except KeyError:
                        # Skip if this combination doesn't exist
                        continue

            # Calculate firing rates by cell type
            cell_type_rates = {}
            for neuron_col in neuron_cols:
                if neuron_col in neuron_to_type:
                    cell_type = neuron_to_type[neuron_col]
                    if cell_type not in cell_type_rates:
                        cell_type_rates[cell_type] = []
                    mean_rate = neural_df[neuron_col].mean()
                    cell_type_rates[cell_type].append(mean_rate)

            # Show results
            print(f"   Mean firing rates by cell type:")
            for cell_type, rates in cell_type_rates.items():
                if rates:
                    print(f"     {cell_type}: {np.mean(rates):.2f} ± {np.std(rates):.2f} Hz (n={len(rates)})")

        return metadata_df
    else:
        print("Neuron metadata DataFrame not found.")
        return None


def demonstrate_anatomical_enhancement():
    """
    Demonstrate the enhanced anatomical information functionality.
    """
    print("\n=== Anatomical Information Enhancement ===")

    # Initialize the data loader
    loader = MOEDataLoader(PATH)

    # Load all sessions first
    print("1. Loading sessions...")
    sessions = loader.load_all_sessions()

    if not sessions:
        print("No sessions loaded. Cannot enhance metadata.")
        return

    # Get enhanced neuron metadata with anatomical information
    print("2. Enhancing neuron metadata with anatomical information...")
    enhanced_metadata = loader.enhance_neuron_metadata()

    if enhanced_metadata.empty:
        print("No enhanced metadata available.")
        return

    print(f"   Enhanced metadata shape: {enhanced_metadata.shape}")
    print(f"   New columns: {list(enhanced_metadata.columns)}")

    # Show sample of enhanced data
    print("\n3. Sample Enhanced Metadata:")
    print(enhanced_metadata.head(10))

    # Get anatomical summary
    print("\n4. Anatomical Summary:")
    anatomical_summary = loader.get_anatomical_summary()

    # Layer distribution
    if 'layer_distribution' in anatomical_summary:
        print(f"\n   Layer Distribution:")
        for layer, count in anatomical_summary['layer_distribution'].items():
            print(f"     {layer}: {count} neurons")

    # Cell type category distribution
    if 'category_distribution' in anatomical_summary:
        print(f"\n   Cell Type Category Distribution:")
        for category, count in anatomical_summary['category_distribution'].items():
            print(f"     {category}: {count} neurons")

    # Cross-tabulation
    if 'layer_category_cross_tab' in anatomical_summary:
        print(f"\n   Layer vs Category Cross-tabulation:")
        cross_tab = anatomical_summary['layer_category_cross_tab']
        # Convert back to DataFrame for better display
        cross_df = pd.DataFrame(cross_tab)
        print(cross_df)

    # Spatial distribution by layer
    if 'layer_spatial_distribution' in anatomical_summary:
        print(f"\n   Spatial Distribution by Layer:")
        layer_spatial = anatomical_summary['layer_spatial_distribution']
        for layer in list(layer_spatial.keys())[:5]:  # Show first 5 layers
            if 'x_position' in layer_spatial[layer] and 'mean' in layer_spatial[layer]['x_position']:
                x_mean = layer_spatial[layer]['x_position']['mean']
                y_mean = layer_spatial[layer]['y_position']['mean']
                print(f"     {layer}: X={x_mean:.1f}, Y={y_mean:.1f}")

    # Demonstrate working with enhanced metadata
    print("\n5. Working with Enhanced Metadata:")

    # Reset index for easier analysis
    metadata_reset = enhanced_metadata.reset_index()

    # Show layer-specific analysis
    print(f"\n   Layer-specific Analysis:")
    for layer in ['L2/3', 'L4', 'L5', 'L6']:
        layer_neurons = metadata_reset[metadata_reset['layer'] == layer]
        if not layer_neurons.empty:
            print(f"     {layer}: {len(layer_neurons)} neurons")
            if 'x_position' in layer_neurons.columns:
                x_range = f"[{layer_neurons['x_position'].min():.1f}, {layer_neurons['x_position'].max():.1f}]"
                y_range = f"[{layer_neurons['y_position'].min():.1f}, {layer_neurons['y_position'].max():.1f}]"
                print(f"       Position range: X{x_range}, Y{y_range}")

    # Show category-specific analysis
    print(f"\n   Category-specific Analysis:")
    for category in ['Pvalb', 'Sst', 'Vip', 'Lamp5']:
        category_neurons = metadata_reset[metadata_reset['major_cell_type'] == category]
        if not category_neurons.empty:
            print(f"     {category}: {len(category_neurons)} neurons")
            if 'x_position' in category_neurons.columns:
                x_range = f"[{category_neurons['x_position'].min():.1f}, {category_neurons['x_position'].max():.1f}]"
                y_range = f"[{category_neurons['y_position'].min():.1f}, {category_neurons['y_position'].max():.1f}]"
                print(f"       Position range: X{x_range}, Y{y_range}")

    # Show how to combine with neural activity data
    print(f"\n6. Combining with Neural Activity Data:")
    dataframes = loader.organize_into_dataframes()

    if 'neural_activity' in dataframes and not enhanced_metadata.empty:
        neural_df = dataframes['neural_activity']

        # Get neuron columns
        neuron_cols = [col for col in neural_df.columns if col.startswith('neuron_')]

        # Calculate mean firing rates for each layer
        print(f"   Mean firing rates by layer:")
        layer_rates = {}

        for layer in metadata_reset['layer'].unique():
            if layer != "Unknown":
                # Get neuron IDs for this layer
                layer_neurons = metadata_reset[metadata_reset['layer'] == layer]
                layer_neuron_ids = layer_neurons['neuron_id'].tolist()

                # Map to neural activity columns
                layer_cols = [f'neuron_{nid.split("_")[1]}' for nid in layer_neuron_ids if nid.startswith('neuron_')]
                layer_cols = [col for col in layer_cols if col in neural_df.columns]

                if layer_cols:
                    mean_rate = neural_df[layer_cols].mean().mean()
                    layer_rates[layer] = mean_rate
                    print(f"     {layer}: {mean_rate:.2f} Hz (n={len(layer_cols)})")

        # Calculate mean firing rates for each category
        print(f"\n   Mean firing rates by cell type category:")
        category_rates = {}

        for category in metadata_reset['major_cell_type'].unique():
            if category != "Unknown":
                # Get neuron IDs for this category
                category_neurons = metadata_reset[metadata_reset['major_cell_type'] == category]
                category_neuron_ids = category_neurons['neuron_id'].tolist()

                # Map to neural activity columns
                category_cols = [f'neuron_{nid.split("_")[1]}' for nid in category_neuron_ids if
                                 nid.startswith('neuron_')]
                category_cols = [col for col in category_cols if col in neural_df.columns]

                if category_cols:
                    mean_rate = neural_df[category_cols].mean().mean()
                    category_rates[category] = mean_rate
                    print(f"     {category}: {mean_rate:.2f} Hz (n={len(category_cols)})")

    return enhanced_metadata


if __name__ == "__main__":
    print("MOE Data Loader Example")
    print("=" * 50)

    # Check if sample data exists, create if not
    if not os.path.exists(PATH) or not os.listdir(PATH):
        print("Creating sample data...")
        create_sample_data()
    else:
        print("Using existing data in PATH/ folder")

    # Demonstrate the data loader
    demonstrate_data_loading()

    # Show custom usage examples
    demonstrate_custom_usage()

    # Demonstrate DataFrame organization
    demonstrate_dataframe_organization()

    # Demonstrate session dataframe saving
    demonstrate_session_dataframe_saving()

    # Demonstrate cell type analysis
    demonstrate_cell_type_analysis()

    # Demonstrate anatomical enhancement
    demonstrate_anatomical_enhancement()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nYou can now:")
    print("1. Modify the data_loader.py to match your specific data format")
    print("2. Use the MOEDataLoader class in your own scripts")
    print("3. Run NeuPRINT with your data by updating the paths")

# %%
