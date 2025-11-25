import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import io
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Molecular Landscape Explorer
# A tool for analyzing structure-activity relationships in chemical datasets

st.set_page_config(
    page_title="Molecular Landscape Explorer",
    layout="wide",
    page_icon="🧪"
)

# Set matplotlib style
sns.set_style("whitegrid")

st.title("Molecular Landscape Explorer")
about_expander = st.expander("About Structure-Activity Landscape Analysis", expanded=True)
with about_expander:
    st.markdown('''
    **Molecular Landscape Explorer** is an interactive tool for visualizing and analyzing 
    structure-activity relationships in chemical compound datasets.  
    
    This application helps identify key patterns in molecular data including:
    - **Activity Cliffs**: Structurally similar compounds with significant activity differences
    - **SAR Zones**: Regions with predictable structure-activity relationships
    - **Scaffold Transitions**: Areas where core molecular structures change
    
    *Understanding these patterns is crucial for medicinal chemistry and drug discovery efforts.*
    ''')

# Sidebar configuration
st.sidebar.subheader("Analysis Configuration")

# Application mode selection
analysis_mode = st.sidebar.radio(
    "Analysis Mode", 
    ["Basic Landscape", "Advanced SAR Analysis"]
)

if analysis_mode == "Basic Landscape":
    fingerprint_config = {
        'ECFP4': 2, 
        'ECFP6': 3, 
        'ECFP8': 4, 
        'ECFP10': 5
    }
    selected_fp = st.sidebar.radio(
        "Molecular Representation", 
        ('ECFP4', 'ECFP6', 'ECFP8', 'ECFP10')
    )
    radius = fingerprint_config[selected_fp]
    fingerprint_size = st.sidebar.selectbox(
        "Fingerprint Dimension", 
        options=[512, 1024, 2048, 4096], 
        index=2
    )
    similarity_cutoff = st.sidebar.selectbox(
        "Similarity Threshold", 
        options=[0.7, 0.5, 0.6, 0.8, 0.9, 1.0]
    )
    activity_cutoff = st.sidebar.selectbox(
        "Activity Difference Threshold", 
        options=[0.5, 1, 1.5, 2, 2.5, 3],
        index=1
    )
else:  # Advanced SAR Analysis
    # Molecular representation selection
    molecular_representation = st.sidebar.selectbox(
        "Molecular Representation Type", 
        ["ECFP4", "ECFP6", "MACCS"], 
        index=0
    )

    # Configuration based on representation type
    if molecular_representation.startswith("ECFP"):
        radius_param = st.sidebar.slider(
            "Morgan Radius", 1, 4, 
            2 if molecular_representation == "ECFP4" else 3
        )
        bit_size = st.sidebar.selectbox(
            "Fingerprint Dimension", 
            [512, 1024, 2048], 
            index=2
        )
    else:  # MACCS
        radius_param = None
        bit_size = 167  # Fixed size for MACCS keys

    visualization_color = st.sidebar.selectbox(
        "Color Mapping", 
        ["SALI", "MaxActivity", "Zone"]
    )
    max_visualization_pairs = st.sidebar.number_input(
        "Maximum pairs for visualization", 
        min_value=2000, max_value=200000, 
        value=10000, step=1000
    )

    # Landscape classification parameters
    similarity_cutoff = st.sidebar.slider(
        "Similarity threshold", 0.1, 0.9, 0.5, 0.05
    )
    activity_cutoff = st.sidebar.slider(
        "Activity threshold", 0.1, 5.0, 1.0, 0.1
    )

# Core computational functions
def compute_similarity(smiles1, smiles2, radius_param, nBits):
    """Calculate Tanimoto similarity between two molecules"""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius_param, nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius_param, nBits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def classify_landscape_region(similarity, activity_diff, sim_threshold, act_threshold):
    """Classify molecular pairs into landscape regions"""
    if similarity >= sim_threshold and activity_diff >= act_threshold:
        return 'Activity Cliffs'
    elif similarity < sim_threshold and activity_diff < act_threshold:
        return 'Scaffold Transitions'
    elif similarity >= sim_threshold and activity_diff < act_threshold:
        return 'Consistent SAR Regions'
    else:
        return 'Baseline Regions'

def analyze_molecular_pairs(df, radius_param, sim_threshold, act_threshold):
    """Analyze all molecular pairs in the dataset"""
    pair_results = []
    for (idx1, row1), (idx2, row2) in combinations(df.iterrows(), 2):
        smi1, act1 = row1['Smiles'], row1['pIC50']
        smi2, act2 = row2['Smiles'], row2['pIC50']
        similarity = compute_similarity(smi1, smi2, int(radius_param), fingerprint_size)
        activity_difference = abs(act1 - act2)
        region = classify_landscape_region(
            similarity, activity_difference, sim_threshold, act_threshold
        )
        pair_results.append((
            row1['Molecule ID'], row2['Molecule ID'], 
            similarity, activity_difference, region
        ))

    results_df = pd.DataFrame(
        pair_results,
        columns=['Molecule_ID1', 'Molecule_ID2', 'Similarity', 
                'Activity_Difference', 'Landscape_Region']
    )
    return results_df

def create_landscape_visualization(results_df):
    """Generate the molecular landscape visualization"""
    region_colors = {
        'Activity Cliffs': 'red', 
        'Scaffold Transitions': 'purple', 
        'Consistent SAR Regions': 'green',
        'Baseline Regions': 'blue'
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for region, color in region_colors.items():
        region_data = results_df[results_df['Landscape_Region'] == region]
        ax.scatter(
            region_data['Similarity'], 
            region_data['Activity_Difference'], 
            c=color, label=region, alpha=0.6
        )

    # Add threshold lines
    ax.axvline(similarity_cutoff, linestyle='dotted', color='black')
    ax.axhline(activity_cutoff, linestyle='dotted', color='black')

    ax.set_xlabel('Structural Similarity')
    ax.set_ylabel('Activity Difference')
    fp_name = {2: 'ECFP4', 3: 'ECFP6', 4: 'ECFP8', 5: 'ECFP10'}[radius_param]
    ax.set_title(f'Molecular Landscape Map (Representation: {fp_name})')
    ax.legend(loc='best')
    ax.grid(False)

    st.pyplot(fig)
    return fig

# Advanced analysis functions
def generate_molecular_descriptors(smiles_list, desc_type, radius_param, n_bits):
    """Generate molecular descriptors/fingerprints"""
    descriptors = []
    valid_indices = []
    problematic_smiles = []
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            problematic_smiles.append(smiles)
            continue
            
        try:
            if desc_type == "ECFP4":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            elif desc_type == "ECFP6":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
            elif desc_type == "MACCS":
                desc = MACCSkeys.GenMACCSKeys(mol)
            else:
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius_param, nBits=n_bits)
            
            descriptors.append(desc)
            valid_indices.append(idx)
        except Exception as e:
            problematic_smiles.append(f"{smiles} (error: {str(e)})")
            continue
    
    return descriptors, valid_indices, problematic_smiles

def compute_pairwise_similarities(descriptors):
    """Compute pairwise similarity matrix"""
    n_molecules = len(descriptors)
    similarity_matrix = np.zeros((n_molecules, n_molecules), dtype=float)
    
    for i in range(n_molecules):
        for j in range(i, n_molecules):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                try:
                    sim_value = DataStructs.TanimotoSimilarity(
                        descriptors[i], descriptors[j]
                    )
                    similarity_matrix[i, j] = sim_value
                    similarity_matrix[j, i] = sim_value
                except Exception:
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
    
    return similarity_matrix

def categorize_sar_regions(pairs_data, sim_threshold, act_threshold):
    """Categorize molecular pairs into SAR regions"""
    categorized_data = pairs_data.copy()
    
    # Initialize region classification
    categorized_data['Zone'] = 'Baseline Regions'
    
    # Activity Cliffs classification
    cliffs_criteria = (
        categorized_data['Similarity'] > sim_threshold
    ) & (
        categorized_data['Activity_Diff'] > act_threshold
    )
    categorized_data.loc[cliffs_criteria, 'Zone'] = 'Activity Cliffs'
    
    # Consistent SAR regions
    consistent_criteria = (
        categorized_data['Similarity'] > sim_threshold
    ) & (
        categorized_data['Activity_Diff'] <= act_threshold
    )
    categorized_data.loc[consistent_criteria, 'Zone'] = 'Consistent SAR Regions'
    
    # Scaffold transitions
    scaffold_criteria = (
        categorized_data['Similarity'] <= sim_threshold
    ) & (
        categorized_data['Activity_Diff'] <= act_threshold
    )
    categorized_data.loc[scaffold_criteria, 'Zone'] = 'Scaffold Transitions'
    
    return categorized_data

def create_enhanced_visualization(plot_data, sim_threshold, act_threshold):
    """Create enhanced landscape visualization"""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create scatter plot with region coloring
    sns.scatterplot(
        data=plot_data.sort_values("Zone"),
        x="Similarity",
        y="Activity_Diff",
        hue="Zone",
        palette={
            "Consistent SAR Regions": "green",
            "Baseline Regions": "blue",
            "Scaffold Transitions": "orange",
            "Activity Cliffs": "red",
        },
        alpha=0.6,
        s=30,
        edgecolor=None,
        ax=ax
    )
    
    # Add threshold guidelines
    ax.axvline(x=sim_threshold, color='red', linestyle='--', alpha=0.8, 
               label=f'Similarity threshold = {sim_threshold}')
    ax.axhline(y=act_threshold, color='blue', linestyle='--', alpha=0.8, 
               label=f'Activity threshold = {act_threshold}')
    
    plt.title("Molecular Landscape Analysis", fontsize=15)
    plt.xlabel("Structural Similarity")
    plt.ylabel("Activity Difference")
    plt.legend(title="SAR Region")
    plt.grid(True)
    
    plt.tight_layout()
    return fig

def perform_advanced_analysis(
    df, smiles_column, activity_column, id_column, 
    desc_type, radius_param, n_bits, sim_threshold, 
    act_threshold, color_scheme, max_pairs
):
    """Perform advanced molecular landscape analysis"""
    
    # Data preparation and validation
    clean_data = df.dropna(subset=[smiles_column, activity_column]).copy()
    if len(clean_data) == 0:
        st.error("No valid data after removing incomplete entries.")
        return None, None, None
        
    # Process activity values
    try:
        activity_values = clean_data[activity_column].astype(float).values
    except Exception as e:
        st.error(f"Activity data processing failed: {e}")
        return None, None, None

    molecule_ids = clean_data[id_column].astype(str).values if id_column != "None" else np.array([f"Mol_{i+1}" for i in range(len(clean_data))])
    smiles_strings = clean_data[smiles_column].astype(str).values

    # Step 1: Generate molecular descriptors
    st.write(f"### Step 1: Computing {desc_type} molecular representations...")
    descriptors, valid_indices, invalid_smiles = generate_molecular_descriptors(
        smiles_strings, desc_type, radius_param, n_bits
    )
    
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid molecular structures identified.")
        with st.expander("View problematic structures"):
            for bad_smiles in invalid_smiles[:10]:
                st.write(bad_smiles)
            if len(invalid_smiles) > 10:
                st.write(f"... and {len(invalid_smiles) - 10} additional structures")
    
    # Filter to valid entries
    activity_values = activity_values[valid_indices]
    molecule_ids = molecule_ids[valid_indices]
    smiles_strings = smiles_strings[valid_indices]
    n_valid = len(descriptors)
    
    if n_valid < 2:
        st.error("Minimum of 2 valid molecules required for analysis.")
        return None, None, None

    st.success(f"✅ Molecular representations computed for {n_valid} compounds.")

    # Step 2: Compute similarity matrix
    st.write("### Step 2: Calculating pairwise molecular similarities...")
    similarity_matrix = compute_pairwise_similarities(descriptors)
    st.success(f"✅ Similarity analysis completed for {n_valid} molecules.")

    # Step 3: Generate molecular pairs and compute landscape indices
    st.write("### Step 3: Generating molecular pairs and landscape metrics...")
    molecular_pairs = []
    min_distance = 1e-2
    
    total_possible_pairs = n_valid * (n_valid - 1) // 2
    progress_display = st.empty()
    progress_indicator = st.progress(0)
    
    pairs_processed = 0
    for i in range(n_valid):
        for j in range(i+1, n_valid):
            if pairs_processed % 1000 == 0:
                progress_display.text(f"Processing pairs: {pairs_processed:,}/{total_possible_pairs:,}")
                progress_indicator.progress(min(pairs_processed / total_possible_pairs, 1.0))
            
            similarity = similarity_matrix[i, j]
            activity_difference = float(abs(activity_values[i] - activity_values[j]))
            max_activity = float(max(activity_values[i], activity_values[j]))
            distance_metric = max(1.0 - similarity, min_distance)
            landscape_index = activity_difference / distance_metric
            
            molecular_pairs.append({
                "Mol1_idx": i, "Mol2_idx": j,
                "Mol1_ID": molecule_ids[i], "Mol2_ID": molecule_ids[j],
                "SMILES1": smiles_strings[i], "SMILES2": smiles_strings[j],
                "Activity1": activity_values[i], "Activity2": activity_values[j],
                "Similarity": similarity, "Activity_Diff": activity_difference,
                "MaxActivity": max_activity, "SALI": landscape_index
            })
            pairs_processed += 1
    
    progress_display.empty()
    progress_indicator.empty()
    
    if not molecular_pairs:
        st.error("No valid molecular pairs generated. Please verify input data.")
        return None, None, None
        
    pairs_dataframe = pd.DataFrame(molecular_pairs)
    st.success(f"✅ Generated {len(pairs_dataframe):,} molecular pair comparisons.")

    # Step 4: Classify pairs into landscape regions
    st.write("### Step 4: Classifying molecular pairs into landscape regions...")
    classified_data = categorize_sar_regions(
        pairs_dataframe, sim_threshold, act_threshold
    )
    
    return classified_data, pairs_dataframe, n_valid

# Data input section
st.subheader("Dataset Input")
data_source = st.radio(
    "Select data source:",
    ("Upload CSV File", "Use Example Dataset"),
    horizontal=True,
)

# Example dataset URL
example_data_url = "https://raw.githubusercontent.com/exampleuser/molecular-data/main/sample_compounds.csv"

# Store data source selection
st.session_state["data_source"] = data_source

uploaded_data = None

if data_source == "Use Example Dataset":
    try:
        response = requests.get(example_data_url)
        if response.status_code == 200:
            uploaded_data = io.StringIO(response.text)
        else:
            st.error(f"Unable to load example dataset (HTTP {response.status_code}).")
            st.stop()
    except Exception as e:
        st.error(f"Error loading example dataset: {e}")
        st.stop()
else:
    uploaded_data = st.file_uploader("Upload CSV File", type=["csv"])

def read_csv_flexible(file_object):
    """Read CSV file with automatic delimiter detection"""
    try:
        return pd.read_csv(file_object)  # Try comma delimiter
    except pd.errors.ParserError:
        return pd.read_csv(file_object, sep=';')  # Try semicolon delimiter

if uploaded_data:
    try:
        input_data = read_csv_flexible(uploaded_data)

        if not input_data.empty:
            st.subheader("Configure Dataset Columns")

            id_column = st.selectbox("Molecule Identifier column:", input_data.columns)
            smiles_column = st.selectbox("Molecular Structure (SMILES) column:", input_data.columns)
            activity_column = st.selectbox("Biological Activity column:", input_data.columns)

            if id_column and smiles_column and activity_column:
                execute_analysis = st.button(f"Execute {analysis_mode} Analysis")

                if execute_analysis:
                    if analysis_mode == "Basic Landscape":
                        analysis_data = input_data[[id_column, smiles_column, activity_column]].rename(
                            columns={
                                id_column: "Molecule ID", 
                                smiles_column: "Smiles", 
                                activity_column: "pIC50"
                            }
                        )

                        landscape_results = analyze_molecular_pairs(
                            analysis_data, radius_param, 
                            similarity_cutoff, activity_cutoff
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Input Dataset")
                            st.dataframe(analysis_data)

                        with col2:
                            st.write("### Landscape Analysis Results")
                            st.dataframe(landscape_results)
                            results_csv = landscape_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results", 
                                data=results_csv, 
                                file_name='molecular_landscape_results.csv',
                                mime='text/csv'
                            )

                        st.subheader("Molecular Landscape Visualization")
                        landscape_plot = create_landscape_visualization(landscape_results)
                        plot_buffer = io.BytesIO()
                        landscape_plot.savefig(plot_buffer, format="png", bbox_inches="tight")
                        plot_buffer.seek(0)

                        st.download_button(
                            label="Download Visualization", 
                            data=plot_buffer, 
                            file_name="Molecular_Landscape_Map.png",
                            mime="image/png"
                        )

                    else:  # Advanced SAR Analysis
                        classified_results, pairs_data, molecule_count = perform_advanced_analysis(
                            input_data, smiles_column, activity_column, id_column, 
                            molecular_representation, radius_param, bit_size,
                            similarity_cutoff, activity_cutoff, 
                            visualization_color, max_visualization_pairs
                        )
                        
                        if classified_results is not None:
                            # Calculate region distribution
                            region_distribution = classified_results['Zone'].value_counts()
                            
                            # Display region statistics
                            st.subheader("📊 Landscape Region Distribution")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Activity Cliffs", region_distribution.get("Activity Cliffs", 0))
                            with col2:
                                st.metric("Consistent SAR", region_distribution.get("Consistent SAR Regions", 0))
                            with col3:
                                st.metric("Scaffold Transitions", region_distribution.get("Scaffold Transitions", 0))
                            with col4:
                                st.metric("Baseline Regions", region_distribution.get("Baseline Regions", 0))

                            # Results visualization section
                            st.markdown("---")
                            st.header("📊 Analysis Visualizations")
                            
                            # Create visualization tabs
                            viz_tabs = st.tabs(["Region Classification", "Interactive Landscape"])
                            
                            with viz_tabs[0]:  # Region classification tab
                                st.subheader("Molecular Landscape Regions")
                                
                                # Manage data size for visualization
                                visualization_data = classified_results
                                if len(classified_results) > max_visualization_pairs:
                                    st.warning(f"Large dataset ({len(classified_results):,} pairs) - sampling {max_visualization_pairs:,} for visualization.")
                                    # Sample proportionally from each region
                                    visualization_data = classified_results.groupby(
                                        'Zone', group_keys=False
                                    ).apply(
                                        lambda x: x.sample(
                                            n=min(len(x), max_visualization_pairs // 4), 
                                            random_state=42
                                        )
                                    )
                                
                                # Generate region visualization
                                region_plot = create_enhanced_visualization(
                                    visualization_data, similarity_cutoff, activity_cutoff
                                )
                                st.pyplot(region_plot)
                                
                                # Download option for region plot
                                buffer = BytesIO()
                                region_plot.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
                                buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Region Map",
                                    data=buffer,
                                    file_name=f"landscape_regions_{molecular_representation}.png",
                                    mime="image/png"
                                )
                                
                                # Region descriptions
                                with st.expander("ℹ️ Region Descriptions"):
                                    st.markdown("""
                                    **Activity Cliffs**: Highly similar structures with significant activity differences  
                                    → Important for understanding activity discontinuities
                                    
                                    **Consistent SAR Regions**: Similar structures with comparable activities  
                                    → Predictable structure-activity relationships
                                    
                                    **Scaffold Transitions**: Different structural scaffolds with similar activities  
                                    → Opportunities for scaffold hopping
                                    
                                    **Baseline Regions**: Diverse structures with varying activities  
                                    → Expected behavior for structurally diverse compounds
                                    """)
                            
                            with viz_tabs[1]:  # Interactive landscape tab
                                st.subheader("Interactive Molecular Landscape")
                                
                                # Prepare data for interactive visualization
                                interactive_data = classified_results
                                if len(classified_results) > max_visualization_pairs:
                                    st.warning(f"Large dataset - sampling {max_visualization_pairs:,} pairs for interactive view.")
                                    interactive_data = classified_results.sample(
                                        n=max_visualization_pairs, random_state=42
                                    )

                                # Create interactive plot
                                interactive_fig = px.scatter(
                                    interactive_data,
                                    x="Similarity",
                                    y="Activity_Diff",
                                    color=visualization_color,
                                    opacity=0.7,
                                    hover_data=[
                                        "Mol1_ID", "Mol2_ID", "Similarity", 
                                        "Activity_Diff", "SALI", "Zone"
                                    ],
                                    title=f"Interactive Molecular Landscape ({molecular_representation})",
                                    width=1000,
                                    height=650,
                                )
                                interactive_fig.update_traces(marker=dict(size=8))
                                
                                # Add threshold guides
                                interactive_fig.add_vline(
                                    x=similarity_cutoff, line_dash="dash", line_color="red"
                                )
                                interactive_fig.add_hline(
                                    y=activity_cutoff, line_dash="dash", line_color="blue"
                                )
                                
                                st.plotly_chart(interactive_fig, use_container_width=True)

                            # Results download section
                            st.markdown("---")
                            st.header("📥 Download Analysis Results")
                            
                            # Download options
                            download_col1, download_col2 = st.columns(2)
                            
                            with download_col1:
                                # Complete results download
                                complete_csv = classified_results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Complete Results (CSV)",
                                    data=complete_csv,
                                    file_name=f"molecular_landscape_analysis_{molecular_representation}.csv",
                                    mime="text/csv"
                                )
                            
                            with download_col2:
                                # Activity cliffs only
                                cliffs_data = classified_results[
                                    classified_results['Zone'] == 'Activity Cliffs'
                                ]
                                cliffs_csv = cliffs_data.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Activity Cliffs Data (CSV)",
                                    data=cliffs_csv,
                                    file_name=f"activity_cliffs_{molecular_representation}.csv",
                                    mime="text/csv"
                                )

                            st.success("🎉 Analysis completed successfully! Download results using the buttons above.")

        else:
            st.error("The provided file is empty or cannot be processed.")

    except Exception as e:
        st.error(f"Analysis error: {e}")

# Information section
info_section = st.expander("Additional Information", expanded=False)
with info_section:
    st.write('''
             #### Technical Support 
             For technical issues or suggestions, please create an issue on our 
             [project repository](https://github.com/exampleuser/molecular-landscape-explorer).

             #### Scientific Context
             Molecular landscape analysis helps identify critical structure-activity 
             relationships in compound datasets, supporting drug discovery and 
             chemical optimization efforts.
             ''')
