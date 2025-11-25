import streamlit as st
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np
import matplotlib.pyplot as plt
import io
import plotly.express as px
from scipy.stats import gaussian_kde
import seaborn as sns

# ==============================================================================
# 1. APP CONFIGURATION & SETUP
# ==============================================================================

st.set_page_config(
    page_title="Molecular Landscape Explorer",
    layout="wide",
    page_icon="🧪"
)

# Set visual style
sns.set_style("whitegrid")

# Initialize Session State for data persistence
# This is crucial for download buttons to work correctly without resetting the app
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'analysis_mode_state' not in st.session_state:
    st.session_state['analysis_mode_state'] = None

# ==============================================================================
# 2. CORE COMPUTATIONAL FUNCTIONS (CACHED FOR PERFORMANCE & REPRODUCIBILITY)
# ==============================================================================

@st.cache_data
def compute_density(x, y):
    """
    Calculate point density using Gaussian KDE for visualization.
    Cached to avoid re-computing on every interaction.
    """
    # Stack data and calculate density
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        return z, idx
    except Exception:
        # Fallback if KDE fails (e.g., singular matrix due to low variance)
        return np.zeros_like(x), np.arange(len(x))

@st.cache_data
def generate_molecular_descriptors(smiles_list, desc_type, radius_param, n_bits):
    """
    Generate molecular descriptors/fingerprints.
    Cached: Running this twice on the same dataset will be instant.
    """
    descriptors = []
    valid_indices = []
    problematic_smiles = []
    
    for idx, smiles in enumerate(smiles_list):
        # Ensure string format
        smiles_str = str(smiles)
        mol = Chem.MolFromSmiles(smiles_str)
        
        if mol is None:
            problematic_smiles.append(smiles_str)
            continue
            
        try:
            if desc_type == "ECFP4":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            elif desc_type == "ECFP6":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
            elif desc_type == "MACCS":
                desc = MACCSkeys.GenMACCSKeys(mol)
            else:
                # Custom case
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius_param, nBits=n_bits)
            
            descriptors.append(desc)
            valid_indices.append(idx)
        except Exception as e:
            problematic_smiles.append(f"{smiles_str} (error: {str(e)})")
            continue
    
    return descriptors, valid_indices, problematic_smiles

@st.cache_data
def compute_similarity_matrix(descriptors):
    """
    Compute full pairwise similarity matrix efficiently.
    """
    n_molecules = len(descriptors)
    similarity_matrix = np.zeros((n_molecules, n_molecules), dtype=float)
    
    # Convert descriptors to list for faster iteration
    fps = list(descriptors)
    
    # Optimized bulk calculation
    for i in range(n_molecules):
        similarity_matrix[i, i] = 1.0 
        if i < n_molecules - 1:
            # Calculate similarity of molecule i against all subsequent molecules
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            
            # Fill the symmetric matrix
            for idx, sim in enumerate(sims):
                j = i + 1 + idx
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            
    return similarity_matrix

@st.cache_data
def process_landscape_data(
    df, smiles_col, act_col, id_col, 
    desc_type, radius, bits, sim_thresh, act_thresh
):
    """
    Main processing pipeline.
    Returns a DataFrame of molecular pairs with analysis metrics.
    """
    # Clean Data: Remove rows with missing critical values
    df_clean = df.dropna(subset=[smiles_col, act_col]).copy()
    
    # Extract arrays
    smiles_arr = df_clean[smiles_col].astype(str).values
    act_arr = df_clean[act_col].astype(float).values
    
    # Handle ID column
    if id_col != "None" and id_col in df_clean.columns:
        ids_arr = df_clean[id_col].astype(str).values 
    else:
        ids_arr = np.array([f"Mol_{i+1}" for i in range(len(df_clean))])

    # 1. Compute Descriptors
    descriptors, valid_idx, invalid_smiles = generate_molecular_descriptors(
        smiles_arr, desc_type, radius, bits
    )
    
    if invalid_smiles:
        st.warning(f"Skipped {len(invalid_smiles)} invalid SMILES strings.")
    
    # Filter arrays to keep only valid molecules
    smiles_arr = smiles_arr[valid_idx]
    act_arr = act_arr[valid_idx]
    ids_arr = ids_arr[valid_idx]
    n_mols = len(descriptors)

    if n_mols < 2:
        return None, "Not enough valid molecules to form pairs."

    # 2. Compute Similarity Matrix
    sim_matrix = compute_similarity_matrix(descriptors)

    # 3. Generate Pairs
    pairs = []
    min_dist = 1e-3 # Small constant to prevent division by zero in SALI calculation
    
    # Warning for large datasets
    if n_mols > 1000:
        st.info(f"Processing large dataset ({n_mols} compounds). Generating ~{n_mols**2//2} pairs...")
    
    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            sim = sim_matrix[i, j]
            act_diff = abs(act_arr[i] - act_arr[j])
            
            # Landscape Classification Logic
            if sim >= sim_thresh and act_diff >= act_thresh:
                zone = 'Activity Cliffs'
            elif sim < sim_thresh and act_diff < act_thresh:
                zone = 'Scaffold Transitions'
            elif sim >= sim_thresh and act_diff < act_thresh:
                zone = 'Consistent SAR Regions'
            else:
                zone = 'Baseline Regions'

            # SALI Score (Structure-Activity Landscape Index)
            sali = act_diff / max(1.0 - sim, min_dist)

            pairs.append({
                "Mol1_ID": ids_arr[i], "Mol2_ID": ids_arr[j],
                "Similarity": sim, 
                "Activity_Diff": act_diff,
                "Max. Activity": max(act_arr[i], act_arr[j]),
                "SALI": sali,
                "Zone": zone
            })

    pairs_df = pd.DataFrame(pairs)
    
    # 4. Calculate Density (if enough points exist)
    if not pairs_df.empty and len(pairs_df) > 5:
        try:
            density, _ = compute_density(pairs_df["Similarity"], pairs_df["Activity_Diff"])
            pairs_df["Density"] = density
        except:
            pairs_df["Density"] = 0.0
    else:
        pairs_df["Density"] = 0.0

    return pairs_df, None

# ==============================================================================
# 3. UI LAYOUT & MAIN LOGIC
# ==============================================================================

st.title("Molecular Landscape Explorer")

with st.expander("About Structure-Activity Landscape Analysis", expanded=False):
    st.markdown('''
    **Molecular Landscape Explorer** helps identify key patterns in molecular data:
    - **Activity Cliffs**: Structurally similar compounds with significant activity differences.
    - **SAR Zones**: Regions with predictable structure-activity relationships.
    - **SAS Map**: Structure-Activity Similarity Map.
    ''')

# --- SIDEBAR CONFIGURATION ---
st.sidebar.subheader("Analysis Configuration")
analysis_mode = st.sidebar.radio("Analysis Mode", ["Basic Landscape", "SAS Map Plot"])

# Initialize default params
radius_param = 2
bit_size = 1024
mol_rep = "ECFP4"

if analysis_mode == "Basic Landscape":
    st.sidebar.info("Basic mode: Quick visual check with standard settings.")
    selected_fp = st.sidebar.selectbox("Fingerprint", ['ECFP4', 'ECFP6'])
    radius_param = 2 if selected_fp == 'ECFP4' else 3
    sim_cutoff = st.sidebar.slider("Similarity Threshold", 0.5, 1.0, 0.7)
    act_cutoff = st.sidebar.slider("Activity Diff Threshold", 0.5, 5.0, 1.0)
    # Map simple selection to full params
    mol_rep = selected_fp

else:  # SAS Map Plot
    st.sidebar.markdown("### Molecular Representation")
    mol_rep = st.sidebar.selectbox("Type", ["ECFP4", "ECFP6", "MACCS"], index=0)
    
    if mol_rep.startswith("ECFP"):
        radius_param = 2 if mol_rep == "ECFP4" else 3
        bit_size = st.sidebar.selectbox("Bit Dimension", [1024, 2048], index=0)
    else:
        radius_param = 0
        bit_size = 167 # Fixed for MACCS

    st.sidebar.markdown("### Visualization Settings")
    # UPDATED: Color mapping options including Density
    viz_color_col = st.sidebar.selectbox(
        "Color Mapping", 
        ["SALI", "Max. Activity", "Density"]
    )
    
    cmap_name = st.sidebar.selectbox(
        "Colormap", 
        ["viridis", "plasma", "inferno", "turbo", "RdYlBu", "jet"],
        index=0
    )
    
    max_viz_pairs = st.sidebar.number_input("Max pairs to plot (for performance)", 2000, 100000, 10000, 1000)
    
    st.sidebar.markdown("### Landscape Thresholds")
    sim_cutoff = st.sidebar.slider("Similarity Cutoff", 0.1, 0.9, 0.7, 0.05)
    act_cutoff = st.sidebar.slider("Activity Cutoff", 0.5, 4.0, 1.0, 0.1)


# --- DATA INPUT ---
st.subheader("Dataset Input")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

def load_csv(file):
    try:
        return pd.read_csv(file)
    except:
        # Fallback for semicolon separated files
        return pd.read_csv(file, sep=';')

if uploaded_file:
    df_input = load_csv(uploaded_file)
    
    if not df_input.empty:
        st.write("Preview of uploaded data:")
        st.dataframe(df_input.head())

        col1, col2, col3 = st.columns(3)
        with col1:
            id_col = st.selectbox("ID Column", ["None"] + list(df_input.columns))
        with col2:
            smiles_col = st.selectbox("SMILES Column", df_input.columns)
        with col3:
            act_col = st.selectbox("Activity Column (e.g. pIC50)", df_input.columns)

        # --- EXECUTION BUTTON ---
        if st.button(f"🚀 Generate {analysis_mode}"):
            # Clear previous results to force new analysis
            st.session_state['analysis_results'] = None
            
            with st.spinner("Calculating Molecular Landscape..."):
                # Unified processing call for both modes
                # This ensures robustness for both Basic and Advanced modes
                results, error_msg = process_landscape_data(
                    df_input, smiles_col, act_col, id_col,
                    mol_rep, radius_param, bit_size, sim_cutoff, act_cutoff
                )
                
                if error_msg:
                    st.error(error_msg)
                else:
                    # Save to session state
                    st.session_state['analysis_results'] = results
                    st.session_state['analysis_mode_state'] = analysis_mode

    # --- RESULTS DISPLAY ---
    # Only display if results exist in session state
    if st.session_state['analysis_results'] is not None:
        results_df = st.session_state['analysis_results']
        current_mode = st.session_state['analysis_mode_state']
        
        st.markdown("---")
        
        if current_mode == "SAS Map Plot":
            st.header("📊 SAS Map Plot Results")
            
            # 1. STATISTICS METRICS
            region_counts = results_df['Zone'].value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Activity Cliffs", int(region_counts.get("Activity Cliffs", 0)))
            m2.metric("Consistent SAR", int(region_counts.get("Consistent SAR Regions", 0)))
            m3.metric("Scaffold Trans.", int(region_counts.get("Scaffold Transitions", 0)))
            m4.metric("Baseline Pairs", int(region_counts.get("Baseline Regions", 0)))
            
            # 2. INTERACTIVE PLOT
            # Sub-sample for plotting speed if data is huge
            if len(results_df) > max_viz_pairs:
                # REPRODUCIBILITY: Use fixed random_state
                plot_df = results_df.sample(n=max_viz_pairs, random_state=42)
                st.caption(f"Note: Displaying a random sample of {max_viz_pairs} pairs for performance.")
            else:
                plot_df = results_df

            # Create Plotly Figure
            fig = px.scatter(
                plot_df,
                x="Similarity",
                y="Activity_Diff",
                color=viz_color_col,  # User selected: SALI, Max. Activity, or Density
                color_continuous_scale=cmap_name,
                title=f"SAS Map: Colored by {viz_color_col}",
                hover_data=["Mol1_ID", "Mol2_ID", "SALI", "Zone"],
                opacity=0.7,
                render_mode='webgl' # Significant performance boost for large scatter plots
            )
            
            # Add Threshold Lines
            fig.add_vline(x=sim_cutoff, line_dash="dash", line_color="gray", annotation_text="Sim. Cutoff")
            fig.add_hline(y=act_cutoff, line_dash="dash", line_color="gray", annotation_text="Act. Cutoff")
            
            # Customize Axis Layout
            fig.update_layout(
                xaxis_title="Structural Similarity (Tanimoto)",
                yaxis_title="Activity Difference",
                height=700,
                legend_title_text=viz_color_col,
                font=dict(family="Arial", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. DOWNLOADS
            st.subheader("📥 Downloads")
            
            d1, d2 = st.columns(2)
            
            # CSV Download
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            with d1:
                st.download_button(
                    label="Download Full Results (CSV)",
                    data=csv_data,
                    file_name="sas_map_results.csv",
                    mime="text/csv"
                )
            
            # HTML Plot Download
            # We write the fig to a buffer
            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            html_bytes = buffer.getvalue().encode()
            
            with d2:
                st.download_button(
                    label="Download Interactive Plot (HTML)",
                    data=html_bytes,
                    file_name="sas_map_plot.html",
                    mime="text/html"
                )
                
        else: # Basic Landscape Mode
            st.subheader("Basic Landscape Plot")
            
            # Static Matplotlib Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {
                'Activity Cliffs': 'red', 
                'Consistent SAR Regions': 'green', 
                'Scaffold Transitions': 'purple', 
                'Baseline Regions': 'blue'
            }
            
            # Plot each zone
            for zone, color in colors.items():
                subset = results_df[results_df['Zone'] == zone]
                if not subset.empty:
                    ax.scatter(
                        subset['Similarity'], 
                        subset['Activity_Diff'], 
                        c=color, 
                        label=zone, 
                        alpha=0.6,
                        edgecolors='none'
                    )
            
            ax.axvline(sim_cutoff, c='k', ls='--', alpha=0.5)
            ax.axhline(act_cutoff, c='k', ls='--', alpha=0.5)
            ax.set_xlabel("Structural Similarity")
            ax.set_ylabel("Activity Difference")
            ax.set_title("Basic Structure-Activity Landscape")
            ax.legend()
            
            st.pyplot(fig)
            
            # Simple CSV Download
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results (CSV)", 
                csv_data, 
                "basic_landscape.csv", 
                "text/csv"
            )
             #### Technical Support 
             For technical issues or suggestions, please create an issue on our 
             [project repository](https://github.com/dasguptaindra/SALI-MAP-Analysis).

             #### Scientific Context
             Molecular landscape analysis helps identify critical structure-activity 
             relationships in compound datasets, supporting drug discovery and 
             chemical optimization efforts.
             ''')


