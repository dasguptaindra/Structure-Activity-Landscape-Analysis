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

sns.set_style("whitegrid")

# Initialize Session State
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'analysis_mode_state' not in st.session_state:
    st.session_state['analysis_mode_state'] = None

# ==============================================================================
# 2. CORE COMPUTATIONAL FUNCTIONS (CACHED)
# ==============================================================================

@st.cache_data
def compute_density(x, y):
    """Calculate point density using Gaussian KDE for visualization."""
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort() # Sort so densest points are plotted on top
        return z, idx
    except Exception:
        return np.zeros_like(x), np.arange(len(x))

@st.cache_data
def generate_molecular_descriptors(smiles_list, desc_type, radius_param, n_bits):
    """Generate molecular descriptors (Cached for speed)."""
    descriptors = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        smiles_str = str(smiles)
        mol = Chem.MolFromSmiles(smiles_str)
        
        if mol is None:
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
        except:
            continue
    
    return descriptors, valid_indices

@st.cache_data
def compute_similarity_matrix(descriptors):
    """Compute full pairwise similarity matrix."""
    n_molecules = len(descriptors)
    similarity_matrix = np.zeros((n_molecules, n_molecules), dtype=float)
    fps = list(descriptors)
    
    for i in range(n_molecules):
        similarity_matrix[i, i] = 1.0 
        if i < n_molecules - 1:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
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
    """Main processing pipeline with updated Zone Names."""
    df_clean = df.dropna(subset=[smiles_col, act_col]).copy()
    
    smiles_arr = df_clean[smiles_col].astype(str).values
    act_arr = df_clean[act_col].astype(float).values
    
    if id_col != "None" and id_col in df_clean.columns:
        ids_arr = df_clean[id_col].astype(str).values 
    else:
        ids_arr = np.array([f"Mol_{i+1}" for i in range(len(df_clean))])

    # 1. Descriptors
    descriptors, valid_idx = generate_molecular_descriptors(
        smiles_arr, desc_type, radius, bits
    )
    
    if len(descriptors) < 2:
        return None, "Not enough valid molecules."

    # Filter arrays
    act_arr = act_arr[valid_idx]
    ids_arr = ids_arr[valid_idx]
    n_mols = len(descriptors)

    # 2. Similarity Matrix
    sim_matrix = compute_similarity_matrix(descriptors)

    # 3. Generate Pairs
    pairs = []
    min_dist = 1e-3
    
    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            sim = sim_matrix[i, j]
            act_diff = abs(act_arr[i] - act_arr[j])
            
            # -------------------------------------------------------
            # ZONE CLASSIFICATION LOGIC (MODIFIED NAMES)
            # -------------------------------------------------------
            if sim >= sim_thresh and act_diff >= act_thresh:
                zone = 'Activity Cliffs'
            elif sim < sim_thresh and act_diff < act_thresh:
                zone = 'Similarity Cliffs'    # Previously 'Scaffold Transitions'
            elif sim >= sim_thresh and act_diff < act_thresh:
                zone = 'Smooth SAR'           # Previously 'Consistent SAR Regions'
            else:
                zone = 'Nondescriptive Zone'  # Previously 'Baseline Regions'

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
    
    # 4. Calculate Density
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
# 3. UI LAYOUT
# ==============================================================================

st.title("Molecular Landscape Explorer")

st.sidebar.subheader("Technical support")
st.sidebar.markdown(
    """For technical issues or suggestions, please create an issue on the 
    [project repository](https://github.com/dasguptaindra/Structure-Activity-Landscape-Analysis)."""
)

# 1. Analysis Mode
analysis_mode = st.sidebar.radio(
    "Analysis Mode", 
    ["Basic Landscape", "SAS Map Plot"] 
)

# Initialize defaults
radius_param = 2
bit_size = 1024
mol_rep = "ECFP4"

if analysis_mode == "Basic Landscape":
    st.sidebar.info("Basic visual check.")
    selected_fp = st.sidebar.selectbox("Fingerprint", ['ECFP4', 'ECFP6'])
    radius_param = 2 if selected_fp == 'ECFP4' else 3
    sim_cutoff = st.sidebar.slider("Similarity Threshold", 0.5, 1.0, 0.7)
    act_cutoff = st.sidebar.slider("Activity Diff Threshold", 0.5, 5.0, 1.0)
    mol_rep = selected_fp

else:  # SAS Map Plot Mode
    st.sidebar.markdown("### Molecular Representation")
    mol_rep = st.sidebar.selectbox("Type", ["ECFP4", "ECFP6", "MACCS"], index=0)
    
    if mol_rep.startswith("ECFP"):
        radius_param = 2 if mol_rep == "ECFP4" else 3
        bit_size = st.sidebar.selectbox("Bit Dimension", [1024, 2048], index=0)
    else:
        radius_param = 0
        bit_size = 167 

    st.sidebar.markdown("### Visualization Settings")
    
    # Color Mapping options
    viz_color_col = st.sidebar.selectbox(
        "Color Mapping", 
        ["SALI", "Max. Activity", "Density"]
    )
    
    cmap_name = st.sidebar.selectbox(
        "Colormap", 
        ["viridis", "plasma", "inferno", "turbo", "RdYlBu", "jet"],
        index=0
    )
    
    max_viz_pairs = st.sidebar.number_input("Max pairs to plot", 2000, 100000, 10000, 1000)
    
    st.sidebar.markdown("### Landscape Thresholds")
    sim_cutoff = st.sidebar.slider("Similarity Cutoff", 0.1, 0.9, 0.7, 0.05)
    act_cutoff = st.sidebar.slider("Activity Cutoff", 0.5, 4.0, 1.0, 0.1)


# --- DATA INPUT ---
st.subheader("Dataset Input")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)
    except:
        df_input = pd.read_csv(uploaded_file, sep=';')
    
    if not df_input.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            id_col = st.selectbox("ID Column", ["None"] + list(df_input.columns))
        with col2:
            smiles_col = st.selectbox("SMILES Column", df_input.columns)
        with col3:
            act_col = st.selectbox("Activity Column", df_input.columns)

        # RUN ANALYSIS
        if st.button(f"🚀 Generate {analysis_mode}"):
            st.session_state['analysis_results'] = None # Clear old
            
            with st.spinner("Calculating..."):
                results, error_msg = process_landscape_data(
                    df_input, smiles_col, act_col, id_col,
                    mol_rep, radius_param, bit_size, sim_cutoff, act_cutoff
                )
                
                if error_msg:
                    st.error(error_msg)
                else:
                    st.session_state['analysis_results'] = results
                    st.session_state['analysis_mode_state'] = analysis_mode

    # DISPLAY RESULTS
    if st.session_state['analysis_results'] is not None:
        results_df = st.session_state['analysis_results']
        current_mode = st.session_state['analysis_mode_state']
        
        st.markdown("---")
        
        if current_mode == "SAS Map Plot":
            st.header("📊 SAS Map Plot Results")
            
            # Stats with Updated Zone Names
            rc = results_df['Zone'].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Activity Cliffs", int(rc.get("Activity Cliffs", 0)))
            c2.metric("Smooth SAR", int(rc.get("Smooth SAR", 0)))
            c3.metric("Similarity Cliffs", int(rc.get("Similarity Cliffs", 0)))
            c4.metric("Nondescriptive", int(rc.get("Nondescriptive Zone", 0)))
            
            # Plotting
            if len(results_df) > max_viz_pairs:
                plot_df = results_df.sample(n=max_viz_pairs, random_state=42)
            else:
                plot_df = results_df

            fig = px.scatter(
                plot_df,
                x="Similarity",
                y="Activity_Diff",
                color=viz_color_col, 
                color_continuous_scale=cmap_name,
                title=f"SAS Map: Colored by {viz_color_col}",
                hover_data=["Mol1_ID", "Mol2_ID", "SALI", "Zone"],
                opacity=0.7,
                render_mode='webgl'
            )
            
            fig.add_vline(x=sim_cutoff, line_dash="dash", line_color="gray")
            fig.add_hline(y=act_cutoff, line_dash="dash", line_color="gray")
            
            # PLOTLY FONT STYLING (Times New Roman)
            fig.update_layout(
                height=700,
                xaxis_title="Similarity",
                yaxis_title="Activity Difference",
                font=dict(family="Times New Roman", size=16),
                title_font=dict(family="Times New Roman", size=24),
                xaxis=dict(
                    title_font=dict(family="Times New Roman", size=20),
                    tickfont=dict(family="Times New Roman", size=16)
                ),
                yaxis=dict(
                    title_font=dict(family="Times New Roman", size=20),
                    tickfont=dict(family="Times New Roman", size=16)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Downloads
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            
            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            html_bytes = buffer.getvalue().encode()
            
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download Data (CSV)", csv_data, "sas_map_results.csv", "text/csv")
            with d2:
                st.download_button("Download Plot (HTML)", html_bytes, "sas_map_plot.html", "text/html")
                
        else: # Basic Mode
            st.subheader("Basic Landscape Plot")
            
            # MATPLOTLIB FONT CONFIGURATION
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["axes.labelsize"] = 18
            plt.rcParams["axes.titlesize"] = 20
            plt.rcParams["xtick.labelsize"] = 14
            plt.rcParams["ytick.labelsize"] = 14
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Updated Colors Dictionary with New Zone Names
            colors = {
                'Activity Cliffs': 'red', 
                'Smooth SAR': 'green', 
                'Similarity Cliffs': 'purple', 
                'Nondescriptive Zone': 'blue'
            }
            
            for zone, color in colors.items():
                subset = results_df[results_df['Zone'] == zone]
                if not subset.empty:
                    ax.scatter(subset['Similarity'], subset['Activity_Diff'], c=color, label=zone, alpha=0.6)
            
            ax.axvline(sim_cutoff, c='k', ls='--')
            ax.axhline(act_cutoff, c='k', ls='--')
            ax.set_xlabel("Similarity")
            ax.set_ylabel("Activity Diff")
            
            # Set legend font
            plt.legend(prop={'family': 'Times New Roman', 'size': 12})
            
            st.pyplot(fig)
            
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_data, "basic_results.csv", "text/csv")




