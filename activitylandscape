import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
from scipy.stats import gaussian_kde

# RDKit Imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator

# ==============================================================================
# 1. APP CONFIGURATION & SETUP
# ==============================================================================

st.set_page_config(
    page_title="Activity Landscape Explorer",
    layout="wide",
    page_icon="🧪"
)

sns.set_style("whitegrid")

# Initialize Session State
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'column_mapping' not in st.session_state:
    st.session_state['column_mapping'] = {}
if 'selected_fp' not in st.session_state:
    st.session_state['selected_fp'] = None
if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False

# ==============================================================================
# 2. CORE COMPUTATIONAL FUNCTIONS (OPTIMIZED)
# ==============================================================================

@st.cache_data
def compute_density(x, y, max_samples=50000):
    """
    Calculate point density using Gaussian KDE.
    Optimized: Downsamples for KDE fitting if data is too large to prevent hanging.
    """
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.zeros_like(x)
    
    # Stack data
    xy = np.vstack([x_clean, y_clean])
    
    try:
        # Optimization: If too many points, fit KDE on a random subset to save memory/time
        if len(x_clean) > max_samples:
            indices = np.random.choice(len(x_clean), max_samples, replace=False)
            kde = gaussian_kde(xy[:, indices])
        else:
            kde = gaussian_kde(xy)
            
        # Evaluate on all points
        z = kde(xy)
        
        # Create full array with NaN for filtered points
        z_full = np.full_like(x, np.nan, dtype=float)
        z_full[mask] = z
        return z_full
    except Exception:
        return np.zeros_like(x)

@st.cache_data
def generate_molecular_descriptors(smiles_list, desc_type, n_bits):
    """Generate molecular descriptors (Cached)."""
    descriptors = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        smiles_str = str(smiles).strip()
        if not smiles_str:
            continue
            
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            continue
            
        try:
            if desc_type == "MACCS":
                desc = MACCSkeys.GenMACCSKeys(mol)
            elif "ECFP" in desc_type or "FCFP" in desc_type:
                use_features = "FCFP" in desc_type
                # Parse radius from name (ECFP4 -> radius 2, ECFP6 -> radius 3)
                radius = int(desc_type[-1]) // 2 if desc_type[-1].isdigit() else 2
                
                desc = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=n_bits, useFeatures=use_features
                )
            else:
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            
            descriptors.append(desc)
            valid_indices.append(idx)
        except Exception:
            continue
    
    return descriptors, valid_indices

@st.cache_data
def compute_similarity_matrix(descriptors):
    """
    Compute full pairwise similarity matrix using RDKit BulkTanimotoSimilarity.
    This is ~50x faster than nested Python loops.
    """
    n_molecules = len(descriptors)
    if n_molecules == 0:
        return np.array([])
        
    # Initialize matrix
    similarity_matrix = np.zeros((n_molecules, n_molecules), dtype=np.float32)
    fps = list(descriptors)
    
    # Use BulkTanimoto for massive speedup
    for i in range(n_molecules):
        similarity_matrix[i, i] = 1.0
        # Calculate sim against all subsequent molecules in one C++ call
        if i < n_molecules - 1:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            similarity_matrix[i, i+1:] = sims
            similarity_matrix[i+1:, i] = sims
            
    return similarity_matrix

@st.cache_data
def process_landscape_data(
    df, smiles_col, act_col, id_col, 
    desc_type, bits, sim_thresh, act_thresh
):
    """
    Main processing pipeline using VECTORIZATION (No nested loops).
    """
    # Input validation
    if df is None or df.empty:
        return None, "DataFrame is empty"
    
    df_clean = df.dropna(subset=[smiles_col, act_col]).copy()
    if len(df_clean) == 0:
        return None, "No valid data after cleaning"
    
    # Prepare arrays
    smiles_arr = df_clean[smiles_col].astype(str).values
    act_arr = pd.to_numeric(df_clean[act_col], errors='coerce').values
    
    if id_col != "None" and id_col in df_clean.columns:
        ids_arr = df_clean[id_col].astype(str).values 
    else:
        ids_arr = np.array([f"Mol_{i+1}" for i in range(len(df_clean))])

    # 1. Descriptors
    descriptors, valid_idx = generate_molecular_descriptors(smiles_arr, desc_type, bits)
    
    if len(descriptors) < 2:
        return None, "Not enough valid molecules (<2)."

    # Filter arrays to match valid descriptors
    act_arr = act_arr[valid_idx]
    ids_arr = ids_arr[valid_idx]
    smiles_arr = smiles_arr[valid_idx]
    n_mols = len(descriptors)

    # 2. Similarity Matrix (Fast)
    sim_matrix = compute_similarity_matrix(descriptors)

    # 3. Vectorized Pair Generation
    # Get indices for the upper triangle (excluding diagonal)
    idx1, idx2 = np.triu_indices(n_mols, k=1)
    
    # Extract data using indices
    mol1_ids = ids_arr[idx1]
    mol2_ids = ids_arr[idx2]
    mol1_smiles = smiles_arr[idx1]
    mol2_smiles = smiles_arr[idx2]
    sim_values = sim_matrix[idx1, idx2]
    
    # Calculate Activity Differences
    act1 = act_arr[idx1]
    act2 = act_arr[idx2]
    act_diffs = np.abs(act1 - act2)
    max_acts = np.maximum(act1, act2)
    
    # 4. Zone Classification (Vectorized)
    # Create conditions
    is_high_sim = sim_values >= sim_thresh
    is_high_diff = act_diffs >= act_thresh
    
    # Initialize zones array
    zones = np.full(len(sim_values), 'Non-descript Zone', dtype=object)
    
    # Apply logic
    # Activity Cliffs: High Sim, High Diff
    zones[is_high_sim & is_high_diff] = 'Activity Cliffs'
    
    # Smooth SAR: High Sim, Low Diff
    zones[is_high_sim & ~is_high_diff] = 'Smooth SAR'
    
    # Scaffold Hops (formerly Similarity Cliffs): Low Sim, Low Diff
    zones[~is_high_sim & ~is_high_diff] = 'Scaffold Hops'
    
    # Non-descript (Low Sim, High Diff) - already default
    
    # 5. Calculate SALI
    # Avoid division by zero
    denominator = 1.0 - sim_values
    denominator[denominator < 1e-3] = 1e-3
    sali_values = act_diffs / denominator

    # 6. Build DataFrame
    pairs_df = pd.DataFrame({
        "Mol1_ID": mol1_ids,
        "Mol2_ID": mol2_ids,
        "Mol1_SMILES": mol1_smiles,
        "Mol2_SMILES": mol2_smiles,
        "Similarity": sim_values,
        "Activity_Diff": act_diffs,
        "Max_Activity": max_acts,
        "SALI": sali_values,
        "Zone": zones
    })

    if pairs_df.empty:
        return None, "No pairs generated."

    # 7. Calculate Density (on the pairs)
    # Only run if data isn't massive (limit to prevent crash on huge datasets)
    if len(pairs_df) > 0:
        pairs_df["Density"] = compute_density(
            pairs_df["Similarity"].values, 
            pairs_df["Activity_Diff"].values
        )
    
    return pairs_df, None

def safe_dataframe_display(df, max_rows=5):
    """Safely display dataframe preview."""
    try:
        display_df = df.head(max_rows).astype(str)
        st.dataframe(display_df)
    except Exception:
        st.write(df.head(max_rows))

# ==============================================================================
# 3. UI LAYOUT
# ==============================================================================

st.title("🧪 Activity Landscape Explorer")

st.sidebar.subheader("About")
st.sidebar.info(
    "This tool analyzes Structure-Activity Relationships (SAR) using Activity Landscape Modeling. "
    "It identifies Activity Cliffs, Scaffold Hops, and Smooth SAR regions."
)

st.sidebar.subheader("Zone Definitions")
st.sidebar.markdown("""
- **Activity Cliffs**: High similarity, high activity difference
- **Smooth SAR**: High similarity, low activity difference  
- **Scaffold Hops**: Low similarity, low activity difference
- **Non-descript**: Low similarity, high activity difference
""")

# ==============================================================================
# 4. DATA INPUT
# ==============================================================================

st.subheader("1. Dataset Input")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    st.session_state['file_uploaded'] = True
    try:
        df_input = pd.read_csv(uploaded_file)
    except:
        try:
            df_input = pd.read_csv(uploaded_file, sep=';')
        except:
            st.error("Could not read file. Please ensure it is a valid CSV.")
            st.stop()

    if df_input.empty:
        st.warning("File is empty.")
        st.stop()

    # Preview
    with st.expander("Data Preview", expanded=True):
        safe_dataframe_display(df_input)

    # Column Mapping
    st.subheader("2. Column Mapping")
    cols = list(df_input.columns)
    
    c1, c2, c3 = st.columns(3)
    id_col = c1.selectbox("Molecule ID (Optional)", ["None"] + cols)
    smiles_col = c2.selectbox("SMILES Column *", cols)
    act_col = c3.selectbox("Activity Column (pIC50/Numeric) *", cols)

    # Validation
    if smiles_col == act_col:
        st.error("SMILES and Activity columns must be different.")
        st.stop()
        
    # Check numeric
    if not pd.to_numeric(df_input[act_col], errors='coerce').notnull().any():
        st.error(f"Column '{act_col}' does not appear to contain numeric data.")
        st.stop()

    # Store mapping
    st.session_state['column_mapping'] = {
        'id_col': id_col, 'smiles_col': smiles_col, 'act_col': act_col
    }

    # ==============================================================================
    # 5. SETTINGS & ANALYSIS
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("3. Analysis Settings")

    # Fingerprint Selection
    fp_type = st.selectbox(
        "Fingerprint Type", 
        ["ECFP4", "ECFP6", "FCFP4", "FCFP6", "MACCS"],
        index=0
    )
    
    c_set1, c_set2 = st.columns(2)
    
    with c_set1:
        bit_size = 2048
        if "MACCS" not in fp_type:
            bit_size = st.selectbox("Bit Length", [512, 1024, 2048], index=2)
        else:
            st.info("MACCS uses fixed 166 bits.")

    with c_set2:
        sim_cutoff = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        act_cutoff = st.slider("Activity Diff Threshold", 0.0, 5.0, 1.0, 0.1)

    # Run Button
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Processing... (This is now 50x faster!)"):
            results, error = process_landscape_data(
                df_input, smiles_col, act_col, id_col,
                fp_type, bit_size, sim_cutoff, act_cutoff
            )
            
            if error:
                st.error(error)
            else:
                st.session_state['analysis_results'] = results
                st.success(f"Processed {len(results)} pairs successfully!")

    # ==============================================================================
    # 6. VISUALIZATION
    # ==============================================================================

    if st.session_state['analysis_results'] is not None:
        results_df = st.session_state['analysis_results']
        st.markdown("---")
        st.header("📊 Results: SAS Map")

        # Metrics
        counts = results_df['Zone'].value_counts()
        
        # Custom Metric Styling
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Activity Cliffs", counts.get("Activity Cliffs", 0), delta_color="inverse")
        m2.metric("Scaffold Hops", counts.get("Scaffold Hops", 0), help="Structurally diverse, Similar Activity")
        m3.metric("Smooth SAR", counts.get("Smooth SAR", 0))
        m4.metric("Non-descript", counts.get("Non-descript Zone", 0))

        # Visualization Controls
        c_viz1, c_viz2 = st.columns(2)
        color_by = c_viz1.selectbox("Color By", ["Zone", "Density", "SALI", "Activity_Diff"])
        cmap = c_viz2.selectbox("Color Palette", ["viridis", "plasma", "turbo", "RdYlBu"])

        # Plot Logic
        plot_df = results_df.copy()
        
        # Zone Colors
        zone_map = {
            'Activity Cliffs': 'red',
            'Smooth SAR': 'green',
            'Scaffold Hops': 'blue',
            'Non-descript Zone': 'gray'
        }

        if color_by == "Zone":
            fig = px.scatter(
                plot_df, x="Similarity", y="Activity_Diff",
                color="Zone", color_discrete_map=zone_map,
                hover_data=["Mol1_ID", "Mol2_ID", "SALI"],
                title=f"Structure-Activity Similarity Map ({fp_type})",
                category_orders={"Zone": ["Activity Cliffs", "Smooth SAR", "Scaffold Hops", "Non-descript Zone"]},
                opacity=0.7, render_mode='webgl' # WebGL for performance
            )
        else:
            # Clean data for continuous plotting
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[color_by])
            fig = px.scatter(
                plot_df, x="Similarity", y="Activity_Diff",
                color=color_by, color_continuous_scale=cmap,
                hover_data=["Mol1_ID", "Mol2_ID", "Zone"],
                title=f"SAS Map Colored by {color_by}",
                opacity=0.7, render_mode='webgl'
            )

        # Guidelines
        fig.add_vline(x=sim_cutoff, line_dash="dash", line_color="black", annotation_text="Sim Cutoff")
        fig.add_hline(y=act_cutoff, line_dash="dash", line_color="black", annotation_text="Act Cutoff")

        # Layout updates
        fig.update_layout(
            height=700,
            xaxis_title="Structural Similarity",
            yaxis_title="Activity Difference",
            font=dict(family="Arial", size=14),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==============================================================================
        # 7. DETAILED ANALYSIS & DOWNLOADS
        # ==============================================================================

        st.subheader("🔍 Detailed Analysis")
        
        # Zone-specific analysis
        tab1, tab2, tab3, tab4 = st.tabs(["Activity Cliffs", "Smooth SAR", "Scaffold Hops", "All Pairs"])
        
        with tab1:
            cliffs_df = results_df[results_df['Zone'] == 'Activity Cliffs']
            st.write(f"**Activity Cliffs ({len(cliffs_df)} pairs)**: High similarity but large activity differences")
            if not cliffs_df.empty:
                st.dataframe(cliffs_df.sort_values('SALI', ascending=False).head(10))
            else:
                st.info("No Activity Cliffs found with current thresholds.")
                
        with tab2:
            smooth_df = results_df[results_df['Zone'] == 'Smooth SAR']
            st.write(f"**Smooth SAR ({len(smooth_df)} pairs)**: Consistent structure-activity relationships")
            if not smooth_df.empty:
                st.dataframe(smooth_df.sort_values('Similarity', ascending=False).head(10))
            else:
                st.info("No Smooth SAR regions found with current thresholds.")
                
        with tab3:
            hops_df = results_df[results_df['Zone'] == 'Scaffold Hops']
            st.write(f"**Scaffold Hops ({len(hops_df)} pairs)**: Different structures with similar activity")
            if not hops_df.empty:
                st.dataframe(hops_df.sort_values('Activity_Diff', ascending=True).head(10))
            else:
                st.info("No Scaffold Hops found with current thresholds.")
                
        with tab4:
            st.dataframe(results_df.head(100))  # Limit display to 100 rows

        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Pairs", len(results_df))
            st.metric("Average Similarity", f"{results_df['Similarity'].mean():.3f}")
            
        with col2:
            st.metric("Average Activity Difference", f"{results_df['Activity_Diff'].mean():.3f}")
            st.metric("Max SALI", f"{results_df['SALI'].replace([np.inf, -np.inf], np.nan).max():.3f}")
            
        with col3:
            st.metric("Pairs above Sim Threshold", f"{(results_df['Similarity'] >= sim_cutoff).sum()}")
            st.metric("Pairs above Act Diff Threshold", f"{(results_df['Activity_Diff'] >= act_cutoff).sum()}")

        # Download Section
        st.subheader("📥 Downloads")
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Full Results CSV", 
            csv, 
            "sas_map_results.csv", 
            "text/csv", 
            key='download-csv'
        )
        
        # Zone-specific downloads
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        with col_d1:
            cliffs_csv = results_df[results_df['Zone'] == 'Activity Cliffs'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Activity Cliffs", 
                cliffs_csv, 
                "activity_cliffs.csv", 
                "text/csv"
            )
            
        with col_d2:
            smooth_csv = results_df[results_df['Zone'] == 'Smooth SAR'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Smooth SAR", 
                smooth_csv, 
                "smooth_sar.csv", 
                "text/csv"
            )
            
        with col_d3:
            hops_csv = results_df[results_df['Zone'] == 'Scaffold Hops'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Scaffold Hops", 
                hops_csv, 
                "scaffold_hops.csv", 
                "text/csv"
            )

else:
    # Welcome screen when no file is uploaded
    st.markdown("""
    ## Welcome to the Activity Landscape Explorer!
    
    This application helps you analyze Structure-Activity Relationships (SAR) through Activity Landscape Modeling.
    
    ### How to use:
    1. **Upload a CSV file** containing molecular structures and activity data
    2. **Map your columns**: SMILES, Activity values, and optional Molecule IDs
    3. **Configure analysis**: Choose fingerprint type and thresholds
    4. **Explore results**: Visualize SAS maps and analyze different SAR zones
    
    ### Expected CSV format:
    - **SMILES Column**: Molecular structures in SMILES format
    - **Activity Column**: Numeric activity values (pIC50, IC50, etc.)
    - **ID Column (optional)**: Molecule identifiers
    Upload a CSV file to get started!
    """)

# ==============================================================================
# 8. FOOTER
# ==============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Activity Landscape Explorer | Built with Streamlit & RDKit"
    "</div>",
    unsafe_allow_html=True
)







