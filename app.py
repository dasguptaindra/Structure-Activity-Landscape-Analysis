import streamlit as st
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdFingerprintGenerator
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

# ==============================================================================
# 2. CORE COMPUTATIONAL FUNCTIONS (CACHED)
# ==============================================================================

@st.cache_data
def compute_density(x, y):
    """Calculate point density using Gaussian KDE for visualization."""
    # Remove NaN/infinite values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.zeros_like(x)
    
    xy = np.vstack([x_clean, y_clean])
    try:
        z = gaussian_kde(xy)(xy)
        # Create full array with NaN for filtered points
        z_full = np.full_like(x, np.nan, dtype=float)
        z_full[mask] = z
        return z_full
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros_like(x)

@st.cache_data
def generate_molecular_descriptors(smiles_list, desc_type, radius_param, n_bits):
    """Generate molecular descriptors (Cached for speed)."""
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
            if desc_type == "ECFP4":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            elif desc_type == "ECFP6":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
            elif desc_type == "ECFP8":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=n_bits)
            elif desc_type == "ECFP10":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=5, nBits=n_bits)
            elif desc_type == "FCFP4":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useFeatures=True)
            elif desc_type == "FCFP6":
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits, useFeatures=True)
            elif desc_type == "MACCS":
                desc = MACCSkeys.GenMACCSKeys(mol)
            elif desc_type == "RDKit":
                desc = Chem.RDKFingerprint(mol, fpSize=n_bits)
            elif desc_type == "Avalon":
                from rdkit.Avalon import pyAvalonTools
                desc = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
            elif desc_type == "AtomPairs":
                desc = Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            elif desc_type == "TopologicalTorsions":
                desc = Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
            elif desc_type == "Pattern":
                desc = Chem.rdmolops.PatternFingerprint(mol, fpSize=n_bits)
            elif desc_type == "Morgan-Count":
                # Count-based Morgan fingerprint
                fp_gen = rdFingerprintGenerator.GetCountFPGenerator(fpSize=n_bits, radius=radius_param)
                desc = fp_gen.GetCountFingerprint(mol)
            elif desc_type == "Morgan-Chirality":
                # Morgan fingerprint with chirality
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius_param, nBits=n_bits, useChirality=True)
            elif desc_type == "Morgan-Features":
                # Morgan fingerprint with feature flags
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius_param, nBits=n_bits, useFeatures=True)
            else:
                # Default to ECFP4
                desc = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            
            descriptors.append(desc)
            valid_indices.append(idx)
        except Exception as e:
            continue
    
    return descriptors, valid_indices

@st.cache_data
def compute_similarity_matrix(descriptors):
    """Compute full pairwise similarity matrix."""
    n_molecules = len(descriptors)
    if n_molecules == 0:
        return np.array([])
        
    similarity_matrix = np.zeros((n_molecules, n_molecules), dtype=float)
    fps = list(descriptors)
    
    for i in range(n_molecules):
        similarity_matrix[i, i] = 1.0 
        for j in range(i + 1, n_molecules):
            try:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            except Exception:
                similarity_matrix[i, j] = 0.0
                similarity_matrix[j, i] = 0.0
            
    return similarity_matrix

@st.cache_data
def process_landscape_data(
    df, smiles_col, act_col, id_col, 
    desc_type, radius, bits, sim_thresh, act_thresh
):
    """Main processing pipeline with updated Zone Names."""
    # Input validation
    if df is None or df.empty:
        return None, "DataFrame is empty"
    
    if smiles_col not in df.columns or act_col not in df.columns:
        return None, "Required columns not found"
    
    df_clean = df.dropna(subset=[smiles_col, act_col]).copy()
    
    if len(df_clean) == 0:
        return None, "No valid data after cleaning"
    
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
        return None, "Not enough valid molecules after descriptor generation."

    # Filter arrays
    act_arr = act_arr[valid_idx]
    ids_arr = ids_arr[valid_idx]
    smiles_arr = smiles_arr[valid_idx]
    n_mols = len(descriptors)

    # 2. Similarity Matrix
    sim_matrix = compute_similarity_matrix(descriptors)
    
    if sim_matrix.size == 0:
        return None, "Failed to compute similarity matrix"

    # 3. Generate Pairs
    pairs = []
    min_dist = 1e-3
    
    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            sim = sim_matrix[i, j]
            act_diff = abs(act_arr[i] - act_arr[j])
            
            # -------------------------------------------------------
            # ZONE CLASSIFICATION LOGIC - FIXED
            # -------------------------------------------------------
            if sim >= sim_thresh and act_diff >= act_thresh:
                zone = 'Activity Cliffs'
            elif sim < sim_thresh and act_diff < act_thresh:
                zone = 'Similarity Cliffs'
            elif sim >= sim_thresh and act_diff < act_thresh:
                zone = 'Smooth SAR'
            else:
                zone = 'Nondescriptive Zone'

            # Calculate SALI with safe division - FIXED
            denominator = max(1.0 - sim, min_dist)
            sali = act_diff / denominator

            pairs.append({
                "Mol1_ID": ids_arr[i], 
                "Mol2_ID": ids_arr[j],
                "Mol1_SMILES": smiles_arr[i],
                "Mol2_SMILES": smiles_arr[j],
                "Mol1_Activity": act_arr[i],
                "Mol2_Activity": act_arr[j],
                "Similarity": sim, 
                "Activity_Diff": act_diff,
                "Max_Activity": max(act_arr[i], act_arr[j]),  # Fixed column name
                "SALI": sali,
                "Zone": zone
            })

    if not pairs:
        return None, "No valid molecular pairs generated"
        
    pairs_df = pd.DataFrame(pairs)
    
    # 4. Calculate Density - FIXED
    if not pairs_df.empty and len(pairs_df) > 5:
        try:
            # Ensure we have valid numeric data for density calculation
            valid_sim = pairs_df["Similarity"].replace([np.inf, -np.inf], np.nan).dropna()
            valid_act = pairs_df["Activity_Diff"].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_sim) > 1 and len(valid_act) > 1:
                density = compute_density(pairs_df["Similarity"].values, pairs_df["Activity_Diff"].values)
                pairs_df["Density"] = density
            else:
                pairs_df["Density"] = 0.0
        except Exception as e:
            pairs_df["Density"] = 0.0
    else:
        pairs_df["Density"] = 0.0

    return pairs_df, None

# ==============================================================================
# 3. UI LAYOUT
# ==============================================================================

st.title("Activity Landscape Explorer")

st.sidebar.subheader("Technical support")
st.sidebar.markdown(
    """For technical issues or suggestions, please create an issue on the 
    [project repository](https://github.com/dasguptaindra/Structure-Activity-Landscape-Analysis)."""
)

# ==============================================================================
# 4. DATA INPUT & COLUMN MAPPING
# ==============================================================================

st.subheader("Dataset Input")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        try:
            df_input = pd.read_csv(uploaded_file, sep=';')
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    if df_input.empty:
        st.warning("Uploaded file is empty")
        st.stop()
    
    # Column mapping interface
    st.subheader("Column Mapping")
    st.info("Please map your CSV columns to the required fields:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_columns = list(df_input.columns)
        id_col = st.selectbox("ID Column (Optional)", ["None"] + available_columns)
    
    with col2:
        smiles_col = st.selectbox("SMILES Column *", available_columns)
    
    with col3:
        act_col = st.selectbox("Activity Column *", available_columns)
    
    # Show column preview
    st.write("**Column Preview:**")
    preview_df = df_input[[smiles_col, act_col] + ([id_col] if id_col != "None" else [])].head()
    st.dataframe(preview_df)
    
    # Validation checks
    validation_passed = True
    validation_errors = []
    
    # Check if required columns are selected
    if smiles_col == act_col:
        validation_errors.append("SMILES Column and Activity Column cannot be the same")
        validation_passed = False
    
    if id_col != "None" and (id_col == smiles_col or id_col == act_col):
        validation_errors.append("ID Column cannot be the same as SMILES or Activity Column")
        validation_passed = False
    
    # Check for valid data in selected columns
    if smiles_col in df_input.columns:
        if df_input[smiles_col].isna().all():
            validation_errors.append(f"SMILES Column '{smiles_col}' contains only empty values")
            validation_passed = False
        else:
            # Quick check for valid SMILES
            sample_smiles = df_input[smiles_col].dropna().iloc[0]
            if Chem.MolFromSmiles(str(sample_smiles)) is None:
                validation_errors.append(f"Sample SMILES '{sample_smiles}' appears to be invalid")
                validation_passed = False
    
    if act_col in df_input.columns:
        if df_input[act_col].isna().all():
            validation_errors.append(f"Activity Column '{act_col}' contains only empty values")
            validation_passed = False
        else:
            # Try to convert to numeric to check if it's actually numeric data
            try:
                pd.to_numeric(df_input[act_col].dropna())
            except ValueError:
                validation_errors.append(f"Activity Column '{act_col}' contains non-numeric values")
                validation_passed = False

    # Display validation errors
    if not validation_passed:
        st.error("**Validation Errors:**")
        for error in validation_errors:
            st.error(f"• {error}")
        st.info("Please correct the column mapping above and try again.")
        st.stop()
    
    # Store column mapping in session state
    st.session_state['column_mapping'] = {
        'id_col': id_col,
        'smiles_col': smiles_col,
        'act_col': act_col
    }
    
    # ==============================================================================
    # 5. FINGERPRINT SELECTION & ANALYSIS SETTINGS
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("Fingerprint Selection & Analysis Settings")
    
    # Fingerprint categories
    fp_categories = {
        "Extended Connectivity Fingerprints (ECFP)": ["ECFP4", "ECFP6", "ECFP8", "ECFP10"],
        "Functional Connectivity Fingerprints (FCFP)": ["FCFP4", "FCFP6"],
        "Morgan Fingerprints (Advanced)": ["Morgan-Count", "Morgan-Chirality", "Morgan-Features"],
        "Other Fingerprints": ["MACCS", "RDKit", "Avalon", "AtomPairs", "TopologicalTorsions", "Pattern"]
    }
    
    # Create tabs for different fingerprint categories
    fp_tabs = st.tabs(list(fp_categories.keys()))
    
    selected_fingerprint = None
    fingerprint_params = {}
    
    for i, (category, fingerprints) in enumerate(fp_categories.items()):
        with fp_tabs[i]:
            st.write(f"**{category}**")
            
            for fp in fingerprints:
                if st.button(f"Select {fp}", key=f"btn_{fp}"):
                    selected_fingerprint = fp
                    st.session_state['selected_fp'] = fp
            
            # Show which fingerprint is currently selected
            if 'selected_fp' in st.session_state and st.session_state['selected_fp'] in fingerprints:
                st.success(f"✅ Currently selected: **{st.session_state['selected_fp']}**")
    
    # If no fingerprint selected yet, show message
    if 'selected_fp' not in st.session_state:
        st.info("👆 Please select a fingerprint type from the tabs above")
        st.stop()
    
    mol_rep = st.session_state['selected_fp']
    
    # Fingerprint-specific parameters
    st.write("**Fingerprint Parameters:**")
    
    # Set default parameters based on fingerprint type
    default_bits = 2048
    default_radius = 2
    
    if mol_rep in ["ECFP4", "FCFP4"]:
        default_radius = 2
    elif mol_rep in ["ECFP6", "FCFP6"]:
        default_radius = 3
    elif mol_rep == "ECFP8":
        default_radius = 4
    elif mol_rep == "ECFP10":
        default_radius = 5
    
    # Show radius parameter for fingerprints that use it
    if mol_rep in ["ECFP4", "ECFP6", "ECFP8", "ECFP10", "FCFP4", "FCFP6", 
                   "Morgan-Count", "Morgan-Chirality", "Morgan-Features"]:
        radius_param = st.slider("Radius Parameter", 1, 6, default_radius, 1,
                               help="Radius for Morgan fingerprint generation. Larger values capture larger molecular environments.")
    else:
        radius_param = default_radius
        st.info(f"Radius parameter not applicable for {mol_rep}")
    
    # Bit size selection
    if mol_rep in ["ECFP4", "ECFP6", "ECFP8", "ECFP10", "FCFP4", "FCFP6", 
                   "RDKit", "Avalon", "AtomPairs", "TopologicalTorsions", "Pattern",
                   "Morgan-Count", "Morgan-Chirality", "Morgan-Features"]:
        bit_size = st.selectbox("Bit Dimension", [512, 1024, 2048, 4096], index=2)
    else:
        # MACCS has fixed size
        bit_size = 167
        st.info(f"MACCS uses fixed 167-bit fingerprint")
    
    # Display fingerprint description
    fp_descriptions = {
        "ECFP4": "Extended Connectivity Fingerprint (radius 2) - captures atom environments up to 2 bonds away",
        "ECFP6": "Extended Connectivity Fingerprint (radius 3) - captures larger atom environments",
        "ECFP8": "Extended Connectivity Fingerprint (radius 4) - very large atom environments",
        "ECFP10": "Extended Connectivity Fingerprint (radius 5) - extensive atom environments",
        "FCFP4": "Functional Class Fingerprint (radius 2) - based on pharmacophoric features",
        "FCFP6": "Functional Class Fingerprint (radius 3) - larger pharmacophoric features",
        "MACCS": "MACCS Keys - 166 predefined structural fragments",
        "RDKit": "RDKit topological fingerprint - based on hashed atom paths",
        "Avalon": "Avalon fingerprint - linear fingerprints for chemical structures",
        "AtomPairs": "Atom Pair fingerprint - captures pairs of atoms and their distances",
        "TopologicalTorsions": "Topological Torsion fingerprint - captures linear sequences of 4 atoms",
        "Pattern": "Pattern fingerprint - based on SMARTS patterns",
        "Morgan-Count": "Morgan count-based fingerprint - uses counts instead of bits",
        "Morgan-Chirality": "Morgan fingerprint with chirality information",
        "Morgan-Features": "Morgan fingerprint using feature definitions"
    }
    
    if mol_rep in fp_descriptions:
        st.info(f"**{mol_rep}**: {fp_descriptions[mol_rep]}")
    
    # Visualization Settings
    st.markdown("---")
    st.subheader("Visualization Settings")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Color Mapping options
        viz_color_col = st.selectbox(
            "Color Mapping", 
            ["Zone", "SALI", "Max_Activity", "Density"]
        )
        
        cmap_name = st.selectbox(
            "Colormap", 
            ["viridis", "plasma", "inferno", "turbo", "RdYlBu", "jet"],
            index=0
        )
    
    with col_viz2:
        max_viz_pairs = st.number_input("Max pairs to plot", 2000, 100000, 10000, 1000)
        
        # Landscape Thresholds
        sim_cutoff = st.slider("Similarity Cutoff", 0.1, 0.9, 0.7, 0.05)
        act_cutoff = st.slider("Activity Cutoff", 0.5, 4.0, 1.0, 0.1)
    
    # Clear cache button to force recalculation when fingerprints change
    if st.button("Clear Cache & Refresh Analysis"):
        st.cache_data.clear()
        st.session_state['analysis_results'] = None
        st.rerun()
    
    # Display current settings
    st.info(f"**Current Analysis Settings:** {mol_rep} | Radius: {radius_param} | Bits: {bit_size} | Similarity Cutoff: {sim_cutoff} | Activity Cutoff: {act_cutoff}")
    
    # RUN ANALYSIS
    if st.button("🚀 Generate SAS Map Plot", type="primary"):
        st.session_state['analysis_results'] = None  # Clear old results
        
        with st.spinner("Calculating molecular descriptors and similarities..."):
            results, error_msg = process_landscape_data(
                df_input, smiles_col, act_col, id_col,
                mol_rep, radius_param, bit_size, sim_cutoff, act_cutoff
            )
            
            if error_msg:
                st.error(f"Analysis failed: {error_msg}")
            elif results is None:
                st.error("No results generated from analysis")
            else:
                st.session_state['analysis_results'] = results
                st.success(f"Analysis complete! Generated {len(results)} molecular pairs using {mol_rep}.")

    # ==============================================================================
    # 6. RESULTS DISPLAY
    # ==============================================================================
    
    if st.session_state['analysis_results'] is not None:
        results_df = st.session_state['analysis_results']
        
        st.markdown("---")
        st.header(f"📊 SAS Map Plot Results - {mol_rep}")
        
        # Stats with Zone Names
        rc = results_df['Zone'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Activity Cliffs", int(rc.get("Activity Cliffs", 0)))
        c2.metric("Smooth SAR", int(rc.get("Smooth SAR", 0)))
        c3.metric("Similarity Cliffs", int(rc.get("Similarity Cliffs", 0)))
        c4.metric("Nondescriptive Zone", int(rc.get("Nondescriptive Zone", 0)))
        
        # Plotting
        if len(results_df) > max_viz_pairs:
            plot_df = results_df.sample(n=max_viz_pairs, random_state=42)
            st.info(f"Displaying {max_viz_pairs} random pairs from {len(results_df)} total pairs")
        else:
            plot_df = results_df

        # Custom color mapping for zones with correct spelling
        zone_colors = {
            'Activity Cliffs': 'red',
            'Smooth SAR': 'green', 
            'Similarity Cliffs': 'blue',
            'Nondescriptive Zone': 'orange'
        }

        # Handle categorical vs continuous color mapping
        if viz_color_col == "Zone":
            fig = px.scatter(
                plot_df,
                x="Similarity",
                y="Activity_Diff",
                color="Zone",
                color_discrete_map=zone_colors,
                title=f"SAS Map ({mol_rep}): Colored by Zone",
                hover_data=["Mol1_ID", "Mol2_ID", "SALI", "Zone"],
                opacity=0.7,
                render_mode='webgl',
                category_orders={"Zone": ["Activity Cliffs", "Smooth SAR", "Similarity Cliffs", "Nondescriptive Zone"]}
            )
        else:
            # For continuous color scales, ensure data is valid
            if viz_color_col in plot_df.columns:
                # Remove infinite values for continuous color scales
                plot_df_clean = plot_df.replace([np.inf, -np.inf], np.nan)
                
                fig = px.scatter(
                    plot_df_clean,
                    x="Similarity",
                    y="Activity_Diff",
                    color=viz_color_col, 
                    color_continuous_scale=cmap_name,
                    title=f"SAS Map ({mol_rep}): Colored by {viz_color_col}",
                    hover_data=["Mol1_ID", "Mol2_ID", "SALI", "Zone"],
                    opacity=0.7,
                    render_mode='webgl'
                )
            else:
                st.error(f"Column '{viz_color_col}' not found in results")
                fig = px.scatter(
                    plot_df,
                    x="Similarity",
                    y="Activity_Diff",
                    color="Zone",
                    color_discrete_map=zone_colors,
                    title=f"SAS Map ({mol_rep}): Colored by Zone",
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
                tickfont=dict(family="Times New Roman", size=16),
                range=[0, 1]  # Fixed range for similarity
            ),
            yaxis=dict(
                title_font=dict(family="Times New Roman", size=20),
                tickfont=dict(family="Times New Roman", size=16)
            ),
            legend=dict(
                title_font=dict(family="Times New Roman", size=14),
                font=dict(family="Times New Roman", size=12)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show zone distribution only (removed statistical summary)
        st.subheader("Zone Distribution")
        zone_dist = results_df['Zone'].value_counts().reset_index()
        zone_dist.columns = ['Zone', 'Count']
        zone_dist['Percentage'] = (zone_dist['Count'] / len(results_df) * 100).round(2)
        st.dataframe(zone_dist)
        
        # Downloads
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        
        buffer = io.StringIO()
        fig.write_html(buffer, include_plotlyjs='cdn')
        html_bytes = buffer.getvalue().encode()
        
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download Data (CSV)", csv_data, f"sas_map_results_{mol_rep}.csv", "text/csv")
        with d2:
            st.download_button("Download Plot (HTML)", html_bytes, f"sas_map_plot_{mol_rep}.html", "text/html")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
