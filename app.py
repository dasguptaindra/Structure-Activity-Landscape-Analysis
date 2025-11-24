# app.py - Advanced SAS Map Streamlit app
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib style
sns.set_style("whitegrid")

st.set_page_config(page_title="Advanced SAS Map (SALI) — Streamlit", layout="wide")
st.title("🧭 Advanced SAS Map Generator — SALI / Activity Cliffs")

# ---------- Sidebar: Upload & Parameters ----------
st.sidebar.header("Input & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Fingerprint type selection
fingerprint_type = st.sidebar.selectbox(
    "Fingerprint Type", 
    ["ECFP4", "ECFP6", "MACCS"], 
    index=0
)

# Conditional parameters based on fingerprint type
if fingerprint_type.startswith("ECFP"):
    radius = st.sidebar.slider("Morgan radius", 1, 4, 2 if fingerprint_type == "ECFP4" else 3)
    n_bits = st.sidebar.selectbox("Fingerprint size (bits)", [512, 1024, 2048], index=2)
else:  # MACCS
    radius = None
    n_bits = 167  # MACCS has fixed size

color_by = st.sidebar.selectbox("Color by", ["SALI", "MaxActivity"])
max_pairs_plot = st.sidebar.number_input("Max pairs to plot", min_value=2000, max_value=200000, value=10000, step=1000)

# Zone classification parameters
st.sidebar.header("Zone Classification")
similarity_threshold = st.sidebar.slider("Similarity threshold", 0.1, 0.9, 0.5, 0.05)
activity_threshold = st.sidebar.slider("Activity threshold", 0.1, 5.0, 1.0, 0.1)

# ---------- Functions ----------
def compute_fingerprints(smiles_list, fp_type, radius, n_bits):
    """Compute fingerprints for a list of SMILES"""
    fps = []
    valid_idx = []
    invalid_smiles = []
    
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid_smiles.append(s)
            continue
            
        try:
            if fp_type == "ECFP4":
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
            elif fp_type == "ECFP6":
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=n_bits)
            elif fp_type == "MACCS":
                fp = MACCSkeys.GenMACCSKeys(m)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
            
            fps.append(fp)
            valid_idx.append(i)
        except Exception as e:
            invalid_smiles.append(f"{s} (error: {str(e)})")
            continue
    
    return fps, valid_idx, invalid_smiles

def compute_similarity_matrix(fps):
    """Compute pairwise Tanimoto similarity matrix"""
    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                try:
                    s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s
                except Exception:
                    sim_matrix[i, j] = 0.0
                    sim_matrix[j, i] = 0.0
    
    return sim_matrix

def classify_zones(pairs_df, similarity_threshold, activity_threshold):
    """Classify pairs into SAR zones"""
    classified_df = pairs_df.copy()
    
    # Initialize Zone column
    classified_df['Zone'] = 'Non-descript Zones'
    
    # Activity Cliffs: High similarity, high activity difference
    cliffs_mask = (classified_df['Similarity'] > similarity_threshold) & (classified_df['Activity_Diff'] > activity_threshold)
    classified_df.loc[cliffs_mask, 'Zone'] = 'Activity Cliffs'
    
    # Smooth SAR Zones: High similarity, low activity difference
    smooth_mask = (classified_df['Similarity'] > similarity_threshold) & (classified_df['Activity_Diff'] <= activity_threshold)
    classified_df.loc[smooth_mask, 'Zone'] = 'Smooth SAR Zones'
    
    # Scaffold Hops: Low similarity, low activity difference
    hops_mask = (classified_df['Similarity'] <= similarity_threshold) & (classified_df['Activity_Diff'] <= activity_threshold)
    classified_df.loc[hops_mask, 'Zone'] = 'Scaffold Hops'
    
    return classified_df

def create_zone_plot(plot_df, similarity_threshold, activity_threshold):
    """Create enhanced zone classification plot"""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create the zone classification plot
    sns.scatterplot(
        data=plot_df.sort_values("Zone"),
        x="Similarity",
        y="Activity_Diff",
        hue="Zone",
        palette={
            "Smooth SAR Zones": "green",
            "Non-descript Zones": "blue",
            "Scaffold Hops": "orange",
            "Activity Cliffs": "red",
        },
        alpha=0.6,
        s=30,
        edgecolor=None,
        ax=ax
    )
    
    # Add threshold lines
    ax.axvline(x=similarity_threshold, color='red', linestyle='--', alpha=0.8, 
               label=f'Similarity threshold = {similarity_threshold}')
    ax.axhline(y=activity_threshold, color='blue', linestyle='--', alpha=0.8, 
               label=f'Activity threshold = {activity_threshold}')
    
    plt.title("Activity Landscape Zones (Tanimoto Similarity vs. Activity Difference)", fontsize=15)
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Absolute Activity Difference")
    plt.legend(title="SAR Zone")
    plt.grid(True)
    
    plt.tight_layout()
    return fig

# ---------- Main UI ----------
if uploaded_file is None:
    st.info("📁 Upload a CSV file containing columns: SMILES and an activity (e.g. pIC50).")
    st.stop()

# Read file
try:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Molecules", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Column selectors
cols = list(df.columns)
smiles_col = st.selectbox("SMILES column", cols, index=0)
activity_col = st.selectbox("Activity column", cols, index=1 if len(cols)>1 else 0)
id_col_opt = ["None"] + cols
id_col = st.selectbox("Optional ID column", id_col_opt, index=0)

# Generate button
if st.button("🚀 Generate SAS map and analyze"):
    st.info("Processing — this may take a while for large datasets.")
    
    # Basic filter and validation
    df_clean = df.dropna(subset=[smiles_col, activity_col]).copy()
    if len(df_clean) == 0:
        st.error("No valid data after removing rows with missing SMILES or activity values.")
        st.stop()
        
    # Parse activities
    try:
        activities = df_clean[activity_col].astype(float).values
    except Exception as e:
        st.error(f"Activity column conversion to float failed: {e}")
        st.stop()

    ids = df_clean[id_col].astype(str).values if id_col != "None" else np.array([f"Mol_{i+1}" for i in range(len(df_clean))])
    smiles_list = df_clean[smiles_col].astype(str).values

    # Step 1: Compute fingerprints
    st.write(f"### Step 1: Computing {fingerprint_type} fingerprints...")
    fps, valid_idx, invalid_smiles = compute_fingerprints(smiles_list, fingerprint_type, radius, n_bits)
    
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid SMILES found and excluded.")
        with st.expander("Show invalid SMILES"):
            for bad_smiles in invalid_smiles[:10]:
                st.write(bad_smiles)
            if len(invalid_smiles) > 10:
                st.write(f"... and {len(invalid_smiles) - 10} more")
    
    # Keep only valid entries
    activities = activities[valid_idx]
    ids = ids[valid_idx]
    smiles_list = smiles_list[valid_idx]
    n = len(fps)
    
    if n < 2:
        st.error("Need at least 2 valid molecules to compute pairs.")
        st.stop()

    st.success(f"✅ Fingerprints computed for {n} molecules using {fingerprint_type}.")

    # Step 2: Compute similarity matrix
    st.write("### Step 2: Computing pairwise Tanimoto similarities...")
    sim_matrix = compute_similarity_matrix(fps)
    st.success(f"✅ Similarity matrix computed for {n} molecules.")

    # Step 3: Build pairs and compute SALI
    st.write("### Step 3: Building pair list and computing SALI...")
    pairs = []
    eps_distance = 1e-2
    
    total_pairs = n * (n - 1) // 2
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    pair_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_count % 1000 == 0:
                progress_text.text(f"Processing pairs: {pair_count:,}/{total_pairs:,}")
                progress_bar.progress(min(pair_count / total_pairs, 1.0))
            
            sim = sim_matrix[i, j]
            act_diff = float(abs(activities[i] - activities[j]))
            max_val = float(max(activities[i], activities[j]))
            distance = max(1.0 - sim, eps_distance)
            sali = act_diff / distance
            
            pairs.append({
                "Mol1_idx": i, "Mol2_idx": j,
                "Mol1_ID": ids[i], "Mol2_ID": ids[j],
                "SMILES1": smiles_list[i], "SMILES2": smiles_list[j],
                "Activity1": activities[i], "Activity2": activities[j],
                "Similarity": sim, "Activity_Diff": act_diff,
                "MaxActivity": max_val, "SALI": sali
            })
            pair_count += 1
    
    progress_text.empty()
    progress_bar.empty()
    
    if not pairs:
        st.error("No valid pairs were generated. Check your data.")
        st.stop()
        
    pairs_df = pd.DataFrame(pairs)
    st.success(f"✅ Created {len(pairs_df):,} molecular pairs.")

    # ---------- Zone Classification ----------
    st.write("### Step 4: Classifying pairs into SAR zones...")
    classified_df = classify_zones(pairs_df, similarity_threshold, activity_threshold)
    
    # Count zones
    zone_counts = classified_df['Zone'].value_counts()
    
    # Display zone statistics
    st.subheader("📊 Zone Classification Results")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Activity Cliffs", zone_counts.get("Activity Cliffs", 0))
    with col2:
        st.metric("Smooth SAR Zones", zone_counts.get("Smooth SAR Zones", 0))
    with col3:
        st.metric("Scaffold Hops", zone_counts.get("Scaffold Hops", 0))
    with col4:
        st.metric("Non-descript Zones", zone_counts.get("Non-descript Zones", 0))

    # ---------- RESULTS VISUALIZATION ----------
    st.markdown("---")
    st.header("📊 Results Visualization")
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Zone Classification Map", "Interactive SAS Map"])
    
    with tabs[0]:  # Zone Classification tab
        st.subheader("Activity Landscape Zone Classification")
        
        # Optionally subsample for plotting
        plot_df = classified_df
        if len(classified_df) > max_pairs_plot:
            st.warning(f"Too many pairs ({len(classified_df):,}) — subsampling {max_pairs_plot:,} for plotting.")
            # Sample proportionally from each zone to maintain distribution
            plot_df = classified_df.groupby('Zone', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max_pairs_plot // 4), random_state=42)
            )
        
        # Create the zone plot
        fig = create_zone_plot(plot_df, similarity_threshold, activity_threshold)
        st.pyplot(fig)
        
        # Add download button for the zone plot
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Zone Classification Map (PNG)",
            data=buf,
            file_name=f"zone_classification_{fingerprint_type}.png",
            mime="image/png"
        )
        
        # Zone descriptions
        with st.expander("ℹ️ Zone Descriptions"):
            st.markdown("""
            **Activity Cliffs**: High structural similarity but large activity differences  
            → Important for understanding SAR discontinuities
            
            **Smooth SAR Zones**: High structural similarity with small activity differences  
            → Predictable structure-activity relationships
            
            **Scaffold Hops**: Low structural similarity and small activity differences  
            → Different scaffolds with similar activity levels
            
            **Non-descript Zones**: Low structural similarity but large activity differences  
            → Expected behavior for structurally diverse compounds
            """)
    
    with tabs[1]:  # Interactive SAS Map tab
        st.subheader("Interactive SAS Activity Landscape Map")
        
        # Optionally subsample for plotting
        plot_df_interactive = classified_df
        if len(classified_df) > max_pairs_plot:
            st.warning(f"Too many pairs ({len(classified_df):,}) — subsampling {max_pairs_plot:,} for plotting.")
            plot_df_interactive = classified_df.sample(n=max_pairs_plot, random_state=42)

        # Create interactive plot
        fig = px.scatter(
            plot_df_interactive,
            x="Similarity",
            y="Activity_Diff",
            color=color_by,
            opacity=0.7,
            hover_data=["Mol1_ID", "Mol2_ID", "Similarity", "Activity_Diff", "SALI", "Zone"],
            title=f"Interactive SAS Map ({fingerprint_type}) — colored by {color_by}",
            width=1000,
            height=650,
        )
        fig.update_traces(marker=dict(size=8))
        
        # Add threshold lines to interactive plot
        fig.add_vline(x=similarity_threshold, line_dash="dash", line_color="red")
        fig.add_hline(y=activity_threshold, line_dash="dash", line_color="blue")
        
        st.plotly_chart(fig, use_container_width=True)

    # ---------- DOWNLOAD SECTION ----------
    st.markdown("---")
    st.header("📥 Download Results")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Full pairs data as CSV
        csv_data = classified_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Pairs with Zones (CSV)",
            data=csv_data,
            file_name=f"SAS_pairs_zones_{fingerprint_type}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Activity cliffs only
        activity_cliffs = classified_df[classified_df['Zone'] == 'Activity Cliffs']
        cliffs_csv = activity_cliffs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Activity Cliffs Only (CSV)",
            data=cliffs_csv,
            file_name=f"activity_cliffs_{fingerprint_type}.csv",
            mime="text/csv"
        )

    st.success("🎉 Analysis complete! Use the download buttons above to save your results.")
