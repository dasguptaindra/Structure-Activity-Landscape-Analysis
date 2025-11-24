# app.py - Advanced SAS Map Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.neighbors import KernelDensity
import plotly.express as px
import os
from io import BytesIO
from PIL import Image
import base64
import textwrap

# Try to import ketcher widget (optional)
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except Exception:
    KETCHER_AVAILABLE = False

st.set_page_config(page_title="Advanced SAS Map (SALI) — Streamlit", layout="wide")
st.title("🧭 Advanced SAS Map Generator — SALI / Activity Cliffs")

# ---------- Sidebar: Upload & Params ----------
st.sidebar.header("Input & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
example_btn = st.sidebar.button("Load example (small sample)")

if example_btn:
    # Create a tiny example if user wants to test quickly
    example = pd.DataFrame({
        "SMILES": ["CCO","CCN","CCCl","CCC(=O)O","c1ccccc1O","CC(=O)O","CC(C)O","CC(C)C(=O)O"],
        "pIC50": [6.2, 5.9, 4.5, 7.2, 5.1, 6.8, 4.9, 7.6],
        "ChEMBL ID": [f"CHEMBL{i}" for i in range(1,9)]
    })
    uploaded_file = BytesIO(example.to_csv(index=False).encode("utf-8"))
    st.sidebar.success("Example loaded — go to main area and press Generate")

radius = st.sidebar.slider("Morgan radius", 1, 4, 2)
n_bits = st.sidebar.selectbox("Fingerprint size (bits)", [512, 1024, 2048], index=2)
color_by = st.sidebar.selectbox("Color by", ["SALI", "MaxActivity", "Density"])
top_n = st.sidebar.number_input("Top cliffs to highlight (top SALI)", min_value=1, max_value=1000, value=20)
kde_bandwidth = st.sidebar.slider("KDE bandwidth (if Density)", 0.02, 1.0, 0.12)
max_pairs_plot = st.sidebar.number_input("Max pairs to plot (subsample if too large)", min_value=2000, max_value=200000, value=30000, step=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Indrasis Das Gupta**")

# ---------- Main UI ----------
if uploaded_file is None:
    st.info("Upload a CSV file containing columns: SMILES and an activity (e.g. pIC50).")
    st.stop()

# Read file
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
# Column selectors
cols = list(df.columns)
smiles_col = st.selectbox("SMILES column", cols, index=0)
activity_col = st.selectbox("Activity column", cols, index=1 if len(cols)>1 else 0)
id_col_opt = ["None"] + cols
id_col = st.selectbox("Optional ID column", id_col_opt, index=0)

# Generate button
if st.button("🚀 Generate SAS map and analyze"):
    st.info("Processing — this may take a while for large datasets. Progress messages show below.")
    # Basic filter
    df = df.dropna(subset=[smiles_col, activity_col]).reset_index(drop=True)
    # parse activities
    try:
        activities = df[activity_col].astype(float).values
    except Exception as e:
        st.error(f"Activity column conversion to float failed: {e}")
        st.stop()

    ids = df[id_col].astype(str).values if id_col != "None" else np.array([f"Mol_{i+1}" for i in range(len(df))])
    smiles_list = df[smiles_col].astype(str).values

    # Compute fingerprints
    st.write("Computing fingerprints...")
    fps = []
    valid_idx = []
    invalid_count = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, s in enumerate(smiles_list):
        status_text.text(f"Parsing SMILES {i+1}/{len(smiles_list)}")
        progress_bar.progress((i + 1) / len(smiles_list))
        
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid_count += 1
            fps.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
        fps.append(fp)
        valid_idx.append(i)
    
    progress_bar.empty()
    status_text.empty()
    
    if invalid_count:
        st.warning(f"{invalid_count} invalid SMILES found and will be excluded.")
    
    # Keep only valid entries
    fps = [fps[i] for i in valid_idx]
    activities = activities[valid_idx]
    ids = ids[valid_idx]
    smiles_list = smiles_list[valid_idx]
    n = len(fps)
    st.success(f"Fingerprints computed for {n} molecules.")

    if n < 2:
        st.error("Need at least 2 valid molecules to compute pairs.")
        st.stop()

    # Pairwise similarity
    st.write("Computing pairwise Tanimoto similarities...")
    sim_matrix = np.zeros((n, n), dtype=float)
    
    # Progress for similarity calculation
    sim_progress = st.progress(0)
    sim_status = st.empty()
    
    # compute triangular matrix
    for i in range(n):
        sim_status.text(f"Computing similarities: {i+1}/{n}")
        sim_progress.progress((i + 1) / n)
        for j in range(i, n):
            s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s
    
    sim_progress.empty()
    sim_status.empty()

    # Build pairs
    st.write("Building pair list and computing SALI ...")
    pairs = []
    eps_distance = 1e-2
    
    pair_progress = st.progress(0)
    pair_status = st.empty()
    total_pairs = n * (n - 1) // 2
    
    pair_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_count % 1000 == 0:  # Update progress periodically
                pair_status.text(f"Processing pairs: {pair_count:,}/{total_pairs:,}")
                pair_progress.progress(pair_count / total_pairs)
            
            sim = sim_matrix[i, j]
            act_diff = float(abs(activities[i] - activities[j]))
            max_val = float(max(activities[i], activities[j]))
            distance = max(1.0 - sim, eps_distance)
            sali = act_diff / distance
            pairs.append({
                "Mol1_idx": i, "Mol2_idx": j,
                "Mol1_ID": ids[i], "Mol2_ID": ids[j],
                "SMILES1": smiles_list[i], "SMILES2": smiles_list[j],
                "Similarity": sim, "Activity_Diff": act_diff,
                "MaxActivity": max_val, "SALI": sali
            })
            pair_count += 1
    
    pair_progress.empty()
    pair_status.empty()
    
    pairs_df = pd.DataFrame(pairs)
    st.write(f"Created {len(pairs_df):,} pairs.")

    # Add Density if requested
    if color_by == "Density":
        st.write("Estimating 2D density (KDE)...")
        xy = np.vstack([pairs_df["Similarity"].values, pairs_df["Activity_Diff"].values]).T
        kde = KernelDensity(bandwidth=float(kde_bandwidth)).fit(xy)
        pairs_df["Density"] = np.exp(kde.score_samples(xy))
        st.success("Density estimated.")

    # Mark top N SALI cliffs
    top_n_use = min(int(top_n), len(pairs_df))
    pairs_df["is_top_cliff"] = False
    if top_n_use > 0:
        top_idxs = pairs_df.nlargest(top_n_use, "SALI").index
        pairs_df.loc[top_idxs, "is_top_cliff"] = True

    # Optionally subsample for plotting to avoid massive figures
    plot_df = pairs_df
    if len(pairs_df) > max_pairs_plot:
        st.warning(f"Too many pairs ({len(pairs_df):,}) — subsampling {max_pairs_plot:,} for plotting, but full table is available for download.")
        # Keep all top cliffs and sample the rest
        top_df = pairs_df[pairs_df["is_top_cliff"]]
        other_df = pairs_df[~pairs_df["is_top_cliff"]].sample(n=max_pairs_plot - len(top_df), random_state=42)
        plot_df = pd.concat([top_df, other_df], ignore_index=True)

    # Plotly SAS Map with highlighted top cliffs
    st.write("Rendering interactive SAS map...")
    # marker size: larger for top cliffs
    plot_df = plot_df.copy()
    plot_df["marker_size"] = np.where(plot_df["is_top_cliff"], 10, 6)
    # symbol for top cliffs
    plot_df["symbol"] = np.where(plot_df["is_top_cliff"], "diamond", "circle")

    color_col = color_by if color_by in plot_df.columns else "SALI"
    fig = px.scatter(
        plot_df,
        x="Similarity",
        y="Activity_Diff",
        color=color_col,
        size="marker_size",
        symbol="symbol",
        hover_data=["Mol1_ID", "Mol2_ID", "Similarity", "Activity_Diff", "SALI"],
        title=f"SAS Map — colored by {color_by} (top {top_n_use} cliffs highlighted)",
        width=1000,
        height=650,
    )
    fig.update_traces(marker=dict(opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

    # Sidebar: table and pair selection
    st.markdown("---")
    st.subheader("Top SALI cliffs (table & selection)")
    top_table = pairs_df.sort_values("SALI", ascending=False).head(200).reset_index(drop=True)
    # show top K
    st.dataframe(top_table[["Mol1_ID","Mol2_ID","Similarity","Activity_Diff","MaxActivity","SALI"]])

    # Pair selection dropdown (by index in pairs_df)
    st.subheader("Inspect a pair (view structures & open in Ketcher)")
    pair_selector_options = [
        f"{idx}: {row.Mol1_ID} — {row.Mol2_ID} (SALI={row.SALI:.2f})"
        for idx, row in pairs_df.sort_values("SALI", ascending=False).head(1000).reset_index().iterrows()
    ]
    if len(pair_selector_options) == 0:
        st.info("No pairs available to inspect.")
        st.stop()

    selected_pair_label = st.selectbox("Choose a pair (top SALI list)", pair_selector_options, index=0)
    # extract index
    sel_idx = int(selected_pair_label.split(":")[0])
    sel_row = pairs_df.loc[sel_idx]

    st.markdown("**Selected pair details**")
    st.write({
        "Mol1": sel_row["Mol1_ID"],
        "Mol2": sel_row["Mol2_ID"],
        "Similarity": float(sel_row["Similarity"]),
        "Activity_Diff": float(sel_row["Activity_Diff"]),
        "SALI": float(sel_row["SALI"]),
    })

    # Simple molecule display without SVG rendering
    col1, col2, col3 = st.columns([3,3,2])
    with col1:
        st.markdown(f"**{sel_row['Mol1_ID']}**")
        st.markdown(f"**SMILES:**")
        st.code(sel_row["SMILES1"], language="text")
        st.markdown(f"**Activity:** {activities[sel_row['Mol1_idx']]:.2f}")

    with col2:
        st.markdown(f"**{sel_row['Mol2_ID']}**")
        st.markdown(f"**SMILES:**")
        st.code(sel_row["SMILES2"], language="text")
        st.markdown(f"**Activity:** {activities[sel_row['Mol2_idx']]:.2f}")

    with col3:
        st.markdown("**Actions**")
        # Download pair CSV
        pair_csv = pd.DataFrame([sel_row]).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download pair CSV", data=pair_csv, file_name="selected_pair.csv", mime="text/csv")
        
        # Show Ketcher if available
        if KETCHER_AVAILABLE:
            st.markdown("Open structures in **Ketcher** (editable):")
            st_ketcher(smiles=sel_row["SMILES1"], height=300, key=f"ketcher1_{sel_idx}")
            st_ketcher(smiles=sel_row["SMILES2"], height=300, key=f"ketcher2_{sel_idx}")
        else:
            st.info("Ketcher not available. Install `streamlit-ketcher` to enable an embedded editor/viewer.")

    # Full pairs download and top lists
    st.markdown("---")
    st.subheader("Download full results")
    full_csv = pairs_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download full pairs CSV", data=full_csv, file_name="SAS_pairs_full.csv", mime="text/csv")

    st.success("Analysis complete.")
