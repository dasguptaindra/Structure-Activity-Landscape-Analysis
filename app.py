import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Draw import MolToImage
import plotly.express as px
from sklearn.neighbors import KernelDensity
from streamlit_ketcher import st_ketcher

import os

# ================= APP HEADER =================
st.set_page_config(page_title="SAS Map & Activity Cliffs Explorer", layout="wide")
st.title("🧪 SAS Map & Activity Cliff Analysis Tool")
st.write("Upload dataset → compute SALI → visualize SAS map → inspect molecule pairs & cliffs")

# ================= FILE UPLOAD =================
file = st.file_uploader("📂 Upload CSV file containing SMILES & pIC50 columns", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.success(f"Data loaded: {df.shape[0]} molecules")

    smiles_col = st.selectbox("Select SMILES Column", df.columns)
    activity_col = st.selectbox("Select Activity Column", df.columns)

    df = df[[smiles_col, activity_col]].dropna()

    # ============ Compute fingerprints ============
    fps, mols = [], []
    for s in df[smiles_col]:
        m = Chem.MolFromSmiles(s)
        if m is None:
            fps.append(None)
            mols.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
            mols.append(m)

    df = df[[smiles_col, activity_col]]
    df["Mol"] = mols
    activities = df[activity_col].astype(float).values
    smiles_list = df[smiles_col].astype(str).values

    # ============ Pairwise Similarity + SALI ============
    pairs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if fps[i] is None or fps[j] is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            act_diff = abs(activities[i] - activities[j])
            sali = act_diff / max(1 - sim, 1e-3)
            pairs.append([i, j, sim, act_diff, sali])

    pairs_df = pd.DataFrame(pairs, columns=["Mol1", "Mol2", "Similarity", "ActivityDiff", "SALI"])

    # ============ Top Cliffs ============
    top_cliffs = pairs_df.sort_values("SALI", ascending=False).head(20)

    # ============ Plot SAS Map ============
    fig = px.scatter(
        pairs_df,
        x="Similarity",
        y="ActivityDiff",
        color="SALI",
        title="Structure–Activity Similarity (SAS) Map",
        color_continuous_scale="Viridis",
        hover_data={"SALI": True, "Similarity": True, "ActivityDiff": True},
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============ Highlight Top Cliffs ============
    st.header("🔥 Top Activity Cliffs (High SALI Pairs)")
    st.dataframe(top_cliffs)

    # ============ Select a Pair to View Structures ============
    st.subheader("🔍 Inspect Selected Molecule Pair")

    pair_index = st.slider("Select Pair Index", 0, len(top_cliffs) - 1, 0)
    mol1_idx = int(top_cliffs.iloc[pair_index]["Mol1"])
    mol2_idx = int(top_cliffs.iloc[pair_index]["Mol2"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Molecule 1**")
        st.image(MolToImage(mols[mol1_idx], size=(300, 300)))
        st.code(smiles_list[mol1_idx])

    with col2:
        st.write("**Molecule 2**")
        st.image(MolToImage(mols[mol2_idx], size=(300, 300)))
        st.code(smiles_list[mol2_idx])

    st.info(f"SALI Score: {top_cliffs.iloc[pair_index]['SALI']:.2f}")

    # ================= Ketcher Editor =================
    st.header("🧬 Draw or Edit Structure (Ketcher Viewer)")
    molfile = st_ketcher()
    st.write("Molfile Output:")
    st.code(molfile)

else:
    st.info("Upload dataset to begin analysis")


st.markdown("---")
st.write("Developed with ❤️ for Chemoinformatics & Drug Discovery")
