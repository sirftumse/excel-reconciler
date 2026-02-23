import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
import time
from difflib import SequenceMatcher

# --- PERFORMANCE CONFIG ---
pd.set_option("styler.render.max_elements", 20000000)

@st.cache_data(show_spinner="Loading datasets...")
def load_excel_data(f1, f2):
    """Loads Excel files and forces all columns to string to prevent decimal issues"""
    try:
        d1 = pd.read_excel(f1, engine='calamine', dtype=str)
        d2 = pd.read_excel(f2, engine='calamine', dtype=str)
    except:
        d1 = pd.read_excel(f1, dtype=str)
        d2 = pd.read_excel(f2, dtype=str)
    return d1, d2

def get_similarity(a, b):
    """Calculates similarity score between two strings (0.0 to 1.0)"""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 Global Two-Way Reconciliation (Ultimate)")
st.markdown("Features: **Visual Highlighting**, **Clean Blanks (No NaN)**, **No .0 Decimals**, and **Fuzzy Name Matching**.")

# Initialize session state
if 'reconciliation_done' not in st.session_state:
    st.session_state.reconciliation_done = False
    st.session_state.mismatch_df = None
    st.session_state.missing_df = None
    st.session_state.stats = None
    st.session_state.fnames = ["File A", "File B"]

# --- FILE UPLOAD ---
col_a, col_b = st.columns(2)
with col_a:
    file1 = st.file_uploader("Upload Master File (File A)", type=['xlsx'])
with col_b:
    file2 = st.file_uploader("Upload Comparison File (File B)", type=['xlsx'])

if file1 and file2:
    fname1, fname2 = file1.name, file2.name
    st.session_state.fnames = [fname1, fname2]
    df1, df2 = load_excel_data(file1, file2)

    st.divider()
    st.subheader("🛠️ Step 1: Map Columns")
    all_h1, all_h2 = df1.columns.tolist(), df2.columns.tolist()
    
    selected_headers = st.multiselect("Select columns to verify:", options=all_h1, default=all_h1)

    mapping = {}
    grid = st.columns(3)
    for i, h in enumerate(selected_headers):
        with grid[i % 3]:
            d_idx = all_h2.index(h) if h in all_h2 else 0
            mapping[h] = st.selectbox(f"'{h}' in {fname2}:", options=all_h2, index=d_idx)

    id_col = st.selectbox("Anchor Column (Unique ID / Enrollment No):", options=selected_headers)

    if st.button("🚀 Run Intelligent Reconciliation"):
        my_bar = st.progress(0, text="Initializing engine...")
        
        def smart_clean(val):
            """Standardizes data: handles blanks as empty strings, removes .0"""
            if pd.isna(val) or str(val).strip().lower() in ['nan', 'none', '']:
                return "" 
            
            val = str(val).strip()
            if val.endswith('.0'):
                val = val[:-2]
            
            val_lower = val.lower()
            try:
                if len(val_lower) > 5 and any(char in val_lower for char in './-'):
                    return pd.to_datetime(val_lower, dayfirst=True).strftime('%Y%m%d')
            except:
                pass
            
            return re.sub(r'[.\-/_,\s]', '', val_lower)

        def get_norm_df(temp_df, headers_list, is_file2=False):
            """Creates a normalized version using empty strings for blanks"""
            actual_cols = [mapping[h] if is_file2 else h for h in headers_list]
            norm = temp_df[actual_cols].copy().fillna("")
            for col in norm.columns:
                norm[col] = norm[col].apply(smart_clean)
            norm.columns = headers_list 
            return norm

        # 1. Normalization
        my_bar.progress(10, text="Cleaning data and removing .0 decimals...")
        df1_norm = get_norm_df(df1, selected_headers, is_file2=False)
        df2_norm = get_norm_df(df2, selected_headers, is_file2=True)

        # 2. Indexing
        my_bar.progress(20, text="Indexing records...")
        ids_a = set(df1_norm[id_col].tolist())
        ids_b = set(df2_norm[id_col].tolist())
        only_in_b = ids_b - ids_a

        f2_id_lookup = {}
        for idx, val in enumerate(df2_norm[id_col]):
            f2_id_lookup.setdefault(val, []).append(idx)

        f2_fp_lookup = {}
        f2_fps = df2_norm.apply(lambda x: "|".join(x), axis=1)
        for idx, fp in enumerate(f2_fps):
            f2_fp_lookup.setdefault(fp, []).append(idx)

        mismatches, missing_entries, used_f2_rows = [], [], set()
        match_count = 0
        f1_fps = df1_norm.apply(lambda x: "|".join(x), axis=1)

        # 3. Deep Scan
        total = len(df1)
        for i in range(total):
            if i % (max(1, total // 20)) == 0:
                my_bar.progress(25 + int((i/total)*60), text="Analyzing files...")

            row1_id = df1_norm.iloc[i][id_col]
            fp = f1_fps.iloc[i]
            
            if fp in f2_fp_lookup and f2_fp_lookup[fp]:
                match_idx = f2_fp_lookup[fp].pop(0)
                used_f2_rows.add(match_idx)
                match_count += 1
                continue 
            
            if row1_id in f2_id_lookup:
                pot_idx = next((idx for idx in f2_id_lookup[row1_id] if idx not in used_f2_rows), None)
                if pot_idx is not None:
                    used_f2_rows.add(pot_idx)
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[pot_idx][h]]
                    
                    diff_str = ",".join(diffs)
                    
                    sim_text = ""
                    for h in diffs:
                        if "name" in h.lower():
                            score = get_similarity(df1.iloc[i][h], df2.iloc[pot_idx][mapping[h]])
                            sim_text = f" (Name Match: {int(score*100)}%)"
                            break

                    reason = f"Diff in: {diff_str}{sim_text}"
                    
                    # Prepare display rows (using empty strings for blanks)
                    r1 = df1.iloc[i][selected_headers].fillna("").to_dict()
                    r1.update({'Source': fname1, 'Status': 'Mismatch', 'Reason': reason, 'Diff_Cols': diff_str})
                    
                    r2 = {h: df2.iloc[pot_idx][mapping[h]] for h in selected_headers}
                    r2 = {k: ("" if pd.isna(v) or str(v).strip().lower() == "nan" else v) for k, v in r2.items()}
                    r2.update({'Source': fname2, 'Status': 'Mismatch', 'Reason': reason, 'Diff_Cols': diff_str})
                    
                    sep = {k: "---" for k in r1.keys()}
                    sep['Diff_Cols'] = ""
                    
                    mismatches.extend([r1, r2, sep])
                    continue

            r_miss = df1.iloc[i][selected_headers].fillna("").to_dict()
            r_miss.update({'Source': fname1, 'Status': f'Not in {fname2}', 'Reason': 'ID Missing'})
            missing_entries.append(r_miss)

        # 4. Reverse Pass
        for j in range(len(df2)):
            if j not in used_f2_rows and df2_norm.iloc[j][id_col] in only_in_b:
                r_extra = {h: df2.iloc[j][mapping[h]] for h in selected_headers}
                r_extra = {k: ("" if pd.isna(v) or str(v).strip().lower() == "nan" else v) for k, v in r_extra.items()}
                r_extra.update({'Source': fname2, 'Status': f'Not in {fname1}', 'Reason': 'ID Missing'})
                missing_entries.append(r_extra)

        my_bar.progress(100, text="Successfully Reconciled!")
        
        st.session_state.mismatch_df = pd.DataFrame(mismatches).astype(str)
        st.session_state.missing_df = pd.DataFrame(missing_entries).astype(str)
        st.session_state.stats = {"Match": match_count, "Conflict": len(mismatches)//3, "Missing": len(missing_entries)}
        st.session_state.reconciliation_done = True
        my_bar.empty()

# --- UI DISPLAY ---
if st.session_state.reconciliation_done:
    st.divider()
    s = st.session_state.stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Identical Rows", s["Match"])
    m2.metric("Data Mismatches", s["Conflict"])
    m3.metric("Total Missing Records", s["Missing"])

    tab1, tab2 = st.tabs(["⚠️ Mismatches (Highlighted)", "❌ Missing Entries"])
    
    with tab1:
        def color_cells(row):
            diff_set = set(row.get('Diff_Cols', "").split(","))
            return ['background-color: #ffcccc' if col in diff_set else '' for col in row.index]

        if not st.session_state.mismatch_df.empty:
            view_df = st.session_state.mismatch_df
            styled = view_df.style.apply(color_cells, axis=1)
            cols_to_show = [c for c in view_df.columns if c != 'Diff_Cols']
            st.dataframe(styled, use_container_width=True, column_order=cols_to_show)
        else:
            st.success("Perfect Match! No data conflicts found.")

    with tab2:
        st.dataframe(st.session_state.missing_df, use_container_width=True)

    st.divider()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not st.session_state.mismatch_df.empty:
            st.session_state.mismatch_df.drop(columns=['Diff_Cols'], errors='ignore').to_excel(writer, index=False, sheet_name="Mismatches")
        st.session_state.missing_df.to_excel(writer, index=False, sheet_name="Missing_Records")
    
    st.download_button("📥 Download Final Excel Report", output.getvalue(), "reconciliation_report.xlsx")
