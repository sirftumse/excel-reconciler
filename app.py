import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

# --- PERFORMANCE CONFIG ---
pd.set_option("styler.render.max_elements", 20000000)

@st.cache_data(show_spinner="Loading data...")
def load_excel_data(f1, f2):
    try:
        d1 = pd.read_excel(f1, engine='calamine')
        d2 = pd.read_excel(f2, engine='calamine')
    except:
        d1 = pd.read_excel(f1)
        d2 = pd.read_excel(f2)
    return d1, d2

st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 Global Two-Way Reconciliation")
st.markdown("Identifies exactly what is in one file but missing from the other using actual filenames.")

col_a, col_b = st.columns(2)
with col_a:
    file1 = st.file_uploader("Upload Master File (File A)", type=['xlsx'])
with col_b:
    file2 = st.file_uploader("Upload Comparison File (File B)", type=['xlsx'])

if file1 and file2:
    # Capture the actual names of the files
    fname1, fname2 = file1.name, file2.name
    df1, df2 = load_excel_data(file1, file2)

    st.divider()
    st.subheader("🛠️ Step 1: Map Columns")
    
    all_headers_f1 = df1.columns.tolist()
    all_headers_f2 = df2.columns.tolist()
    
    selected_headers = st.multiselect("Select columns to verify:", options=all_headers_f1, default=all_headers_f1)

    mapping = {}
    grid = st.columns(3)
    for i, h in enumerate(selected_headers):
        with grid[i % 3]:
            d_idx = all_headers_f2.index(h) if h in all_headers_f2 else 0
            mapping[h] = st.selectbox(f"'{h}' in {fname2}:", options=all_headers_f2, index=d_idx)

    id_col = st.selectbox("Anchor Column (Unique ID / Enrollment No):", options=selected_headers)

    if st.button("🚀 Run Secure Two-Way Search"):
        
        def smart_clean(val):
            val = str(val).strip().lower()
            val = re.sub(r'[.\-/_,\s]', '', val)
            return val if val not in ['nan', '', 'none'] else 'empty'

        def get_norm_df(temp_df, headers_list, is_file2=False):
            actual_cols = [mapping[h] if is_file2 else h for h in headers_list]
            norm = temp_df[actual_cols].copy().fillna("")
            for col in norm.columns:
                norm[col] = norm[col].apply(smart_clean)
            norm.columns = headers_list 
            return norm

        # 1. Normalize
        df1_norm = get_norm_df(df1, selected_headers, is_file2=False)
        df2_norm = get_norm_df(df2, selected_headers, is_file2=True)

        # 2. Build Lookups
        f2_id_lookup = {}
        for idx, val in enumerate(df2_norm[id_col]):
            f2_id_lookup.setdefault(val, []).append(idx)

        f1_id_set = set(df1_norm[id_col].tolist()) 

        f2_fp_lookup = {}
        f2_fps = df2_norm.apply(lambda x: "|".join(x), axis=1)
        for idx, fp in enumerate(f2_fps):
            f2_fp_lookup.setdefault(fp, []).append(idx)

        final_results = []
        used_f2_rows = set()
        stats = {"Match": 0, "Conflict": 0, "Miss1": 0, "Miss2": 0}

        # --- FORWARD PASS (A -> B) ---
        f1_fps = df1_norm.apply(lambda x: "|".join(x), axis=1)
        for i in range(len(df1)):
            row1_id = df1_norm.iloc[i][id_col]
            fp = f1_fps.iloc[i]
            
            if fp in f2_fp_lookup and f2_fp_lookup[fp]:
                match_idx = f2_fp_lookup[fp].pop(0)
                used_f2_rows.add(match_idx)
                stats["Match"] += 1
                continue 
            
            if row1_id in f2_id_lookup:
                potential_idx = None
                for idx in f2_id_lookup[row1_id]:
                    if idx not in used_f2_rows:
                        potential_idx = idx
                        break
                
                if potential_idx is not None:
                    used_f2_rows.add(potential_idx)
                    stats["Conflict"] += 1
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[potential_idx][h]]
                    reason = f"Diff in: {', '.join(diffs)}"
                    
                    r1 = df1.iloc[i][selected_headers].to_dict()
                    r1.update({'Source File': fname1, 'Status': 'Data Conflict', 'Reason': reason})
                    r2 = {h: df2.iloc[potential_idx][mapping[h]] for h in selected_headers}
                    r2.update({'Source File': fname2, 'Status': 'Data Conflict', 'Reason': reason})
                    final_results.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    continue

            # This entry is in A but not in B
            stats["Miss1"] += 1
            r_miss = df1.iloc[i][selected_headers].to_dict()
            r_miss.update({'Source File': fname1, 'Status': f'Not in {fname2}', 'Reason': 'ID Missing in Comparison'})
            final_results.append(r_miss)

        # --- REVERSE PASS (B -> A) ---
        for j in range(len(df2)):
            if j not in used_f2_rows:
                row2_id = df2_norm.iloc[j][id_col]
                if row2_id not in f1_id_set:
                    stats["Miss2"] += 1
                    r_extra = {h: df2.iloc[j][mapping[h]] for h in selected_headers}
                    r_extra.update({'Source File': fname2, 'Status': f'Not in {fname1}', 'Reason': 'ID Missing in Master'})
                    final_results.append(r_extra)

        # --- UI DISPLAY ---
        res_df = pd.DataFrame(final_results).astype(object)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exact Matches", stats["Match"])
        m2.metric("Conflicts", stats["Conflict"])
        m3.metric(f"Only in {fname1}", stats["Miss1"])
        m4.metric(f"Only in {fname2}", stats["Miss2"])
        
        st.dataframe(res_df, width='stretch')

        if not res_df.empty:
            output = BytesIO()
            if len(res_df) > 1000000:
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results (CSV)", csv, "reconciliation.csv")
            else:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button("📥 Download Results (Excel)", output.getvalue(), "reconciliation.xlsx")
