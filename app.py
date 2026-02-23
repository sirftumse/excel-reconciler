import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
import time

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
st.markdown("Finds matches, conflicts, and missing records with **Live Progress Tracking**.")

# Initialize session state for persistent results
if 'reconciliation_done' not in st.session_state:
    st.session_state.reconciliation_done = False
    st.session_state.mismatch_df = None
    st.session_state.missing_df = None
    st.session_state.stats = None
    st.session_state.fnames = ["File A", "File B"]

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
    
    all_h1 = df1.columns.tolist()
    all_h2 = df2.columns.tolist()
    
    selected_headers = st.multiselect("Select columns to verify:", options=all_h1, default=all_h1)

    mapping = {}
    grid = st.columns(3)
    for i, h in enumerate(selected_headers):
        with grid[i % 3]:
            d_idx = all_h2.index(h) if h in all_h2 else 0
            mapping[h] = st.selectbox(f"'{h}' in {fname2}:", options=all_h2, index=d_idx)

    id_col = st.selectbox("Anchor Column (Unique ID / Enrollment No):", options=selected_headers)

    if st.button("🚀 Run Secure Two-Way Search"):
        
        # --- PROGRESS BAR INITIALIZATION ---
        progress_text = "Processing data... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
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
        my_bar.progress(10, text="Step 1/4: Normalizing data formats...")
        df1_norm = get_norm_df(df1, selected_headers, is_file2=False)
        df2_norm = get_norm_df(df2, selected_headers, is_file2=True)

        # 2. Build Lookups
        my_bar.progress(25, text="Step 2/4: Building search index...")
        f2_id_lookup = {}
        for idx, val in enumerate(df2_norm[id_col]):
            f2_id_lookup.setdefault(val, []).append(idx)

        f1_id_set = set(df1_norm[id_col].tolist()) 

        f2_fp_lookup = {}
        f2_fps = df2_norm.apply(lambda x: "|".join(x), axis=1)
        for idx, fp in enumerate(f2_fps):
            f2_fp_lookup.setdefault(fp, []).append(idx)

        mismatches = []
        missing_entries = []
        used_f2_rows = set()
        match_count = 0

        # --- FORWARD PASS (File A -> File B) ---
        f1_fps = df1_norm.apply(lambda x: "|".join(x), axis=1)
        total_rows = len(df1)
        
        for i in range(total_rows):
            # Update progress bar every 10% for performance
            if i % (max(1, total_rows // 10)) == 0:
                perc = 25 + int((i / total_rows) * 50)
                my_bar.progress(perc, text=f"Step 3/4: Comparing {fname1} (Row {i}/{total_rows})")

            row1_id = df1_norm.iloc[i][id_col]
            fp = f1_fps.iloc[i]
            
            if fp in f2_fp_lookup and f2_fp_lookup[fp]:
                match_idx = f2_fp_lookup[fp].pop(0)
                used_f2_rows.add(match_idx)
                match_count += 1
                continue 
            
            if row1_id in f2_id_lookup:
                potential_idx = next((idx for idx in f2_id_lookup[row1_id] if idx not in used_f2_rows), None)
                if potential_idx is not None:
                    used_f2_rows.add(potential_idx)
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[potential_idx][h]]
                    reason = f"Diff in: {', '.join(diffs)}"
                    
                    r1 = df1.iloc[i][selected_headers].to_dict()
                    r1.update({'Source File': fname1, 'Status': 'Mismatch', 'Reason': reason})
                    r2 = {h: df2.iloc[potential_idx][mapping[h]] for h in selected_headers}
                    r2.update({'Source File': fname2, 'Status': 'Mismatch', 'Reason': reason})
                    mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    continue

            r_miss = df1.iloc[i][selected_headers].to_dict()
            r_miss.update({'Source File': fname1, 'Status': f'Missing in {fname2}', 'Reason': 'ID not found'})
            missing_entries.append(r_miss)

        # --- REVERSE PASS (File B -> File A) ---
        my_bar.progress(85, text=f"Step 4/4: Scanning {fname2} for unique entries...")
        for j in range(len(df2)):
            if j not in used_f2_rows:
                row2_id = df2_norm.iloc[j][id_col]
                if row2_id not in f1_id_set:
                    r_extra = {h: df2.iloc[j][mapping[h]] for h in selected_headers}
                    r_extra.update({'Source File': fname2, 'Status': f'Missing in {fname1}', 'Reason': 'ID not found'})
                    missing_entries.append(r_extra)

        my_bar.progress(100, text="Reconciliation Complete!")
        time.sleep(1)
        my_bar.empty() # Remove progress bar after finishing

        # Store in session state
        st.session_state.mismatch_df = pd.DataFrame(mismatches).astype(object)
        st.session_state.missing_df = pd.DataFrame(missing_entries).astype(object)
        st.session_state.stats = {
            "Match": match_count, 
            "Conflict": len(mismatches) // 3, 
            "Missing": len(missing_entries)
        }
        st.session_state.reconciliation_done = True

# --- UI DISPLAY SECTION ---
if st.session_state.reconciliation_done:
    st.divider()
    s = st.session_state.stats
    fn = st.session_state.fnames
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Identical Rows", s["Match"])
    m2.metric("Data Conflicts", s["Conflict"])
    m3.metric("Total Missing Records", s["Missing"])

    tab1, tab2 = st.tabs(["⚠️ View Mismatches", "❌ View Missing Entries"])
    
    with tab1:
        st.write("### Data Mismatches")
        search_m = st.text_input("Search Mismatches:", key="search_m")
        df_m = st.session_state.mismatch_df
        if search_m and not df_m.empty:
            df_m = df_m[df_m.apply(lambda r: r.astype(str).str.contains(search_m, case=False).any(), axis=1)]
        st.dataframe(df_m, use_container_width=True)

    with tab2:
        st.write(f"### Records missing from either file")
        search_mis = st.text_input("Search Missing Entries:", key="search_mis")
        df_mis = st.session_state.missing_df
        if search_mis and not df_mis.empty:
            df_mis = df_mis[df_mis.apply(lambda r: r.astype(str).str.contains(search_mis, case=False).any(), axis=1)]
        st.dataframe(df_mis, use_container_width=True)

    st.divider()
    st.write("### 📥 Download Split Report")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.mismatch_df.to_excel(writer, index=False, sheet_name="Mismatches")
        st.session_state.missing_df.to_excel(writer, index=False, sheet_name="Missing_Entries")
    
    st.download_button(
        label="Download Multi-Sheet Excel Report",
        data=output.getvalue(),
        file_name="reconciliation_final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
