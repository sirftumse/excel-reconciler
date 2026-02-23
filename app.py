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
    try:
        # We force 'str' type on load to prevent IDs from getting .0 decimals
        d1 = pd.read_excel(f1, engine='calamine', dtype=str)
        d2 = pd.read_excel(f2, engine='calamine', dtype=str)
    except:
        d1 = pd.read_excel(f1, dtype=str)
        d2 = pd.read_excel(f2, dtype=str)
    return d1, d2

def get_similarity(a, b):
    """Calculates similarity score between two strings (0.0 to 1.0)"""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 Global Two-Way Reconciliation (Ultimate)")
st.markdown("Fixed: **Automatic .0 decimal removal**. Includes **Fuzzy Matching** & **Progress Tracking**.")

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
        # --- PROGRESS BAR INITIALIZATION ---
        my_bar = st.progress(0, text="Initializing engine...")
        
        def smart_clean(val):
            """Removes .0, standardizes dates, and strips punctuation"""
            val = str(val).strip()
            
            # Remove trailing .0 from IDs/Numbers
            if val.endswith('.0'):
                val = val[:-2]
            
            val_lower = val.lower()
            
            # Logic: Date Standardizer
            try:
                if len(val_lower) > 5 and any(char in val_lower for char in './-'):
                    return pd.to_datetime(val_lower, dayfirst=True).strftime('%Y%m%d')
            except:
                pass
            
            # Punctuation & Space Strip
            return re.sub(r'[.\-/_,\s]', '', val_lower)

        def get_norm_df(temp_df, headers_list, is_file2=False):
            actual_cols = [mapping[h] if is_file2 else h for h in headers_list]
            norm = temp_df[actual_cols].copy().fillna("")
            for col in norm.columns:
                norm[col] = norm[col].apply(smart_clean)
            norm.columns = headers_list 
            return norm

        # 1. Normalization
        my_bar.progress(10, text="Step 1/4: Cleaning data & removing decimals...")
        df1_norm = get_norm_df(df1, selected_headers, is_file2=False)
        df2_norm = get_norm_df(df2, selected_headers, is_file2=True)

        # 2. Set-Based Pre-Check
        my_bar.progress(20, text="Step 2/4: Calculating missing ID sets...")
        ids_a = set(df1_norm[id_col].tolist())
        ids_b = set(df2_norm[id_col].tolist())
        only_in_b = ids_b - ids_a

        # 3. Indexing
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

        # 4. Deep Scan Forward (A -> B)
        total = len(df1)
        for i in range(total):
            if i % (max(1, total // 20)) == 0:
                my_bar.progress(25 + int((i/total)*60), text=f"Step 3/4: Analyzing {fname1} (Row {i}/{total})")

            row1_id = df1_norm.iloc[i][id_col]
            fp = f1_fps.iloc[i]
            
            # Case 1: Exact Match
            if fp in f2_fp_lookup and f2_fp_lookup[fp]:
                match_idx = f2_fp_lookup[fp].pop(0)
                used_f2_rows.add(match_idx)
                match_count += 1
                continue 
            
            # Case 2: Conflict (Same ID, Different Data)
            if row1_id in f2_id_lookup:
                pot_idx = next((idx for idx in f2_id_lookup[row1_id] if idx not in used_f2_rows), None)
                if pot_idx is not None:
                    used_f2_rows.add(pot_idx)
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[pot_idx][h]]
                    
                    # Fuzzy Logic for Names
                    similarity_text = ""
                    for h in diffs:
                        if "name" in h.lower():
                            score = get_similarity(df1.iloc[i][h], df2.iloc[pot_idx][mapping[h]])
                            similarity_text = f" (Name Match: {int(score*100)}%)"
                            break

                    reason = f"Diff in: {', '.join(diffs)}{similarity_text}"
                    r1 = df1.iloc[i][selected_headers].to_dict()
                    r1.update({'Source': fname1, 'Status': 'Mismatch', 'Reason': reason})
                    r2 = {h: df2.iloc[pot_idx][mapping[h]] for h in selected_headers}
                    r2.update({'Source': fname2, 'Status': 'Mismatch', 'Reason': reason})
                    mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    continue

            # Case 3: ID in A but not B
            r_miss = df1.iloc[i][selected_headers].to_dict()
            r_miss.update({'Source': fname1, 'Status': f'Not in {fname2}', 'Reason': 'ID Missing'})
            missing_entries.append(r_miss)

        # 5. Reverse Pass for unique IDs in B
        my_bar.progress(90, text="Step 4/4: Finding unique records in comparison file...")
        for j in range(len(df2)):
            if j not in used_f2_rows and df2_norm.iloc[j][id_col] in only_in_b:
                r_extra = {h: df2.iloc[j][mapping[h]] for h in selected_headers}
                r_extra.update({'Source': fname2, 'Status': f'Not in {fname1}', 'Reason': 'ID Missing'})
                missing_entries.append(r_extra)

        my_bar.progress(100, text="Successfully Reconciled!")
        time.sleep(1)
        my_bar.empty()

        st.session_state.mismatch_df = pd.DataFrame(mismatches).astype(object)
        st.session_state.missing_df = pd.DataFrame(missing_entries).astype(object)
        st.session_state.stats = {"Match": match_count, "Conflict": len(mismatches)//3, "Missing": len(missing_entries)}
        st.session_state.reconciliation_done = True

# --- UI DISPLAY SECTION ---
if st.session_state.reconciliation_done:
    st.divider()
    s = st.session_state.stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Identical Rows", s["Match"])
    m2.metric("Data Mismatches", s["Conflict"])
    m3.metric("Total Missing Records", s["Missing"])

    tab1, tab2 = st.tabs(["⚠️ Mismatches & Conflicts", "❌ Missing Entries"])
    
    with tab1:
        st.dataframe(st.session_state.mismatch_df, use_container_width=True)

    with tab2:
        st.dataframe(st.session_state.missing_df, use_container_width=True)

    st.divider()
    st.write("### 📥 Download Split Excel Report")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.mismatch_df.to_excel(writer, index=False, sheet_name="Mismatches")
        st.session_state.missing_df.to_excel(writer, index=False, sheet_name="Missing_Records")
    
    st.download_button(
        label="Download Final Report",
        data=output.getvalue(),
        file_name="reconciliation_final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
