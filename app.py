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
        d1 = pd.read_excel(f1, engine='calamine')
        d2 = pd.read_excel(f2, engine='calamine')
    except:
        d1 = pd.read_excel(f1)
        d2 = pd.read_excel(f2)
    return d1, d2

def get_similarity(a, b):
    """Calculates how similar two names are (0.0 to 1.0)"""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 Global Two-Way Reconciliation (Ultimate)")
st.markdown("Combines **Fuzzy Matching**, **Date Standardization**, and **Live Tracking**.")

# Initialize session state
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

    if st.button("🚀 Run Intelligent Search"):
        # --- PROGRESS BAR ---
        my_bar = st.progress(0, text="Initializing...")
        
        def smart_clean(val):
            """Logic 1 & 2: Date Standardization & Punctuation Strip"""
            val_str = str(val).strip().lower()
            # Try parsing as date to normalize 29.04.2003 vs 29-04-2003
            try:
                if len(val_str) > 5: # basic check for date-like length
                    return pd.to_datetime(val_str).strftime('%Y%m%d')
            except:
                pass
            # Fallback to stripping punctuation
            return re.sub(r'[.\-/_,\s]', '', val_str)

        def get_norm_df(temp_df, headers_list, is_file2=False):
            actual_cols = [mapping[h] if is_file2 else h for h in headers_list]
            norm = temp_df[actual_cols].copy().fillna("")
            for col in norm.columns:
                norm[col] = norm[col].apply(smart_clean)
            norm.columns = headers_list 
            return norm

        # Step 1: Normalization
        my_bar.progress(10, text="Normalizing date and text formats...")
        df1_norm = get_norm_df(df1, selected_headers, is_file2=False)
        df2_norm = get_norm_df(df2, selected_headers, is_file2=True)

        # Step 2: Instant Set-Check (Logic 4)
        my_bar.progress(20, text="Calculating missing ID sets...")
        ids_a = set(df1_norm[id_col].tolist())
        ids_b = set(df2_norm[id_col].tolist())
        only_in_b = ids_b - ids_a

        # Step 3: Indexing for Speed
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

        # Step 4: Intelligent Forward Scan (A -> B)
        total = len(df1)
        for i in range(total):
            # Update progress bar
            if i % (max(1, total // 20)) == 0:
                my_bar.progress(25 + int((i/total)*60), text=f"Deep Scanning {fname1}...")

            row1_id = df1_norm.iloc[i][id_col]
            fp = f1_fps.iloc[i]
            
            # Exact Match
            if fp in f2_fp_lookup and f2_fp_lookup[fp]:
                match_idx = f2_fp_lookup[fp].pop(0)
                used_f2_rows.add(match_idx)
                match_count += 1
                continue 
            
            # ID Match but Data Conflict
            if row1_id in f2_id_lookup:
                pot_idx = next((idx for idx in f2_id_lookup[row1_id] if idx not in used_f2_rows), None)
                if pot_idx is not None:
                    used_f2_rows.add(pot_idx)
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[pot_idx][h]]
                    
                    # Logic 3: Fuzzy Name Similarity Score
                    similarity_text = ""
                    for h in diffs:
                        if "name" in h.lower():
                            score = get_similarity(df1.iloc[i][h], df2.iloc[pot_idx][mapping[h]])
                            similarity_text = f" (Name Similarity: {int(score*100)}%)"
                            break

                    reason = f"Diff in: {', '.join(diffs)}{similarity_text}"
                    r1 = df1.iloc[i][selected_headers].to_dict()
                    r1.update({'Source': fname1, 'Status': 'Mismatch', 'Reason': reason})
                    r2 = {h: df2.iloc[pot_idx][mapping[h]] for h in selected_headers}
                    r2.update({'Source': fname2, 'Status': 'Mismatch', 'Reason': reason})
                    mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    continue

            # Definitely Missing in B
            r_miss = df1.iloc[i][selected_headers].to_dict()
            r_miss.update({'Source': fname1, 'Status': f'Not in {fname2}', 'Reason': 'ID Missing'})
            missing_entries.append(r_miss)

        # Reverse Scan (Entries unique to B)
        my_bar.progress(90, text=f"Scanning {fname2} for orphans...")
        for j in range(len(df2)):
            if j not in used_f2_rows and df2_norm.iloc[j][id_col] in only_in_b:
                r_extra = {h: df2.iloc[j][mapping[h]] for h in selected_headers}
                r_extra.update({'Source': fname2, 'Status': f'Not in {fname1}', 'Reason': 'ID Missing'})
                missing_entries.append(r_extra)

        my_bar.progress(100, text="Processing Complete!")
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
    fn = st.session_state.fnames
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Exact Row Matches", s["Match"])
    m2.metric("Data Conflicts (Pairs)", s["Conflict"])
    m3.metric("Total Missing Records", s["Missing"])

    tab1, tab2 = st.tabs(["⚠️ Conflicts & Fuzzy Mismatches", "❌ Missing Entries (Orphans)"])
    
    with tab1:
        search_m = st.text_input("Filter Mismatches:", key="search_m")
        df_m = st.session_state.mismatch_df
        if search_m and not df_m.empty:
            df_m = df_m[df_m.apply(lambda r: r.astype(str).str.contains(search_m, case=False).any(), axis=1)]
        st.dataframe(df_m, use_container_width=True)

    with tab2:
        search_mis = st.text_input("Filter Missing Records:", key="search_mis")
        df_mis = st.session_state.missing_df
        if search_mis and not df_mis.empty:
            df_mis = df_mis[df_mis.apply(lambda r: r.astype(str).str.contains(search_mis, case=False).any(), axis=1)]
        st.dataframe(df_mis, use_container_width=True)

    st.divider()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.mismatch_df.to_excel(writer, index=False, sheet_name="Mismatches")
        st.session_state.missing_df.to_excel(writer, index=False, sheet_name="Missing_Entries")
    
    st.download_button("📥 Download Final Report (Excel)", output.getvalue(), "reconciliation_final.xlsx")
