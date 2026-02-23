import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

# --- PERFORMANCE CONFIG ---
pd.set_option("styler.render.max_elements", 20000000)

@st.cache_data(show_spinner="Loading huge files...")
def load_excel_data(f1, f2):
    try:
        d1 = pd.read_excel(f1, engine='calamine')
        d2 = pd.read_excel(f2, engine='calamine')
    except:
        d1 = pd.read_excel(f1)
        d2 = pd.read_excel(f2)
    return d1, d2

st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 Global Search: ID-First Verification")
st.markdown("This version prioritizes the **ID (Anchor)** to ensure you never match different students.")

col_a, col_b = st.columns(2)
with col_a:
    file1 = st.file_uploader("Upload Master File", type=['xlsx'])
with col_b:
    file2 = st.file_uploader("Upload Comparison File", type=['xlsx'])

if file1 and file2:
    fname1, fname2 = file1.name, file2.name
    df1, df2 = load_excel_data(file1, file2)

    st.divider()
    st.subheader("🛠️ Step 1: Map Columns")
    
    all_headers_f1 = df1.columns.tolist()
    all_headers_f2 = df2.columns.tolist()
    
    # User selects columns to compare (Include Name, DOB, etc. here)
    selected_headers = st.multiselect("Select columns to verify:", options=all_headers_f1, default=all_headers_f1)

    mapping = {}
    if selected_headers:
        grid = st.columns(3)
        for i, h in enumerate(selected_headers):
            with grid[i % 3]:
                d_idx = all_headers_f2.index(h) if h in all_headers_f2 else 0
                mapping[h] = st.selectbox(f"'{h}' in {fname2}:", options=all_headers_f2, index=d_idx)

    # CRITICAL: Anchor must be the Enrollment No or EMPCode
    id_col = st.selectbox("Anchor Column (Choose Unique ID / Enrollment No):", options=selected_headers)

    if st.button("🚀 Run Secure Global Search"):
        
        # --- SMART CLEANING ENGINE (Fixes dots, dashes, and commas) ---
        def smart_clean(val):
            val = str(val).strip().lower()
            # Removes punctuation so "A. Khan" and "A-Khan" both become "akhan"
            val = re.sub(r'[.\-/_,\s]', '', val)
            return val if val != 'nan' and val != '' else 'empty'

        def get_norm_df(temp_df, headers_list, is_file2=False):
            actual_cols = [mapping[h] if is_file2 else h for h in headers_list]
            norm = temp_df[actual_cols].copy().fillna("")
            for col in norm.columns:
                norm[col] = norm[col].apply(smart_clean)
            norm.columns = headers_list 
            return norm

        # 1. Normalize both files
        df1_norm = get_norm_df(df1, selected_headers, is_file2=False)
        df2_norm = get_norm_df(df2, selected_headers, is_file2=True)

        # 2. Build ID-based Lookup (Anchor Search)
        # This ensures we find the person by ID first, ignoring name differences initially
        f2_id_lookup = {}
        for idx, val in enumerate(df2_norm[id_col]):
            f2_id_lookup.setdefault(val, []).append(idx)

        # 3. Build Full Fingerprint for Exact Matches
        f1_fp = df1_norm.apply(lambda x: "|".join(x), axis=1)
        f2_fp = df2_norm.apply(lambda x: "|".join(x), axis=1)
        
        f2_exact_lookup = {}
        for idx, fp in enumerate(f2_fp):
            f2_exact_lookup.setdefault(fp, []).append(idx)

        final_results = []
        used_f2_rows = set()
        stats = {"Match": 0, "Conflict": 0, "Missing": 0}

        progress_bar = st.progress(0)
        total_rows = len(df1)

        # --- MATCHING LOOP ---
        for i in range(total_rows):
            if i % 1000 == 0: progress_bar.progress((i + 1) / total_rows)

            current_fp = f1_fp.iloc[i]
            row1_id = df1_norm.iloc[i][id_col]
            
            # STEP A: Try to find EXACT match anywhere
            if current_fp in f2_exact_lookup and f2_exact_lookup[current_fp]:
                match_idx = f2_exact_lookup[current_fp].pop(0)
                used_f2_rows.add(match_idx)
                stats["Match"] += 1
                continue 
            
            # STEP B: If not exact, find the same ID to check for Conflicts (Mismatch)
            if row1_id in f2_id_lookup:
                potential_idx = None
                for idx in f2_id_lookup[row1_id]:
                    if idx not in used_f2_rows:
                        potential_idx = idx
                        break
                
                if potential_idx is not None:
                    used_f2_rows.add(potential_idx)
                    stats["Conflict"] += 1
                    
                    # Find exactly which columns are different (Name, DOB, etc.)
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[potential_idx][h]]
                    reason = f"Diff in: {', '.join(diffs)}"

                    r1 = df1.iloc[i][selected_headers].to_dict()
                    r1.update({'Source': fname1, 'Status': 'Mismatch', 'Reason': reason, 'Excel Row': i+2})
                    
                    r2_raw = df2.iloc[potential_idx]
                    r2 = {h: r2_raw[mapping[h]] for h in selected_headers}
                    r2.update({'Source': fname2, 'Status': 'Mismatch', 'Reason': reason, 'Excel Row': potential_idx+2})
                    
                    final_results.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    continue

            # STEP C: Not found by ID at all
            stats["Missing"] += 1
            r_miss = df1.iloc[i][selected_headers].to_dict()
            r_miss.update({'Source': fname1, 'Status': 'Missing', 'Reason': 'ID not found anywhere', 'Excel Row': i+2})
            final_results.append(r_miss)

        progress_bar.empty()
        res_df = pd.DataFrame(final_results).astype(object)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Identical Found ✅", stats["Match"])
        m2.metric("Conflicts (Same ID) ⚠️", stats["Conflict"])
        m3.metric("Not Found (Missing) ❌", stats["Missing"])
        
        st.dataframe(res_df, width='stretch')

        if not res_df.empty:
            output = BytesIO()
            if len(res_df) > 1000000:
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download All (CSV)", csv, "verification_report.csv")
            else:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button("📥 Download All (Excel)", output.getvalue(), "verification_report.xlsx")
