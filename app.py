import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

# --- PERFORMANCE CONFIG ---
pd.set_option("styler.render.max_elements", 20000000)

@st.cache_data(show_spinner="Loading files...")
def load_excel_data(f1, f2):
    try:
        d1 = pd.read_excel(f1, engine='calamine')
        d2 = pd.read_excel(f2, engine='calamine')
    except:
        d1 = pd.read_excel(f1)
        d2 = pd.read_excel(f2)
    return d1, d2

st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 Global Data Search & Match")

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
    
    selected_headers = st.multiselect("Select columns to compare:", options=all_headers_f1, default=all_headers_f1)

    mapping = {}
    if selected_headers:
        grid = st.columns(3)
        for i, h in enumerate(selected_headers):
            with grid[i % 3]:
                d_idx = all_headers_f2.index(h) if h in all_headers_f2 else 0
                mapping[h] = st.selectbox(f"'{h}' in {fname2}:", options=all_headers_f2, index=d_idx)

    id_col = st.selectbox("Anchor Column (Unique ID):", options=selected_headers)

    if st.button("🚀 Run Global Search"):
        
        # --- SMART CLEANING ENGINE ---
        def smart_clean(val):
            # 1. Convert to string and lowercase
            val = str(val).strip().lower()
            # 2. Remove all punctuation (dots, dashes, slashes, commas)
            # This makes "15.05.1995" and "15-05-1995" look like "15051995"
            val = re.sub(r'[.\-/_,\s]', '', val)
            return val if val != 'nan' else ''

        def get_fingerprint_df(temp_df, headers_list, is_file2=False):
            actual_cols = [mapping[h] if is_file2 else h for h in headers_list]
            norm = temp_df[actual_cols].copy().fillna("")
            
            # Apply the cleaning to every cell
            for col in norm.columns:
                norm[col] = norm[col].apply(smart_clean)
            
            norm.columns = headers_list 
            return norm

        # 1. Create Cleaned DataFrames for matching
        df1_norm = get_fingerprint_df(df1, selected_headers, is_file2=False)
        df2_norm = get_fingerprint_df(df2, selected_headers, is_file2=True)

        f1_fp = df1_norm.apply(lambda x: "|".join(x), axis=1)
        f2_fp = df2_norm.apply(lambda x: "|".join(x), axis=1)
        
        f2_lookup = {}
        for idx, fp in enumerate(f2_fp):
            f2_lookup.setdefault(fp, []).append(idx)

        f2_id_lookup = {}
        for idx, val in enumerate(df2_norm[id_col]):
            f2_id_lookup.setdefault(val, []).append(idx)

        final_results = []
        used_f2_rows = set()
        stats = {"Match": 0, "Mismatch": 0, "Missing": 0}

        progress_bar = st.progress(0)
        total_rows = len(df1)

        # --- MATCHING LOOP ---
        for i in range(total_rows):
            if i % 1000 == 0: progress_bar.progress((i + 1) / total_rows)

            current_fp = f1_fp.iloc[i]
            
            # Match Step
            if current_fp in f2_lookup and f2_lookup[current_fp]:
                match_idx = f2_lookup[current_fp].pop(0)
                used_f2_rows.add(match_idx)
                stats["Match"] += 1
                continue 
            
            # Conflict Step
            row1_id = df1_norm.iloc[i][id_col]
            if row1_id in f2_id_lookup:
                potential_idx = None
                for idx in f2_id_lookup[row1_id]:
                    if idx not in used_f2_rows:
                        potential_idx = idx
                        break
                
                if potential_idx is not None:
                    used_f2_rows.add(potential_idx)
                    stats["Mismatch"] += 1
                    
                    diffs = [h for h in selected_headers if df1_norm.iloc[i][h] != df2_norm.iloc[potential_idx][h]]
                    reason = f"Diff in: {', '.join(diffs)}"

                    r1 = df1.iloc[i][selected_headers].to_dict()
                    r1.update({'Source': fname1, 'Status': 'Mismatch', 'Reason': reason, 'Excel Row': i+2})
                    
                    r2_raw = df2.iloc[potential_idx]
                    r2 = {h: r2_raw[mapping[h]] for h in selected_headers}
                    r2.update({'Source': fname2, 'Status': 'Mismatch', 'Reason': reason, 'Excel Row': potential_idx+2})
                    
                    final_results.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    continue

            # Missing Step
            stats["Missing"] += 1
            r_miss = df1.iloc[i][selected_headers].to_dict()
            r_miss.update({'Source': fname1, 'Status': 'Missing', 'Reason': 'ID not found', 'Excel Row': i+2})
            final_results.append(r_miss)

        progress_bar.empty()
        res_df = pd.DataFrame(final_results).astype(object)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Matches", stats["Match"])
        m2.metric("Conflicts", stats["Mismatch"])
        m3.metric("Missing", stats["Missing"])
        
        st.dataframe(res_df, width='stretch')

        if not res_df.empty:
            output = BytesIO()
            if len(res_df) > 1048570:
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download (CSV)", csv, "report.csv")
            else:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button("📥 Download (Excel)", output.getvalue(), "report.xlsx")
