import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Deep Reconciler Pro", layout="wide")
st.title("📂 Advanced Reconciliation & Conflict Suite")

# 1. File Uploaders
col_a, col_b = st.columns(2)
with col_a:
    file1 = st.file_uploader("Upload File 1", type=['xlsx'])
with col_b:
    file2 = st.file_uploader("Upload File 2", type=['xlsx'])

if file1 and file2:
    fname1 = file1.name
    fname2 = file2.name
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    st.divider()
    
    # 2. Settings
    st.subheader("🛠️ Step 1: Mapping & Comparison Setup")
    all_headers = df1.columns.tolist()
    selected_headers = st.multiselect(
        "Select columns for logic:",
        options=all_headers,
        default=[h for h in all_headers if h.lower() not in ['#', 'transaction_id', 'transaction-id']]
    )

    mapping = {}
    if selected_headers:
        grid = st.columns(3)
        for i, h in enumerate(selected_headers):
            with grid[i % 3]:
                d_idx = 0
                for idx, col in enumerate(df2.columns):
                    if col.strip().lower().replace("-","").replace("_","") == h.strip().lower().replace("-","").replace("_",""):
                        d_idx = idx
                        break
                mapping[h] = st.selectbox(f"'{h}' in {fname2}:", options=df2.columns.tolist(), index=d_idx, key=f"map_{h}")

    id_col = st.selectbox("Anchor Column (e.g. Employee Empcode or Subject):", options=selected_headers)

    st.divider()

    if selected_headers and st.button("🚀 Generate Full Report"):
        
        # --- LOGIC ---
        def normalize_cell(value, header):
            if pd.isna(value) or value == "": return ""
            val = str(value).strip()
            if "group" in str(header).lower():
                return ",".join(sorted([p.strip() for p in val.split(",")]))
            if isinstance(value, datetime): return value.strftime('%d/%m/%Y')
            return val.lower()

        # Build Lookups
        f2_full_map = {}
        f2_anchor_map = {}
        used_f2_indices = set()
        
        for idx, row in df2.iterrows():
            fk = "|".join([normalize_cell(row[mapping[h]], h) for h in selected_headers])
            anchor_val = normalize_cell(row[mapping[id_col]], id_col)
            f2_full_map.setdefault(fk, []).append(idx)
            f2_anchor_map.setdefault(anchor_val, []).append(idx)

        final_results = []
        stats = {"exact": 0, "mismatch": 0, "missing_f2": 0, "missing_f1": 0}

        # Process File 1
        for idx_f1, row_f1 in df1.iterrows():
            fk_f1 = "|".join([normalize_cell(row_f1[h], h) for h in selected_headers])
            
            # 1. EXACT
            if fk_f1 in f2_full_map and f2_full_map[fk_f1]:
                match_idx = f2_full_map[fk_f1].pop(0)
                used_f2_indices.add(match_idx)
                stats["exact"] += 1
                continue 

            # 2. NEAR MATCH (Conflict Dashboard Logic)
            anchor_f1 = normalize_cell(row_f1[id_col], id_col)
            potential_match_idx = None
            if anchor_f1 in f2_anchor_map:
                for cand_idx in f2_anchor_map[anchor_f1]:
                    if cand_idx not in used_f2_indices:
                        potential_match_idx = cand_idx
                        break
            
            if potential_match_idx is not None:
                used_f2_indices.add(potential_match_idx)
                row_f2 = df2.loc[potential_match_idx]
                diffs = [h for h in selected_headers if normalize_cell(row_f1[h], h) != normalize_cell(row_f2[mapping[h]], h)]
                reason = f"Diff in: {', '.join(diffs)}" if diffs else "Duplicate Key"

                r1 = row_f1[selected_headers].to_dict()
                r1.update({'Source': fname1, 'Status': 'Data Mismatch', 'Reason': reason, 'Row #': idx_f1 + 2})
                final_results.append(r1)
                
                r2 = {h: row_f2[mapping[h]] for h in selected_headers}
                r2.update({'Source': fname2, 'Status': 'Data Mismatch', 'Reason': reason, 'Row #': potential_match_idx + 2})
                final_results.append(r2)
                final_results.append({k: "" for k in r1.keys()})
                stats["mismatch"] += 1
                continue

            # 3. MISSING IN FILE 2
            r_m = row_f1[selected_headers].to_dict()
            r_m.update({'Source': fname1, 'Status': f'Missing in {fname2}', 'Reason': 'N/A', 'Row #': idx_f1 + 2})
            final_results.append(r_m)
            stats["missing_f2"] += 1

        # 4. MISSING IN FILE 1
        for idx_f2, row_f2 in df2.iterrows():
            if idx_f2 not in used_f2_indices:
                r_u = {h: row_f2[mapping[h]] for h in selected_headers}
                r_u.update({'Source': fname2, 'Status': f'Missing in {fname1}', 'Reason': 'N/A', 'Row #': idx_f2 + 2})
                final_results.append(r_u)
                stats["missing_f1"] += 1

        # --- TABS FOR DASHBOARDS ---
        tab1, tab2 = st.tabs(["📊 Summary Dashboard", "⚖️ Conflict/Near-Match Dashboard"])
        
        res_df = pd.DataFrame(final_results)

        with tab1:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exact Matches", stats["exact"])
            m2.metric("Conflicts", stats["mismatch"], delta_color="inverse")
            m3.metric(f"Missing in {fname2}", stats["missing_f2"], delta_color="inverse")
            m4.metric(f"Missing in {fname1}", stats["missing_f1"], delta_color="inverse")
            
            st.subheader("Combined Discrepancy View")
            def color_all(row):
                if row['Status'] == 'Data Mismatch': return ['background-color: #f4cccc'] * len(row)
                if 'Missing' in str(row['Status']): return ['background-color: #ffe5e5'] * len(row)
                return [None] * len(row)
            st.dataframe(res_df.style.apply(color_all, axis=1), use_container_width=True)

        with tab2:
            st.subheader("Near-Match Conflicts")
            st.info("These rows share the same Anchor but have differences in other columns.")
            # Filter specifically for Mismatches only
            near_match_df = res_df[res_df['Status'].isin(['Data Mismatch', ''])]
            def color_near(row):
                if row['Status'] == 'Data Mismatch': return ['background-color: #fce4ec; color: #880e4f'] * len(row)
                if row['Status'] == "": return ['background-color: #eeeeee'] * len(row)
                return [None] * len(row)
            st.dataframe(near_match_df.style.apply(color_near, axis=1), use_container_width=True)

        # Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            res_df.to_excel(writer, index=False)
        st.download_button("📥 Download Final Report", output.getvalue(), "reconciled_data.xlsx")

else:
    st.info("Please upload both files to continue.")