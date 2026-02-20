import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime

# --- PERFORMANCE & STYLING CONFIG ---
pd.set_option("styler.render.max_elements", 20000000)

@st.cache_data(show_spinner="Loading files into memory...")
def load_excel_data(f1, f2):
    try:
        d1 = pd.read_excel(f1, engine='calamine')
        d2 = pd.read_excel(f2, engine='calamine')
    except:
        d1 = pd.read_excel(f1)
        d2 = pd.read_excel(f2)
    return d1, d2

# --- APP INTERFACE ---
st.set_page_config(page_title="Deep Reconciler Pro ⚡", layout="wide")
st.title("🔍 Data Reconciliation & Conflict Suite")

col_a, col_b = st.columns(2)
with col_a:
    file1 = st.file_uploader("Upload Master File (Excel)", type=['xlsx'])
with col_b:
    file2 = st.file_uploader("Upload Comparison File (Excel)", type=['xlsx'])

if file1 and file2:
    fname1, fname2 = file1.name, file2.name
    df1, df2 = load_excel_data(file1, file2)

    st.divider()
    
    # --- MAPPING ---
    st.subheader("🛠️ Step 1: Mapping & Comparison Setup")
    all_headers = df1.columns.tolist()
    selected_headers = st.multiselect(
        "Select columns to include in comparison:",
        options=all_headers,
        default=[h for h in all_headers if h.lower() not in ['#', 'id', 'created_by']]
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

    id_col = st.selectbox("Anchor Column (The ID used to pair rows):", options=selected_headers)

    if selected_headers and st.button("🚀 Generate Reconciled Dashboards"):
        
        # --- LOGIC ENGINE ---
        def fast_normalize(temp_df, headers_list):
            normalized = temp_df[headers_list].copy().fillna("").astype(str)
            for col in headers_list:
                normalized[col] = normalized[col].str.strip().str.lower()
            return normalized

        df1_norm = fast_normalize(df1, selected_headers)
        df2_mapped_cols = [mapping[h] for h in selected_headers]
        df2_norm = fast_normalize(df2, df2_mapped_cols)
        df2_norm.columns = selected_headers 

        df1_fp = df1_norm.apply(lambda x: "|".join(x), axis=1)
        df2_fp = df2_norm.apply(lambda x: "|".join(x), axis=1)
        df1_anchors = df1_norm[id_col]
        df2_anchors = df2_norm[id_col]

        f2_full_lookup = {}
        for idx, fp in enumerate(df2_fp):
            f2_full_lookup.setdefault(fp, []).append(idx)

        f2_anchor_lookup = {}
        for idx, anc in enumerate(df2_anchors):
            f2_anchor_lookup.setdefault(anc, []).append(idx)

        used_f2_indices = set()
        final_results = []
        stats = {"exact": 0, "mismatch": 0, "missing_f2": 0, "missing_f1": 0}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx_f1 in range(len(df1)):
            if idx_f1 % 1000 == 0:
                progress_bar.progress((idx_f1 + 1) / len(df1))
                status_text.text(f"Processing... {idx_f1+1} rows analyzed")

            fp_f1 = df1_fp.iloc[idx_f1]
            if fp_f1 in f2_full_lookup and f2_full_lookup[fp_f1]:
                match_idx = f2_full_lookup[fp_f1].pop(0)
                used_f2_indices.add(match_idx)
                stats["exact"] += 1
                continue 

            anchor_f1 = df1_anchors.iloc[idx_f1]
            potential_match_idx = None
            if anchor_f1 in f2_anchor_lookup:
                for cand_idx in f2_anchor_lookup[anchor_f1]:
                    if cand_idx not in used_f2_indices:
                        potential_match_idx = cand_idx
                        break
            
            if potential_match_idx is not None:
                used_f2_indices.add(potential_match_idx)
                row_f1 = df1.iloc[idx_f1][selected_headers].to_dict()
                row_f2_raw = df2.iloc[potential_match_idx]
                row_f2 = {h: row_f2_raw[mapping[h]] for h in selected_headers}
                
                diffs = [h for h in selected_headers if df1_norm.at[idx_f1, h] != df2_norm.at[potential_match_idx, h]]
                reason = f"Diff in: {', '.join(diffs)}"

                row_f1.update({'Source': fname1, 'Status': 'Data Mismatch', 'Reason': reason, 'Excel Row': idx_f1 + 2})
                final_results.append(row_f1)
                
                row_f2.update({'Source': fname2, 'Status': 'Data Mismatch', 'Reason': reason, 'Excel Row': potential_match_idx + 2})
                final_results.append(row_f2)
                final_results.append({k: "" for k in row_f1.keys()}) 
                stats["mismatch"] += 1
                continue

            r_m = df1.iloc[idx_f1][selected_headers].to_dict()
            r_m.update({'Source': fname1, 'Status': f'Missing in {fname2}', 'Reason': 'N/A', 'Excel Row': idx_f1 + 2})
            final_results.append(r_m)
            stats["missing_f2"] += 1

        for idx_f2 in range(len(df2)):
            if idx_f2 not in used_f2_indices:
                row_f2_raw = df2.iloc[idx_f2]
                r_u = {h: row_f2_raw[mapping[h]] for h in selected_headers}
                r_u.update({'Source': fname2, 'Status': f'Missing in {fname1}', 'Reason': 'N/A', 'Excel Row': idx_f2 + 2})
                final_results.append(r_u)
                stats["missing_f1"] += 1

        progress_bar.empty()
        status_text.success("Comparison Finished!")
        
        # --- RESULTS RENDER SAFETY CHECK ---
        if final_results:
            res_df = pd.DataFrame(final_results).astype(object)
        else:
            # Create an empty dataframe with expected columns if no results
            cols = selected_headers + ['Source', 'Status', 'Reason', 'Excel Row']
            res_df = pd.DataFrame(columns=cols)

        tab1, tab2 = st.tabs(["📊 Full Summary", "⚖️ Conflicts Only"])

        with tab1:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exact ✅", stats["exact"])
            m2.metric("Conflicts ⚠️", stats["mismatch"])
            m3.metric("Missing F2 ❌", stats["missing_f2"])
            m4.metric("Missing F1 ❓", stats["missing_f1"])
            st.dataframe(res_df, width='stretch')

        with tab2:
            # SAFETY: Check if 'Status' column exists before filtering
            if 'Status' in res_df.columns:
                conflicts = res_df[res_df['Status'] == 'Data Mismatch']
                st.dataframe(conflicts, width='stretch')
            else:
                st.write("No mismatches found.")

        # --- SMART EXPORT ---
        st.divider()
        st.subheader("📥 Export Results")
        
        if not res_df.empty:
            if len(res_df) > 1048570:
                st.warning("⚠️ Result has 1,048,576+ rows. Exporting as CSV.")
                csv_data = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Final Report (CSV)", csv_data, "reconciled_results.csv", "text/csv")
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Results')
                st.download_button("📥 Download Final Report (Excel)", output.getvalue(), "reconciled_results.xlsx")
        else:
            st.warning("Nothing to export.")

else:
    st.info("Upload two Excel files to start.")
