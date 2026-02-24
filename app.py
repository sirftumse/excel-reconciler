import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
from difflib import SequenceMatcher

# --- PERFORMANCE CONFIG ---
pd.set_option("styler.render.max_elements", 2000000)

@st.cache_data(show_spinner="Loading datasets...")
def load_excel_data(f1, f2):
    try:
        d1 = pd.read_excel(f1, engine='calamine', dtype=str).fillna("")
        d2 = pd.read_excel(f2, engine='calamine', dtype=str).fillna("")
    except:
        d1 = pd.read_excel(f1, dtype=str).fillna("")
        d2 = pd.read_excel(f2, dtype=str).fillna("")
    return d1, d2

def get_similarity(str1, str2):
    if not str1 or not str2: return 0
    return SequenceMatcher(None, str(str1), str(str2)).quick_ratio()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")
st.title("🔍 High-Speed Robust Reconciliation")

# --- FILE UPLOAD ---
col_a, col_b = st.columns(2)
with col_a:
    file1 = st.file_uploader("Upload Master File (File A)", type=['xlsx'])
with col_b:
    file2 = st.file_uploader("Upload Comparison File (File B)", type=['xlsx'])

if file1 and file2:
    df1, df2 = load_excel_data(file1, file2)
    all_h1, all_h2 = df1.columns.tolist(), df2.columns.tolist()

    st.divider()
    st.subheader("🛠️ Step 1: Define Row Identity (Anchors)")
    
    selected_headers = st.multiselect("Select columns to verify/show:", options=all_h1, default=all_h1)
    suggested_anchors = [h for h in selected_headers if any(x in h.lower() for x in ['session', 'subject', 'date', 'timing'])]
    
    anchor_cols = st.multiselect(
        "Select Anchor Columns (Used for exact match):", 
        options=selected_headers, 
        default=suggested_anchors
    )

    st.subheader("🛠️ Step 2: Confirm Mapping")
    mapping = {}
    missing_mappings = []
    options_with_blank = ["-- Select Column --"] + all_h2
    
    grid = st.columns(3)
    for i, h in enumerate(selected_headers):
        with grid[i % 3]:
            h_lower = h.lower()
            try:
                d_idx = [x.lower() for x in all_h2].index(h_lower) + 1
            except:
                d_idx = 0
            
            val = st.selectbox(f"'{h}' maps to:", options=options_with_blank, index=d_idx, key=f"map_{h}")
            if val == "-- Select Column --":
                mapping[h] = None
                missing_mappings.append(h)
            else:
                mapping[h] = val

    if st.button("🚀 Run Optimized Search"):
        if missing_mappings:
            st.error(f"❌ Map these columns first: {', '.join(missing_mappings)}")
        else:
            # --- PRE-PROCESSING ---
            def anchor_clean(val):
                return re.sub(r'[^a-z0-9]', '', str(val).strip().lower())

            def precision_clean(val):
                v = str(val).strip()
                if v.isdigit() and len(v) == 5:
                    try: v = pd.to_datetime(int(v), unit='D', origin='1899-12-30').strftime('%d-%m-%Y')
                    except: pass
                return v 

            mismatches, missing_entries = [], []
            used_f2 = set()
            matched_f1 = set()
            match_count = 0
            
            my_bar = st.progress(0, text="Starting...")

            # STAGE 1: Exact Anchor Matching (Very Fast)
            my_bar.progress(10, text="Stage 1: Exact Matching...")
            f2_idx_map = {}
            for idx, row in df2.iterrows():
                fp = "|".join([anchor_clean(row[mapping[a]]) for a in anchor_cols])
                f2_idx_map.setdefault(fp, []).append(idx)

            for i, row1 in df1.iterrows():
                f1_fp = "|".join([anchor_clean(row1[a]) for a in anchor_cols])
                if f1_fp in f2_idx_map and f2_idx_map[f1_fp]:
                    target_idx = f2_idx_map[f1_fp].pop(0)
                    matched_f1.add(i)
                    used_f2.add(target_idx)
                    row2 = df2.iloc[target_idx]
                    
                    diffs = [c for c in selected_headers if precision_clean(row1[c]) != precision_clean(row2[mapping[c]])]
                    if not diffs:
                        match_count += 1
                    else:
                        r1 = row1[selected_headers].to_dict()
                        r1.update({'Source': file1.name, 'Status': 'Mismatch', 'Diff_Cols': ",".join(diffs)})
                        r2 = {h: row2[mapping[h]] for h in selected_headers}
                        r2.update({'Source': file2.name, 'Status': 'Mismatch', 'Diff_Cols': ",".join(diffs)})
                        mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])

            # STAGE 2: Optimized Best-Pair Search
            unmatched_f1 = [i for i in df1.index if i not in matched_f1]
            available_f2_indices = [j for j in df2.index if j not in used_f2]
            
            # Create a localized dictionary for available F2 rows to speed up lookups
            f2_remaining = df2.iloc[available_f2_indices]
            
            total_unmatched = len(unmatched_f1)
            for idx, i in enumerate(unmatched_f1):
                if idx % 5 == 0: # Update UI less frequently to save time
                    p_val = 40 + int((idx / total_unmatched) * 55) if total_unmatched > 0 else 90
                    my_bar.progress(p_val, text=f"Stage 2: Scanning candidates ({idx+1}/{total_unmatched})...")
                
                row1 = df1.iloc[i]
                sub1 = str(row1.get('Subject', '')).lower()
                
                best_score, best_idx = -1, -1
                
                # SPEED TRICK: Only compare rows in B that have at least some similarity
                # We use a fast Subject filter first
                for j_idx, row2 in f2_remaining.iterrows():
                    if j_idx in used_f2: continue
                    
                    sub2 = str(row2.get(mapping.get('Subject', ''), '')).lower()
                    
                    # Quick check: If subjects are totally different, don't waste time on SequenceMatcher
                    if sub1[:3] == sub2[:3] or sub1[-3:] == sub2[-3:]:
                        score = get_similarity(sub1, sub2)
                        if score > best_score:
                            best_score, best_idx = score, j_idx
                
                if best_idx != -1 and best_score > 0.5:
                    used_f2.add(best_idx)
                    row2 = df2.iloc[best_idx]
                    diffs = [c for c in selected_headers if precision_clean(row1[c]) != precision_clean(row2[mapping[c]])]
                    
                    r1 = row1[selected_headers].to_dict()
                    r1.update({'Source': file1.name, 'Status': 'Fuzzy Match', 'Diff_Cols': ",".join(diffs)})
                    r2 = {h: row2[mapping[h]] for h in selected_headers}
                    r2.update({'Source': file2.name, 'Status': 'Fuzzy Match', 'Diff_Cols': ",".join(diffs)})
                    mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])
                else:
                    r_miss = row1[selected_headers].to_dict()
                    r_miss.update({'Source': file1.name, 'Status': 'Missing in B'})
                    missing_entries.append(r_miss)

            # Final Extras
            for j in df2.index:
                if j not in used_f2:
                    row2 = df2.iloc[j]
                    r_extra = {h: row2[mapping[h]] for h in selected_headers}
                    r_extra.update({'Source': file2.name, 'Status': 'Extra in B'})
                    missing_entries.append(r_extra)

            my_bar.progress(100, text="Finished!")
            st.success(f"Done! Matches: {match_count} | Conflicts: {len(mismatches)//3}")
            
            t1, t2 = st.tabs(["⚠️ Mismatches", "❌ Missing/Extra"])
            with t1:
                if mismatches:
                    df_m = pd.DataFrame(mismatches)
                    def color_cells(row):
                        diff_set = set(str(row.get('Diff_Cols', "")).split(","))
                        return ['background-color: #ffcccc' if col in diff_set else '' for col in row.index]
                    st.dataframe(df_m.style.apply(color_cells, axis=1), use_container_width=True)
            with t2:
                if missing_entries:
                    st.dataframe(pd.DataFrame(missing_entries), use_container_width=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if mismatches: pd.DataFrame(mismatches).to_excel(writer, index=False, sheet_name="Mismatches")
                if missing_entries: pd.DataFrame(missing_entries).to_excel(writer, index=False, sheet_name="Missing_Extra")
            st.download_button("📥 Download Report", output.getvalue(), "report.xlsx")
