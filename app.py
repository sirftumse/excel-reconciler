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
    d1.columns = [str(c).strip() for c in d1.columns]
    d2.columns = [str(c).strip() for c in d2.columns]
    return d1, d2

def get_similarity(str1, str2):
    if not str1 or not str2: return 0
    return SequenceMatcher(None, str(str1), str(str2)).quick_ratio()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Matcher Pro ⚡", layout="wide")

# Initialize Session State for persistence
if "reset_count" not in st.session_state: st.session_state.reset_count = 0
if "last_selected_headers" not in st.session_state: st.session_state.last_selected_headers = []

def trigger_reset():
    st.session_state.reset_count += 1
    st.session_state.last_selected_headers = [] # Wipe memory on manual reset

# --- UI HEADER ---
col_t, col_r = st.columns([4, 1])
with col_t:
    st.title("🔍 High-Speed Robust Reconciliation")
with col_r:
    st.button("♻️ Reset All Selections", on_click=trigger_reset, use_container_width=True)

file1 = st.file_uploader("Upload Master File (File A)", type=['xlsx'])
file2 = st.file_uploader("Upload Comparison File (File B)", type=['xlsx'])

if file1 and file2:
    df1, df2 = load_excel_data(file1, file2)
    all_h1, all_h2 = df1.columns.tolist(), df2.columns.tolist()

    st.divider()
    st.subheader("🛠️ Step 1: Define Row Identity (Anchors)")
    
    # PERSISTENCE LOGIC:
    # If we have a previous selection, use only those that exist in the current file.
    # If no previous selection (or reset was pressed), default to all columns.
    current_defaults = [h for h in st.session_state.last_selected_headers if h in all_h1]
    
    if not current_defaults:
        current_defaults = all_h1

    selected_headers = st.multiselect(
        "Select columns to verify/show:", 
        options=all_h1, 
        default=current_defaults,
        key=f"header_sel_{st.session_state.reset_count}"
    )
    # Update memory for the next file change
    st.session_state.last_selected_headers = selected_headers

    # Anchor Logic (Filtered by selected_headers)
    anchor_cols = st.multiselect(
        "Select Anchor Columns (Used for exact match):", 
        options=selected_headers, 
        default=[h for h in selected_headers if any(x in h.lower() for x in ['session', 'subject', 'date', 'timing', 'enrollment', 'no', 'code'])],
        key=f"anchor_sel_{st.session_state.reset_count}"
    )

    st.subheader("🛠️ Step 2: Confirm Mapping")
    mapping, missing_mappings = {}, []
    options_with_blank = ["-- Select Column --"] + all_h2
    
    grid = st.columns(3)
    for i, h in enumerate(selected_headers):
        with grid[i % 3]:
            try: d_idx = [x.lower() for x in all_h2].index(h.lower()) + 1
            except: d_idx = 0
            
            val = st.selectbox(f"'{h}' maps to:", options=options_with_blank, index=d_idx, key=f"map_{h}_{st.session_state.reset_count}")
            if val == "-- Select Column --":
                mapping[h] = None; missing_mappings.append(h)
            else:
                mapping[h] = val

    if st.button("🚀 Run Optimized Search"):
        if not anchor_cols:
            st.error("❌ Please select at least one Anchor Column.")
        elif missing_mappings:
            st.error(f"❌ Map these columns: {', '.join(missing_mappings)}")
        else:
            def anchor_clean(val):
                return re.sub(r'[^a-z0-9]', '', str(val).strip().lower().replace('.', '-'))

            def precision_clean(val):
                v = str(val).strip()
                if v.isdigit() and len(v) == 5:
                    try: v = pd.to_datetime(int(v), unit='D', origin='1899-12-30').strftime('%d-%m-%Y')
                    except: pass
                return v.replace('.', '-')

            mismatches, missing_entries, perfect_matches_paired = [], [], []
            used_f2, matched_f1, match_count = set(), set(), 0
            bar = st.progress(0, text="Processing...")

            # Stage 1: Exact Match
            f2_map = {}
            for idx, row in df2.iterrows():
                fp = "|".join([anchor_clean(row[mapping[a]]) for a in anchor_cols])
                f2_map.setdefault(fp, []).append(idx)

            for i, row1 in df1.iterrows():
                f1_fp = "|".join([anchor_clean(row1[a]) for a in anchor_cols])
                if f1_fp in f2_map and f2_map[f1_fp]:
                    target_idx = f2_map[f1_fp].pop(0)
                    matched_f1.add(i); used_f2.add(target_idx)
                    row2 = df2.iloc[target_idx]
                    diffs = [c for c in selected_headers if precision_clean(row1[c]) != precision_clean(row2[mapping[c]])]
                    
                    r1, r2 = row1[selected_headers].to_dict(), {h: row2[mapping[h]] for h in selected_headers}
                    if not diffs: 
                        match_count += 1
                        r1.update({'Source': file1.name, 'Status': 'Perfect Match'})
                        r2.update({'Source': file2.name, 'Status': 'Perfect Match'})
                        perfect_matches_paired.extend([r1, r2, {k: "---" for k in r1.keys()}])
                    else:
                        r1.update({'Source': file1.name, 'Status': 'Mismatch', 'Diff_Cols': ",".join(diffs)})
                        r2.update({'Source': file2.name, 'Status': 'Mismatch', 'Diff_Cols': ",".join(diffs)})
                        mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])

            # Stage 2: Fuzzy Match
            unmatched = [i for i in df1.index if i not in matched_f1]
            f2_rem = df2[~df2.index.isin(used_f2)]
            for idx, i in enumerate(unmatched):
                if idx % 20 == 0: bar.progress(40 + int((idx/len(unmatched))*55) if len(unmatched)>0 else 95)
                row1 = df1.iloc[i]
                s_key = 'Subject' if 'Subject' in row1 else (selected_headers[0] if selected_headers else None)
                v1 = str(row1.get(s_key, '')).lower()
                best_s, best_i = -1, -1
                for j_idx, row2 in f2_rem.iterrows():
                    v2 = str(row2.get(mapping.get(s_key, ''), '')).lower()
                    if not v1 or v1[:3] == v2[:3]:
                        score = get_similarity(v1, v2)
                        if score > best_s: best_s, best_i = score, j_idx
                if best_i != -1 and best_s > 0.5:
                    used_f2.add(best_i); row2 = df2.iloc[best_i]
                    diffs = [c for c in selected_headers if precision_clean(row1[c]) != precision_clean(row2[mapping[c]])]
                    r1, r2 = row1[selected_headers].to_dict(), {h: row2[mapping[h]] for h in selected_headers}
                    r1.update({'Source': file1.name, 'Status': 'Fuzzy Match', 'Diff_Cols': ",".join(diffs)})
                    r2.update({'Source': file2.name, 'Status': 'Fuzzy Match', 'Diff_Cols': ",".join(diffs)})
                    mismatches.extend([r1, r2, {k: "---" for k in r1.keys()}])
                else:
                    r_m = row1[selected_headers].to_dict()
                    r_m.update({'Source': file1.name, 'Status': f'Missing in {file2.name}'})
                    missing_entries.append(r_m)

            # Extra entries in B
            for j in df2.index:
                if j not in used_f2:
                    row2 = df2.iloc[j]
                    r_e = {h: row2[mapping[h]] for h in selected_headers}
                    r_e.update({'Source': file2.name, 'Status': f'Extra in {file2.name}'})
                    missing_entries.append(r_e)

            bar.progress(100)
            
            # UPGRADED SUMMARY (BOLD RED IF > 0)
            m_count = len(mismatches)//3
            mis_count = len(missing_entries)
            mismatch_style = "color: red; font-weight: bold;" if m_count > 0 else "color: black;"
            missing_style = "color: red; font-weight: bold;" if mis_count > 0 else "color: black;"
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db;">
                <span style="color: black;">Matches: {match_count}</span> | 
                <span style="{mismatch_style}">Conflicts: {m_count}</span> | 
                <span style="{missing_style}">Missing/Extra: {mis_count}</span>
            </div>
            """, unsafe_allow_html=True)
            
            t1, t2 = st.tabs(["⚠️ Mismatches", "❌ Missing/Extra"])
            with t1:
                if mismatches: st.dataframe(pd.DataFrame(mismatches).style.apply(lambda row: ['background-color: #ffcccc' if col in str(row.get('Diff_Cols', "")).split(",") else '' for col in row.index], axis=1), use_container_width=True)
                else: st.success("No conflicts found!")
            with t2:
                if missing_entries: st.dataframe(pd.DataFrame(missing_entries), use_container_width=True)
                else: st.success("No missing entries found!")

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pd.DataFrame({"Metric": ["Matches", "Conflicts", "Missing"], "Count": [match_count, m_count, mis_count]}).to_excel(writer, index=False, sheet_name="Summary")
                if mismatches: pd.DataFrame(mismatches).to_excel(writer, index=False, sheet_name="Mismatches")
                if missing_entries: pd.DataFrame(missing_entries).to_excel(writer, index=False, sheet_name="Missing_Extra")
                if perfect_matches_paired: pd.DataFrame(perfect_matches_paired).to_excel(writer, index=False, sheet_name="Perfect_Matches")
            st.download_button("📥 Download Final Report", output.getvalue(), "reconciliation_report.xlsx")
