# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import io
from sklearn.preprocessing import MinMaxScaler # Added for site scoring

# --- Page Configuration ---
st.set_page_config(page_title="Recruitment Forecasting Tool", layout="wide")
st.title("📊 Recruitment Forecasting Tool")

# --- Helper Functions (Data Parsing & Timestamping) ---
# These functions encapsulate the logic we developed previously.

@st.cache_data # Cache the result of parsing the funnel definition
def parse_funnel_definition(uploaded_file):
    """Parses the wide-format Stage & Status Breakdown CSV."""
    if uploaded_file is None: return None, None, None 
    try:
        bytes_data = uploaded_file.getvalue()
        try: stringio = io.StringIO(bytes_data.decode("utf-8"))
        except UnicodeDecodeError:
             st.warning("UTF-8 decoding failed for Funnel Definition file, trying latin-1.")
             stringio = io.StringIO(bytes_data.decode("latin-1")) 
        df_funnel_def = pd.read_csv(stringio, sep='\t', header=None) 
        parsed_funnel_definition = {}
        parsed_ordered_stages = []
        ts_col_map = {} 
        for col_idx in df_funnel_def.columns:
            column_data = df_funnel_def[col_idx]
            stage_name = column_data.iloc[0]
            if pd.isna(stage_name) or str(stage_name).strip() == "": continue
            stage_name = str(stage_name).strip().replace('"', '')
            parsed_ordered_stages.append(stage_name)
            statuses = column_data.iloc[1:].dropna().astype(str).apply(lambda x: x.strip().replace('"', '')).tolist()
            statuses = [s for s in statuses if s] 
            if stage_name not in statuses: statuses.append(stage_name)
            parsed_funnel_definition[stage_name] = statuses
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            ts_col_map[stage_name] = clean_ts_name
        if not parsed_ordered_stages: 
             st.error("Could not parse any stages from the Funnel Definition file.")
             return None, None, None
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}")
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    if pd.isna(dt_str): return pd.NaT 
    dt_str_cleaned = str(dt_str).strip()
    tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned)
    parsed_dt = pd.to_datetime(dt_str_no_tz, errors='coerce') 
    return parsed_dt

def parse_history_string(history_str):
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    raw_lines = str(history_str).strip().split('\n')
    parsed_events = []
    for line in raw_lines:
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups(); name = name.strip()
            dt_obj = parse_datetime_with_timezone(dt_str.strip()) 
            if name and pd.notna(dt_obj): 
                try: py_dt = dt_obj.to_pydatetime(); parsed_events.append((name, py_dt)) 
                except AttributeError: pass 
    try: parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e: st.warning(f"History sort warning: {e}")
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps) 
    for stage, statuses in funnel_def.items():
        for status in statuses: status_to_stage_map[status] = stage
    all_events = []
    stage_hist = row.get(parsed_stage_history_col, [])
    status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist])
    try: all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e: pass 
    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue 
        event_stage = None
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage) 
            if ts_col_name and pd.isna(timestamps[ts_col_name]): 
                timestamps[ts_col_name] = event_dt 
    return pd.Series(timestamps, dtype='datetime64[ns]')

@st.cache_data 
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    """Loads, cleans, parses history, calculates timestamps."""
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None: return None
    df = _df_raw.copy() 
    submitted_on_col = None
    if "Submitted On" in df.columns: submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns:
         df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
         submitted_on_col = "Submitted On"
    else:
         if "Submitted On" not in df.columns: st.error("Missing 'Submitted On'/'Referral Date'."); return None
         else: submitted_on_col = "Submitted On"
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df); df.dropna(subset=["Submitted On_DT"], inplace=True); rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: st.warning(f"Dropped {rows_dropped} rows due to unparseable date.")
    if df.empty: st.error("No valid data remaining after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']
    parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string); parsed_cols[col_name] = parsed_col_name 
        else: st.warning(f"History column '{col_name}' not found.")
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History'); parsed_status_hist_col = parsed_cols.get('Lead Status History')
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        if 'Lead Stage History' not in df.columns and 'Lead Status History' not in df.columns: st.error("Neither history column found.")