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
# (parse_funnel_definition, parse_datetime_with_timezone, 
#  parse_history_string, get_stage_timestamps, preprocess_referral_data 
#  remain the same as in the previous version - Turn 56)

@st.cache_data 
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
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None:
        st.warning("Preprocessing cannot proceed: Missing inputs.")
        return None
    df = _df_raw.copy() 
    submitted_on_col = None
    if "Submitted On" in df.columns: submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns:
         st.info("Using 'Referral Date' column as 'Submitted On'.")
         df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
         submitted_on_col = "Submitted On"
    else:
         if "Submitted On" not in df.columns: 
              st.error("Missing required 'Submitted On' or 'Referral Date' column.")
              return None
         else: submitted_on_col = "Submitted On"
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True) 
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: st.warning(f"Dropped {rows_dropped} rows due to unparseable '{submitted_on_col}' date.")
    if df.empty: st.error("No valid data remaining after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']
    parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string) 
            parsed_cols[col_name] = parsed_col_name 
        else: st.warning(f"History column '{col_name}' not found.")
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History')
    parsed_status_hist_col = parsed_cols.get('Lead Status History')
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        if 'Lead Stage History' not in df.columns and 'Lead Status History' not in df.columns: st.error("Neither history column found.")
        else: st.error("History columns failed to parse.")
        return None 
    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    old_ts_cols = [col for col in df.columns if col.startswith('TS_')]
    df.drop(columns=old_ts_cols, inplace=True, errors='ignore')
    df = pd.concat([df, timestamp_cols_df], axis=1)
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce') 
    st.success("Referral Data Preprocessed Successfully.")
    return df

# --- Calculation Functions for App Sections ---

# @st.cache_data # Consider caching if inputs (df, spend) don't change often
def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    """ Calculates historical monthly cohort metrics."""
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    if not isinstance(monthly_ad_spend_input, dict): return pd.DataFrame()
    if "Submission_Month" not in _processed_df.columns: return pd.DataFrame()
    processed_df = _processed_df.copy() 
    try:
        cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals")
        cohort_summary = cohort_summary.set_index("Submission_Month")
        cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0) 
        reached_stage_cols_map = {}
        for stage_name in ordered_stages:
            ts_col = ts_col_map.get(stage_name)
            if ts_col and ts_col in processed_df.columns:
                reached_col_cleaned = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                reached_stage_cols_map[stage_name] = reached_col_cleaned
                if pd.api.types.is_datetime64_any_dtype(processed_df[ts_col]):
                     reached_stage_count = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
                     cohort_summary[reached_col_cleaned] = reached_stage_count
                else: cohort_summary[reached_col_cleaned] = 0
        cohort_summary = cohort_summary.fillna(0)
        for col in cohort_summary.columns:
            if col != "Ad Spend": cohort_summary[col] = cohort_summary[col].astype(int)
        cohort_summary["Ad Spend"] = cohort_summary["Ad Spend"].astype(float)
        pof_reached_col = reached_stage_cols_map.get("Passed Online Form")
        if pof_reached_col and pof_reached_col in cohort_summary.columns:
            cohort_summary.rename(columns={pof_reached_col: "Pre-Screener Qualified"}, inplace=True, errors='ignore') 
            base_count_col = "Pre-Screener Qualified"
        elif "Total Qualified Referrals" in cohort_summary.columns :
            cohort_summary.rename(columns={"Total Qualified Referrals": "Pre-Screener Qualified"}, inplace=True, errors='ignore')
            base_count_col = "Pre-Screener Qualified"
        else: base_count_col = None 
        proforma_metrics = pd.DataFrame(index=cohort_summary.index)
        if base_count_col and base_count_col in cohort_summary.columns: 
            proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
            proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col]
            proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col].replace(0, np.nan)).round(2) 
            for stage_orig, reached_col in reached_stage_cols_map.items():
                 metric_name = "Pre-Screener Qualified" if stage_orig == "Passed Online Form" else f"Total {stage_orig}"
                 if reached_col in cohort_summary.columns and metric_name not in proforma_metrics.columns:
                     proforma_metrics[metric_name] = cohort_summary[reached_col]
            sts_col = reached_stage_cols_map.get("Sent To Site"); appt_col = reached_stage_cols_map.get("Appointment Scheduled"); icf_col = reached_stage_cols_map.get("Signed ICF")
            if sts_col in cohort_summary.columns: proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col] / cohort_summary[base_count_col].replace(0, np.nan))
            if sts_col in cohort_summary.columns and appt_col in cohort_summary.columns: proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col] / cohort_summary[sts_col].replace(0, np.nan))
            if appt_col in cohort_summary.columns and icf_col in cohort_summary.columns: proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col] / cohort_summary[appt_col].replace(0, np.nan))
            if icf_col in cohort_summary.columns:
                proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col] / cohort_summary[base_count_col].replace(0, np.nan))
                proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col].replace(0, np.nan)).round(2)
        return proforma_metrics
    except Exception as e: st.error(f"Error during ProForma calculation: {e}"); return pd.DataFrame()

# @st.cache_data # Avoid caching until logic is stable and inputs defined
def calculate_site_metrics(_processed_df, ordered_stages, ts_col_map):
    """Calculates basic and advanced metrics per site."""
    st.write("Calculating site metrics...") # Progress indicator
    if _processed_df is None or _processed_df.empty or 'Site' not in _processed_df.columns:
        st.warning("Cannot calculate site metrics: Missing data or 'Site' column.")
        return pd.DataFrame() 

    processed_df = _processed_df.copy()
    site_metrics_list = []
    site_groups = processed_df.groupby('Site')

    # Define key stages & statuses
    qual_stage = "Passed Online Form"; sts_stage = "Sent To Site"; appt_stage = "Appointment Scheduled"
    icf_stage = "Signed ICF"; sf_stage = "Screen Failed"
    ts_qual_col = ts_col_map.get(qual_stage); ts_sts_col = ts_col_map.get(sts_stage)
    ts_appt_col = ts_col_map.get(appt_stage); ts_icf_col = ts_col_map.get(icf_stage)
    ts_sf_col = ts_col_map.get(sf_stage)
    site_contact_attempt_statuses = ["Site Contact Attempt 1"] # Assumption
    post_sts_progress_stages = ["Appointment Scheduled", "Signed ICF", "Enrolled", "Screen Failed"] 
    
    # Check required TS columns exist
    required_ts_cols = [ts_qual_col, ts_sts_col, ts_appt_col, ts_icf_col, ts_sf_col]
    if not all(col in processed_df.columns for col in required_ts_cols if col is not None):
         st.warning(f"One or more required timestamp columns missing ({[col for col in required_ts_cols if col not in processed_df.columns]}). Some site metrics may be inaccurate.")
         # Pad missing columns with NaT if absolutely necessary, though better to check upstream
         for col in required_ts_cols:
             if col and col not in processed_df.columns: processed_df[col] = pd.NaT


    for site_name, group in site_groups:
        metrics = {'Site': site_name}
        # Basic Counts & Conversions
        metrics['Total Qualified'] = group.shape[0] 
        reached_sts = group[ts_sts_col].notna().sum() if ts_sts_col else 0
        reached_appt = group[ts_appt_col].notna().sum() if ts_appt_col else 0
        reached_icf = group[ts_icf_col].notna().sum() if ts_icf_col else 0
        metrics['Reached StS'] = reached_sts; metrics['Reached Appt'] = reached_appt; metrics['Reached ICF'] = reached_icf
        total_qual=metrics['Total Qualified']
        metrics['Qual -> ICF %'] = (reached_icf / total_qual) if total_qual > 0 else 0
        metrics['StS -> Appt %'] = (reached_appt / reached_sts) if reached_sts > 0 else 0
        metrics['Appt -> ICF %'] = (reached_icf / reached_appt) if reached_appt > 0 else 0
            
        # Lag Times
        def calculate_avg_lag(df, col_from, col_to):
            if not col_from or not col_to or col_from not in df or col_to not in df: return np.nan
            valid_df = df.dropna(subset=[col_from, col_to])
            if valid_df.empty: return np.nan
            diff = valid_df[col_to] - valid_df[col_from]; diff_positive = diff[diff >= pd.Timedelta(days=0)] # Allow zero lag
            if diff_positive.empty: return np.nan
            return diff_positive.mean().total_seconds() / (60*60*24)
        metrics['Lag Qual -> ICF (Days)'] = calculate_avg_lag(group, ts_qual_col, ts_icf_col) # Overall lag

        # Advanced Metrics
        ttc_times = []; funnel_movement_steps = []
        sent_to_site_group = group.dropna(subset=[ts_sts_col]) if ts_sts_col else pd.DataFrame() # Ensure empty df if no StS
        if not sent_to_site_group.empty and 'Parsed_Lead_Status_History' in sent_to_site_group:
            for idx, row in sent_to_site_group.iterrows():
                ts_sent = row[ts_sts_col]; first_contact_ts = pd.NaT
                if row['Parsed_Lead_Status_History']:
                    for status_name, event_dt in row['Parsed_Lead_Status_History']:
                        if status_name in site_contact_attempt_statuses and event_dt >= ts_sent:
                            first_contact_ts = event_dt; break
                if pd.notna(first_contact_ts):
                    time_diff = first_contact_ts - ts_sent
                    if time_diff >= pd.Timedelta(0): ttc_times.append(time_diff.total_seconds() / (60*60*24))                 
                stages_reached_post_sts = set()
                if 'Parsed_Lead_Stage_History' in row and row['Parsed_Lead_Stage_History']:
                    for stage_name, event_dt in row['Parsed_Lead_Stage_History']:
                        if stage_name in post_sts_progress_stages and event_dt > ts_sent: stages_reached_post_sts.add(stage_name)
                funnel_movement_steps.append(len(stages_reached_post_sts))
        metrics['Avg TTC (Days)'] = np.mean(ttc_times) if ttc_times else np.nan
        metrics['Avg Funnel Movement Steps'] = np.mean(funnel_movement_steps) if funnel_movement_steps else 0
        
        # Screen Fail Rate
        site_sfs = group[ts_sf_col].notna().sum() if ts_sf_col else 0
        metrics['Site Screen Fail %'] = (site_sfs / reached_icf) if reached_icf > 0 else 0.0 # Treat rate as 0 if no ICFs
        
        site_metrics_list.append(metrics)

    site_metrics_df = pd.DataFrame(site_metrics_list)
    st.success("Site metrics calculated.")
    return site_metrics_df 

# @st.cache_data # Avoid caching until logic is stable
def score_sites(_site_metrics_df, weights):
    """Applies normalization and weighting to score sites."""
    st.write("Applying site scoring...") # Progress indicator
    if _site_metrics_df is None or _site_metrics_df.empty:
        st.warning("Cannot score sites: No site metrics available.")
        return pd.DataFrame()
        
    site_metrics_df = _site_metrics_df.copy() # Work on copy
    
    # Metrics to use for scoring (match keys in weights)
    metrics_to_scale = list(weights.keys())
    lower_is_better = ["Avg TTC (Days)", "Site Screen Fail %"]

    # Ensure all required columns exist, fill NaN appropriately before scaling
    for col in metrics_to_scale:
        if col not in site_metrics_df.columns:
            st.warning(f"Metric column '{col}' needed for scoring not found. Assigning 0.")
            site_metrics_df[col] = 0 # Add column with 0 if missing
        
        # Fill NaNs based on whether lower is better
        if col in lower_is_better:
            # Fill NaN TTC/SFR with a high value (e.g., max observed + buffer or a reasonable large number)
            max_val = site_metrics_df[col].max()
            fill_val = max_val + 1 if pd.notna(max_val) else 999 # Default high if no non-NaN values
            site_metrics_df[col].fillna(fill_val, inplace=True)
        else:
            # Fill NaN for rates/steps (higher is better) with 0
            site_metrics_df[col].fillna(0, inplace=True)

    # Check if any metrics have zero variance (all sites same value)
    # Add small noise to prevent NaN during scaling if needed, or handle in scaler
    scaled_metrics = pd.DataFrame(index=site_metrics_df.index) # Initialize scaled df
    for col in metrics_to_scale:
         if site_metrics_df[col].min() == site_metrics_df[col].max():
             # Handle zero variance: assign neutral score (0.5) or skip scaling for this metric
             scaled_metrics[col] = 0.5 
             st.caption(f"Note: Metric '{col}' has same value for all sites; contributing neutral score.")
         else:
             scaler = MinMaxScaler()
             # Reshape needed for single column scaling
             scaled_values = scaler.fit_transform(site_metrics_df[[col]]) 
             scaled_metrics[col] = scaled_values.flatten() # Store scaled values (0-1)

    # Invert scales for 'lower is better' metrics
    for col in lower_is_better:
        if col in scaled_metrics.columns:
            scaled_metrics[col] = 1 - scaled_metrics[col]

    # Calculate Weighted Score
    site_metrics_df['Score_Raw'] = 0
    total_weight_used = 0 # Track sum of weights applied
    for metric, weight in weights.items():
         if metric in scaled_metrics.columns:
             site_metrics_df['Score_Raw'] += scaled_metrics[metric] * weight
             total_weight_used += abs(weight) # Use absolute for sum magnitude if using signed weights
             
    # Scale score to 0-100 based on theoretical min/max or observed range
    # Using observed range:
    min_raw_score = site_metrics_df['Score_Raw'].min()
    max_raw_score = site_metrics_df['Score_Raw'].max()
    if max_raw_score > min_raw_score:
         site_metrics_df['Score'] = ((site_metrics_df['Score_Raw'] - min_raw_score) / (max_raw_score - min_raw_score)) * 100
    elif pd.notna(max_raw_score): # Handle case where all scores are the same
         site_metrics_df['Score'] = 50.0 # Assign a mid-range score
    else: # Handle all NaN case
         site_metrics_df['Score'] = 0.0

    # Assign Grades
    def assign_grade(score):
        if pd.isna(score): return 'N/A'
        if score >= 97: return 'A+'
        elif score >= 90: return 'A'
        elif score >= 87: return 'B+'
        elif score >= 80: return 'B'
        elif score >= 77: return 'C+'
        elif score >= 70: return 'C'
        elif score >= 67: return 'D+'
        elif score >= 60: return 'D'
        else: return 'F'
    site_metrics_df['Grade'] = site_metrics_df['Score'].apply(assign_grade)

    # Prepare final output table
    site_metrics_df.reset_index(inplace=True) # Ensure 'Site' is a column
    site_metrics_df.sort_values('Score', ascending=False, inplace=True)
    st.success("Site scoring complete.")
    return site_metrics_df 

# Placeholder - Logic from Turn 40/42 will go here
# @st.cache_data 
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs): 
     st.write("Projection calculation logic to be implemented here.")
     # Example: Use projection_inputs['spend'], projection_inputs['cpqr'] etc.
     return pd.DataFrame({'Month': ['Apr-25', 'May-25'],'Projected ICF': [0, 0], 'Info': ['Calculation Pending']}) 


# --- Streamlit UI ---

# Sidebar for Inputs
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV/TSV)", type=["csv", "tsv"], key="referral_uploader")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv", "csv"], key="funnel_uploader") 
    st.divider()
    
    # --- Inputs for Calculations ---
    st.subheader("Historical Ad Spend (Monthly)")
    st.info("Enter **historical** spend for past months found in data.")
    # TODO: Make this section dynamic based on data
    ad_spend_input_dict_manual = {}
    spend_month_str_1 = st.text_input("Spend Month 1 (YYYY-MM)", "2025-02", key="spend_m1_str")
    spend_val_1 = st.number_input(f"Spend for {spend_month_str_1}", value=45000.0, step=1000.0, format="%.2f", key="spend_v1")
    spend_month_str_2 = st.text_input("Spend Month 2 (YYYY-MM)", "2025-03", key="spend_m2_str")
    spend_val_2 = st.number_input(f"Spend for {spend_month_str_2}", value=60000.0, step=1000.0, format="%.2f", key="spend_v2")
    ad_spend_input_dict = {}
    try: ad_spend_input_dict[pd.Period(spend_month_str_1, freq='M')] = spend_val_1
    except Exception: st.warning(f"Invalid format for Month 1")
    try: ad_spend_input_dict[pd.Period(spend_month_str_2, freq='M')] = spend_val_2
    except Exception: st.warning(f"Invalid format for Month 2")
    st.caption("Note: Ad Spend input currently supports only 2 months.")

    st.divider()
    st.subheader("Site Scoring Weights")
    weights = {}
    weights["Qual -> ICF %"] = st.slider("Weight: Qualified -> ICF %", 0, 100, 20, key='w_qicf') / 100.0
    weights["Avg TTC (Days)"] = st.slider("Weight: Avg Time to Contact (Lower Better)", 0, 100, 25, key='w_ttc') / 100.0 # Keep positive, inversion handled in score_sites
    weights["Avg Funnel Movement Steps"] = st.slider("Weight: Avg Funnel Movement Steps", 0, 100, 5, key='w_fms') / 100.0
    weights["Site Screen Fail %"] = st.slider("Weight: Site Screen Fail % (Lower Better)", 0, 100, 5, key='w_sfr') / 100.0 # Keep positive
    weights["StS -> Appt %"] = st.slider("Weight: StS -> Appt Sched %", 0, 100, 30, key='w_sa') / 100.0
    weights["Appt -> ICF %"] = st.slider("Weight: Appt Sched -> ICF %", 0, 100, 15, key='w_ai') / 100.0
    
    total_weight = sum(weights.values())
    st.caption(f"Current Total Weight: {total_weight*100:.0f}%")
    # Note: Weights don't strictly need to sum to 100 as score is rescaled, but it helps conceptually.


# --- Main App Logic & Display ---
referral_data_processed = None 
funnel_definition, ordered_stages, ts_col_map = None, None, None 

if uploaded_referral_file is not None and uploaded_funnel_def_file is not None:
    funnel_definition, ordered_stages, ts_col_map = parse_funnel_definition(uploaded_funnel_def_file)

    if funnel_definition and ordered_stages and ts_col_map: 
        try:
             bytes_data = uploaded_referral_file.getvalue()
             try: decoded_data = bytes_data.decode("utf-8")
             except UnicodeDecodeError: decoded_data = bytes_data.decode("latin-1") 
             stringio = io.StringIO(decoded_data)
             try:
                  # Assuming CSV based on last successful run
                  referrals_raw_df = pd.read_csv(stringio, sep=',', header=0, on_bad_lines='warn', low_memory=False) 
                  # DEBUG LINE REMOVED: st.write("Columns found by pandas:", referrals_raw_df.columns.tolist()) 
                  st.success("Referral Data Loaded (assuming CSV with header).")
                  referral_data_processed = preprocess_referral_data(referrals_raw_df, funnel_definition, ordered_stages, ts_col_map)
             except Exception as read_err:
                  st.error(f"Error reading referral file (assuming CSV, header=0): {read_err}")
        except Exception as e:
             st.error(f"An unexpected error occurred during referral data loading: {e}")

# --- Display Sections ---
if referral_data_processed is not None and not referral_data_processed.empty:
    st.markdown("---")
    st.success("Data loaded and preprocessed. Displaying analysis sections.") 
    
    tab1, tab2, tab3 = st.tabs(["📅 Monthly ProForma", "🏆 Site Performance", "📈 Projections"])

    with tab1:
        st.header("Monthly ProForma (Historical Cohorts)")
        st.write("Shows historical performance based on the month referrals were submitted.")
        proforma_df = calculate_proforma_metrics(referral_data_processed, ordered_stages, ts_col_map, ad_spend_input_dict) 
        if not proforma_df.empty:
            proforma_display = proforma_df.transpose()
            proforma_display.columns = [str(col) for col in proforma_display.columns] 
            format_dict = {}
            for idx in proforma_display.index:
                 if 'Cost' in idx or 'Spend' in idx: format_dict[idx] = "${:,.2f}"
                 elif '%' in idx: format_dict[idx] = "{:.1%}"
                 elif 'Total' in idx or 'Qualified' in idx or 'Reached' in idx: format_dict[idx] = "{:,.0f}"
            st.dataframe(proforma_display.style.format(format_dict, na_rep='-'))
            try:
                 csv = proforma_df.reset_index().to_csv(index=False).encode('utf-8')
                 st.download_button(label="Download ProForma Data as CSV", data=csv, file_name='monthly_proforma.csv', mime='text/csv', key='download_proforma')
            except Exception as e: st.warning(f"Could not generate download button: {e}")
        else: st.warning("Could not generate ProForma table.")

    with tab2:
        st.header("Site Performance Ranking")
        st.write("Calculates metrics and scores for each site based on weights set in sidebar.")
        
        # Calculate Site Metrics (cached if underlying data doesn't change)
        # Consider caching calculate_site_metrics if _processed_df doesn't change
        site_metrics_calculated = calculate_site_metrics(referral_data_processed, ordered_stages, ts_col_map) 
        
        if not site_metrics_calculated.empty:
            # Score Sites using weights from sidebar
            # Note: score_sites function applies MinMax scaling etc.
            ranked_sites_df = score_sites(site_metrics_calculated, weights) 
            
            st.subheader("Site Ranking")
            # Select and format columns for display
            display_cols = ['Site', 'Score', 'Grade', 'Total Qualified', 'Qual -> ICF %', 'Avg TTC (Days)', 
                            'Avg Funnel Movement Steps', 'StS -> Appt %', 'Appt -> ICF %', 'Site Screen Fail %']
            display_cols = [col for col in display_cols if col in ranked_sites_df.columns] # Ensure columns exist
            final_ranked_display = ranked_sites_df[display_cols].copy()

            # Formatting
            final_ranked_display['Score'] = final_ranked_display['Score'].round(1)
            percent_cols = [col for col in final_ranked_display.columns if '%' in col]
            lag_cols = [col for col in final_ranked_display.columns if 'TTC' in col]
            step_cols = [col for col in final_ranked_display.columns if 'Steps' in col]
            
            for col in percent_cols: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
            for col in lag_cols: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
            for col in step_cols: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
            
            st.dataframe(final_ranked_display.style.format(na_rep='-'))

            # Add download button
            try:
                 csv_sites = final_ranked_display.to_csv(index=False).encode('utf-8')
                 st.download_button(label="Download Site Ranking Data as CSV", data=csv_sites, file_name='site_ranking.csv', mime='text/csv', key='download_sites')
            except Exception as e: st.warning(f"Could not generate site download button: {e}")

        else:
            st.warning("Could not calculate site metrics to display ranking.")


    with tab3:
        st.header("Projections")
        st.write("Forecasts future performance based on assumptions.")
        st.info("Projection calculation logic needs to be implemented.")
        
        # --- Add Projection Inputs to Sidebar ---
        # Example:
        # with st.sidebar:
        #    st.divider()
        #    st.subheader("Projection Assumptions")
        #    proj_horizon = st.slider("Projection Horizon (Months)", 1, 24, 12)
        #    proj_cpqr = st.number_input("Assumed CPQR ($)", value=120.0, step=10.0)
        #    # Add inputs for future spend, conversion rates etc.

        # --- Call projection function (when implemented) ---
        # projection_inputs = { 'horizon': proj_horizon, 'cpqr': proj_cpqr, ... } 
        # projected_icfs = calculate_projections(referral_data_processed, ordered_stages, ts_col_map, projection_inputs)
        # st.dataframe(projected_icfs)

        # Placeholder display
        st.dataframe(pd.DataFrame({'Month': ['Apr-25', 'May-25'],'Info': ['Calculation Pending']}))

elif not uploaded_referral_file or not uploaded_funnel_def_file:
    st.info("👋 Welcome! Please upload both the Referral Data (CSV) and Funnel Definition (TSV) files using the sidebar to begin.")