# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import io
from sklearn.preprocessing import MinMaxScaler # For site scoring
import traceback # To print full tracebacks for debugging

# --- Page Configuration ---
st.set_page_config(page_title="Recruitment Forecasting Tool", layout="wide")
st.title("📊 Recruitment Forecasting Tool")

# --- Helper Functions (Data Parsing & Timestamping) ---

@st.cache_data 
def parse_funnel_definition(uploaded_file):
    st.write("DEBUG: Running parse_funnel_definition")
    if uploaded_file is None: return None, None, None 
    try:
        bytes_data = uploaded_file.getvalue(); stringio = io.StringIO(bytes_data.decode("utf-8", errors='replace')) 
        df_funnel_def = pd.read_csv(stringio, sep='\t', header=None) 
        parsed_funnel_definition = {}; parsed_ordered_stages = []; ts_col_map = {} 
        for col_idx in df_funnel_def.columns:
            column_data = df_funnel_def[col_idx]; stage_name = column_data.iloc[0]
            if pd.isna(stage_name) or str(stage_name).strip() == "": continue
            stage_name = str(stage_name).strip().replace('"', ''); parsed_ordered_stages.append(stage_name)
            statuses = column_data.iloc[1:].dropna().astype(str).apply(lambda x: x.strip().replace('"', '')).tolist()
            statuses = [s for s in statuses if s]; 
            if stage_name not in statuses: statuses.append(stage_name)
            parsed_funnel_definition[stage_name] = statuses
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"; ts_col_map[stage_name] = clean_ts_name
        if not parsed_ordered_stages: st.error("Could not parse stages."); return None, None, None
        st.write("DEBUG: Exiting parse_funnel_definition successfully")
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e: 
        st.error(f"Error parsing Funnel Definition file: {e}")
        st.exception(e)
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    if pd.isna(dt_str): return pd.NaT 
    dt_str_cleaned = str(dt_str).strip(); tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned); parsed_dt = pd.to_datetime(dt_str_no_tz, errors='coerce') 
    return parsed_dt

def parse_history_string(history_str):
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    raw_lines = str(history_str).strip().split('\n'); parsed_events = []
    for line in raw_lines:
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups(); name = name.strip(); dt_obj = parse_datetime_with_timezone(dt_str.strip()) 
            if name and pd.notna(dt_obj): 
                try: py_dt = dt_obj.to_pydatetime(); parsed_events.append((name, py_dt)) 
                except AttributeError: pass 
    try: parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e: pass 
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}; status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps) 
    for stage, statuses in funnel_def.items():
        for status in statuses: status_to_stage_map[status] = stage
    all_events = []; stage_hist = row.get(parsed_stage_history_col, []); status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist])
    try: all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e: pass 
    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue 
        event_stage = None; 
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage) 
            if ts_col_name and pd.isna(timestamps[ts_col_name]): timestamps[ts_col_name] = event_dt 
    return pd.Series(timestamps, dtype='datetime64[ns]')

@st.cache_data 
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    """Loads, cleans, parses history, calculates timestamps."""
    st.write("DEBUG: Running preprocess_referral_data")
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None: return None
    df = _df_raw.copy(); submitted_on_col = None
    if "Submitted On" in df.columns: submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True); submitted_on_col = "Submitted On"
    else: 
        if "Submitted On" not in df.columns: st.error("Missing 'Submitted On'/'Referral Date'."); return None
        else: submitted_on_col = "Submitted On"
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df); df.dropna(subset=["Submitted On_DT"], inplace=True); rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: st.warning(f"Dropped {rows_dropped} rows due to unparseable date.")
    if df.empty: st.error("No valid data remaining after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']; parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string); parsed_cols[col_name] = parsed_col_name 
        else: st.warning(f"History column '{col_name}' not found.")
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History'); parsed_status_hist_col = parsed_cols.get('Lead Status History')
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        if 'Lead Stage History' not in df.columns and 'Lead Status History' not in df.columns: st.error("Neither history column found.")
        else: st.error("History columns failed to parse.")
        return None 
    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    old_ts_cols = [col for col in df.columns if col.startswith('TS_')]; df.drop(columns=old_ts_cols, inplace=True, errors='ignore')
    df = pd.concat([df, timestamp_cols_df], axis=1)
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce') 
    st.success("Referral Data Preprocessed Successfully.")
    st.write("DEBUG: Exiting preprocess_referral_data successfully")
    return df

# --- Calculation Functions for App Sections ---

# @st.cache_data 
def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    """ Calculates historical monthly cohort metrics."""
    st.write("DEBUG: Running calculate_proforma_metrics")
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
        base_count_col = None 
        if pof_reached_col and pof_reached_col in cohort_summary.columns:
            cohort_summary.rename(columns={pof_reached_col: "Pre-Screener Qualified"}, inplace=True, errors='ignore') 
            base_count_col = "Pre-Screener Qualified"
        elif "Total Qualified Referrals" in cohort_summary.columns :
            cohort_summary.rename(columns={"Total Qualified Referrals": "Pre-Screener Qualified"}, inplace=True, errors='ignore')
            base_count_col = "Pre-Screener Qualified"
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
        st.write("DEBUG: Exiting calculate_proforma_metrics successfully")
        return proforma_metrics
    except Exception as e: 
        st.error(f"ProForma Calc Error: {e}"); st.exception(e)
        return pd.DataFrame() # Return empty DataFrame on error

# @st.cache_data 
def calculate_site_metrics(_processed_df, ordered_stages, ts_col_map):
    """Calculates basic and advanced metrics per site."""
    st.write("DEBUG: Running calculate_site_metrics")
    if _processed_df is None or _processed_df.empty or 'Site' not in _processed_df.columns: return pd.DataFrame() 
    processed_df = _processed_df.copy(); site_metrics_list = []
    try: 
        site_groups = processed_df.groupby('Site')
        qual_stage="Passed Online Form"; sts_stage="Sent To Site"; appt_stage="Appointment Scheduled"; icf_stage="Signed ICF"; sf_stage="Screen Failed"
        ts_qual_col=ts_col_map.get(qual_stage); ts_sts_col=ts_col_map.get(sts_stage); ts_appt_col=ts_col_map.get(appt_stage); ts_icf_col=ts_col_map.get(icf_stage); ts_sf_col=ts_col_map.get(sf_stage)
        site_contact_attempt_statuses = ["Site Contact Attempt 1"]; post_sts_progress_stages = ["Appointment Scheduled", "Signed ICF", "Enrolled", "Screen Failed"] 
        required_ts_cols = [ts_qual_col, ts_sts_col, ts_appt_col, ts_icf_col, ts_sf_col]
        for col in required_ts_cols:
            if col and col not in processed_df.columns: processed_df[col] = pd.NaT; processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
        for site_name, group in site_groups:
            metrics = {'Site': site_name}; metrics['Total Qualified'] = group.shape[0] 
            reached_sts = group[ts_sts_col].notna().sum() if ts_sts_col else 0
            reached_appt = group[ts_appt_col].notna().sum() if ts_appt_col else 0
            reached_icf = group[ts_icf_col].notna().sum() if ts_icf_col else 0
            metrics['Reached StS'] = reached_sts; metrics['Reached Appt'] = reached_appt; metrics['Reached ICF'] = reached_icf 
            total_qual=metrics['Total Qualified']
            metrics['Qual -> ICF %'] = (reached_icf / total_qual) if total_qual > 0 else 0.0
            metrics['StS -> Appt %'] = (reached_appt / reached_sts) if reached_sts > 0 else 0.0
            metrics['Appt -> ICF %'] = (reached_icf / reached_appt) if reached_appt > 0 else 0.0
            def calculate_avg_lag(df, col_from, col_to):
                if not col_from or not col_to or col_from not in df or col_to not in df or not pd.api.types.is_datetime64_any_dtype(df[col_from]) or not pd.api.types.is_datetime64_any_dtype(df[col_to]): return np.nan
                valid_df = df.dropna(subset=[col_from, col_to]);
                if valid_df.empty: return np.nan
                diff = valid_df[col_to] - valid_df[col_from]; diff_positive = diff[diff >= pd.Timedelta(days=0)] 
                if diff_positive.empty: return np.nan
                return diff_positive.mean().total_seconds() / (60*60*24)
            metrics['Lag Qual -> ICF (Days)'] = calculate_avg_lag(group, ts_qual_col, ts_icf_col) 
            ttc_times = []; funnel_movement_steps = []
            sent_to_site_group = group.dropna(subset=[ts_sts_col]) if ts_sts_col and ts_sts_col in group else pd.DataFrame()
            parsed_status_col = f"Parsed_Lead_Status_History"; parsed_stage_col = f"Parsed_Lead_Stage_History"
            if not sent_to_site_group.empty and parsed_status_col in sent_to_site_group:
                for idx, row in sent_to_site_group.iterrows():
                    ts_sent = row[ts_sts_col]; first_contact_ts = pd.NaT
                    history_list = row.get(parsed_status_col, [])
                    if history_list: 
                        for status_name, event_dt in history_list:
                            if status_name in site_contact_attempt_statuses and pd.notna(event_dt) and pd.notna(ts_sent) and event_dt >= ts_sent:
                                first_contact_ts = event_dt; break
                    if pd.notna(first_contact_ts) and pd.notna(ts_sent):
                        time_diff = first_contact_ts - ts_sent
                        if time_diff >= pd.Timedelta(0): ttc_times.append(time_diff.total_seconds() / (60*60*24))                 
                    stages_reached_post_sts = set()
                    stage_history_list = row.get(parsed_stage_col, [])
                    if stage_history_list and pd.notna(ts_sent): 
                         for stage_name, event_dt in stage_history_list:
                             if stage_name in post_sts_progress_stages and pd.notna(event_dt) and event_dt > ts_sent: stages_reached_post_sts.add(stage_name)
                    funnel_movement_steps.append(len(stages_reached_post_sts))
            metrics['Avg TTC (Days)'] = np.mean(ttc_times) if ttc_times else np.nan
            metrics['Avg Funnel Movement Steps'] = np.mean(funnel_movement_steps) if funnel_movement_steps else 0
            site_sfs = group[ts_sf_col].notna().sum() if ts_sf_col and ts_sf_col in group else 0
            metrics['Site Screen Fail %'] = (site_sfs / reached_icf) if reached_icf > 0 else 0.0 
            site_metrics_list.append(metrics)
        site_metrics_df = pd.DataFrame(site_metrics_list)
        st.write("DEBUG: Exiting calculate_site_metrics successfully")
        return site_metrics_df 
    except Exception as e: st.error(f"Error calculating site metrics: {e}"); st.exception(e); return pd.DataFrame()

# @st.cache_data 
def score_sites(_site_metrics_df, weights):
    """Applies normalization and weighting to score sites using percentile grading."""
    st.write("DEBUG: Running score_sites")
    if _site_metrics_df is None or _site_metrics_df.empty: return pd.DataFrame()
    try: 
        site_metrics_df = _site_metrics_df.copy() 
        if 'Site' not in site_metrics_df.columns:
             if site_metrics_df.index.name == 'Site': site_metrics_df.reset_index(inplace=True)
             else: return pd.DataFrame()
        site_metrics_df.set_index('Site', inplace=True) 
        metrics_to_scale = list(weights.keys())
        lower_is_better = ["Avg TTC (Days)", "Site Screen Fail %"]
        for col in metrics_to_scale:
            if col not in site_metrics_df.columns:
                site_metrics_df[col] = 0 if col not in lower_is_better else np.nan 
            if col in lower_is_better:
                max_val = site_metrics_df[col].max(); fill_val = max_val + 1 if pd.notna(max_val) and max_val > 0 else 999 
                site_metrics_df[col].fillna(fill_val, inplace=True)
            else: site_metrics_df[col].fillna(0, inplace=True)
        scaled_metrics = pd.DataFrame(index=site_metrics_df.index) 
        if not site_metrics_df.empty and len(site_metrics_df) > 0: 
            for col in metrics_to_scale:
                 if col in site_metrics_df.columns: 
                     min_val = site_metrics_df[col].min(); max_val = site_metrics_df[col].max()
                     if min_val == max_val: scaled_metrics[col] = 0.5 
                     elif pd.notna(min_val) and pd.notna(max_val): 
                         scaler = MinMaxScaler()
                         scaled_values = scaler.fit_transform(site_metrics_df[[col]]) 
                         scaled_metrics[col] = scaled_values.flatten() 
                     else: scaled_metrics[col] = 0.5
                 else: scaled_metrics[col] = 0.5 
            for col in lower_is_better:
                if col in scaled_metrics.columns: scaled_metrics[col] = 1 - scaled_metrics[col]
        site_metrics_df['Score_Raw'] = 0; total_weight_applied = 0
        for metric, weight in weights.items():
             if metric in scaled_metrics.columns:
                 positive_weight = abs(weight) 
                 scaled_metrics[metric].fillna(0.5, inplace=True) 
                 site_metrics_df['Score_Raw'] += scaled_metrics[metric] * positive_weight
                 total_weight_applied += positive_weight
        if total_weight_applied > 0: site_metrics_df['Score'] = (site_metrics_df['Score_Raw'] / total_weight_applied) * 100
        else: site_metrics_df['Score'] = 0.0
        site_metrics_df['Score'].fillna(0.0, inplace=True)
        if len(site_metrics_df) > 1: 
            site_metrics_df['Score_Rank_Percentile'] = site_metrics_df['Score'].rank(pct=True)
            bins = [0, 0.10, 0.25, 0.60, 0.85, 1.0]; labels = ['F', 'D', 'C', 'B', 'A']
            try: site_metrics_df['Grade'] = pd.qcut(site_metrics_df['Score_Rank_Percentile'], q=bins, labels=labels, duplicates='drop') 
            except ValueError: 
                 st.warning("Using fixed score ranges for grading (percentile failed).")
                 def assign_grade_fallback(score): 
                     if pd.isna(score): return 'N/A'
                     score = round(score)
                     if score >= 90: return 'A'; elif score >= 80: return 'B'; elif score >= 70: return 'C'; elif score >= 60: return 'D'; else: return 'F'
                 site_metrics_df['Grade'] = site_metrics_df['Score'].apply(assign_grade_fallback)
            site_metrics_df['Grade'] = site_metrics_df['Grade'].astype(str).replace('nan', 'N/A') 
        elif len(site_metrics_df) == 1: site_metrics_df['Grade'] = 'N/A' 
        else: site_metrics_df['Grade'] = []
        site_metrics_df.reset_index(inplace=True); site_metrics_df.sort_values('Score', ascending=False, inplace=True)
        st.write("DEBUG: Exiting score_sites successfully")
        return site_metrics_df 
    except Exception as e: st.error(f"Error during Site Scoring: {e}"); st.exception(e); return _site_metrics_df.reset_index() if _site_metrics_df is not None else pd.DataFrame()

# --- Projection Function with Debugging ---
@st.cache_data 
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs): 
    """Calculates projections based on inputs and historical data."""
    
    st.write("DEBUG: Running calculate_projections") # DEBUG
    st.write("DEBUG: Projection Inputs Received:", projection_inputs) # DEBUG
    
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    required_keys = ['horizon', 'spend_dict', 'cpqr', 'conv_rates', 'use_rolling_rates', 'rolling_window']
    if not isinstance(projection_inputs, dict) or not all(k in projection_inputs for k in required_keys): return pd.DataFrame()
        
    processed_df = _processed_df.copy(); horizon = projection_inputs['horizon']
    future_spend_dict = projection_inputs['spend_dict']; assumed_cpqr = projection_inputs['cpqr']
    manual_conv_rates = projection_inputs['conv_rates']; use_rolling_rates = projection_inputs['use_rolling_rates']
    rolling_window = projection_inputs['rolling_window']
    if assumed_cpqr <= 0: st.error("CPQR must be > 0."); return pd.DataFrame()

    lag_results = {}
    for i in range(len(ordered_stages) - 1):
        stage_from = ordered_stages[i]; stage_to = ordered_stages[i+1]; ts_col_from = ts_col_map.get(stage_from); ts_col_to = ts_col_map.get(stage_to)
        if ts_col_from in processed_df.columns and ts_col_to in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df[ts_col_from]) and pd.api.types.is_datetime64_any_dtype(processed_df[ts_col_to]):
            valid_ts_df = processed_df.dropna(subset=[ts_col_from, ts_col_to])
            if not valid_ts_df.empty:
                time_diff = valid_ts_df[ts_col_to] - valid_ts_df[ts_col_from]; diff_positive = time_diff[time_diff >= pd.Timedelta(days=0)] 
                if not diff_positive.empty: lag_results[f"{stage_from} -> {stage_to}"] = diff_positive.mean().total_seconds() / (60*60*24)
                else: lag_results[f"{stage_from} -> {stage_to}"] = np.nan
            else: lag_results[f"{stage_from} -> {stage_to}"] = np.nan
        else: lag_results[f"{stage_from} -> {stage_to}"] = np.nan
    start_stage = ordered_stages[0]; end_stage = "Signed ICF"; ts_col_start = ts_col_map.get(start_stage); ts_col_end = ts_col_map.get(end_stage)
    if ts_col_start in processed_df.columns and ts_col_end in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df[ts_col_start]) and pd.api.types.is_datetime64_any_dtype(processed_df[ts_col_end]):
         valid_ts_df_overall = processed_df.dropna(subset=[ts_col_start, ts_col_end])
         if not valid_ts_df_overall.empty:
             time_diff_overall = valid_ts_df_overall[ts_col_end] - valid_ts_df_overall[ts_col_start]; diff_positive = time_diff_overall[time_diff_overall >= pd.Timedelta(days=0)]
             if not diff_positive.empty: lag_results[f"{start_stage} -> {end_stage}"] = diff_positive.mean().total_seconds() / (60*60*24)
             else: lag_results[f"{start_stage} -> {end_stage}"] = np.nan
         else: lag_results[f"{start_stage} -> {end_stage}"] = np.nan
    else: lag_results[f"{start_stage} -> {end_stage}"] = np.nan
    st.write(f"DEBUG: Overall Lag (Qual->ICF): {lag_results.get(f'{start_stage} -> {end_stage}')}") # DEBUG

    projection_conv_rates = {}
    if use_rolling_rates and "Submission_Month" in processed_df.columns:
        hist_counts = processed_df.groupby("Submission_Month").size().to_frame(name="Total Qualified Referrals") 
        for stage_name in ordered_stages:
             ts_col = ts_col_map.get(stage_name)
             if ts_col and ts_col in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df[ts_col]):
                  reached_col_cleaned = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                  hist_counts[reached_col_cleaned] = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
        hist_counts = hist_counts.fillna(0)
        pof_hist_col = f"Reached_{ordered_stages[0].replace(' ', '_').replace('(', '').replace(')', '')}"
        base_hist_col = pof_hist_col if pof_hist_col in hist_counts.columns else "Total Qualified Referrals"
        valid_historical_rates_found = False
        for i in range(len(ordered_stages) - 1):
            stage_from = ordered_stages[i]; stage_to = ordered_stages[i+1]
            col_from_key = ordered_stages[i] if i > 0 else ordered_stages[0] # Base on actual first stage for 'from' if not POF itself
            col_from = f"Reached_{col_from_key.replace(' ', '_').replace('(', '').replace(')', '')}"
            if i == 0 : col_from = base_hist_col # Use the base for the very first step

            col_to = f"Reached_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"
            rate_key = f"{stage_from} -> {stage_to}"
            if col_from in hist_counts.columns and col_to in hist_counts.columns:
                 monthly_rate = (hist_counts[col_to] / hist_counts[col_from].replace(0, np.nan)).fillna(0)
                 actual_window = min(rolling_window, len(monthly_rate))
                 if actual_window > 0:
                     rolling_avg_rate = monthly_rate.rolling(window=actual_window, min_periods=1).mean()
                     if not rolling_avg_rate.empty:
                          latest_rolling_rate = rolling_avg_rate.iloc[-1]
                          projection_conv_rates[rate_key] = latest_rolling_rate if pd.notna(latest_rolling_rate) else 0.0
                          valid_historical_rates_found = True
                     else: projection_conv_rates[rate_key] = 0.0 
                 else: projection_conv_rates[rate_key] = 0.0 
            else: projection_conv_rates[rate_key] = 0.0 
        if not valid_historical_rates_found: projection_conv_rates = manual_conv_rates 
    else: projection_conv_rates = manual_conv_rates
    st.write("DEBUG: Rates used for projection:", projection_conv_rates) # DEBUG

    try:
        last_historical_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df and not processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
        proj_start_month = last_historical_month + 1 
        future_months = pd.period_range(start=proj_start_month, periods=horizon, freq='M')
        projection_cohorts = pd.DataFrame(index=future_months)
        projection_cohorts['Forecasted_Ad_Spend'] = [future_spend_dict.get(m, 0) for m in future_months] 
        projection_cohorts['Forecasted_PSQ'] = (projection_cohorts['Forecasted_Ad_Spend'] / assumed_cpqr).round(0).astype(int)
        last_stage_proj_col = 'Forecasted_PSQ'; icf_stage_name = "Signed ICF" 
        icf_proj_col = f"Projected_{icf_stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        for i in range(len(ordered_stages) - 1):
            stage_from = ordered_stages[i]; stage_to = ordered_stages[i+1]
            conv_rate = projection_conv_rates.get(f"{stage_from} -> {stage_to}", 0) 
            proj_col_to = f"Projected_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"
            if last_stage_proj_col in projection_cohorts.columns: 
                proj_counts = (projection_cohorts[last_stage_proj_col] * conv_rate)
                projection_cohorts[proj_col_to] = proj_counts.round(0).fillna(0).astype(int) 
                last_stage_proj_col = proj_col_to 
            else: projection_cohorts[proj_col_to] = 0 
            if stage_to == icf_stage_name: break 
        
        overall_lag_days = lag_results.get(f"{ordered_stages[0]} -> {icf_stage_name}") 
        if pd.isna(overall_lag_days):
            cumulative_lag = 0; valid_lag_path = True
            try: 
                icf_index = ordered_stages.index(icf_stage_name)
                for i in range(icf_index):
                    step_lag = lag_results.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}")
                    if pd.isna(step_lag): valid_lag_path = False; break
                    cumulative_lag += step_lag
                if valid_lag_path: overall_lag_days = cumulative_lag
                else: overall_lag_days = 30 
            except ValueError: overall_lag_days = 30
        lag_in_months = int(np.round(overall_lag_days / 30.4375)) if pd.notna(overall_lag_days) else 0 
        
        projection_results = pd.DataFrame(index=future_months); projection_results['Projected_ICF_Landed'] = 0 
        if icf_proj_col in projection_cohorts.columns:
            for start_month_idx, start_month in enumerate(projection_cohorts.index):
                projected_icfs_for_cohort = projection_cohorts.iloc[start_month_idx][icf_proj_col]
                try:
                     landing_month_period = start_month + lag_in_months 
                     if landing_month_period in projection_results.index:
                         projection_results.loc[landing_month_period, 'Projected_ICF_Landed'] += projected_icfs_for_cohort
                except (ValueError, TypeError): pass 
            projection_results['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'].fillna(0).astype(int)
            projection_cohorts['Projected_CPICF_Cohort'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col].replace(0, np.nan)).round(2)
            display_df = pd.DataFrame(index=future_months)
            display_df['Forecasted_Ad_Spend'] = projection_cohorts['Forecasted_Ad_Spend']
            display_df['Forecasted_Qual_Referrals'] = projection_cohorts['Forecasted_PSQ']
            display_df['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'] 
            cpicf_cohort_series = projection_cohorts['Projected_CPICF_Cohort']
            cpicf_shifted_index = cpicf_cohort_series.index + lag_in_months 
            valid_shifted_indices = cpicf_shifted_index[cpicf_shifted_index.isin(display_df.index)]
            if not valid_shifted_indices.empty and not cpicf_cohort_series.loc[valid_shifted_indices - lag_in_months].empty:
                cpicf_shifted_series = cpicf_cohort_series.loc[valid_shifted_indices - lag_in_months].copy()
                cpicf_shifted_series.index = valid_shifted_indices 
                display_df['Projected_CPICF_Cohort_Source'] = cpicf_shifted_series 
            else: display_df['Projected_CPICF_Cohort_Source'] = np.nan # If no valid mapping
            
            st.write("DEBUG: Exiting calculate_projections successfully.") # DEBUG
            return display_df
        else: st.error("DEBUG: Could not calculate projected ICF column in cohorts."); return pd.DataFrame()          
    except Exception as e: 
        st.error(f"Projection calc error: {e}"); st.exception(e)
        return pd.DataFrame()


# --- Streamlit UI ---
# (Sidebar remains the same as Turn 64)
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"], key="referral_uploader")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv"], key="funnel_uploader") 
    st.divider()
    with st.expander("Historical Ad Spend"):
        st.info("Enter **historical** spend for past months found in data.")
        ad_spend_input_dict_manual = {}
        spend_month_str_1 = st.text_input("Month 1 (YYYY-MM)", "2025-02", key="spend_m1_str")
        spend_val_1 = st.number_input(f"Spend {spend_month_str_1}", value=45000.0, step=1000.0, format="%.2f", key="spend_v1")
        spend_month_str_2 = st.text_input("Month 2 (YYYY-MM)", "2025-03", key="spend_m2_str")
        spend_val_2 = st.number_input(f"Spend {spend_month_str_2}", value=60000.0, step=1000.0, format="%.2f", key="spend_v2")
        ad_spend_input_dict = {}
        try: ad_spend_input_dict[pd.Period(spend_month_str_1, freq='M')] = spend_val_1
        except Exception: st.warning(f"Invalid format for Month 1")
        try: ad_spend_input_dict[pd.Period(spend_month_str_2, freq='M')] = spend_val_2
        except Exception: st.warning(f"Invalid format for Month 2")
        st.caption("Ad Spend input method needs improvement.")
    st.divider()
    with st.expander("Site Scoring Weights"):
        weights_input = {} 
        weights_input["Qual -> ICF %"] = st.slider("Qual -> ICF %", 0, 100, 20, key='w_qicf') 
        weights_input["Avg TTC (Days)"] = st.slider("Avg Time to Contact", 0, 100, 25, key='w_ttc') 
        weights_input["Avg Funnel Movement Steps"] = st.slider("Avg Funnel Movement Steps", 0, 100, 5, key='w_fms') 
        weights_input["Site Screen Fail %"] = st.slider("Site Screen Fail %", 0, 100, 5, key='w_sfr') 
        weights_input["StS -> Appt %"] = st.slider("StS -> Appt Sched %", 0, 100, 30, key='w_sa') 
        weights_input["Appt -> ICF %"] = st.slider("Appt Sched -> ICF %", 0, 100, 15, key='w_ai') 
        total_weight_input = sum(abs(w) for w in weights_input.values()) 
        if total_weight_input > 0: weights_normalized = {k: v / total_weight_input for k, v in weights_input.items()}
        else: weights_normalized = {k: 0 for k in weights_input} 
        st.caption(f"Weights normalized. Lower is better for TTC & Screen Fail %.")
    st.divider()
    with st.expander("Projection Assumptions", expanded=True): 
        proj_horizon = st.number_input("Projection Horizon (Months)", min_value=1, max_value=36, value=12, step=1, key='proj_horizon')
        proj_cpqr = st.number_input("Assumed CPQR ($)", min_value=1.0, value=120.0, step=10.0, format="%.2f", key='proj_cpqr')
        st.write("Future Monthly Ad Spend:")
        current_date = pd.Timestamp.now(); proj_start_month_ui = pd.Period(current_date, freq='M') + 1 
        if proj_horizon <= 0: proj_horizon = 1 
        future_months_ui = pd.period_range(start=proj_start_month_ui, periods=proj_horizon, freq='M')
        spend_df_for_editor = pd.DataFrame({'Month': future_months_ui.strftime('%Y-%m'), 'Planned_Spend': [20000.0] * proj_horizon }) 
        st.caption("Edit planned spend per month:")
        edited_spend_df = st.data_editor(spend_df_for_editor, key='proj_spend_editor', use_container_width=True, num_rows="fixed")
        proj_spend_dict = {}
        if 'Month' in edited_spend_df.columns and 'Planned_Spend' in edited_spend_df.columns:
             for index, row in edited_spend_df.iterrows():
                 try: proj_spend_dict[pd.Period(row['Month'], freq='M')] = row['Planned_Spend']
                 except Exception as e: pass 
        else: st.error("Edited spend table columns missing.")
        st.write("Conversion Rate Assumption:")
        rate_assumption_method = st.radio( "Use Rates Based On:", ('Manual Input Below', 'Rolling Historical Average'), key='rate_method', horizontal=True )
        proj_conv_rates_input = {} 
        use_rolling_flag = False; rolling_window_months = 0
        if rate_assumption_method == 'Rolling Historical Average':
            use_rolling_flag = True
            rolling_window_months = st.selectbox("Select Rolling Window (Months):", [1, 3, 6], index=1, key='rolling_window') 
            st.caption(f"NOTE: Manual rates below will be ignored. Using {rolling_window_months}-month rolling average.")
        else: st.caption("Using manually input rates below.")
        cols_rate = st.columns(2)
        with cols_rate[0]:
             proj_conv_rates_input["Passed Online Form -> Pre-Screening Activities"] = st.slider("Qual -> PreScreen %", 0.0, 100.0, 100.0, step=0.1, format="%.1f%%", key='cr_qps') / 100.0
             proj_conv_rates_input["Pre-Screening Activities -> Sent To Site"] = st.slider("PreScreen -> StS %", 0.0, 100.0, 17.0, step=0.1, format="%.1f%%", key='cr_pssts') / 100.0
        with cols_rate[1]:
             proj_conv_rates_input["Sent To Site -> Appointment Scheduled"] = st.slider("StS -> Appt %", 0.0, 100.0, 33.0, step=0.1, format="%.1f%%", key='cr_sa') / 100.0
             proj_conv_rates_input["Appointment Scheduled -> Signed ICF"] = st.slider("Appt -> ICF %", 0.0, 100.0, 35.0, step=0.1, format="%.1f%%", key='cr_ai') / 100.0


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
                  referrals_raw_df = pd.read_csv(stringio, sep=',', header=0, on_bad_lines='warn', low_memory=False) 
                  st.success("Referral Data Loaded (assuming CSV with header).")
                  referral_data_processed = preprocess_referral_data(referrals_raw_df, funnel_definition, ordered_stages, ts_col_map)
             except Exception as read_err: st.error(f"Error reading referral file: {read_err}")
        except Exception as e: st.error(f"Error loading data: {e}")

# --- Display Sections ---
if referral_data_processed is not None and not referral_data_processed.empty:
    st.markdown("---")
    st.success("Data loaded and preprocessed.") 
    tab1, tab2, tab3 = st.tabs(["📅 Monthly ProForma", "🏆 Site Performance", "📈 Projections"])
    with tab1:
        st.header("Monthly ProForma (Historical Cohorts)")
        proforma_df = calculate_proforma_metrics(referral_data_processed, ordered_stages, ts_col_map, ad_spend_input_dict) 
        if not proforma_df.empty:
            proforma_display = proforma_df.transpose(); proforma_display.columns = [str(col) for col in proforma_display.columns] 
            format_dict = {}; 
            for idx in proforma_display.index:
                 if 'Cost' in idx or 'Spend' in idx: format_dict[idx] = "${:,.2f}"
                 elif '%' in idx: format_dict[idx] = "{:.1%}"
                 elif 'Total' in idx or 'Qualified' in idx or 'Reached' in idx: format_dict[idx] = "{:,.0f}"
            st.dataframe(proforma_display.style.format(format_dict, na_rep='-'))
            try:
                 csv = proforma_df.reset_index().to_csv(index=False).encode('utf-8')
                 st.download_button(label="Download ProForma Data", data=csv, file_name='monthly_proforma.csv', mime='text/csv', key='dl_proforma')
            except Exception as e: st.warning(f"Download button error: {e}")
        else: st.warning("Could not generate ProForma table.")
    with tab2:
        st.header("Site Performance Ranking")
        site_metrics_calculated = calculate_site_metrics(referral_data_processed, ordered_stages, ts_col_map) 
        if not site_metrics_calculated.empty:
            ranked_sites_df = score_sites(site_metrics_calculated, weights_normalized) 
            st.subheader("Site Ranking")
            display_cols = ['Site', 'Score', 'Grade', 'Total Qualified', 'Reached StS', 'Reached Appt', 'Reached ICF', 'Qual -> ICF %', 'Avg TTC (Days)', 'Avg Funnel Movement Steps', 'StS -> Appt %', 'Appt -> ICF %', 'Site Screen Fail %']
            display_cols = [col for col in ranked_sites_df.columns if col in display_cols] 
            final_ranked_display = ranked_sites_df[display_cols].copy()
            final_ranked_display['Score'] = final_ranked_display['Score'].round(1)
            percent_cols=[c for c in final_ranked_display if '%' in c]; lag_cols=[c for c in final_ranked_display if 'TTC' in c]; step_cols=[c for c in final_ranked_display if 'Steps' in c]; count_cols=['Total Qualified', 'Reached StS', 'Reached Appt', 'Reached ICF'] 
            for col in percent_cols: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
            for col in lag_cols: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
            for col in step_cols: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
            for col in count_cols: 
                if col in final_ranked_display.columns: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x==x else '-') 
            st.dataframe(final_ranked_display.style.format(na_rep='-'))
            try:
                 csv_sites = final_ranked_display.to_csv(index=False).encode('utf-8')
                 st.download_button(label="Download Site Ranking", data=csv_sites, file_name='site_ranking.csv', mime='text/csv', key='dl_sites')
            except Exception as e: st.warning(f"Download button error: {e}")
        else: st.warning("Could not calculate site metrics.")
    with tab3:
        st.header("Projections")
        st.write("Forecasts future performance based on assumptions set in sidebar.")
        projection_inputs = {
            'horizon': proj_horizon, 'spend_dict': proj_spend_dict, 
            'cpqr': proj_cpqr, 'conv_rates': proj_conv_rates_input,
            'use_rolling_rates': use_rolling_flag, 'rolling_window': rolling_window_months
        }
        projection_results_df = calculate_projections(referral_data_processed, ordered_stages, ts_col_map, projection_inputs)
        if projection_results_df is not None and not projection_results_df.empty and isinstance(projection_results_df, pd.DataFrame): 
            st.subheader("Projected Monthly ICFs & Cohort CPICF")
            display_cols_proj = ['Forecasted_Ad_Spend', 'Forecasted_Qual_Referrals', 'Projected_ICF_Landed', 'Projected_CPICF_Cohort_Source']
            results_display = projection_results_df[[col for col in display_cols_proj if col in projection_results_df.columns]].copy() 
            results_display.index = results_display.index.strftime('%Y-%m') 
            if 'Forecasted_Ad_Spend' in results_display: results_display['Forecasted_Ad_Spend'] = results_display['Forecasted_Ad_Spend'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
            if 'Projected_CPICF_Cohort_Source' in results_display: results_display['Projected_CPICF_Cohort_Source'] = results_display['Projected_CPICF_Cohort_Source'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
            if 'Forecasted_Qual_Referrals' in results_display: results_display['Forecasted_Qual_Referrals'] = results_display['Forecasted_Qual_Referrals'].fillna(0).astype(int).map('{:,}'.format) 
            if 'Projected_ICF_Landed' in results_display: results_display['Projected_ICF_Landed'] = results_display['Projected_ICF_Landed'].fillna(0).astype(int).map('{:,}'.format) 
            st.dataframe(results_display.style.format(na_rep='-'))
            if 'Projected_ICF_Landed' in projection_results_df.columns:
                 st.subheader("Projected ICFs Landed Over Time")
                 chart_data = projection_results_df[['Projected_ICF_Landed']].copy()
                 if isinstance(chart_data.index, pd.PeriodIndex): chart_data.index = chart_data.index.to_timestamp() 
                 chart_data['Projected_ICF_Landed'] = pd.to_numeric(chart_data['Projected_ICF_Landed'], errors='coerce').fillna(0)
                 st.line_chart(chart_data)
            try:
                 csv_proj = results_display.reset_index().to_csv(index=False).encode('utf-8')
                 st.download_button(label="Download Projection Data", data=csv_proj, file_name='projection.csv', mime='text/csv', key='dl_proj')
            except Exception as e: st.warning(f"Download button error: {e}")
        elif isinstance(projection_results_df, str): st.warning(projection_results_df) 
        else: st.warning("Could not calculate projections.")
elif not uploaded_referral_file or not uploaded_funnel_def_file:
    st.info("👋 Welcome! Please upload both the Referral Data (CSV) and Funnel Definition (TSV) files using the sidebar to begin.")