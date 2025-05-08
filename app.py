# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import io

# --- Page Configuration ---
st.set_page_config(page_title="Recruitment Forecasting Tool", layout="wide")
st.title("📊 Recruitment Forecasting Tool")

# --- Helper Functions (Data Parsing & Timestamping) ---
# These functions encapsulate the logic we developed previously.

@st.cache_data # Cache the result of parsing the funnel definition
def parse_funnel_definition(uploaded_file):
    """Parses the wide-format Stage & Status Breakdown CSV."""
    if uploaded_file is None:
        return None, None
    try:
        # Read directly from the uploaded file object
        df_funnel_def = pd.read_csv(uploaded_file, sep='\t', header=None) # Assuming tab-separated based on paste
        
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
            ts_col_map[stage_name] = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}" 
            
        if not parsed_ordered_stages: # Basic validation
             st.error("Could not parse any stages from the Funnel Definition file. Check format.")
             return None, None, None

        st.success("Funnel Definition Parsed Successfully.")
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
        
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}")
        st.error("Please ensure it's a tab-separated file with Stages in the first row and Statuses below.")
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    # (Same function as before)
    if pd.isna(dt_str): return None
    dt_str_cleaned = str(dt_str).strip()
    tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned)
    parsed_dt = pd.to_datetime(dt_str_no_tz, errors='coerce').to_pydatetime() # Flexible parsing
    return parsed_dt

def parse_history_string(history_str):
    # (Same function as before)
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(?:[apAP][mM])?\s*[A-Za-z]{3,}(?:T)?)")
    raw_lines = str(history_str).strip().split('\n')
    parsed_events = []
    for line in raw_lines:
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups(); name = name.strip()
            dt_obj = parse_datetime_with_timezone(dt_str.strip())
            if name and dt_obj: parsed_events.append((name, dt_obj))
    parsed_events.sort(key=lambda x: x[1] if x[1] is not None else datetime.min)
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    # (Same function as before)
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps) # Handle case where funnel_def is None
    for stage, statuses in funnel_def.items():
        for status in statuses: status_to_stage_map[status] = stage
    all_events = []
    if parsed_stage_history_col in row and row[parsed_stage_history_col]:
        all_events.extend([(name, dt) for name, dt in row[parsed_stage_history_col]])
    if parsed_status_history_col in row and row[parsed_status_history_col]:
        all_events.extend([(name, dt) for name, dt in row[parsed_status_history_col]])
    all_events.sort(key=lambda x: x[1] if x[1] is not None else datetime.min)
    for event_name, event_dt in all_events:
        if event_dt is pd.NaT or event_dt is None: continue
        event_stage = None
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage) # Use .get for safety
            if ts_col_name and pd.isna(timestamps[ts_col_name]): 
                timestamps[ts_col_name] = event_dt
    return pd.Series(timestamps)

@st.cache_data # Cache the processed referral data
def preprocess_referral_data(df, funnel_def, ordered_stages, ts_col_map):
    """Parses history, calculates timestamps."""
    if df is None or funnel_def is None or ordered_stages is None or ts_col_map is None:
        return None
        
    # 1. Check for Submitted On Date
    if "Submitted On" not in df.columns:
        if "Referral Date" in df.columns:
             st.info("Using 'Referral Date' column as 'Submitted On'.")
             df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
        else:
             st.error("Missing required 'Submitted On' or 'Referral Date' column.")
             return None
             
    df["Submitted On"] = df["Submitted On"].apply(lambda x: parse_datetime_with_timezone(str(x)))
    df.dropna(subset=["Submitted On"], inplace=True) # Remove rows where submission date couldn't be parsed
    df["Submission_Month"] = df["Submitted On"].dt.to_period('M')
    
    # 2. Parse History Columns
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']
    parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].apply(parse_history_string)
            parsed_cols[col_name] = parsed_col_name # Store mapping for timestamping
        else:
             st.warning(f"History column '{col_name}' not found.")

    # 3. Calculate Timestamps
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History')
    parsed_status_hist_col = parsed_cols.get('Lead Status History')
    
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        st.error("No history columns found to parse for timestamps.")
        return None

    timestamp_cols_df = df.apply(
        lambda row: get_stage_timestamps(
            row, 
            parsed_stage_hist_col, 
            parsed_status_hist_col,
            funnel_def, 
            ordered_stages, 
            ts_col_map
        ), axis=1
    )
    
    # Drop old TS columns before adding new ones
    old_ts_cols = [col for col in df.columns if col.startswith('TS_')]
    df.drop(columns=old_ts_cols, inplace=True, errors='ignore')
    
    df = pd.concat([df, timestamp_cols_df], axis=1)
    
    # Ensure TS columns are datetime type
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns:
             df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
             
    st.success("Referral Data Preprocessed Successfully.")
    return df

# --- Placeholder Function for ProForma Calc ---
# (Logic from Turn 34/40 will go here)
def calculate_proforma_metrics(processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    """ Calculates historical monthly cohort metrics."""
    if processed_df is None or not isinstance(monthly_ad_spend_input, dict):
        st.warning("Cannot calculate ProForma: Missing data or invalid Ad Spend input.")
        return pd.DataFrame() # Return empty dataframe

    cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals")
    cohort_summary = cohort_summary.set_index("Submission_Month")
    
    # Use the Ad Spend input dictionary
    cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0)
        
    reached_stage_cols_map = {}
    for stage_name in ordered_stages:
        ts_col = ts_col_map.get(stage_name)
        if ts_col and ts_col in processed_df.columns:
            reached_col_cleaned = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            reached_stage_cols_map[stage_name] = reached_col_cleaned
            reached_stage_count = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
            cohort_summary[reached_col_cleaned] = reached_stage_count
        
    cohort_summary = cohort_summary.fillna(0)
    for col in cohort_summary.columns:
        if col != "Ad Spend": cohort_summary[col] = cohort_summary[col].astype(int)
    cohort_summary["Ad Spend"] = cohort_summary["Ad Spend"].astype(float)
        
    pof_reached_col = reached_stage_cols_map.get("Passed Online Form")
    if pof_reached_col and pof_reached_col in cohort_summary.columns:
        cohort_summary.rename(columns={pof_reached_col: "Pre-Screener Qualified"}, inplace=True)
        base_count_col = "Pre-Screener Qualified"
    elif "Total Qualified Referrals" in cohort_summary.columns :
        cohort_summary.rename(columns={"Total Qualified Referrals": "Pre-Screener Qualified"}, inplace=True)
        base_count_col = "Pre-Screener Qualified"
    else:
        base_count_col = None 

    proforma_metrics = pd.DataFrame(index=cohort_summary.index)
        
    if base_count_col: 
        proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
        proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col]
        proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col].replace(0, pd.NA)).round(2)

        sts_col_name = reached_stage_cols_map.get("Sent To Site")
        if sts_col_name in cohort_summary.columns: proforma_metrics["Total StS"] = cohort_summary[sts_col_name]
             
        appt_col_name = reached_stage_cols_map.get("Appointment Scheduled")
        if appt_col_name in cohort_summary.columns: proforma_metrics["Total Appt Scheduled"] = cohort_summary[appt_col_name]
             
        icf_col_name = reached_stage_cols_map.get("Signed ICF")
        if icf_col_name in cohort_summary.columns: proforma_metrics["Total ICF"] = cohort_summary[icf_col_name]

        enrolled_col_name = reached_stage_cols_map.get("Enrolled")
        if enrolled_col_name in cohort_summary.columns: proforma_metrics["Total Enrolled"] = cohort_summary[enrolled_col_name]
             
        sf_col_name = reached_stage_cols_map.get("Screen Failed")
        if sf_col_name in cohort_summary.columns: proforma_metrics["Total Screen Failed"] = cohort_summary[sf_col_name]

        lost_col_name = reached_stage_cols_map.get("Lost")
        if lost_col_name in cohort_summary.columns: proforma_metrics["Total Lost"] = cohort_summary[lost_col_name]

        # --- Conversion Rates ---
        if sts_col_name in cohort_summary.columns:
            proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col_name] / cohort_summary[base_count_col].replace(0, pd.NA))
        if sts_col_name in cohort_summary.columns and appt_col_name in cohort_summary.columns:
            proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col_name] / cohort_summary[sts_col_name].replace(0, pd.NA))
        if appt_col_name in cohort_summary.columns and icf_col_name in cohort_summary.columns:
            proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col_name] / cohort_summary[appt_col_name].replace(0, pd.NA))
        if icf_col_name in cohort_summary.columns:
            proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col_name] / cohort_summary[base_count_col].replace(0, pd.NA))
            proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col_name].replace(0, pd.NA)).round(2)

    return proforma_metrics

# --- Streamlit UI ---

# Sidebar for Inputs
with st.sidebar:
    st.header("⚙️ Inputs")
    
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV/TSV)", type=["csv", "tsv"])
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv", "csv"]) # Assume TSV based on paste

    # Placeholder for Ad Spend Input (more sophisticated input needed later)
    st.subheader("Ad Spend (Monthly)")
    spend_month_1 = st.number_input("Spend Month 1 (e.g., Apr 2025)", value=20000.0, step=1000.0, format="%.2f")
    spend_month_2 = st.number_input("Spend Month 2 (e.g., May 2025)", value=20000.0, step=1000.0, format="%.2f")
    # Add more months or a better input method like a table
    # For now, create a dictionary for the ProForma function using dummy future months if needed
    # This part needs improvement for actual use with historical data
    ad_spend_input_dict = {
         # We need actual historical months here from data - this is just placeholder logic
         pd.Period('2025-02', freq='M'): 45000.00, 
         pd.Period('2025-03', freq='M'): 60000.00
         # In reality, we'd get months from data or allow user mapping
    }

    st.info("Placeholder Ad Spend used for calculations currently. Actual input fields needed.")


# --- Main App Logic ---

# Load and process data only if both files are uploaded
if uploaded_referral_file is not None and uploaded_funnel_def_file is not None:
    
    # Parse Funnel Definition
    funnel_def, ordered_stages, ts_col_map = parse_funnel_definition(uploaded_funnel_def_file)

    # Load and Preprocess Referral Data
    try:
         # Use StringIO to handle the uploaded file object's content
         stringio = io.StringIO(uploaded_referral_file.getvalue().decode("utf-8"))
         referrals_raw_df = pd.read_csv(stringio, sep=None, engine='python') # Auto-detect separator
         st.success("Referral Data Loaded.")
         
         processed_referrals_df = preprocess_referral_data(
             referrals_raw_df.copy(), # Pass a copy to avoid modifying cached data
             funnel_def, 
             ordered_stages, 
             ts_col_map
         )

         if processed_referrals_df is not None:
             st.markdown("---")
             st.header("📅 Monthly ProForma")
             
             # Calculate ProForma Metrics
             proforma_df = calculate_proforma_metrics(
                 processed_referrals_df, 
                 ordered_stages, 
                 ts_col_map, 
                 ad_spend_input_dict # Pass the ad spend dictionary
             )

             if not proforma_df.empty:
                 # Transpose and Format for display
                 proforma_display = proforma_df.transpose()
                 proforma_display.columns = [str(col) for col in proforma_display.columns] # Format month columns
                 
                 # Apply formatting (similar to previous step)
                 st.dataframe(proforma_display.style.format(na_rep='-', 
                      formatter={col: "${:,.2f}" for col in proforma_display.index if 'Cost' in col or 'Spend' in col} |
                                {col: "{:,.0f}" for col in proforma_display.index if 'Total' in col or 'Qualified' in col} |
                                {col: "{:.1%}" for col in proforma_display.index if '%' in col} 
                 ))
             else:
                 st.warning("Could not generate ProForma table.")


             # Placeholder for other sections
             st.markdown("---")
             st.header("📈 Projections (Placeholder)")
             st.write("Projection logic will be added here.")

             st.markdown("---")
             st.header("🏆 Site Performance (Placeholder)")
             st.write("Site ranking logic will be added here.")
             
         else:
             st.warning("Referral data preprocessing failed. Cannot proceed.")
             
    except Exception as e:
         st.error(f"Error loading or processing referral data: {e}")
         st.error("Please ensure referral file is a valid CSV/TSV.")

else:
    st.info("Please upload both the Referral Data and Funnel Definition files using the sidebar.")