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
        return None, None, None # Return three Nones
    try:
        # Read directly from the uploaded file object
        bytes_data = uploaded_file.getvalue()
        # Detect encoding or assume utf-8, allow fallback
        try:
            stringio = io.StringIO(bytes_data.decode("utf-8"))
        except UnicodeDecodeError:
             st.warning("UTF-8 decoding failed for Funnel Definition file, trying latin-1.")
             stringio = io.StringIO(bytes_data.decode("latin-1")) # Fallback encoding
             
        df_funnel_def = pd.read_csv(stringio, sep='\t', header=None) # Explicitly tab-separated
        
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
            # Create clean column name for timestamps
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            ts_col_map[stage_name] = clean_ts_name
            
        if not parsed_ordered_stages: # Basic validation
             st.error("Could not parse any stages from the Funnel Definition file. Check format (should be tab-separated).")
             return None, None, None

        # st.success("Funnel Definition Parsed Successfully.") # Keep UI less verbose during run
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
        
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}")
        st.error("Please ensure it's a tab-separated file with Stages in the first row and Statuses below.")
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    # (Function to parse various date formats, stripping timezone)
    if pd.isna(dt_str): return pd.NaT # Return NaT for consistency
    dt_str_cleaned = str(dt_str).strip()
    tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned)
    # Use pd.to_datetime as a flexible parser, coercing errors to NaT
    parsed_dt = pd.to_datetime(dt_str_no_tz, errors='coerce') 
    # Return pandas Timestamp or NaT directly
    return parsed_dt


def parse_history_string(history_str):
    # (Function to parse newline-separated history strings)
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    # More robust pattern to handle variations
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    
    raw_lines = str(history_str).strip().split('\n')
    parsed_events = []
    malformed_lines = 0
    for line in raw_lines:
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups(); name = name.strip()
            dt_obj = parse_datetime_with_timezone(dt_str.strip()) # Returns pd.Timestamp or NaT
            # Only add if date is valid
            if name and pd.notna(dt_obj): 
                # Convert to python datetime before adding to list if needed downstream, 
                # but keeping as Timestamp might be fine for sorting/comparison
                try:
                    py_dt = dt_obj.to_pydatetime() 
                    parsed_events.append((name, py_dt)) 
                except AttributeError: # Handle potential NaT conversion if it occurs
                     pass 
            # Optionally log malformed dates here if needed for debugging
            # elif name: print(f"Debug date parse fail: {dt_str}") 
        else:
             malformed_lines += 1
    
    # Optional warning about lines that didn't match the pattern
    # if malformed_lines > 0: print(f"Debug: {malformed_lines} history lines didn't match expected pattern.")
            
    # Sort by datetime object, handling potential NaT values
    try:
        # Sort using datetime.min for NaT comparisons
        parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e:
        st.warning(f"Could not sort all history events due to data type issue: {e}. Some events might be out of order.")
    
    return parsed_events


def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    # (Function to determine first timestamp for each stage)
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps) # Handle case where funnel_def is None
    
    for stage, statuses in funnel_def.items():
        for status in statuses: status_to_stage_map[status] = stage
        
    all_events = []
    # Use .get() on row to handle potentially missing parsed history columns gracefully
    stage_hist = row.get(parsed_stage_history_col, [])
    status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist])
    
    # Sort all events chronologically
    try:
         # Ensure comparison works with datetime or NaT
         all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e:
         # Log or handle error if sorting fails due to mixed types not handled above
         pass 

    for event_name, event_dt in all_events:
        # Ensure event_dt is a comparable datetime object or NaT
        if pd.isna(event_dt): continue 
        
        event_stage = None
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage) 
            if ts_col_name and pd.isna(timestamps[ts_col_name]): 
                timestamps[ts_col_name] = event_dt # Assign the first valid timestamp
                
    # Convert dictionary to pandas Series before returning
    # Ensure output series has datetime64[ns] dtype where possible, or object if NaT is present
    return pd.Series(timestamps, dtype='datetime64[ns]')


@st.cache_data # Cache the processed referral data
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    """Loads, cleans, parses history, calculates timestamps."""
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None:
        st.warning("Preprocessing cannot proceed: Missing key inputs (dataframe or funnel definition).")
        return None
        
    df = _df_raw.copy() # Work on a copy

    # 1. Check for Submitted On Date
    submitted_on_col = None
    if "Submitted On" in df.columns:
        submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns:
         st.info("Using 'Referral Date' column as 'Submitted On'.")
         df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
         submitted_on_col = "Submitted On"
    else:
         # Check again AFTER potential rename
         if "Submitted On" not in df.columns: 
              st.error("Missing required 'Submitted On' or 'Referral Date' column in Referral Data.")
              # st.write("Columns found:", df.columns.tolist()) # Debugging already added in main block
              return None
         else: # Renaming worked
              submitted_on_col = "Submitted On"
             
    # Convert to datetime, coercing errors. Store in a new column first.
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True) # Remove rows where submission date couldn't be parsed
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        st.warning(f"Dropped {rows_dropped} rows due to unparseable '{submitted_on_col}' date.")
        
    if df.empty:
         st.error("No valid referral data remaining after date parsing.")
         return None
         
    # Use _DT version for dates
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    
    # 2. Parse History Columns
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']
    parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            # Ensure Apply works even if column has mixed types initially
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string) 
            parsed_cols[col_name] = parsed_col_name 
        else:
             st.warning(f"History column '{col_name}' not found.")

    # 3. Calculate Timestamps
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History')
    parsed_status_hist_col = parsed_cols.get('Lead Status History')
    
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        # Check if relevant columns just don't exist
        if 'Lead Stage History' not in df.columns and 'Lead Status History' not in df.columns:
             st.error("Neither 'Lead Stage History' nor 'Lead Status History' columns found in data.")
        else: # Columns exist but weren't parsed (shouldn't happen with current logic)
             st.error("History columns found but failed to parse for timestamps.")
        return None # Cannot proceed without history

    # Apply timestamp function
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
    
    # Concatenate new timestamp columns
    df = pd.concat([df, timestamp_cols_df], axis=1)
    
    # Ensure TS columns are datetime type
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns:
             # Important: Ensure it's datetime AFTER concatenation
             df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce') 
             
    st.success("Referral Data Preprocessed Successfully.")
    return df

# --- Calculation Functions for App Sections ---

# @st.cache_data # Cache calculation results if inputs don't change
def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    """ Calculates historical monthly cohort metrics."""
    # Add input validation
    if _processed_df is None or _processed_df.empty:
        st.warning("Cannot calculate ProForma: No valid processed referral data.")
        return pd.DataFrame()
    if not isinstance(monthly_ad_spend_input, dict):
         st.warning("Cannot calculate ProForma: Invalid Ad Spend input (must be a dictionary).")
         return pd.DataFrame()
    if "Submission_Month" not in _processed_df.columns:
        st.error("Cannot calculate ProForma: 'Submission_Month' column missing.")
        return pd.DataFrame()
        
    processed_df = _processed_df.copy() # Work on copy

    # --- Cohort Summary Calculation --- (Based on logic from Turn 40)
    try:
        cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals")
        cohort_summary = cohort_summary.set_index("Submission_Month")
        
        # Map Ad Spend - Ensure keys match PeriodIndex
        cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0) 
            
        reached_stage_cols_map = {}
        for stage_name in ordered_stages:
            ts_col = ts_col_map.get(stage_name)
            if ts_col and ts_col in processed_df.columns:
                reached_col_cleaned = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                reached_stage_cols_map[stage_name] = reached_col_cleaned
                # Ensure timestamp column is datetime before using dropna
                if pd.api.types.is_datetime64_any_dtype(processed_df[ts_col]):
                     reached_stage_count = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
                     cohort_summary[reached_col_cleaned] = reached_stage_count
                else:
                     st.warning(f"Timestamp column {ts_col} is not datetime type. Cannot count reached stage '{stage_name}'.")
                     cohort_summary[reached_col_cleaned] = 0 # Assign 0 if TS column invalid
            
        cohort_summary = cohort_summary.fillna(0)
        # Convert count columns safely to int
        for col in cohort_summary.columns:
            if col != "Ad Spend": 
                try:
                    cohort_summary[col] = cohort_summary[col].astype(int)
                except ValueError: 
                     cohort_summary[col] = pd.to_numeric(cohort_summary[col], errors='coerce').fillna(0).astype(int)
                     st.warning(f"Had trouble converting column {col} to integer, check data.")

        cohort_summary["Ad Spend"] = cohort_summary["Ad Spend"].astype(float)
        
        # Determine base count column
        pof_reached_col = reached_stage_cols_map.get("Passed Online Form")
        if pof_reached_col and pof_reached_col in cohort_summary.columns:
            cohort_summary.rename(columns={pof_reached_col: "Pre-Screener Qualified"}, inplace=True, errors='ignore') # Ignore error if already renamed
            base_count_col = "Pre-Screener Qualified"
        elif "Total Qualified Referrals" in cohort_summary.columns :
            cohort_summary.rename(columns={"Total Qualified Referrals": "Pre-Screener Qualified"}, inplace=True, errors='ignore')
            base_count_col = "Pre-Screener Qualified"
        else:
            base_count_col = None 
            st.error("Base count column for ProForma not found.")
            return pd.DataFrame()

        # --- Build final metrics table ---
        proforma_metrics = pd.DataFrame(index=cohort_summary.index)
            
        # Check if base_count_col exists before proceeding
        if base_count_col and base_count_col in cohort_summary.columns: 
            proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
            proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col]
            proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col].replace(0, np.nan)).round(2) # Use np.nan for division

            # Safely add columns if they exist in cohort_summary
            for stage_orig, reached_col in reached_stage_cols_map.items():
                 metric_name = "Pre-Screener Qualified" if stage_orig == "Passed Online Form" else f"Total {stage_orig}"
                 if reached_col in cohort_summary.columns and metric_name not in proforma_metrics.columns:
                     proforma_metrics[metric_name] = cohort_summary[reached_col]

            # --- Conversion Rates ---
            # Ensure required columns exist before calculating rates
            sts_col = reached_stage_cols_map.get("Sent To Site")
            appt_col = reached_stage_cols_map.get("Appointment Scheduled")
            icf_col = reached_stage_cols_map.get("Signed ICF")

            if sts_col in cohort_summary.columns:
                proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col] / cohort_summary[base_count_col].replace(0, np.nan))
            if sts_col in cohort_summary.columns and appt_col in cohort_summary.columns:
                proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col] / cohort_summary[sts_col].replace(0, np.nan))
            if appt_col in cohort_summary.columns and icf_col in cohort_summary.columns:
                proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col] / cohort_summary[appt_col].replace(0, np.nan))
            if icf_col in cohort_summary.columns:
                proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col] / cohort_summary[base_count_col].replace(0, np.nan))
                proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col].replace(0, np.nan)).round(2)
        else:
             st.error("Base count column missing, cannot calculate detailed ProForma metrics.")
             return pd.DataFrame()

        return proforma_metrics

    except Exception as e:
         st.error(f"Error during ProForma calculation: {e}")
         return pd.DataFrame()


# Placeholder - Logic from Turn 44 will go here
# @st.cache_data 
def calculate_site_metrics(_processed_df, ordered_stages, ts_col_map):
     # Add detailed implementation later
     st.write("Site metrics calculation function to be implemented here.")
     if _processed_df is None or _processed_df.empty or 'Site' not in _processed_df.columns:
         return pd.DataFrame({'Site': [], 'Info': []}) # Return empty placeholder if data invalid
     # Example structure:
     # site_metrics = _processed_df.groupby('Site').agg(...) 
     # return site_metrics
     # For now, return placeholder matching expected columns for scoring
     sites = _processed_df['Site'].unique()
     return pd.DataFrame({
         'Site': sites, 
         'Qual -> ICF %': [0.0] * len(sites),
         'Avg TTC (Days)': [99] * len(sites),
         'Avg Funnel Movement Steps': [0] * len(sites),
         'Site Screen Fail %': [0.0] * len(sites),
         'StS -> Appt %': [0.0] * len(sites),
         'Appt -> ICF %': [0.0] * len(sites),
         'Total Qualified': [0] * len(sites) # Add base columns needed for display/context
         }).set_index('Site') # Set index for consistency if needed by scoring


# Placeholder - Logic from Turn 46 will go here
# @st.cache_data 
def score_sites(_site_metrics_df, weights):
     st.write("Site scoring logic to be implemented here.")
     # Logic from Turn 44 goes here...
     # Add score and grade columns
     site_metrics_df = _site_metrics_df.copy() # Work on copy
     if not site_metrics_df.empty:
          # Add placeholder columns if they don't exist from calculate_site_metrics placeholder
          if 'Score' not in site_metrics_df.columns: site_metrics_df['Score'] = 0 
          if 'Grade' not in site_metrics_df.columns: site_metrics_df['Grade'] = 'N/A'
          return site_metrics_df.reset_index() # Ensure 'Site' is a column
     return pd.DataFrame({'Site': [],'Score': [], 'Grade': [], 'Info':[]}) # Return empty placeholder


# Placeholder - Logic from Turn 40/42 will go here
# @st.cache_data 
# --- CORRECTED Function Definition ---
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs): # Use dict for inputs
     # Removed the invalid comment -> /* add projection inputs */ 
     st.write("Projection calculation logic to be implemented here.")
     # Logic from Turn 40/42 goes here...
     # Example: Use projection_inputs['spend'], projection_inputs['cpqr'] etc.
     # Need to define the structure of projection_inputs clearly
     return pd.DataFrame({'Month': ['Apr-25', 'May-25'],'Projected ICF': [0, 0], 'Info': ['Calculation Pending','Calculation Pending']}) # Placeholder


# --- Streamlit UI ---

# Sidebar for Inputs
with st.sidebar:
    st.header("⚙️ Setup")
    
    # Use unique keys for file uploaders
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV/TSV)", type=["csv", "tsv"], key="referral_uploader")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv", "csv"], key="funnel_uploader") 

    st.divider()
    
    # --- Inputs for Calculations (Placeholders/Examples) ---
    st.subheader("Historical Ad Spend (Monthly)")
    st.info("Enter **historical** spend for past months found in data. (Input method needs improvement).")
    # Example - needs dynamic generation based on data months
    # Using fixed keys for demonstration - this needs to adapt to actual data months
    # A better approach uses st.data_editor or loops through months found in data
    ad_spend_input_dict_manual = {}
    # Example for manual input (replace with dynamic logic later)
    spend_month_str_1 = st.text_input("Spend Month 1 (YYYY-MM)", "2025-02", key="spend_m1_str")
    spend_val_1 = st.number_input(f"Spend for {spend_month_str_1}", value=45000.0, step=1000.0, format="%.2f", key="spend_v1")
    spend_month_str_2 = st.text_input("Spend Month 2 (YYYY-MM)", "2025-03", key="spend_m2_str")
    spend_val_2 = st.number_input(f"Spend for {spend_month_str_2}", value=60000.0, step=1000.0, format="%.2f", key="spend_v2")
    
    # Convert manual inputs to Period keys - needs error handling
    ad_spend_input_dict = {}
    try:
         m1_period = pd.Period(spend_month_str_1, freq='M')
         ad_spend_input_dict[m1_period] = spend_val_1
    except Exception: st.warning(f"Invalid format for Month 1: {spend_month_str_1}")
    try:
         m2_period = pd.Period(spend_month_str_2, freq='M')
         ad_spend_input_dict[m2_period] = spend_val_2
    except Exception: st.warning(f"Invalid format for Month 2: {spend_month_str_2}")
    
    st.caption("Note: Ad Spend input currently supports only 2 months.")


# --- Main App Logic & Display ---
referral_data_processed = None # Initialize
funnel_definition, ordered_stages, ts_col_map = None, None, None # Initialize

# Load and process data only if both files are uploaded
if uploaded_referral_file is not None and uploaded_funnel_def_file is not None:
    
    # Step 1: Parse Funnel Definition (Cached)
    funnel_definition, ordered_stages, ts_col_map = parse_funnel_definition(uploaded_funnel_def_file)

    # Step 2: Load and Preprocess Referral Data (Cached)
    if funnel_definition and ordered_stages and ts_col_map: 
        try:
             bytes_data = uploaded_referral_file.getvalue()
             try:
                 decoded_data = bytes_data.decode("utf-8")
             except UnicodeDecodeError:
                 st.warning("UTF-8 decoding failed for referral data, trying latin-1.")
                 decoded_data = bytes_data.decode("latin-1") 

             stringio = io.StringIO(decoded_data)
             
             # --- Using Modified pd.read_csv ---
             try:
                  # Explicitly use Tab separator and Header row 0
                  # Added low_memory=False which can sometimes help with parsing complex files
                  referrals_raw_df = pd.read_csv(stringio, sep='\t', header=0, on_bad_lines='warn', low_memory=False) 
                  
                  # --- ADDED DEBUGGING LINE ---
                  st.write("Columns found by pandas:", referrals_raw_df.columns.tolist()) 
                  # --- END DEBUGGING LINE ---
                  
                  st.success("Referral Data Loaded (assuming TSV with header).")
                  
                  # --- Run Preprocessing ---
                  referral_data_processed = preprocess_referral_data(
                      referrals_raw_df, 
                      funnel_definition, 
                      ordered_stages, 
                      ts_col_map
                  )

             except Exception as read_err:
                  st.error(f"Error reading referral file with specified settings (TSV, header=0): {read_err}")
                  st.error("Please ensure the file is tab-separated and the first row is a correct header. Check for inconsistencies mentioned previously.")
                 
        except Exception as e:
             st.error(f"An unexpected error occurred during referral data loading: {e}")

# --- Display Sections ---
# Display content only if data has been successfully processed
if referral_data_processed is not None and not referral_data_processed.empty:
    st.markdown("---")
    st.success("Data loaded and preprocessed. Displaying analysis sections.") # Confirmation message
    
    # Use tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📅 Monthly ProForma", "🏆 Site Performance", "📈 Projections"])

    with tab1:
        st.header("Monthly ProForma (Historical Cohorts)")
        st.write("Shows historical performance based on the month referrals were submitted.")
             
        # Calculate ProForma Metrics
        proforma_df = calculate_proforma_metrics(
             referral_data_processed, 
             ordered_stages, 
             ts_col_map, 
             ad_spend_input_dict # Pass the ad spend dictionary from sidebar
         )

        if not proforma_df.empty:
            # Transpose and Format for display
            proforma_display = proforma_df.transpose()
            # Convert PeriodIndex columns to string for display
            proforma_display.columns = [str(col) for col in proforma_display.columns] 
                 
            # Apply formatting (more robustly)
            format_dict = {}
            for idx in proforma_display.index:
                 if 'Cost' in idx or 'Spend' in idx: format_dict[idx] = "${:,.2f}"
                 elif '%' in idx: format_dict[idx] = "{:.1%}"
                 elif 'Total' in idx or 'Qualified' in idx or 'Reached' in idx: format_dict[idx] = "{:,.0f}" # Format counts
                 # Add other formats if needed

            st.dataframe(proforma_display.style.format(format_dict, na_rep='-'))
            
            # Add download button for the proforma table
            try:
                 # Use the untransposed data for a more standard CSV layout
                 csv = proforma_df.reset_index().to_csv(index=False).encode('utf-8')
                 st.download_button(
                      label="Download ProForma Data as CSV",
                      data=csv,
                      file_name='monthly_proforma.csv',
                      mime='text/csv',
                      key='download_proforma' # Add unique key
                  )
            except Exception as e:
                 st.warning(f"Could not generate download button: {e}")
                 
        else:
            st.warning("Could not generate ProForma table based on the processed data.")


    with tab2:
        st.header("Site Performance Ranking")
        st.write("Calculates metrics and scores for each site.")
        
        st.info("Site scoring calculation logic needs to be implemented in the 'calculate_site_metrics' and 'score_sites' functions.")
        
        # --- Call site metric calculation functions (when implemented) ---
        site_metrics_calc = calculate_site_metrics(referral_data_processed, ordered_stages, ts_col_map) 
        
        # Placeholder for weights input - In a real app, get this from sidebar widgets
        site_score_weights_example = {
             "Qual -> ICF %": 0.20, "Avg TTC (Days)": -0.25, "Avg Funnel Movement Steps": 0.05,
             "Site Screen Fail %": -0.05, "StS -> Appt %": 0.30, "Appt -> ICF %": 0.15
        } # Get weights from sidebar eventually
        
        ranked_sites = score_sites(site_metrics_calc, site_score_weights_example) # Pass calculated metrics
        
        st.dataframe(ranked_sites) # Display results (currently placeholders from functions)


    with tab3:
        st.header("Projections")
        st.write("Forecasts future performance based on assumptions.")
        
        st.info("Projection calculation logic needs to be implemented in the 'calculate_projections' function using inputs from the sidebar.")
        
        # --- Call projection function (when implemented) ---
        # Placeholder inputs - In real app, get from sidebar widgets
        projection_inputs_example = { 'horizon': 12, 'spend': {1: 20000, 2: 20000}, 'cpqr': 120, 'conv_rates':{...} } # Example structure
        # projected_icfs = calculate_projections(referral_data_processed, ordered_stages, ts_col_map, projection_inputs_example)
        # st.dataframe(projected_icfs)

        # Placeholder display
        st.dataframe(pd.DataFrame({
            'Month': ['Apr-25', 'May-25', 'Jun-25'],
            'Info': ['Calculation Pending', 'Calculation Pending','Calculation Pending']
            }))

# Display initial message if files are not uploaded
elif not uploaded_referral_file or not uploaded_funnel_def_file:
    st.info("👋 Welcome! Please upload both the Referral Data and Funnel Definition files using the sidebar to begin.")

# Add a footer or other info if desired
# st.markdown("---")
# st.caption("App v0.2")