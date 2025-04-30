import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set custom theme colors based on 1nHealth brand guidelines
st.set_page_config(page_title="Recruitment Forecast Dashboard", layout="wide")
st.markdown("""
    <style>
        :root {
            --primary-color: #53CA97;
            --secondary-color: #7991C6;
            --text-color: #1B2222;
            --background-color: #F8F8F8;
        }
        html, body, [class*="css"]  {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: "Manrope", sans-serif;
        }
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
        }
        .stSlider>div>div>div[role=slider] {
            background-color: var(--secondary-color);
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            border: 1px solid var(--primary-color);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Clinical Trial Recruitment Forecasting Tool")

st.markdown("""
Upload your referral data and site list to begin tracking:
- Study progress toward ICF goals
- Weekly/monthly ICF forecasts
- Top-of-funnel trends and risks
- Site and source-level performance
""")

# --- Upload Section ---
referral_file = st.file_uploader("📥 Upload Referral Data (CSV)", type="csv")
site_file = st.file_uploader("📥 Upload Site Master List (CSV)", type="csv")

if referral_file and site_file:
    try:
        referrals = pd.read_csv(referral_file)
        sites = pd.read_csv(site_file)

        if 'Lead Stage History' not in referrals.columns or 'Referral Date' not in referrals.columns:
            st.error("Referral data must include 'Lead Stage History' and 'Referral Date' columns.")
        else:
            referrals['Signed ICF'] = referrals['Lead Stage History'].str.contains("Signed ICF", case=False, na=False).astype(int)
            referrals['Referral Date'] = pd.to_datetime(referrals['Referral Date'], errors='coerce')
            today = datetime.today().date()

            # --- ICF Goal Inputs ---
            st.subheader("🎯 Set Campaign Goal")
            target_icfs = st.number_input("Total ICF Goal", value=150)
            current_icfs = referrals['Signed ICF'].sum()
            icfs_remaining = target_icfs - current_icfs

            st.write(f"**Current Signed ICFs:** {current_icfs}")
            st.write(f"**Remaining to goal:** {icfs_remaining if icfs_remaining > 0 else 0}")

            # --- Referral to ICF Forecast ---
            st.subheader("🔮 Forecasting")
            conversion_rate = st.slider("Estimated Referral → ICF Conversion Rate", 0.005, 0.10, 0.04)
            weekly_referrals = st.slider("Average Weekly Referrals", 10, 1000, 260)
            avg_lag_days = st.slider("Avg Lag (Referral → ICF in days)", 30, 120, 63)

            needed_referrals = int(icfs_remaining / conversion_rate)
            weeks_to_fill = needed_referrals / weekly_referrals
            days_to_finish = int(weeks_to_fill * 7 + avg_lag_days)
            projected_end = datetime.today() + timedelta(days=days_to_finish)

            st.write(f"Estimated referrals needed: **{needed_referrals}**")
            st.write(f"Estimated weeks to fill funnel: **{weeks_to_fill:.1f} weeks**")
            st.write(f"📆 **Projected Campaign Completion:** ~{projected_end.date()}")

            # --- ICF Forecast by Week ---
            st.subheader("📆 Projected Weekly ICFs")
            future_weeks = pd.date_range(start=today + timedelta(days=avg_lag_days), periods=10, freq='W')
            projected_icfs = [round(weekly_referrals * conversion_rate, 1)] * len(future_weeks)
            forecast_df = pd.DataFrame({'Week': future_weeks, 'Projected ICFs': projected_icfs})
            st.line_chart(forecast_df.set_index('Week'))

            # --- Top-of-Funnel Trend Alerts ---
            st.subheader("📉 Funnel Trend Alerts")
            if 'Pre-screening Date' in referrals.columns and 'Appointment Date' in referrals.columns:
                referrals['Week'] = pd.to_datetime(referrals['Referral Date']).dt.to_period('W').dt.start_time
                weekly = referrals.groupby('Week').agg(
                    total=('Referral Date', 'count'),
                    prescreens=('Pre-screening Date', lambda x: x.notna().sum()),
                    appointments=('Appointment Date', lambda x: x.notna().sum())
                ).reset_index()
                weekly['ref_to_prescreen'] = weekly['prescreens'] / weekly['total']
                weekly['ref_to_appt'] = weekly['appointments'] / weekly['total']

                st.line_chart(weekly.set_index('Week')[['ref_to_prescreen', 'ref_to_appt']])

                if len(weekly) >= 2:
                    recent = weekly.iloc[-1]
                    prior = weekly.iloc[-2]
                    prescreen_drop = recent['ref_to_prescreen'] < prior['ref_to_prescreen'] * 0.85
                    appt_drop = recent['ref_to_appt'] < prior['ref_to_appt'] * 0.85

                    if prescreen_drop:
                        st.error("🚨 Drop in referral→pre-screen rate this week!")
                    if appt_drop:
                        st.error("🚨 Drop in referral→appointment rate this week!")

            # --- Site Performance Snapshot ---
            st.subheader("🏥 Site-Level Performance")
            if 'Site Number' in referrals.columns:
                site_stats = referrals.groupby('Site Number').agg(
                    Referrals=('Referral Date', 'count'),
                    ICFs=('Signed ICF', 'sum')
                ).reset_index()
                st.dataframe(site_stats.sort_values(by='ICFs', ascending=False))

            # --- Source-Level Performance ---
            st.subheader("📣 Ad Source Impact")
            if 'Ad Source' in referrals.columns:
                source_stats = referrals.groupby('Ad Source').agg(
                    Referrals=('Referral Date', 'count'),
                    ICFs=('Signed ICF', 'sum')
                ).reset_index()
                source_stats['Conversion Rate'] = source_stats['ICFs'] / source_stats['Referrals']
                st.dataframe(source_stats.sort_values(by='ICFs', ascending=False))

    except Exception as e:
        st.error(f"An error occurred while processing your files: {e}")
else:
    st.warning("⬆️ Please upload both the referral and site list to begin analysis.")
