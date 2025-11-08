"""
Enhanced Interactive DST Horner Plot Analyst (Smart Auto MTR Detection)
Professional web application for Drill Stem Test analysis
V9.4 - Auto-Solve for Flow Capacity (kh) and Transmissibility
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64
import graphviz

# --- Page configuration ---
st.set_page_config(
    page_title="DST Horner Analyst (Smart-Fit)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4; /* Primary brand color */
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6; /* Light gray background */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.5rem;
    }
    .stMetric:last-of-type {
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    /* Style for the override expander */
    .st-expander {
        border: 1px solid #ddd !important;
        border-radius: 0.5rem !important;
    }
    /* Horizontal Radio Buttons */
    div[role="radiogroup"] {
        flex-direction: row !important;
        justify-content: space-evenly;
    }
</style>
""", unsafe_allow_html=True)

# --- "SMART" Auto MTR detection function (V9.1) ---
def find_best_mtr(df, min_points=3, min_r_squared_threshold=0.995):
    """
    Automatically find the *true* straight line segment (MTR) by searching
    all possible contiguous sub-segments of the data.

    V9.1 Logic:
    1. Loop through *every possible segment* (from index i to j)
       that has at least `min_points`.
    2. Check if the segment's R-squared is "very straight"
       (i.e., > `min_r_squared_threshold` set by the user).
    3. From all the segments that pass, find the one that is the
       *longest* (has the most points).
    4. If there's a tie in length, pick the one with the *highest* R-squared.

    This logic correctly ignores WBS and Boundary curves (low R²) and
    finds the MTR in the middle. It also prefers a 4-point line
    (R²=0.999) over a 3-point line (R²=1.000) because it is longer.
    """
    best_regression = None
    best_fit_df = None
    best_num_points = 0
    best_r_squared = 0.0

    n = len(df)

    # Loop through all possible start points
    for i in range(n - min_points + 1):
        # Loop through all possible end points
        for j in range(i + min_points, n + 1):

            fit_df_loop = df.iloc[i:j].copy()
            num_points_in_loop = len(fit_df_loop)

            try:
                regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)

                # Check for invalid regression results
                if not np.isfinite(regression.slope):
                    continue

                r_squared = regression.rvalue ** 2

                # --- NEW LOGIC ---
                if r_squared >= min_r_squared_threshold:
                    # This segment is "very straight"

                    if num_points_in_loop > best_num_points:
                        # This is the new *longest* straight line found so far
                        best_num_points = num_points_in_loop
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared
                    elif num_points_in_loop == best_num_points and r_squared > best_r_squared:
                        # If two segments have the same max length,
                        # pick the one that's *even straighter*
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared

            except Exception:
                # Ignore errors from linregress
                continue

    if best_regression is None:
        # Fallback: If no line was found (e.g., threshold too high or data too noisy)
        # Just find the *best R-squared* regardless of threshold.
        st.warning(f"Could not find an MTR with R² > {min_r_squared_threshold:,.3f}. Falling back to best-fit line (longest line with highest R²).")
        for i in range(n - min_points + 1):
            for j in range(i + min_points, n + 1):
                fit_df_loop = df.iloc[i:j].copy()
                num_points_in_loop = len(fit_df_loop)
                try:
                    regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)
                    if not np.isfinite(regression.slope): continue
                    r_squared = regression.rvalue ** 2

                    if num_points_in_loop > best_num_points:
                        # Prioritize length
                        best_num_points = num_points_in_loop
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared
                    elif num_points_in_loop == best_num_points and r_squared > best_r_squared:
                         # Then prioritize R-squared
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared
                except Exception:
                    continue

    st.session_state.mtr_r2_used = best_r_squared
    return best_fit_df, best_regression


# --- Main Analysis Function ---
def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text, dt_unit,
                     m_override, pi_override, k_override, S_override,
                     mtr_sensitivity): # <-- NEW SENSITIVITY
    """
    Performs the complete DST analysis based on user inputs.
    Calculations are now modular and can be overridden.
    V9.4 can auto-solve for Pwf, kh, and Transmissibility.
    """

    # --- 0. Initialize Results Dictionary ---
    results = {
        'm': None, 'pi': None,
        'k': None, 'kh_calc': None, 'transmissibility_calc': None, # <-- NEW
        'S': None, 'FE': None, 'ri': None, 'r_squared': None, 'dP_skin': None,
        'pwf_final_calc': None
    }
    mtr_info = None
    fig_horner = None
    fig_residuals = None
    df = None
    fit_df = None # Initialize
    k = 0 # Initialize
    pwf_final_to_use = pwf_final # Use the input value by default

    # --- 1. Essential Input Validation ---
    # V9.4 - Only tp and Qo are *truly* essential for the first steps
    if tp <= 0:
        st.error("Essential parameter 'Total Flow Time, tp (min)' must be a positive value.")
        return results, fig_horner, df, mtr_info, fig_residuals

    # Clear previous results from session state
    st.session_state.results = None
    st.session_state.figure = None
    st.session_state.figure_residuals = None
    st.session_state.dataframe = None
    st.session_state.mtr_info = None
    st.session_state.mtr_r2_used = None

    tp_hr = tp / 60.0  # tp is *always* in minutes, so tp_hr is correct

    # --- 2. Parse DST Data ---
    try:
        data_io = io.StringIO(data_text)
        df = pd.read_csv(
            data_io,
            sep=r'[,\s]+',
            engine='python',
            header=None,
            names=['dt', 'pwsf']
        )
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 3:
            st.error(f"Please enter at least 3 valid data points (you have {len(df)}).")
            return results, fig_horner, df, mtr_info, fig_residuals
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}. Please check the format (e.g., '5, 965').")
        return results, fig_horner, df, mtr_info, fig_residuals

    # --- 2b. NEW: Unit Conversion ---
    if dt_unit == "hours":
        st.warning("Converting input Δt from hours to minutes for calculation.")
        df['dt_calc'] = df['dt'] * 60.0
    else:
        df['dt_calc'] = df['dt'] # Use minutes as is

    # --- 3. Calculate Horner Time ---
    delta_t = df['dt_calc'].values # Use the converted dt
    pwsf = df['pwsf'].values
    horner_time = (tp + delta_t) / delta_t # This ratio is now unit-consistent
    log_horner_time = np.log10(horner_time)
    df['horner_time'] = horner_time
    df['log_horner_time'] = log_horner_time
    df = df.sort_values(by='log_horner_time', ascending=True).reset_index(drop=True)

    # --- 4. Determine m, pi (OVERRIDE LOGIC) ---
    if m_override > 0 and pi_override > 0:
        st.info(f"Using user-provided m = {m_override} and pi = {pi_override}")
        m_abs = m_override
        m_slope = -m_override # Horner slope is negative
        pi = pi_override
        r_squared = None # Cannot calculate R² if we don't regress
        fit_df = None
        mtr_info = None
    else:
        # Run Smart MTR Detection (V9.1)
        fit_df, regression = find_best_mtr(df, min_points=3, min_r_squared_threshold=mtr_sensitivity)

        if fit_df is None or regression is None:
            st.error("Automatic straight-line detection failed. The data may be too noisy.")
            return results, fig_horner, df, mtr_info, fig_residuals

        m_abs = abs(regression.slope)
        m_slope = regression.slope
        pi = regression.intercept
        r_squared = regression.rvalue ** 2

        mtr_info = {
            'num_points': len(fit_df),
            'start_dt_orig': float(fit_df['dt'].iloc[0]), # Original dt unit
            'end_dt_orig': float(fit_df['dt'].iloc[-1]), # Original dt unit
            'used_rows': fit_df.index.tolist()
        }


    # --- 5. Base Calculations (Always possible) ---
    results['m'] = m_abs
    results['pi'] = pi
    results['r_squared'] = r_squared

    # Calculate residuals for all points (for plotting)
    df['predicted_pwsf'] = pi + m_slope * df['log_horner_time']
    df['residual'] = df['pwsf'] - df['predicted_pwsf']

    # --- 6. Permeability Calculation (V9.4 MODULAR LOGIC) ---
    # This logic block determines k, kh, or T.

    # k: Permeability (md)
    # kh: Flow Capacity (md-ft)
    # T: Transmissibility (md-ft/cp)

    if k_override > 0:
        st.info(f"Using user-provided k = {k_override} md")
        k = k_override
        results['k'] = k

    # Check if we can calculate k (ALL essentials must be > 0)
    elif h > 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            k = (162.6 * (Qo * mu_o * Bo)) / (m_abs * h)
            results['k'] = k
        except Exception as e:
            st.warning(f"Could not calculate Permeability (k): {e}")
            k = 0 # Set k to 0 so subsequent calcs fail gracefully

    # Check if we can calculate kh (h=0, but other fluids > 0)
    elif h == 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            kh_calc = (162.6 * (Qo * mu_o * Bo)) / m_abs
            results['kh_calc'] = kh_calc
            st.success("`h` is 0, solving for Flow Capacity (kh) instead of k.")
        except Exception as e:
            st.warning(f"Could not calculate Flow Capacity (kh): {e}")

    # Check if we can calculate T (h=0, mu_o=0, but flow > 0)
    elif h == 0 and mu_o == 0 and Qo > 0 and Bo > 0:
        try:
            transmissibility_calc = (162.6 * (Qo * Bo)) / m_abs
            results['transmissibility_calc'] = transmissibility_calc
            st.success("`h` and `μo` are 0, solving for Transmissibility (kh/μo) instead of k.")
        except Exception as e:
            st.warning(f"Could not calculate Transmissibility: {e}")

    # --- 7. Skin, dP_Skin, FE Calculation (OVERRIDE LOGIC) ---
    S_calculated = False # Flag
    S = 0.0 # Initialize S

    # --- NEW: V9.2 Auto-Solve for Pwf ---
    # Check if we should *calculate* Pwf
    # This requires S_override, k, and all optionals
    if S_override != 0 and pwf_final == 0 and k > 0 and phi > 0 and Ct > 0 and rw > 0:
        try:
            st.success("Solving for `Pwf (Final Flow P)` based on `S_override`...")
            S = S_override
            results['S'] = S
            S_calculated = True

            # Rearrange the Skin equation to solve for Pwf
            log_term_B = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
            A = (S / 1.151) + log_term_B - 3.23
            pwf_final_calc = pi - (A * m_abs)
            results['pwf_final_calc'] = pwf_final_calc
            pwf_final_to_use = pwf_final_calc # Use this calculated value

        except Exception as e:
            st.warning(f"Could not auto-solve for Pwf: {e}")
            S_calculated = False # Failed, so we can't proceed

    # --- Standard Skin Calculation ---
    elif S_override != 0: # User provided S, but not trying to solve for Pwf
        st.info(f"Using user-provided S = {S_override}")
        S = S_override
        results['S'] = S
        S_calculated = True
    else:
        # Try to calculate S normally
        # This requires k > 0 and all optionals
        if k > 0 and phi > 0 and Ct > 0 and rw > 0 and pwf_final_to_use > 0:
            try:
                log_term = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
                S = 1.151 * (((pi - pwf_final_to_use) / m_abs) - log_term + 3.23)
                results['S'] = S
                S_calculated = True
            except Exception as e:
                st.warning(f"Could not calculate Skin (S): {e}. Check optional parameters.")

    # Now, calculate dP_skin and FE *if* we have a skin value AND k
    if S_calculated and k > 0 and h > 0 and pwf_final_to_use > 0:
        try:
            dP_skin = (141.2 * (Qo * mu_o * Bo) / (k * h)) * S
            results['dP_skin'] = dP_skin

            if (pi - pwf_final_to_use) != 0:
                FE = (pi - pwf_final_to_use - dP_skin) / (pi - pwf_final_to_use)
                results['FE'] = FE
            else:
                st.warning("Could not calculate Flow Efficiency (FE): (pi - pwf) is zero.")
        except Exception as e:
            st.warning(f"Could not calculate dP_skin or FE: {e}")


    # --- 8. Radius of Investigation (Depends on optionals) ---
    if k > 0 and phi > 0 and Ct > 0 and mu_o > 0: # mu_o is also needed
        try:
            # tp is already in minutes, use it directly
            ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))
            results['ri'] = ri
        except Exception as e:
            st.warning(f"Could not calculate Radius of Investigation (ri): {e}")

    # --- 9. Create the Horner Plot (NEW AESTHETICS) ---
    fig_horner, ax = plt.subplots(figsize=(10, 7))
    fig_horner.patch.set_facecolor('#ffffff') # Set outer bg to white
    ax.set_facecolor('#f8f9fa') # Set inner plot bg to light gray

    # Plot all data points (semi-transparent blue)
    ax.scatter(df['horner_time'], df['pwsf'], s=60, color='#007bff', edgecolor='black',
               linewidths=0.5, label='All DST Data', zorder=5, alpha=0.6)

    # Only plot red MTR dots if MTR was *calculated* (bolder red)
    if fit_df is not None and mtr_info is not None:
        ax.scatter(fit_df['horner_time'], fit_df['pwsf'], s=90, color='#dc3545',
                   edgecolor='black', linewidths=1.0, label=f"Auto-Detected MTR (n={mtr_info['num_points']})", zorder=6)

    ax.set_xscale('log')
    ax.invert_xaxis()

    # DYNAMIC AXIS LOGIC
    min_ht_plot = 1.0
    max_ht_plot = df['horner_time'].max()
    ax.set_xlim(left=max_ht_plot * 1.5, right=min_ht_plot * 0.9)
    min_y_plot = df['pwsf'].min()
    max_y_plot = max(df['pwsf'].max(), pi)
    y_padding = (max_y_plot - min_y_plot) * 0.10 # 10% padding
    y_padding = max(y_padding, 20.0) # at least 20 psi
    ax.set_ylim(bottom=min_y_plot - y_padding, top=max_y_plot + y_padding)

    # Plot regression lines
    x_line_log = np.array([np.log10(max_ht_plot * 1.5), 0])
    y_line = pi + m_slope * x_line_log # Uses m_slope (negative)

    label_r2 = f"R² = {r_squared:.3f}" if r_squared is not None else "R² = N/A"

    # Bolder MTR line
    ax.plot(10 ** x_line_log, y_line, color='black',
            label=f'MTR (m = {m_abs:.2f}, {label_r2})', zorder=4, linewidth=3.0, alpha=0.8)
    # Bolder, more distinct pi line
    ax.axhline(pi, color='#28a745', linestyle='--',
               label=f'Extrapolated $p_i$ = {pi:.1f} psi', linewidth=2.5, zorder=4)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Custom gridlines (FIXED: Made more visible)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)
    ax.grid(which='major', linestyle='-', linewidth='0.8', color='#cccccc', alpha=0.9)

    ax.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12, fontweight='bold')
    ax.set_title('Horner Plot Analysis', fontsize=16, fontweight='bold', pad=20)

    # Modern legend
    legend = ax.legend(loc='lower left', frameon=True, framealpha=0.9, facecolor='white', shadow=True)
    legend.get_frame().set_edgecolor('lightgray')

    # Remove all spines (borders)
    ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)

    # Add a subtle axis line at the bottom (new Y min)
    ax.axhline(min_y_plot - y_padding, color='black', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    # --- 10. Create the Residuals Plot (NEW AESTHETICS) ---
    fig_residuals, ax_res = plt.subplots(figsize=(10, 5))
    fig_residuals.patch.set_facecolor('#ffffff') # Set outer bg to white
    ax_res.set_facecolor('#f8f9fa') # Set inner plot bg to light gray

    # Plot points
    ax_res.scatter(df['horner_time'], df['residual'], s=60, color='#007bff',
                   edgecolor='black', linewidths=0.5, label='All Data Residuals', zorder=5, alpha=0.6)

    if fit_df is not None and mtr_info is not None:
        mtr_df_res = df.loc[mtr_info['used_rows']]
        ax_res.scatter(mtr_df_res['horner_time'], mtr_df_res['residual'], s=90, color='#dc3545',
                       edgecolor='black', linewidths=1.0, label='MTR Point Residuals', zorder=6)

    # Plot perfect fit line
    ax_res.axhline(0, color='black', linestyle='--', linewidth=2.0, label='Perfect Fit (Residual = 0)', alpha=0.8)

    ax_res.set_xscale('log')
    ax_res.invert_xaxis()
    ax_res.set_xlim(left=max_ht_plot * 1.1, right=min_ht_plot * 0.9)

    # Dynamic Y-axis for residuals
    max_abs_residual = df['residual'].abs().max()
    # Handle case where residuals are 0 or NaN
    if not np.isfinite(max_abs_residual) or max_abs_residual == 0:
        max_abs_residual = 1.0 # default value
    res_padding = max(max_abs_residual * 0.15, 5.0) # 15% padding, min 5 psi
    ax_res.set_ylim(-max_abs_residual - res_padding, max_abs_residual + res_padding)

    ax_res.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12, fontweight='bold')
    ax_res.set_ylabel('Pressure Residual (psi)\n(Actual - Predicted)', fontsize=12, fontweight='bold')
    ax_res.set_title('MTR Residuals Plot', fontsize=16, fontweight='bold', pad=20)

    # Modern legend
    legend = ax_res.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', shadow=True)
    legend.get_frame().set_edgecolor('lightgray')

    # Custom gridlines (FIXED: Made more visible)
    ax_res.grid(which='major', linestyle='-', linewidth='0.8', color='#cccccc', alpha=0.9)
    ax_res.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)

    ax_res.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    # Add a subtle axis line at the bottom
    ax_res.axhline(-max_abs_residual - res_padding, color='black', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    # --- 11. Return all 5 objects ---
    # We return the df with the *original* dt column for display
    df_to_return = df.drop(columns=['dt_calc'])
    return results, fig_horner, df_to_return, mtr_info, fig_residuals

def get_table_download_link(df):
    """Generate a link to download the processed data as CSV"""
    # Make sure we don't save the 'dt_calc' column
    df_to_save = df.copy()
    if 'dt_calc' in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=['dt_calc'])

    csv = df_to_save.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv">Download processed data as CSV</a>'
    return href

def format_metric(value, unit, format_str=":.2f"):
    """Helper function to format metrics, handling None values."""
    if value is None:
        return "Not Calculated"

    # Build the format specifier string, e.g., "{:.2f}"
    format_spec = "{" + format_str + "}"

    try:
        # Apply the format
        formatted_value = format_spec.format(value)
    except (ValueError, TypeError):
        # Fallback in case of an unexpected error
        formatted_value = str(value)

    return f"{formatted_value} {unit}"

# --- Main Application UI ---
def main():
    st.markdown('<h1 class="main-header">Interactive DST Horner Plot Analyst 📈</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    A professional web application for Drill Stem Test (DST) analysis using the Horner method.
    This tool automates the calculation of key reservoir properties from pressure buildup data,
    now with **smart automatic MTR detection** and **flexible modular calculations**.
    """)

    # --- Initialize session state ---
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'figure' not in st.session_state:
        st.session_state.figure = None
    if 'figure_residuals' not in st.session_state:
        st.session_state.figure_residuals = None # NEW
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    if 'mtr_info' not in st.session_state:
        st.session_state.mtr_info = None
    if 'mtr_r2_used' not in st.session_state: # NEW
        st.session_state.mtr_r2_used = None

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("📊 Input Parameters")
        with st.form(key='input_form'):

            # --- V9.3: Combined Section ---
            st.subheader("Reservoir & Test Properties")
            st.caption("Enter all known values. Leave at 0 to auto-calculate.")

            col1, col2 = st.columns(2)
            with col1:
                h = st.number_input("Pay Thickness, h (ft)", value=10.0, min_value=0.0, format="%.2f", step=1.0,
                                    help="Essential for k. If 0, will solve for kh.")
                Qo = st.number_input("Flow Rate, Qo (bbl/d)", value=135.0, min_value=0.0, format="%.2f", step=1.0,
                                     help="Essential for k and kh.")
                mu_o = st.number_input("Viscosity, μo (cp)", value=1.5, min_value=0.0, format="%.2f", step=0.1,
                                       help="Essential for k and kh. If 0, will solve for T.")
                Bo = st.number_input("FVF, Bo (RB/STB)", value=1.15, min_value=0.0, format="%.3f", step=0.01,
                                     help="Essential for k, kh, and T.")
                phi = st.number_input("Porosity, φ", value=0.10, min_value=0.0, max_value=1.0, format="%.3f", step=0.01,
                                      help="Set to 0 if unknown. Skin/FE/ri cannot be calculated.")
                Ct = st.number_input("Compressibility, Ct (psi⁻¹)", value=8.4e-6, min_value=0.0, format="%.2e", step=1e-7,
                                     help="Set to 0 if unknown. Skin/FE/ri cannot be calculated.")
            with col2:
                rw = st.number_input("Wellbore Radius, rw (ft)", value=0.333, min_value=0.0, format="%.3f", step=0.01,
                                     help="Set to 0 if unknown. Skin/FE cannot be calculated.")
                pwf_final = st.number_input("Final Flow P, Pwf (psi)", value=350.0, min_value=0.0, format="%.1f", step=1.0,
                                            help="Set to 0 if unknown. Will be auto-solved if S is provided.")
                m_override = st.number_input("Horner Slope 'm'", value=0.0, min_value=0.0, format="%.2f",
                                             help="Override: Leave at 0 to auto-detect MTR.")
                pi_override = st.number_input("Initial Pressure 'pi'", value=0.0, min_value=0.0, format="%.1f",
                                              help="Override: Leave at 0 to auto-extrapolate.")
                k_override = st.number_input("Permeability 'k' (md)", value=0.0, min_value=0.0, format="%.2f",
                                             help="Override: Leave at 0 to auto-calculate from m.")
                S_override = st.number_input("Skin 'S'", value=0.0, format="%.2f",
                                            help="Override: Enter non-zero value. If Pwf=0, will solve for Pwf.")

            st.markdown("---")
            st.subheader("DST Test Parameters")
            tp = st.number_input("Total Flow Time, tp (min)", value=60.0, min_value=0.1, format="%.1f",
                                help="The *total* duration of the flow period (tp) in minutes.",
                                step=1.0)

            # --- NEW: MTR Sensitivity Slider ---
            mtr_sensitivity = st.slider(
                "MTR Detection Sensitivity (Min R²)",
                min_value=0.950,
                max_value=1.000,
                value=0.995, # Default
                step=0.001,
                format="%.3f",
                help="Only used if m and pi are 0. Lower = less strict (finds longer, less straight lines). Higher = more strict."
            )

            st.subheader("Pressure Buildup Data")
            # --- NEW: dt Unit Selector ---
            dt_unit = st.radio(
                "Input Δt Time Unit",
                ("minutes", "hours"),
                index=0, # Default to minutes
                horizontal=True,
                help="Select the time unit for the 'Δt' column in your data."
            )

            default_data = """5, 965
10, 1215
15, 1405
20, 1590
25, 1685
30, 1725
35, 1740
40, 1753
45, 1765"""
            data_text = st.text_area(
                "Shut-in Data (Δt, Pwsf [psi])",
                value=default_data,
                height=200,
                help="Enter one 'time, pressure' pair per line. Example: '5, 965'"
            )

            submitted = st.form_submit_button("🚀 Run Analysis")

    # --- Perform analysis when form is submitted ---
    if submitted:
        # Clear previous run's warnings
        st.query_params.clear() # <-- UPDATED from st.experimental_set_query_params()

        with st.spinner("Auto-detecting MTR and performing analysis..."):
            results, figure, dataframe, mtr_info, fig_residuals = perform_analysis(
                h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text,
                dt_unit, # <-- Pass dt_unit
                m_override, pi_override, k_override, S_override, # Pass overrides
                mtr_sensitivity # <-- Pass sensitivity
            )

            # Store results regardless of success, to show partials
            st.session_state.results = results
            st.session_state.figure = figure
            st.session_state.figure_residuals = fig_residuals
            st.session_state.dataframe = dataframe
            st.session_state.mtr_info = mtr_info

            if results['m'] is not None:
                st.success("Analysis completed!")
            else:
                st.error("Analysis failed. Please check input parameters and data.")


    # --- Main panel with results and plots ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("📈 Analysis Results")

        if st.session_state.results:
            results = st.session_state.results
            mtr_info = st.session_state.mtr_info

            # Metrics
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.metric("Horner Slope 'm'", format_metric(results['m'], "psi/cycle", ":.2f"))
            st.metric("Initial Reservoir Pressure, pᵢ", format_metric(results['pi'], "psi", ":.1f"))

            # --- NEW: Modular k, kh, T display ---
            if results['k'] is not None:
                st.metric("Formation Permeability, k", format_metric(results['k'], "md", ":.2f"))
            if results['kh_calc'] is not None:
                st.metric("Flow Capacity, kh (CALCULATED)", format_metric(results['kh_calc'], "md-ft", ":.1f"))
            if results['transmissibility_calc'] is not None:
                st.metric("Transmissibility, T (CALCULATED)", format_metric(results['transmissibility_calc'], "md-ft/cp", ":.1f"))

            st.metric("Skin Factor, S", format_metric(results['S'], "", ":.2f"))

            # --- NEW: Show calculated Pwf ---
            if results['pwf_final_calc'] is not None:
                st.metric("Final Flow P, Pwf (CALCULATED)", format_metric(results['pwf_final_calc'], "psi", ":.1f"))

            st.metric("Pressure Drop (Skin), ΔP_skin", format_metric(results['dP_skin'], "psi", ":.1f"))
            st.metric("Flow Efficiency, FE", format_metric(results['FE'], "", ":.3f"))
            st.metric("Radius of Investigation, rᵢ", format_metric(results['ri'], "ft", ":.1f"))

            r2_val = results['r_squared']
            r2_display = f"{r2_val:.4f}" if r2_val is not None else "N/A (Overridden)"
            st.metric("Regression R-squared", r2_display)

            st.markdown('</div>', unsafe_allow_html=True)

            # MTR Info
            if mtr_info:
                r2_info = st.session_state.mtr_r2_used
                st.info(f"**Auto-MTR Successful:** Found best line (n={mtr_info['num_points']}) with R² = {r2_info:.4f}.\n\nThis line is from Δt = {mtr_info['end_dt_orig']} to {mtr_info['start_dt_orig']} {dt_unit}.")
            elif m_override > 0:
                 st.info(f"**Manual Override:** Using provided m = {m_override} and pi = {pi_override}.")

            # Interpretation
            st.subheader("📋 Interpretation")
            if results['S'] is not None:
                if results['S'] < -3: skin_interpretation = "Highly stimulated well (excellent)"
                elif results['S'] < 0: skin_interpretation = "Stimulated well (good)"
                elif results['S'] < 3: skin_interpretation = "Undamaged well"
                elif results['S'] < 10: skin_interpretation = "Damaged well"
                else: skin_interpretation = "Severely damaged well"
                st.write(f"**Skin Factor Interpretation:** {skin_interpretation}")
            else:
                st.write("**Skin Factor Interpretation:** Skin not calculated (k or other params missing).")

            if results['FE'] is not None:
                if results['FE'] > 1.0: fe_interpretation = "Well is stimulated"
                elif results['FE'] > 0.8: fe_interpretation = "Good flow efficiency"
                elif results['FE'] > 0.5: fe_interpretation = "Moderate damage"
                else: fe_interpretation = "Poor flow efficiency"
                st.write(f"**Flow Efficiency:** {fe_interpretation}")
            else:
                st.write("**Flow Efficiency:** FE not calculated (k or other params missing).")

        else:
            st.info("👈 Enter parameters and click 'Run Analysis' to see results")

    with col2:
        st.header("Analysis Outputs")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Horner Plot",
            "📉 Residuals Plot",
            "📥 Data Table",
            "⚙️ Methodology",
            "🧪 Formulas"
        ])
        with tab1:
            if st.session_state.figure:
                st.pyplot(st.session_state.figure, dpi=300)
            else:
                st.info("The Horner plot will appear here after analysis")

        with tab2:
            if st.session_state.figure_residuals:
                st.pyplot(st.session_state.figure_residuals, dpi=300)
                st.markdown("""
                **How to read this plot:**
                * This plot shows the *difference* (residual) between your actual pressure data and the pressure predicted by the straight MTR line.
                * Points near the **`Perfect Fit (Residual = 0)`** line are well-described by the MTR.
                * The **red points** (if shown) are the ones automatically detected and used by the app.
                """)
            else:
                st.info("The residuals plot will appear here after analysis")

        with tab3:
            if st.session_state.dataframe is not None:
                st.subheader("Processed Data")
                df = st.session_state.dataframe.copy()
                mtr_rows = st.session_state.mtr_info['used_rows'] if st.session_state.mtr_info else []
                df['MTR'] = ["✅" if i in df.index and mtr_rows and i in mtr_rows else "" for i in df.index]

                columns_order = ['dt', 'pwsf', 'horner_time', 'log_horner_time', 'predicted_pwsf', 'residual', 'MTR']
                # Ensure all columns exist before reordering
                df_cols = [col for col in columns_order if col in df.columns]
                df = df[df_cols]

                st.dataframe(df.style.format({
                    'dt': '{:.2f}', # Changed to 2f to show fractional hours
                    'pwsf': '{:.1f}',
                    'horner_time': '{:.2f}',
                    'log_horner_time': '{:.3f}',
                    'predicted_pwsf': '{:.1f}',
                    'residual': '{:.2f}',
                }))
                st.markdown("---")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            else:
                st.info("The processed data table will appear here")

        with tab4:
            st.subheader("Analysis Methodology (Visual Flowchart)")
            st.markdown("---")

            # --- Visual Flowchart using Graphviz ---
            try:
                dot = graphviz.Digraph(comment='DST Analysis Flow Chart')
                dot.attr(rankdir='TB', bgcolor='transparent', newrank='true') # Top-to-Bottom layout

                # Define node styles
                node_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#f0f2f6', 'fontname': 'Inter'}
                start_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#e6f3ff', 'fontname': 'Inter'}
                brain_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#fffbe6', 'fontname': 'Inter'}
                decision_style = {'shape': 'diamond', 'style': 'filled', 'fillcolor': '#f0f2f6', 'fontname': 'Inter'}
                result_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#e6ffed', 'fontname': 'Inter'}
                manual_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#f0e6ff', 'fontname': 'Inter'}
                calc_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#e6fff9', 'fontname': 'Inter'}


                # Define nodes
                dot.node('A', '1. Start: Inputs & Data', **start_style)
                dot.node('B', '2. Convert Δt to Minutes (if needed)', **node_style)
                dot.node('C', '3. Calc & Sort Horner Time', **node_style)

                dot.node('D', '4. m & pi Overridden?', **decision_style)
                dot.node('D_yes', 'Use User-Provided m & pᵢ', **manual_style)
                dot.node('D_no', 'Run Smart MTR Detection\n(V9.1 "Best Segment" Logic)', **brain_style)
                dot.node('D_calc', 'Get m & pᵢ from MTR', **node_style)

                dot.node('E', '5. k Overridden?', **decision_style)
                dot.node('E_yes', 'Use User-Provided k', **manual_style)
                dot.node('E_no', '6. All Essentials?\n(h, Qo, μ, Bo > 0)', **decision_style)
                dot.node('E_no_yes', 'Calculate k', **calc_style)
                dot.node('E_no_no', '7. Can Calc kh?\n(h=0, Qo, μ, Bo > 0)', **decision_style)
                dot.node('E_no_no_yes', 'Calculate kh\n(Flow Capacity)', **calc_style)
                dot.node('E_no_no_no', '8. Can Calc T?\n(h=0, μ=0, Qo, Bo > 0)', **decision_style)
                dot.node('E_no_no_no_yes', 'Calculate T\n(Transmissibility)', **calc_style)
                dot.node('E_no_no_no_no', 'Cannot Calculate\nk, kh, or T', **node_style)

                dot.node('F', '9. S Overridden?', **decision_style)
                dot.node('F_yes', 'Use User-Provided S', **manual_style)
                dot.node('F_no', 'Try to Calculate S', **node_style)

                dot.node('G', '10. Auto-Solve for Pwf?', **decision_style)
                dot.node('G_yes', 'Calculate Pwf (from S)', **calc_style)
                dot.node('G_no', 'Use Pwf input', **node_style)

                dot.node('H', '11. Calculate Final Properties\n(ΔP_skin, FE, rᵢ)', **node_style)
                dot.node('I', '12. End: Display Results & Plots', **result_style)

                # Define edges (arrows)
                dot.edge('A', 'B')
                dot.edge('B', 'C')
                dot.edge('C', 'D')

                dot.edge('D', 'D_yes', label='  Yes  ')
                dot.edge('D', 'D_no', label='  No  ')
                dot.edge('D_no', 'D_calc')

                dot.edge('D_yes', 'E')
                dot.edge('D_calc', 'E')

                dot.edge('E', 'E_yes', label='  Yes  ')
                dot.edge('E', 'E_no', label='  No  ')
                dot.edge('E_no', 'E_no_yes', label='  Yes  ')
                dot.edge('E_no', 'E_no_no', label='  No  ')

                dot.edge('E_no_no', 'E_no_no_yes', label='  Yes  ')
                dot.edge('E_no_no', 'E_no_no_no', label='  No  ')

                dot.edge('E_no_no_no', 'E_no_no_no_yes', label='  Yes  ')
                dot.edge('E_no_no_no', 'E_no_no_no_no', label='  No  ')

                # Edges leading to Skin calculation
                dot.edge('E_yes', 'F')
                dot.edge('E_no_yes', 'F')
                dot.edge('E_no_no_yes', 'H') # Skip Skin if k is unknown
                dot.edge('E_no_no_no_yes', 'H') # Skip Skin
                dot.edge('E_no_no_no_no', 'H') # Skip Skin

                dot.edge('F', 'F_yes', label='  Yes  ')
                dot.edge('F', 'F_no', label='  No  ')
                dot.edge('F_yes', 'G')
                dot.edge('F_no', 'G')

                dot.edge('G', 'G_yes', label='  Yes  \n(S_override set,\nPwf = 0,\nparams OK)')
                dot.edge('G', 'G_no', label='  No  ')

                dot.edge('G_yes', 'H')
                dot.edge('G_no', 'H')

                dot.edge('H', 'I')

                # Render the chart
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"""
                **Flowchart Error:** Could not render the visual flowchart.
                This usually means the `graphviz` system library is not installed on your computer.
                
                *Error details: {e}*
                
                The analysis will still work correctly.
                """)

            st.markdown("---")
            st.subheader("Formula Key")
            st.markdown("**Step 3: Horner Time**")
            st.latex(r"\text{Horner Time} = \frac{t_p + \Delta t}{\Delta t} \quad (\text{units must be consistent})")
            st.markdown("**Step 5-8: Permeability (k), Flow Capacity (kh), Transmissibility (T)**")
            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h} \quad (\text{if } h, Q_o, \mu_o, B_o > 0)")
            st.latex(r"kh = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m} \quad (\text{if } h=0)")
            st.latex(r"T = \frac{kh}{\mu_o} = \frac{162.6 \cdot Q_o \cdot B_o}{m} \quad (\text{if } h=0, \mu_o=0)")

            st.markdown("**Step 9-10: Skin (S) & Pwf**")
            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
            st.caption("This equation is rearranged to solve for `Pwf` if `S` is provided.")
            st.markdown("**Step 11: Final Properties**")
            st.latex(r"\Delta P_{skin} = \frac{141.2 \cdot Q_o \cdot \mu_o \cdot B_o}{k \cdot h} \cdot S")
            st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
            st.latex(r"r_i = \sqrt{\frac{k \cdot t_p}{57600 \cdot \phi \cdot \mu_o \cdot C_t}}")


        with tab5:
            st.subheader("Key Formulas")
            st.latex(r"p_{ws} = p_i - m \log\left(\frac{t_p + \Delta t}{\Delta t}\right)")
            st.caption("Horner Equation (m = slope)")
            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")
            st.caption("Permeability (k)")
            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
            st.caption("Skin Factor (S) - *Requires optional parameters*")
            st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
            st.caption("Flow Efficiency (FE) - *Requires optional parameters*")

    st.markdown("---")
    st.markdown(
        "**DST Horner Analyst** • Built with Python 🐍 and Streamlit • "
        "For professional petroleum engineering analysis"
    )

if __name__ == "__main__":
    main()
