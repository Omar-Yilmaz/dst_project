"""
Enhanced Interactive DST Horner Plot Analyst (Smart Auto MTR Detection)
Professional web application for Drill Stem Test analysis
V9.9 - Refactored MTR, State Fix, Pyplot Warning Fix
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64
# import graphviz <-- FIX 1: Removed unused import

# --- Page configuration ---
st.set_page_config(
    page_title="DST Horner Analyst (Smart-Fit)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling (V9.8 - Theme-Aware) ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Overall Font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background */
    .main .block-container {
        /* Use Streamlit's default background color */
        background-color: var(--background-color);
        padding-top: 2rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        /* Use Streamlit's secondary background color */
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--gray-200); /* Use theme-aware border */
        box-shadow: 2px 0px 10px rgba(0,0,0,0.03);
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.25rem;
        /* Use Streamlit's default text color */
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Result Box */
    .result-box {
        /* Use Streamlit's secondary background color */
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid var(--gray-200); /* Use theme-aware border */
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Metric styling */
    .stMetric {
        border-bottom: 1px solid var(--gray-100);
        padding-bottom: 1rem;
        padding-top: 0.5rem;
    }
    .stMetric:last-of-type {
        border-bottom: none;
    }
    .stMetric > label {
        font-weight: 600;
        color: var(--text-color); /* Use theme-aware text */
        opacity: 0.7;
    }
    .stMetric > div {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-color); /* Use theme-aware text */
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        border-bottom-color: var(--gray-300);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.0rem; /* <-- Slightly smaller to fit all 5 */
        font-weight: 600;
        padding: 0.75rem 0.75rem; /* <-- Reduced padding */
        color: var(--text-color); /* Use theme-aware text */
        opacity: 0.7;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color); /* Use theme's primary color */
        border-bottom: 3px solid var(--primary-color);
        opacity: 1.0;
    }
    
    /* Form & Input Widgets */
    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    /* All input boxes */
    .stNumberInput input, .stTextArea textarea {
        background-color: var(--background-color); /* Use main bg */
        border-radius: 0.5rem;
        border: 1px solid var(--gray-300);
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
        color: var(--text-color); /* Ensure text is visible */
    }
    .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(0,104,201,0.2);
    }
    
    /* Slider */
    [data-testid="stSlider"] {
        background-color: var(--background-color); /* Use main bg */
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--gray-300);
    }

    /* Horizontal Radio Buttons */
    div[role="radiogroup"] {
        flex-direction: row !important;
        justify-content: space-evenly;
        background-color: var(--background-color); /* Use main bg */
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid var(--gray-300);
    }

    /* Main Submit Button */
    [data-testid="stFormSubmitButton"] button {
        background-image: linear-gradient(to right, #0068c9 0%, #007bff 100%);
        color: white;
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 14px rgba(0,104,201,0.25);
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        opacity: 0.9;
        box-shadow: 0 6px 16px rgba(0,104,201,0.3);
        transform: translateY(-1px);
    }
    
    /* Expander for Interpretation */
    .st-expander {
        border: 1px solid var(--gray-300) !important;
        border-radius: 0.75rem !important;
        background-color: var(--secondary-background-color);
    }
    .st-expander header {
        font-weight: 600;
        color: var(--text-color);
    }

</style>
""", unsafe_allow_html=True)

# --- "SMART" Auto MTR detection function (V9.9 Refactor) ---
def find_best_mtr(df, min_points=3, min_r_squared_threshold=0.995):
    """
    Automatically find the *true* straight line segment (MTR) by searching
    all possible contiguous sub-segments of the data.

    V9.9 Logic (Single-Pass O(N^2) Refactor):
    1. Loop through *every possible segment* (from index i to j)
       that has at least `min_points`.
    2. Track two "best" lines simultaneously:
       a) `best_threshold_pass`: The longest line that *meets the R² threshold*.
       b) `best_overall`: The longest line *regardless* of threshold (as a fallback).
    3. "Longest" (more points) is prioritized first, then "straighter" (higher R²).
    4. After checking all segments, return `best_threshold_pass` if it was found.
    5. If no line met the threshold, issue a warning and return `best_overall`.
    """
    # FIX 4: Refactored function
    best_threshold_pass = {'reg': None, 'df': None, 'points': 0, 'r2': 0.0}
    best_overall = {'reg': None, 'df': None, 'points': 0, 'r2': 0.0}

    n = len(df)

    if n < min_points:
        st.session_state.mtr_r2_used = None
        return None, None

    # Loop through all possible start points
    for i in range(n - min_points + 1):
        # Loop through all possible end points
        for j in range(i + min_points, n + 1):

            fit_df_loop = df.iloc[i:j].copy()
            num_points_in_loop = len(fit_df_loop)

            try:
                regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)
                
                # Add check for non-finite values from regression
                if not np.isfinite(regression.slope) or not np.isfinite(regression.rvalue):
                    continue
                    
                r_squared = regression.rvalue ** 2

                # --- Logic to update a "best" dictionary ---
                def update_best(best_dict):
                    if num_points_in_loop > best_dict['points']:
                        # This is the new *longest* line
                        best_dict['points'] = num_points_in_loop
                        best_dict['reg'] = regression
                        best_dict['df'] = fit_df_loop
                        best_dict['r2'] = r_squared
                    elif num_points_in_loop == best_dict['points'] and r_squared > best_dict['r2']:
                        # Same length, but *straighter*
                        best_dict['reg'] = regression
                        best_dict['df'] = fit_df_loop
                        best_dict['r2'] = r_squared

                # 1. Check against the overall best (fallback)
                update_best(best_overall)

                # 2. Check if it passes the threshold
                if r_squared >= min_r_squared_threshold:
                    update_best(best_threshold_pass)

            except Exception:
                # Ignore errors from linregress
                continue

    if best_threshold_pass['reg'] is not None:
        # Success: Found a line that meets the threshold
        st.session_state.mtr_r2_used = best_threshold_pass['r2']
        return best_threshold_pass['df'], best_threshold_pass['reg']
    elif best_overall['reg'] is not None:
        # Fallback: No line met the threshold, use the best-fit line
        st.warning(f"Could not find an MTR with R² > {min_r_squared_threshold:,.3f}. Falling back to best-fit line (n={best_overall['points']}, R²={best_overall['r2']:.4f}).")
        st.session_state.mtr_r2_used = best_overall['r2']
        return best_overall['df'], best_overall['reg']
    else:
        # Complete failure
        st.session_state.mtr_r2_used = None
        return None, None


# --- Main Analysis Function ---
def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text, dt_unit,
                     m_override, pi_override, k_override, S_override,
                     mtr_sensitivity):
    """
    Performs the complete DST analysis based on user inputs.
    Calculations are now modular and can be overridden.
    V9.4 can auto-solve for Pwf, kh, and Transmissibility.
    """

    # --- 0. Initialize Results Dictionary ---
    results = {
        'm': None, 'pi': None,
        'k': None, 'kh_calc': None, 'transmissibility_calc': None,
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

    # --- 4. Calculate Pressure Derivative (V9.5) ---
    # (Removed)

    # --- 5. Determine m, pi (OVERRIDE LOGIC) ---
    if m_override > 0 and pi_override > 0:
        st.info(f"Using user-provided m = {m_override} and pi = {pi_override}")
        m_abs = m_override
        m_slope = -m_override # Horner slope is negative
        pi = pi_override
        r_squared = None # Cannot calculate R² if we don't regress
        fit_df = None
        mtr_info = None
    else:
        # Run Smart MTR Detection (V9.9)
        fit_df, regression = find_best_mtr(df, min_points=3, min_r_squared_threshold=mtr_sensitivity)

        if fit_df is None or regression is None:
            st.error("Automatic straight-line detection failed. The data may be too noisy or no valid line found.")
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


    # --- 6. Base Calculations (Always possible) ---
    results['m'] = m_abs
    results['pi'] = pi
    results['r_squared'] = r_squared

    # Calculate residuals for all points (for plotting)
    df['predicted_pwsf'] = pi + m_slope * df['log_horner_time']
    df['residual'] = df['pwsf'] - df['predicted_pwsf']

    # --- 7. Permeability Calculation (V9.4 MODULAR LOGIC) ---
    if k_override > 0:
        st.info(f"Using user-provided k = {k_override} md")
        k = k_override
        results['k'] = k
    elif h > 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            k = (162.6 * (Qo * mu_o * Bo)) / (m_abs * h)
            results['k'] = k
        except Exception as e:
            st.warning(f"Could not calculate Permeability (k): {e}")
            k = 0
    elif h == 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            kh_calc = (162.6 * (Qo * mu_o * Bo)) / m_abs
            results['kh_calc'] = kh_calc
            st.success("`h` is 0, solving for Flow Capacity (kh) instead of k.")
        except Exception as e:
            st.warning(f"Could not calculate Flow Capacity (kh): {e}")
    elif h == 0 and mu_o == 0 and Qo > 0 and Bo > 0:
        try:
            transmissibility_calc = (162.6 * (Qo * Bo)) / m_abs
            results['transmissibility_calc'] = transmissibility_calc
            st.success("`h` and `μo` are 0, solving for Transmissibility (kh/μo) instead of k.")
        except Exception as e:
            st.warning(f"Could not calculate Transmissibility: {e}")

    # --- 8. Skin, dP_Skin, FE Calculation (OVERRIDE LOGIC) ---
    S_calculated = False
    S = 0.0

    if S_override != 0 and pwf_final == 0 and k > 0 and phi > 0 and Ct > 0 and rw > 0:
        try:
            st.success("Solving for `Pwf (Final Flow P)` based on `S_override`...")
            S = S_override
            results['S'] = S
            S_calculated = True

            log_term_B = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
            A = (S / 1.151) + log_term_B - 3.23
            pwf_final_calc = pi - (A * m_abs)
            results['pwf_final_calc'] = pwf_final_calc
            pwf_final_to_use = pwf_final_calc

        except Exception as e:
            st.warning(f"Could not auto-solve for Pwf: {e}")
            S_calculated = False

    elif S_override != 0:
        st.info(f"Using user-provided S = {S_override}")
        S = S_override
        results['S'] = S
        S_calculated = True
    else:
        if k > 0 and phi > 0 and Ct > 0 and rw > 0 and pwf_final_to_use > 0:
            try:
                log_term = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
                S = 1.151 * (((pi - pwf_final_to_use) / m_abs) - log_term + 3.23)
                results['S'] = S
                S_calculated = True
            except Exception as e:
                st.warning(f"Could not calculate Skin (S): {e}. Check optional parameters.")

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


    # --- 9. Radius of Investigation (Depends on optionals) ---
    if k > 0 and phi > 0 and Ct > 0 and mu_o > 0: # mu_o is also needed
        try:
            ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))
            results['ri'] = ri
        except Exception as e:
            st.warning(f"Could not calculate Radius of Investigation (ri): {e}")

    # --- 10. Create the Horner Plot (THEME-AWARE) ---
    plt.rcParams['font.family'] = 'Inter' # Match CSS font

    # --- Get theme colors from Streamlit (for plots) ---
    theme_text_color = plt.rcParams.get('text.color', 'black')
    theme_bg_color = plt.rcParams.get('figure.facecolor', 'white')
    theme_grid_color = plt.rcParams.get('axes.grid', True)
    theme_grid_color = 'gray' if theme_grid_color else 'gray' # fallback

    fig_horner, ax = plt.subplots(figsize=(10, 7))
    fig_horner.patch.set_facecolor(theme_bg_color)
    ax.set_facecolor(theme_bg_color)

    ax.scatter(df['horner_time'], df['pwsf'], s=60, color='#007bff', edgecolor=theme_text_color,
               linewidths=0.5, label='All DST Data', zorder=5, alpha=0.6)

    if fit_df is not None and mtr_info is not None:
        ax.scatter(fit_df['horner_time'], fit_df['pwsf'], s=90, color='#dc3545',
                   edgecolor=theme_text_color, linewidths=1.0, label=f"Auto-Detected MTR (n={mtr_info['num_points']})", zorder=6)

    ax.set_xscale('log')
    ax.invert_xaxis()

    min_ht_plot = 1.0
    max_ht_plot = df['horner_time'].max()
    ax.set_xlim(left=max_ht_plot * 1.5, right=min_ht_plot * 0.9)
    min_y_plot = df['pwsf'].min()
    max_y_plot = max(df['pwsf'].max(), pi if pi is not None else 0)
    y_padding = (max_y_plot - min_y_plot) * 0.10 # 10% padding
    y_padding = max(y_padding, 20.0) # at least 20 psi
    ax.set_ylim(bottom=min_y_plot - y_padding, top=max_y_plot + y_padding)

    x_line_log = np.array([np.log10(max_ht_plot * 1.5), 0])
    y_line = pi + m_slope * x_line_log # Uses m_slope (negative)

    label_r2 = f"R² = {r_squared:.3f}" if r_squared is not None else "R² = N/A"

    ax.plot(10 ** x_line_log, y_line, color=theme_text_color,
            label=f'MTR (m = {m_abs:.2f}, {label_r2})', zorder=4, linewidth=3.0, alpha=0.8)
    ax.axhline(pi, color='#28a745', linestyle='--',
               label=f'Extrapolated $p_i$ = {pi:.1f} psi', linewidth=2.5, zorder=4)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # THEME-AWARE Gridlines
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color=theme_grid_color, alpha=0.3)
    ax.grid(which='major', linestyle='-', linewidth='0.8', color=theme_grid_color, alpha=0.5)

    ax.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12, fontweight='bold')
    ax.set_title('Horner Plot Analysis', fontsize=16, fontweight='bold', pad=20)

    # THEME-AWARE Legend
    legend = ax.legend(loc='lower left', frameon=True, framealpha=0.8, shadow=True)
    legend.get_frame().set_edgecolor(theme_grid_color)
    legend.get_frame().set_facecolor(theme_bg_color)

    ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)

    plt.tight_layout()

    # --- 11. Create the Residuals Plot (THEME-AWARE) ---
    fig_residuals, ax_res = plt.subplots(figsize=(10, 5))
    fig_residuals.patch.set_facecolor(theme_bg_color)
    ax_res.set_facecolor(theme_bg_color)

    ax_res.scatter(df['horner_time'], df['residual'], s=60, color='#007bff',
                   edgecolor=theme_text_color, linewidths=0.5, label='All Data Residuals', zorder=5, alpha=0.6)

    if fit_df is not None and mtr_info is not None:
        mtr_df_res = df.loc[mtr_info['used_rows']]
        ax_res.scatter(mtr_df_res['horner_time'], mtr_df_res['residual'], s=90, color='#dc3545',
                       edgecolor=theme_text_color, linewidths=1.0, label='MTR Point Residuals', zorder=6)

    ax_res.axhline(0, color=theme_text_color, linestyle='--', linewidth=2.0, label='Perfect Fit (Residual = 0)', alpha=0.8)

    ax_res.set_xscale('log')
    ax_res.invert_xaxis()
    ax_res.set_xlim(left=max_ht_plot * 1.1, right=min_ht_plot * 0.9)

    max_abs_residual = df['residual'].abs().max()
    if not np.isfinite(max_abs_residual) or max_abs_residual == 0:
        max_abs_residual = 1.0
    res_padding = max(max_abs_residual * 0.15, 5.0)
    ax_res.set_ylim(-max_abs_residual - res_padding, max_abs_residual + res_padding)

    ax_res.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12, fontweight='bold')
    ax_res.set_ylabel('Pressure Residual (psi)\n(Actual - Predicted)', fontsize=12, fontweight='bold')
    ax_res.set_title('MTR Residuals Plot', fontsize=16, fontweight='bold', pad=20)

    legend = ax_res.legend(loc='best', frameon=True, framealpha=0.8, shadow=True)
    legend.get_frame().set_edgecolor(theme_grid_color)
    legend.get_frame().set_facecolor(theme_bg_color)

    # THEME-AWARE Gridlines
    ax_res.grid(which='major', linestyle='-', linewidth='0.8', color=theme_grid_color, alpha=0.5)
    ax_res.grid(which='minor', linestyle=':', linewidth='0.5', color=theme_grid_color, alpha=0.3)

    ax_res.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    plt.tight_layout()

    # --- 12. Create the Log-Log Derivative Plot (V9.5) ---
    # (Removed)

    # --- 13. Return all 5 objects ---
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
        st.session_state.figure_residuals = None
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    if 'mtr_info' not in st.session_state:
        st.session_state.mtr_info = None
    if 'mtr_r2_used' not in st.session_state:
        st.session_state.mtr_r2_used = None
    if 'dt_unit_used' not in st.session_state: # <-- FIX 3: Initialize
        st.session_state.dt_unit_used = "minutes"


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
                                     help="Essential for k and kh. If 0, calculation will fail.")
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
        st.query_params.clear() 

        with st.spinner("Auto-detecting MTR and performing analysis..."):
            results, figure, dataframe, mtr_info, fig_residuals = perform_analysis(
                h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text,
                dt_unit,
                m_override, pi_override, k_override, S_override,
                mtr_sensitivity
            )

            # Store results regardless of success, to show partials
            st.session_state.results = results
            st.session_state.figure = figure
            st.session_state.figure_residuals = fig_residuals
            st.session_state.dataframe = dataframe
            st.session_state.mtr_info = mtr_info
            st.session_state.dt_unit_used = dt_unit # <-- FIX 3: Store unit in state

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
                # <-- FIX 3: Read unit from session state -->
                dt_unit_used = st.session_state.get('dt_unit_used', 'minutes')
                st.info(f"**Auto-MTR Successful:** Found best line (n={mtr_info['num_points']}) with R² = {r2_info:.4f}.\n\nThis line is from Δt = {mtr_info['end_dt_orig']} to {mtr_info['start_dt_orig']} {dt_unit_used}.")
            elif m_override > 0:
                 st.info(f"**Manual Override:** Using provided m = {m_override} and pi = {pi_override}.")

            # --- NEW: Interpretation as Expander ---
            with st.expander("Show Interpretation"):
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
                # <-- FIX 2: Added use_container_width=True -->
                st.pyplot(st.session_state.figure, use_container_width=True)
            else:
                st.info("The Horner plot will appear here after analysis")

        with tab2:
            if st.session_state.figure_residuals:
                # <-- FIX 2: Added use_container_width=True -->
                st.pyplot(st.session_state.figure_residuals, use_container_width=True)
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
                
                # Ensure MTR column logic is robust
                df['MTR'] = ""
                if mtr_rows:
                    df.loc[df.index.isin(mtr_rows), 'MTR'] = "✅"

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
            st.subheader("Analysis Methodology")
            st.markdown(
                """
                The app follows a modular, 6-step process to "solve for everything possible":

                **1. Data Prep:**
                Your input data (`tp` and `Δt`) is read and all time units are consistently converted to **minutes**. The `Horner Time` and `log_horner_time` are calculated for every data point.

                **2. Find MTR (Slope & Pressure):**
                * **If** you provide an `m` and `pi` override, the app uses them.
                * **Else,** the app's "Smart MTR Detection" (V9.9) runs. It mathematically searches *every possible line segment* in your data to find the **longest** one that is also **straighter** than the "MTR Detection Sensitivity" R² value. This allows it to ignore WBS and boundary curves.

                **3. Calculate Permeability (k):**
                * **If** you provide a `k` override, the app uses it.
                * **Else,** it tries to solve for `k`. If `h` is 0, it solves for `kh`. If `h` and `μo` are 0, it solves for `T (kh/μo)`.

                **4. Calculate Skin (S):**
                * **If** you provide an `S` override, the app uses it.
                * **Else,** it tries to calculate `S` *only if* it has all the necessary components (`k`, `phi`, `Ct`, `rw`, `Pwf`).

                **5. Auto-Solve for Pwf (Final Flow P):**
                * A special case! **If** you provided an `S` override but left `Pwf` as 0, the app rearranges the Skin equation and *solves for Pwf* for you.

                **6. Final Results:**
                The app calculates any remaining properties (like `ΔP_skin`, `FE`, and `rᵢ`) based on the values from the previous steps and displays all available results.
                """
            )
            st.markdown("---")
            st.subheader("Formula Key")
            st.markdown("**Horner Time**")
            st.latex(r"\text{Horner Time} = \frac{t_p + \Delta t}{\Delta t} \quad (\text{units must be consistent})")

            st.markdown("**Horner Slope (m)**")
            st.latex(r"m = \frac{P_{ws_1} - P_{ws_{10}}}{\log(10) - \log(1)}")
            st.caption("This is the 'by hand' calculation per log cycle. The app uses linear regression on the MTR points for a more accurate value.")

            st.markdown("**Permeability (k), Flow Capacity (kh), Transmissibility (T)**")
            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h} \quad (\text{if } h, Q_o, \mu_o, B_o > 0)")
            st.latex(r"kh = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m} \quad (\text{if } h=0)")
            st.latex(r"T = \frac{kh}{\mu_o} = \frac{162.6 \cdot Q_o \cdot B_o}{m} \quad (\text{if } h=0, \mu_o=0)")


        with tab5:
            st.subheader("Key Formulas")
            st.markdown("**Core Equation**")
            st.latex(r"p_{ws} = p_i - m \log\left(\frac{t_p + \Delta t}{\Delta t}\right)")
            st.caption("Horner Equation (m = slope)")

            st.markdown("**Reservoir Properties**")
            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")
            st.caption("Permeability (k)")
            st.latex(r"r_i = \sqrt{\frac{k \cdot t_p}{57600 \cdot \phi \cdot \mu_o \cdot C_t}}")
            st.caption("Radius of Investigation (rᵢ)")

            st.markdown("**Damage & Efficiency**")
            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
            st.caption("Skin Factor (S) - *Requires all optional parameters*")
            st.latex(r"\Delta P_{skin} = \frac{141.2 \cdot Q_o \cdot \mu_o \cdot B_o}{k \cdot h} \cdot S")
            st.caption("Pressure Drop due to Skin (ΔP_skin)")
            st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
            st.caption("Flow Efficiency (FE)")

    st.markdown("---")
    st.markdown(
        "**DST Horner Analyst** • Built with Python 🐍 and Streamlit • "
        "For professional petroleum engineering analysis"
    )

if __name__ == "__main__":
    main()
