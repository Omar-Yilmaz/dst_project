"""
Enhanced Interactive DST Horner Plot Analyst (Smart Auto MTR Detection)
Professional web application for Drill Stem Test analysis
V10.0 - Enhanced with exports, validation, and better UX
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64
from datetime import datetime

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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
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
    div[role="radiogroup"] {
        flex-direction: row !important;
        justify-content: space-evenly;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Validation Functions ---
def validate_pressure_data(df):
    """Validate pressure buildup data for physical consistency"""
    warnings = []

    # Check for monotonic increase (allowing small fluctuations)
    pressure_diff = df['pwsf'].diff()
    decreasing_points = (pressure_diff < -5).sum()  # Allow 5 psi noise

    if decreasing_points > len(df) * 0.2:  # More than 20% decreasing
        warnings.append("⚠️ Pressure appears to decrease during buildup. Check data quality.")

    # Check for unrealistic pressure values
    if df['pwsf'].min() < 0:
        warnings.append("❌ Negative pressure values detected. Data may be invalid.")

    if df['pwsf'].max() > 20000:
        warnings.append("⚠️ Extremely high pressures detected (>20,000 psi). Verify data.")

    # Check for duplicate time points
    if df['dt'].duplicated().any():
        warnings.append("⚠️ Duplicate time points detected. Consider averaging or removing.")

    # Check time spacing
    time_ratios = df['dt'].iloc[1:].values / df['dt'].iloc[:-1].values
    if (time_ratios > 10).any():
        warnings.append("⚠️ Large gaps in time spacing detected. May affect MTR detection.")

    return warnings

def check_parameter_consistency(h, Qo, mu_o, Bo, phi, Ct, rw, pwf_final, pi):
    """Check if parameters are physically reasonable"""
    warnings = []

    if h > 1000:
        warnings.append("⚠️ Very thick pay zone (>1000 ft). Verify h value.")

    if Qo > 10000:
        warnings.append("⚠️ Very high flow rate (>10,000 bbl/d). Verify Qo.")

    if mu_o < 0.1 or mu_o > 1000:
        warnings.append("⚠️ Unusual viscosity value. Typical range: 0.1-1000 cp.")

    if Bo < 1.0 or Bo > 3.0:
        warnings.append("⚠️ Unusual FVF value. Typical range: 1.0-3.0 RB/STB.")

    if phi > 0.4:
        warnings.append("⚠️ Very high porosity (>40%). Verify φ value.")

    if pwf_final > pi and pi > 0:
        warnings.append("❌ Final flowing pressure > Initial pressure. Data inconsistency.")

    return warnings

# --- Export Functions ---
def export_results_to_txt(results, mtr_info, input_params):
    """Generate a formatted text report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format R-squared properly
    r2_text = f"{results['r_squared']:.4f}" if results['r_squared'] is not None else "N/A"

    # Format optional results
    k_text = f"{results['k']:.2f} md" if results['k'] else "Not calculated"
    kh_text = f"{results['kh_calc']:.1f} md-ft" if results['kh_calc'] else "Not calculated"
    t_text = f"{results['transmissibility_calc']:.1f} md-ft/cp" if results['transmissibility_calc'] else "Not calculated"
    s_text = f"{results['S']:.2f}" if results['S'] is not None else "Not calculated"
    dp_text = f"{results['dP_skin']:.1f} psi" if results['dP_skin'] else "Not calculated"
    fe_text = f"{results['FE']:.3f}" if results['FE'] else "Not calculated"
    ri_text = f"{results['ri']:.1f} ft" if results['ri'] else "Not calculated"

    report = f"""
DST HORNER ANALYSIS REPORT
Generated: {timestamp}
{'='*60}

INPUT PARAMETERS:
{'-'*60}
Pay Thickness (h):          {input_params['h']:.2f} ft
Flow Rate (Qo):             {input_params['Qo']:.2f} bbl/d
Viscosity (μo):             {input_params['mu_o']:.2f} cp
FVF (Bo):                   {input_params['Bo']:.3f} RB/STB
Porosity (φ):               {input_params['phi']:.3f}
Compressibility (Ct):       {input_params['Ct']:.2e} psi⁻¹
Wellbore Radius (rw):       {input_params['rw']:.3f} ft
Total Flow Time (tp):       {input_params['tp']:.1f} min
Final Flowing Pressure:     {input_params['pwf_final']:.1f} psi

ANALYSIS RESULTS:
{'-'*60}
Horner Slope (m):           {results['m']:.2f} psi/cycle
Initial Pressure (pi):      {results['pi']:.1f} psi
R-squared:                  {r2_text}

Permeability (k):           {k_text}
Flow Capacity (kh):         {kh_text}
Transmissibility (T):       {t_text}

Skin Factor (S):            {s_text}
Pressure Drop (ΔP_skin):    {dp_text}
Flow Efficiency (FE):       {fe_text}
Radius of Investigation:    {ri_text}

MTR INFORMATION:
{'-'*60}
"""

    if mtr_info:
        report += f"""Points in MTR:              {mtr_info['num_points']}
Time Range:                 {mtr_info['start_dt_orig']:.2f} to {mtr_info['end_dt_orig']:.2f} {input_params['dt_unit']}
"""
    else:
        report += "Manual override used\n"

    report += f"\n{'='*60}\n"

    return report

def get_plot_download_link(fig, filename):
    """Generate download link for matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

def copy_to_clipboard_button(text, label="📋 Copy Results"):
    """Create a button that copies text to clipboard"""
    st.code(text, language=None)
    st.caption(f"☝️ {label} - Select and copy the text above")

# --- Smart Auto MTR detection function ---
def find_best_mtr(df, min_points=3, min_r_squared_threshold=0.995):
    """
    Automatically find the true straight line segment (MTR) with enhanced logic
    """
    best_regression = None
    best_fit_df = None
    best_num_points = 0
    best_r_squared = 0.0

    n = len(df)

    for i in range(n - min_points + 1):
        for j in range(i + min_points, n + 1):

            fit_df_loop = df.iloc[i:j].copy()
            num_points_in_loop = len(fit_df_loop)

            try:
                regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)

                if not np.isfinite(regression.slope):
                    continue

                r_squared = regression.rvalue ** 2

                if r_squared >= min_r_squared_threshold:
                    if num_points_in_loop > best_num_points:
                        best_num_points = num_points_in_loop
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared
                    elif num_points_in_loop == best_num_points and r_squared > best_r_squared:
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared

            except Exception:
                continue

    if best_regression is None:
        st.warning(f"Could not find an MTR with R² > {min_r_squared_threshold:.3f}. Using best available fit.")
        for i in range(n - min_points + 1):
            for j in range(i + min_points, n + 1):
                fit_df_loop = df.iloc[i:j].copy()
                num_points_in_loop = len(fit_df_loop)
                try:
                    regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)
                    if not np.isfinite(regression.slope): continue
                    r_squared = regression.rvalue ** 2

                    if num_points_in_loop > best_num_points:
                        best_num_points = num_points_in_loop
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared
                    elif num_points_in_loop == best_num_points and r_squared > best_r_squared:
                        best_regression = regression
                        best_fit_df = fit_df_loop
                        best_r_squared = r_squared
                except Exception:
                    continue

    st.session_state.mtr_r2_used = best_r_squared
    return best_fit_df, best_regression

# --- Main Analysis Function ---
def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text, dt_unit,
                     m_override, pi_override, k_override, S_override, mtr_sensitivity):
    """
    Performs the complete DST analysis with enhanced validation and error handling
    """

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
    fit_df = None
    k = 0
    pwf_final_to_use = pwf_final
    validation_warnings = []

    # --- Essential Input Validation ---
    if tp <= 0:
        st.error("❌ Total Flow Time (tp) must be positive.")
        return results, fig_horner, df, mtr_info, fig_residuals, validation_warnings

    st.session_state.results = None
    st.session_state.figure = None
    st.session_state.figure_residuals = None
    st.session_state.dataframe = None
    st.session_state.mtr_info = None
    st.session_state.mtr_r2_used = None

    tp_hr = tp / 60.0

    # --- Parse DST Data with better error handling ---
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
            st.error(f"❌ Need at least 3 valid data points (found {len(df)}).")
            return results, fig_horner, df, mtr_info, fig_residuals, validation_warnings

        # Validate data quality
        data_warnings = validate_pressure_data(df)
        validation_warnings.extend(data_warnings)

    except Exception as e:
        st.error(f"❌ Data parsing error: {str(e)}")
        return results, fig_horner, df, mtr_info, fig_residuals, validation_warnings

    # --- Unit Conversion ---
    if dt_unit == "hours":
        st.info("ℹ️ Converting Δt from hours to minutes.")
        df['dt_calc'] = df['dt'] * 60.0
    else:
        df['dt_calc'] = df['dt']

    # --- Calculate Horner Time ---
    delta_t = df['dt_calc'].values
    pwsf = df['pwsf'].values
    horner_time = (tp + delta_t) / delta_t
    log_horner_time = np.log10(horner_time)
    df['horner_time'] = horner_time
    df['log_horner_time'] = log_horner_time
    df = df.sort_values(by='log_horner_time', ascending=True).reset_index(drop=True)

    # --- Determine m, pi ---
    if m_override > 0 and pi_override > 0:
        st.info(f"ℹ️ Using manual values: m = {m_override}, pi = {pi_override}")
        m_abs = m_override
        m_slope = -m_override
        pi = pi_override
        r_squared = None
        fit_df = None
        mtr_info = None
    else:
        fit_df, regression = find_best_mtr(df, min_points=3, min_r_squared_threshold=mtr_sensitivity)

        if fit_df is None or regression is None:
            st.error("❌ MTR detection failed. Data may be too noisy or non-linear.")
            return results, fig_horner, df, mtr_info, fig_residuals, validation_warnings

        m_abs = abs(regression.slope)
        m_slope = regression.slope
        pi = regression.intercept
        r_squared = regression.rvalue ** 2

        mtr_info = {
            'num_points': len(fit_df),
            'start_dt_orig': float(fit_df['dt'].iloc[0]),
            'end_dt_orig': float(fit_df['dt'].iloc[-1]),
            'used_rows': fit_df.index.tolist()
        }

        # Warn if R² is lower than expected
        if r_squared < 0.99:
            validation_warnings.append(f"⚠️ MTR fit quality is moderate (R² = {r_squared:.4f}). Consider data quality.")

    # --- Base Calculations ---
    results['m'] = m_abs
    results['pi'] = pi
    results['r_squared'] = r_squared

    df['predicted_pwsf'] = pi + m_slope * df['log_horner_time']
    df['residual'] = df['pwsf'] - df['predicted_pwsf']

    # --- Check parameter consistency ---
    param_warnings = check_parameter_consistency(h, Qo, mu_o, Bo, phi, Ct, rw, pwf_final, pi)
    validation_warnings.extend(param_warnings)

    # --- Permeability Calculation ---
    if k_override > 0:
        st.info(f"ℹ️ Using manual k = {k_override} md")
        k = k_override
        results['k'] = k
    elif h > 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            k = (162.6 * (Qo * mu_o * Bo)) / (m_abs * h)
            results['k'] = k

            # Sanity check on k
            if k > 10000:
                validation_warnings.append("⚠️ Very high permeability (>10,000 md). Verify calculation.")
            elif k < 0.01:
                validation_warnings.append("⚠️ Very low permeability (<0.01 md). Verify calculation.")

        except Exception as e:
            st.warning(f"Could not calculate k: {e}")
            k = 0
    elif h == 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            kh_calc = (162.6 * (Qo * mu_o * Bo)) / m_abs
            results['kh_calc'] = kh_calc
            st.success("✅ Calculated Flow Capacity (kh) since h = 0")
        except Exception as e:
            st.warning(f"Could not calculate kh: {e}")
    elif h == 0 and mu_o == 0 and Qo > 0 and Bo > 0:
        try:
            transmissibility_calc = (162.6 * (Qo * Bo)) / m_abs
            results['transmissibility_calc'] = transmissibility_calc
            st.success("✅ Calculated Transmissibility since h = 0 and μo = 0")
        except Exception as e:
            st.warning(f"Could not calculate Transmissibility: {e}")

    # --- Skin, dP_Skin, FE Calculation ---
    S_calculated = False
    S = 0.0

    if S_override != 0 and pwf_final == 0 and k > 0 and phi > 0 and Ct > 0 and rw > 0:
        try:
            st.success("✅ Auto-solving for Pwf based on provided Skin...")
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
        st.info(f"ℹ️ Using manual S = {S_override}")
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
                st.warning(f"Could not calculate Skin: {e}")

    if S_calculated and k > 0 and h > 0 and pwf_final_to_use > 0:
        try:
            dP_skin = (141.2 * (Qo * mu_o * Bo) / (k * h)) * S
            results['dP_skin'] = dP_skin

            if (pi - pwf_final_to_use) != 0:
                FE = (pi - pwf_final_to_use - dP_skin) / (pi - pwf_final_to_use)
                results['FE'] = FE

                if FE < 0:
                    validation_warnings.append("⚠️ Negative Flow Efficiency. Check Skin calculation.")
            else:
                st.warning("Cannot calculate FE: (pi - pwf) is zero.")
        except Exception as e:
            st.warning(f"Could not calculate dP_skin or FE: {e}")

    # --- Radius of Investigation ---
    if k > 0 and phi > 0 and Ct > 0 and mu_o > 0:
        try:
            ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))
            results['ri'] = ri
        except Exception as e:
            st.warning(f"Could not calculate ri: {e}")

    # --- Create Horner Plot ---
    fig_horner, ax = plt.subplots(figsize=(10, 7))
    fig_horner.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')

    ax.scatter(df['horner_time'], df['pwsf'], s=60, color='#007bff', edgecolor='black',
               linewidths=0.5, label='All DST Data', zorder=5, alpha=0.6)

    if fit_df is not None and mtr_info is not None:
        ax.scatter(fit_df['horner_time'], fit_df['pwsf'], s=90, color='#dc3545',
                   edgecolor='black', linewidths=1.0, label=f"Auto-Detected MTR (n={mtr_info['num_points']})", zorder=6)

    ax.set_xscale('log')
    ax.invert_xaxis()

    min_ht_plot = 1.0
    max_ht_plot = df['horner_time'].max()
    ax.set_xlim(left=max_ht_plot * 1.5, right=min_ht_plot * 0.9)
    min_y_plot = df['pwsf'].min()
    max_y_plot = max(df['pwsf'].max(), pi)
    y_padding = max((max_y_plot - min_y_plot) * 0.10, 20.0)
    ax.set_ylim(bottom=min_y_plot - y_padding, top=max_y_plot + y_padding)

    x_line_log = np.array([np.log10(max_ht_plot * 1.5), 0])
    y_line = pi + m_slope * x_line_log

    label_r2 = f"R² = {r_squared:.3f}" if r_squared is not None else "R² = N/A"

    ax.plot(10 ** x_line_log, y_line, color='black',
            label=f'MTR (m = {m_abs:.2f}, {label_r2})', zorder=4, linewidth=3.0, alpha=0.8)
    ax.axhline(pi, color='#28a745', linestyle='--',
               label=f'Extrapolated $p_i$ = {pi:.1f} psi', linewidth=2.5, zorder=4)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)
    ax.grid(which='major', linestyle='-', linewidth='0.8', color='#cccccc', alpha=0.9)

    ax.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12, fontweight='bold')
    ax.set_title('Horner Plot Analysis', fontsize=16, fontweight='bold', pad=20)

    legend = ax.legend(loc='lower left', frameon=True, framealpha=0.9, facecolor='white', shadow=True)
    legend.get_frame().set_edgecolor('lightgray')

    ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
    ax.axhline(min_y_plot - y_padding, color='black', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    # --- Create Residuals Plot ---
    fig_residuals, ax_res = plt.subplots(figsize=(10, 5))
    fig_residuals.patch.set_facecolor('#ffffff')
    ax_res.set_facecolor('#f8f9fa')

    ax_res.scatter(df['horner_time'], df['residual'], s=60, color='#007bff',
                   edgecolor='black', linewidths=0.5, label='All Data Residuals', zorder=5, alpha=0.6)

    if fit_df is not None and mtr_info is not None:
        mtr_df_res = df.loc[mtr_info['used_rows']]
        ax_res.scatter(mtr_df_res['horner_time'], mtr_df_res['residual'], s=90, color='#dc3545',
                       edgecolor='black', linewidths=1.0, label='MTR Point Residuals', zorder=6)

    ax_res.axhline(0, color='black', linestyle='--', linewidth=2.0, label='Perfect Fit (Residual = 0)', alpha=0.8)

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

    legend = ax_res.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', shadow=True)
    legend.get_frame().set_edgecolor('lightgray')

    ax_res.grid(which='major', linestyle='-', linewidth='0.8', color='#cccccc', alpha=0.9)
    ax_res.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)

    ax_res.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax_res.axhline(-max_abs_residual - res_padding, color='black', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    df_to_return = df.drop(columns=['dt_calc'])
    return results, fig_horner, df_to_return, mtr_info, fig_residuals, validation_warnings

def get_table_download_link(df):
    """Generate a link to download the processed data as CSV"""
    df_to_save = df.copy()
    if 'dt_calc' in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=['dt_calc'])

    csv = df_to_save.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv">📥 Download Data Table (CSV)</a>'
    return href

def format_metric(value, unit, format_str=":.2f"):
    """Helper function to format metrics"""
    if value is None:
        return "Not Calculated"

    format_spec = "{" + format_str + "}"

    try:
        formatted_value = format_spec.format(value)
    except (ValueError, TypeError):
        formatted_value = str(value)

    return f"{formatted_value} {unit}"

# --- Main Application ---
def main():
    st.markdown('<h1 class="main-header">Interactive DST Horner Plot Analyst 📈</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    A professional web application for Drill Stem Test (DST) analysis using the Horner method.
    Features **smart MTR detection**, **data validation**, and **export capabilities**.
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
    if 'validation_warnings' not in st.session_state:
        st.session_state.validation_warnings = []
    if 'input_params' not in st.session_state:
        st.session_state.input_params = {}

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("📊 Input Parameters")

        # Add example datasets
        st.subheader("Quick Start")
        example_choice = st.selectbox(
            "Load Example Dataset",
            ["None", "Typical Well", "Damaged Well", "Stimulated Well"],
            help="Load pre-configured example data"
        )

        if example_choice == "Typical Well":
            default_h, default_Qo, default_mu = 10.0, 135.0, 1.5
            default_data = """5, 965
10, 1215
15, 1405
20, 1590
25, 1685
30, 1725
35, 1740
40, 1753
45, 1765"""
        elif example_choice == "Damaged Well":
            default_h, default_Qo, default_mu = 15.0, 85.0, 2.0
            default_data = """10, 1200
20, 1450
30, 1600
40, 1700
50, 1760
60, 1800
70, 1825
80, 1840"""
        elif example_choice == "Stimulated Well":
            default_h, default_Qo, default_mu = 12.0, 250.0, 1.2
            default_data = """5, 1100
10, 1300
15, 1425
20, 1510
25, 1565
30, 1600
35, 1620
40, 1635"""
        else:
            default_h, default_Qo, default_mu = 10.0, 135.0, 1.5
            default_data = """5, 965
10, 1215
15, 1405
20, 1590
25, 1685
30, 1725
35, 1740
40, 1753
45, 1765"""

        with st.form(key='input_form'):

            st.subheader("Reservoir & Test Properties")
            st.caption("Enter all known values. Leave at 0 to auto-calculate.")

            col1, col2 = st.columns(2)
            with col1:
                h = st.number_input("Pay Thickness, h (ft)", value=default_h, min_value=0.0, format="%.2f", step=1.0,
                                    help="Essential for k. If 0, will solve for kh.")
                Qo = st.number_input("Flow Rate, Qo (bbl/d)", value=default_Qo, min_value=0.0, format="%.2f", step=1.0,
                                     help="Essential for k and kh.")
                mu_o = st.number_input("Viscosity, μo (cp)", value=default_mu, min_value=0.0, format="%.2f", step=0.1,
                                       help="Essential for k and kh.")
                Bo = st.number_input("FVF, Bo (RB/STB)", value=1.15, min_value=0.0, format="%.3f", step=0.01,
                                     help="Essential for k, kh, and T.")
                phi = st.number_input("Porosity, φ", value=0.10, min_value=0.0, max_value=1.0, format="%.3f", step=0.01,
                                      help="Set to 0 if unknown.")
                Ct = st.number_input("Compressibility, Ct (psi⁻¹)", value=8.4e-6, min_value=0.0, format="%.2e", step=1e-7,
                                     help="Set to 0 if unknown.")
            with col2:
                rw = st.number_input("Wellbore Radius, rw (ft)", value=0.333, min_value=0.0, format="%.3f", step=0.01,
                                     help="Set to 0 if unknown.")
                pwf_final = st.number_input("Final Flow P, Pwf (psi)", value=350.0, min_value=0.0, format="%.1f", step=1.0,
                                            help="Set to 0 if unknown.")
                m_override = st.number_input("Horner Slope 'm'", value=0.0, min_value=0.0, format="%.2f",
                                             help="Override: Leave at 0 to auto-detect.")
                pi_override = st.number_input("Initial Pressure 'pi'", value=0.0, min_value=0.0, format="%.1f",
                                              help="Override: Leave at 0 to auto-extrapolate.")
                k_override = st.number_input("Permeability 'k' (md)", value=0.0, min_value=0.0, format="%.2f",
                                             help="Override: Leave at 0 to auto-calculate.")
                S_override = st.number_input("Skin 'S'", value=0.0, format="%.2f",
                                            help="Override: Enter non-zero value.")

            st.markdown("---")
            st.subheader("DST Test Parameters")
            tp = st.number_input("Total Flow Time, tp (min)", value=60.0, min_value=0.1, format="%.1f",
                                help="Total duration of flow period.",
                                step=1.0)

            mtr_sensitivity = st.slider(
                "MTR Detection Sensitivity (Min R²)",
                min_value=0.950,
                max_value=1.000,
                value=0.995,
                step=0.001,
                format="%.3f",
                help="Lower = less strict. Higher = more strict."
            )

            st.subheader("Pressure Buildup Data")
            dt_unit = st.radio(
                "Input Δt Time Unit",
                ("minutes", "hours"),
                index=0,
                horizontal=True,
                help="Select time unit for Δt column"
            )

            data_text = st.text_area(
                "Shut-in Data (Δt, Pwsf [psi])",
                value=default_data,
                height=200,
                help="Enter one 'time, pressure' pair per line"
            )

            submitted = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

    # --- Perform analysis when form is submitted ---
    if submitted:
        with st.spinner("🔍 Analyzing data..."):
            # Store input parameters for export
            st.session_state.input_params = {
                'h': h, 'Qo': Qo, 'mu_o': mu_o, 'Bo': Bo, 'rw': rw,
                'phi': phi, 'Ct': Ct, 'pwf_final': pwf_final, 'tp': tp,
                'dt_unit': dt_unit
            }

            results, figure, dataframe, mtr_info, fig_residuals, validation_warnings = perform_analysis(
                h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text,
                dt_unit, m_override, pi_override, k_override, S_override, mtr_sensitivity
            )

            st.session_state.results = results
            st.session_state.figure = figure
            st.session_state.figure_residuals = fig_residuals
            st.session_state.dataframe = dataframe
            st.session_state.mtr_info = mtr_info
            st.session_state.validation_warnings = validation_warnings

            if results['m'] is not None:
                st.success("✅ Analysis completed successfully!")
            else:
                st.error("❌ Analysis failed. Check inputs and data.")

    # --- Display validation warnings ---
    if st.session_state.validation_warnings:
        with st.expander("⚠️ Data Quality Warnings", expanded=True):
            for warning in st.session_state.validation_warnings:
                st.markdown(f"- {warning}")

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

            if results['k'] is not None:
                st.metric("Formation Permeability, k", format_metric(results['k'], "md", ":.2f"))
            if results['kh_calc'] is not None:
                st.metric("Flow Capacity, kh", format_metric(results['kh_calc'], "md-ft", ":.1f"))
            if results['transmissibility_calc'] is not None:
                st.metric("Transmissibility, T", format_metric(results['transmissibility_calc'], "md-ft/cp", ":.1f"))

            st.metric("Skin Factor, S", format_metric(results['S'], "", ":.2f"))

            if results['pwf_final_calc'] is not None:
                st.metric("Final Flow P, Pwf (CALC)", format_metric(results['pwf_final_calc'], "psi", ":.1f"))

            st.metric("Pressure Drop (Skin), ΔP_skin", format_metric(results['dP_skin'], "psi", ":.1f"))
            st.metric("Flow Efficiency, FE", format_metric(results['FE'], "", ":.3f"))
            st.metric("Radius of Investigation, rᵢ", format_metric(results['ri'], "ft", ":.1f"))

            r2_val = results['r_squared']
            r2_display = f"{r2_val:.4f}" if r2_val is not None else "N/A"
            st.metric("Regression R-squared", r2_display)

            st.markdown('</div>', unsafe_allow_html=True)

            # MTR Info
            if mtr_info:
                r2_info = st.session_state.mtr_r2_used
                st.info(f"**Auto-MTR:** {mtr_info['num_points']} points, R² = {r2_info:.4f}\n\nΔt range: {mtr_info['end_dt_orig']:.2f} to {mtr_info['start_dt_orig']:.2f} {st.session_state.input_params.get('dt_unit', 'min')}")
            elif m_override > 0:
                st.info(f"**Manual Override:** m = {m_override}, pi = {pi_override}")

            # Interpretation
            st.subheader("📋 Interpretation")
            if results['S'] is not None:
                if results['S'] < -3:
                    skin_interpretation = "✅ Highly stimulated well (excellent)"
                    interpretation_color = "success"
                elif results['S'] < 0:
                    skin_interpretation = "✅ Stimulated well (good)"
                    interpretation_color = "success"
                elif results['S'] < 3:
                    skin_interpretation = "✅ Undamaged well"
                    interpretation_color = "info"
                elif results['S'] < 10:
                    skin_interpretation = "⚠️ Damaged well"
                    interpretation_color = "warning"
                else:
                    skin_interpretation = "❌ Severely damaged well"
                    interpretation_color = "error"

                if interpretation_color == "success":
                    st.success(f"**Skin:** {skin_interpretation}")
                elif interpretation_color == "warning":
                    st.warning(f"**Skin:** {skin_interpretation}")
                elif interpretation_color == "error":
                    st.error(f"**Skin:** {skin_interpretation}")
                else:
                    st.info(f"**Skin:** {skin_interpretation}")
            else:
                st.info("**Skin:** Not calculated (missing parameters)")

            if results['FE'] is not None:
                if results['FE'] > 1.0:
                    fe_interpretation = "✅ Well is stimulated"
                elif results['FE'] > 0.8:
                    fe_interpretation = "✅ Good flow efficiency"
                elif results['FE'] > 0.5:
                    fe_interpretation = "⚠️ Moderate damage"
                else:
                    fe_interpretation = "❌ Poor flow efficiency"
                st.write(f"**Flow Efficiency:** {fe_interpretation}")
            else:
                st.info("**Flow Efficiency:** Not calculated")

            # Export buttons
            st.markdown("---")
            st.subheader("📥 Export Results")

            if st.session_state.input_params:
                report_text = export_results_to_txt(
                    results,
                    mtr_info,
                    st.session_state.input_params
                )

                # Text report download
                b64_report = base64.b64encode(report_text.encode()).decode()
                href_report = f'<a href="data:text/plain;base64,{b64_report}" download="DST_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">📄 Download Text Report</a>'
                st.markdown(href_report, unsafe_allow_html=True)

                # Quick copy section
                with st.expander("📋 Quick Copy Results"):
                    copy_to_clipboard_button(report_text, "Select and copy results above")

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
                st.markdown("---")
                st.markdown(get_plot_download_link(st.session_state.figure, "Horner_Plot.png"), unsafe_allow_html=True)
            else:
                st.info("The Horner plot will appear here after analysis")

        with tab2:
            if st.session_state.figure_residuals:
                st.pyplot(st.session_state.figure_residuals, dpi=300)
                st.markdown("---")
                st.markdown(get_plot_download_link(st.session_state.figure_residuals, "Residuals_Plot.png"), unsafe_allow_html=True)
                st.markdown("""
                **How to read this plot:**
                * Shows difference between actual and predicted pressure
                * Points near zero line = good MTR fit
                * Red points = MTR points used in regression
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
                df_cols = [col for col in columns_order if col in df.columns]
                df = df[df_cols]

                st.dataframe(df.style.format({
                    'dt': '{:.2f}',
                    'pwsf': '{:.1f}',
                    'horner_time': '{:.2f}',
                    'log_horner_time': '{:.3f}',
                    'predicted_pwsf': '{:.1f}',
                    'residual': '{:.2f}',
                }), use_container_width=True)

                st.markdown("---")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            else:
                st.info("The data table will appear here")

        with tab4:
            st.subheader("Analysis Methodology")
            st.markdown("""
            This app uses the Horner method for DST analysis with the following key features:
            
            **Smart MTR Detection (V9.1):**
            - Automatically finds the longest straight-line segment
            - Prioritizes line length over perfect R² to avoid short segments
            - Ignores wellbore storage and boundary effects
            
            **Modular Calculations:**
            - Auto-calculates k, kh, or T depending on available parameters
            - Can solve for Pwf if Skin is provided
            - Handles partial parameter sets gracefully
            
            **Data Validation:**
            - Checks for physically impossible values
            - Warns about data quality issues
            - Validates parameter consistency
            """)

            st.markdown("---")
            st.subheader("Formula Key")
            st.markdown("**Horner Time:**")
            st.latex(r"\text{Horner Time} = \frac{t_p + \Delta t}{\Delta t}")

            st.markdown("**Horner Slope (m):**")
            st.latex(r"m = \frac{P_{ws_1} - P_{ws_{10}}}{\log(10) - \log(1)}")

            st.markdown("**Permeability:**")
            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")

            st.markdown("**Skin Factor:**")
            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")

        with tab5:
            st.subheader("Key Formulas")
            st.latex(r"p_{ws} = p_i - m \log\left(\frac{t_p + \Delta t}{\Delta t}\right)")
            st.caption("Horner Equation")

            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")
            st.caption("Permeability")

            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
            st.caption("Skin Factor")

            st.latex(r"\Delta P_{skin} = \frac{141.2 \cdot Q_o \cdot \mu_o \cdot B_o}{k \cdot h} \cdot S")
            st.caption("Pressure Drop due to Skin")

            st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
            st.caption("Flow Efficiency")

            st.latex(r"r_i = \sqrt{\frac{k \cdot t_p}{57600 \cdot \phi \cdot \mu_o \cdot C_t}}")
            st.caption("Radius of Investigation")

    st.markdown("---")
    st.markdown(
        "**DST Horner Analyst V10.0** • Enhanced with exports, validation & better UX • "
        "Built with Python 🐍 and Streamlit"
    )

if __name__ == "__main__":
    main()
