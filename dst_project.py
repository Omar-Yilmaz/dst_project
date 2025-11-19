"""
Enhanced Interactive DST Horner Plot Analyst (Smart Auto MTR Detection)
Professional web application for Drill Stem Test analysis
V10.4 - Complete Restoration with Correct Header (No Icon, a.jpg Right)
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64
import os
from datetime import datetime

# --- Page configuration ---
st.set_page_config(
    page_title="DST Horner Analyst (Smart-Fit)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper: Robust Image Finder ---
def find_image_path(keywords):
    """
    Searches the current directory for an image file that matches ANY of the keywords.
    Returns the filename if found, else None.
    """
    try:
        files = os.listdir('.')
        # Priority check for exact matches first
        for k in keywords:
            if k in files: return k

        # Fuzzy search
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                for k in keywords:
                    if k.lower() in f.lower():
                        return f
    except Exception:
        pass
    return None

# --- Data Validation Functions ---
def validate_pressure_data(df):
    """Validate pressure buildup data for physical consistency"""
    warnings = []
    if len(df) < 2: return warnings

    pressure_diff = df['pwsf'].diff()
    decreasing_points = (pressure_diff < -5).sum()

    if decreasing_points > len(df) * 0.2:
        warnings.append("⚠️ Pressure appears to decrease during buildup. Check data quality.")
    if df['pwsf'].min() < 0:
        warnings.append("❌ Negative pressure values detected. Data may be invalid.")
    if df['pwsf'].max() > 20000:
        warnings.append("⚠️ Extremely high pressures detected (>20,000 psi). Verify data.")
    if df['dt'].duplicated().any():
        warnings.append("⚠️ Duplicate time points detected. Consider averaging or removing.")

    if len(df) > 1:
        time_ratios = df['dt'].iloc[1:].values / df['dt'].iloc[:-1].values
        if (time_ratios > 10).any():
            warnings.append("⚠️ Large gaps in time spacing detected. May affect MTR detection.")

    return warnings

def check_parameter_consistency(h, Qo, mu_o, Bo, phi, Ct, rw, pwf_final, pi):
    """Check if parameters are physically reasonable"""
    warnings = []
    if h > 1000: warnings.append("⚠️ Very thick pay zone (>1000 ft). Verify h value.")
    if Qo > 10000: warnings.append("⚠️ Very high flow rate (>10,000 bbl/d). Verify Qo.")
    if mu_o < 0.1 or mu_o > 1000: warnings.append("⚠️ Unusual viscosity value. Typical range: 0.1-1000 cp.")
    if Bo < 1.0 or Bo > 3.0: warnings.append("⚠️ Unusual FVF value. Typical range: 1.0-3.0 RB/STB.")
    if phi > 0.4: warnings.append("⚠️ Very high porosity (>40%). Verify φ value.")
    if pwf_final > pi and pi > 0: warnings.append("❌ Final flowing pressure > Initial pressure. Data inconsistency.")
    return warnings

# --- Export Functions ---
def export_results_to_txt(results, mtr_info, input_params):
    """Generate a formatted text report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r2_text = f"{results['r_squared']:.4f}" if results['r_squared'] is not None else "N/A"

    # Format optional results
    k_text = f"{results['k']:.2f} md" if results['k'] else "Not calculated"
    kh_text = f"{results['kh_calc']:.1f} md-ft" if results['kh_calc'] else "Not calculated"
    t_text = f"{results['transmissibility_calc']:.1f} md-ft/cp" if results['transmissibility_calc'] else "Not calculated"

    # Handle None values safely
    def safe_fmt(val, fmt="{:.2f}"):
        return fmt.format(val) if val is not None else "N/A"

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
Horner Slope (m):           {safe_fmt(results['m'])} psi/cycle
Initial Pressure (pi):      {safe_fmt(results['pi'], "{:.1f}")} psi
R-squared:                  {r2_text}

Permeability (k):           {k_text}
Flow Capacity (kh):         {kh_text}
Transmissibility (T):       {t_text}

Skin Factor (S):            {safe_fmt(results['S'])}
Pressure Drop (ΔP_skin):    {safe_fmt(results['dP_skin'], "{:.1f}")} psi
Flow Efficiency (FE):       {safe_fmt(results['FE'], "{:.3f}")}
Damage Ratio (DR):          {safe_fmt(results['DR'], "{:.3f}")}
Productivity Index (PI):    {safe_fmt(results['PI'], "{:.3f}")} bbl/d/psi
Radius of Investigation:    {safe_fmt(results['ri'], "{:.1f}")} ft

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

def generate_excel_download(results, mtr_info, input_params, df):
    """Generate an Excel file with parameters and results"""
    output = io.BytesIO()

    # Create DataFrames for Parameters and Results
    param_data = {k: [v] for k, v in input_params.items()}
    param_df = pd.DataFrame(param_data).T.reset_index()
    param_df.columns = ['Parameter', 'Value']

    res_data = {k: [v] for k, v in results.items()}
    res_df = pd.DataFrame(res_data).T.reset_index()
    res_df.columns = ['Metric', 'Value']

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write sheets
            res_df.to_excel(writer, sheet_name='Results Summary', index=False)
            param_df.to_excel(writer, sheet_name='Input Parameters', index=False)

            # Process main data
            df_out = df.copy()
            if 'dt_calc' in df_out.columns:
                df_out = df_out.drop(columns=['dt_calc'])
            df_out.to_excel(writer, sheet_name='Pressure Data', index=False)

        return output.getvalue()
    except Exception as e:
        return None

def get_plot_download_link(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;">📥 Download {filename}</a>'
    return href

def copy_to_clipboard_button(text, label="📋 Copy Results"):
    st.code(text, language=None)
    st.caption(f"☝️ {label} - Select and copy the text above")

# --- Smart Auto MTR detection function ---
def find_best_mtr(df, min_points=3, min_r_squared_threshold=0.995):
    best_regression = None
    best_fit_df = None
    best_num_points = 0
    best_r_squared = 0.0
    n = len(df)

    # Logic: Capture the longest segment that meets the R2 threshold
    for i in range(n - min_points + 1):
        for j in range(i + min_points, n + 1):
            fit_df_loop = df.iloc[i:j].copy()
            num_points_in_loop = len(fit_df_loop)
            try:
                regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)
                if not np.isfinite(regression.slope): continue
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

    # Fallback: If no segment meets strict threshold, find best available based on a score
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

                    # Score prioritizes length but penalizes bad fit
                    score = num_points_in_loop * (r_squared**2)
                    current_best_score = best_num_points * (best_r_squared**2)

                    if score > current_best_score:
                        best_num_points = num_points_in_loop
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
    results = {
        'm': None, 'pi': None,
        'k': None, 'kh_calc': None, 'transmissibility_calc': None,
        'S': None, 'FE': None, 'ri': None, 'r_squared': None, 'dP_skin': None,
        'pwf_final_calc': None, 'PI': None, 'DR': None
    }
    mtr_info = None
    fig_horner = None
    fig_residuals = None
    df = None
    fit_df = None
    k = 0
    pwf_final_to_use = pwf_final
    validation_warnings = []

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

    try:
        data_io = io.StringIO(data_text)
        df = pd.read_csv(data_io, sep=r'[,\s]+', engine='python', header=None, names=['dt', 'pwsf'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 3:
            st.error(f"❌ Need at least 3 valid data points (found {len(df)}).")
            return results, fig_horner, df, mtr_info, fig_residuals, validation_warnings
        data_warnings = validate_pressure_data(df)
        validation_warnings.extend(data_warnings)
    except Exception as e:
        st.error(f"❌ Data parsing error: {str(e)}")
        return results, fig_horner, df, mtr_info, fig_residuals, validation_warnings

    if dt_unit == "hours":
        st.info("ℹ️ Converting Δt from hours to minutes.")
        df['dt_calc'] = df['dt'] * 60.0
    else:
        df['dt_calc'] = df['dt']

    delta_t = df['dt_calc'].values
    pwsf = df['pwsf'].values
    horner_time = (tp + delta_t) / delta_t
    log_horner_time = np.log10(horner_time)
    df['horner_time'] = horner_time
    df['log_horner_time'] = log_horner_time
    df = df.sort_values(by='log_horner_time', ascending=True).reset_index(drop=True)

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
        if r_squared < 0.99:
            validation_warnings.append(f"⚠️ MTR fit quality is moderate (R² = {r_squared:.4f}). Consider data quality.")

    results['m'] = m_abs
    results['pi'] = pi
    results['r_squared'] = r_squared
    df['predicted_pwsf'] = pi + m_slope * df['log_horner_time']
    df['residual'] = df['pwsf'] - df['predicted_pwsf']

    param_warnings = check_parameter_consistency(h, Qo, mu_o, Bo, phi, Ct, rw, pwf_final, pi)
    validation_warnings.extend(param_warnings)

    # --- Modular Calculations (Restored from original 123.py) ---
    if k_override > 0:
        st.info(f"ℹ️ Using manual k = {k_override} md")
        k = k_override
        results['k'] = k
    elif h > 0 and Qo > 0 and mu_o > 0 and Bo > 0:
        try:
            k = (162.6 * (Qo * mu_o * Bo)) / (m_abs * h)
            results['k'] = k
            if k > 10000: validation_warnings.append("⚠️ Very high permeability (>10,000 md). Verify calculation.")
            elif k < 0.01: validation_warnings.append("⚠️ Very low permeability (<0.01 md). Verify calculation.")
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
                if FE > 0:
                    results['DR'] = 1.0 / FE
                results['PI'] = Qo / (pi - pwf_final_to_use)

                if FE < 0: validation_warnings.append("⚠️ Negative Flow Efficiency. Check Skin calculation.")
            else:
                st.warning("Cannot calculate FE: (pi - pwf) is zero.")
        except Exception as e:
            st.warning(f"Could not calculate dP_skin or FE: {e}")

    if k > 0 and phi > 0 and Ct > 0 and mu_o > 0:
        try:
            ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))
            results['ri'] = ri
        except Exception as e:
            st.warning(f"Could not calculate ri: {e}")

    # --- Plots with improved style ---
    # Use a style context that mimics the requested 'whitegrid' look
    with plt.style.context('seaborn-v0_8-whitegrid'):

        # Horner Plot
        fig_horner, ax = plt.subplots(figsize=(12, 8)) # Increased size

        # Plot Data
        ax.scatter(df['horner_time'], df['pwsf'], s=90, color='#3498db', edgecolor='white', linewidths=0.8, label='All Data', zorder=5, alpha=0.8)

        if fit_df is not None and mtr_info is not None:
            ax.scatter(fit_df['horner_time'], fit_df['pwsf'], s=120, color='#e74c3c', edgecolor='white', linewidths=1.0, label=f"MTR (n={mtr_info['num_points']})", zorder=6)

        ax.set_xscale('log')
        ax.invert_xaxis()

        # Axis Limits
        min_ht_plot = 1.0
        max_ht_plot = df['horner_time'].max()
        ax.set_xlim(left=max_ht_plot * 1.5, right=min_ht_plot * 0.9)

        min_y_plot = df['pwsf'].min()
        max_y_plot = max(df['pwsf'].max(), pi)
        y_padding = max((max_y_plot - min_y_plot) * 0.10, 20.0)
        ax.set_ylim(bottom=min_y_plot - y_padding, top=max_y_plot + y_padding)

        # Regression Line
        x_line_log = np.array([np.log10(max_ht_plot * 1.5), 0])
        y_line = pi + m_slope * x_line_log
        label_r2 = f"R² = {r_squared:.3f}" if r_squared is not None else "R² = N/A"

        # Annotate the equation of the line
        eq_text = f"Pws = {pi:.1f} - {m_abs:.2f} log(HT)"
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.plot(10 ** x_line_log, y_line, color='#2c3e50', label=f'Regression Line', zorder=4, linewidth=2.5, alpha=0.8)

        # Highlight Pi (Initial Pressure) line only, as requested
        ax.axhline(pi, color='#2ecc71', linestyle='--', label=f'Extrapolated Pi = {pi:.1f} psi', linewidth=2.0, zorder=4)

        # Styling
        ax.set_xlabel('Horner Time $(t_p + \Delta t) / \Delta t$', fontsize=13, fontweight='bold')
        ax.set_ylabel('Pressure (psi)', fontsize=13, fontweight='bold')
        ax.set_title('Horner Plot Analysis', fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc', fontsize=11)
        ax.grid(True, which='major', linestyle='-', alpha=0.8)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)

        plt.tight_layout()

        # Residuals Plot
        fig_residuals, ax_res = plt.subplots(figsize=(12, 5)) # Increased size

        ax_res.scatter(df['horner_time'], df['residual'], s=70, color='#3498db', edgecolor='white', linewidths=0.5, label='Residuals', zorder=5, alpha=0.7)

        if fit_df is not None and mtr_info is not None:
            mtr_df_res = df.loc[mtr_info['used_rows']]
            ax_res.scatter(mtr_df_res['horner_time'], mtr_df_res['residual'], s=100, color='#e74c3c', edgecolor='white', linewidths=1.0, label='MTR Points', zorder=6)

        ax_res.axhline(0, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_res.set_xscale('log')
        ax_res.invert_xaxis()
        ax_res.set_xlim(left=max_ht_plot * 1.1, right=min_ht_plot * 0.9)

        max_abs_residual = df['residual'].abs().max()
        if not np.isfinite(max_abs_residual) or max_abs_residual == 0: max_abs_residual = 1.0
        res_padding = max(max_abs_residual * 0.15, 5.0)
        ax_res.set_ylim(-max_abs_residual - res_padding, max_abs_residual + res_padding)

        ax_res.set_xlabel('Horner Time', fontsize=12, fontweight='bold')
        ax_res.set_ylabel('Residual (psi)', fontsize=12, fontweight='bold')
        ax_res.set_title('Regression Residuals', fontsize=13, fontweight='bold')
        ax_res.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
        ax_res.grid(True, which='major', linestyle='-', alpha=0.8)
        ax_res.grid(True, which='minor', linestyle=':', alpha=0.4)

        plt.tight_layout()

    df_to_return = df.drop(columns=['dt_calc'])
    return results, fig_horner, df_to_return, mtr_info, fig_residuals, validation_warnings

def get_table_download_link(df):
    df_to_save = df.copy()
    if 'dt_calc' in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=['dt_calc'])
    csv = df_to_save.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv" style="text-decoration:none;">📥 Download Data Table (CSV)</a>'
    return href

def format_metric(value, unit, format_str=":.2f"):
    if value is None: return "Not Calculated"
    format_spec = "{" + format_str + "}"
    try: formatted_value = format_spec.format(value)
    except (ValueError, TypeError): formatted_value = str(value)
    return f"{formatted_value} {unit}"

# --- Main Application ---
def main():
    try:
        # ---------------------------------------------------------
        # 🛠️ HEADER CONFIGURATION - EDIT EVERYTHING HERE 🛠️
        # ---------------------------------------------------------
        CONFIG = {
            # --- Text Content ---
            "TITLE": "DST Horner Plot Analyst",
            # Subtitle split into two lines for cleaner visual matching the image
            "SUBTITLE_LINE1": "University of Kirkuk | College of Engineering",
            "SUBTITLE_LINE2": "Petroleum Engineering Department",

            "DEVELOPERS": "Developed by: Bilal Rabah & Omar Yilmaz",
            "SUPERVISOR": "Supervised by: Lec. Mohammed Yashar",
            "DATE": "Date: November 2025",
            "ICON": None, # ICON REMOVED COMPLETELY - CLEAN TEXT ONLY

            # --- Visual Colors (Hex Codes) ---
            "BG_GRADIENT_1": "#1F4E78",      # Dark Blue
            "BG_GRADIENT_2": "#2c5f8d",      # Lighter Blue
            "TEXT_COLOR": "#ffffff",         # White

            # --- Image Paths ---
            "LEFT_LOGO_QUERY": ['eng', 'logo', 'triangle'],
            "RIGHT_LOGO_QUERY": ['anniversary', '22', 'right', 'a.png']
        }
        # ---------------------------------------------------------

        # --- DYNAMIC CSS INJECTION ---
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

            /* Main Blue Header Card */
            .blue-header-card {{
                background: linear-gradient(135deg, {CONFIG['BG_GRADIENT_1']} 0%, {CONFIG['BG_GRADIENT_2']} 100%);
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                color: {CONFIG['TEXT_COLOR']};
                margin-bottom: 2rem;
            }}
            
            /* REMOVED header-icon STYLE */
            
            .header-title {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 2.2rem;
                font-weight: 800;
                margin: 0;
                color: {CONFIG['TEXT_COLOR']};
                text-shadow: 0px 2px 4px rgba(0,0,0,0.2);
            }}
            
            .header-subtitle {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 1.1rem;
                font-weight: 500;
                margin-top: 0.5rem;
                color: #e0e0e0;
                line-height: 1.4;
            }}
            
            .header-dev {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 0.85rem;
                font-style: italic;
                margin-top: 1rem;
                color: #b0c4de;
                border-top: 1px solid rgba(255,255,255,0.2);
                padding-top: 0.5rem;
            }}
            
            /* Result Boxes */
            .result-container {{
                background-color: #ffffff !important;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 5px solid {CONFIG['BG_GRADIENT_1']};
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            /* Metric Styling */
            div[data-testid="stMetricValue"] {{
                font-weight: 700;
                color: {CONFIG['BG_GRADIENT_1']} !important;
                font-family: 'Segoe UI', sans-serif;
            }}
            
            div[data-testid="stMetricLabel"] {{
                font-weight: 600;
                color: #5D6D7E !important;
                font-family: 'Segoe UI', sans-serif;
            }}
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                height: 3rem;
                white-space: pre-wrap;
                border-radius: 4px 4px 0 0;
                padding-left: 1rem;
                padding-right: 1rem;
                font-weight: 600;
            }}
            
            div[data-testid="stImage"] {{
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }}
        </style>
        """, unsafe_allow_html=True)

        # --- HEADER SECTION (Dynamic with Config) ---
        c1, c2, c3 = st.columns([1, 3, 1])

        with c1:
            eng_path = find_image_path(CONFIG['LEFT_LOGO_QUERY'])
            if eng_path: st.image(eng_path, use_container_width=True)

        with c2:
            # CLEAN HEADER WITHOUT ICON
            st.markdown(f"""
                <div class="blue-header-card">
                    <div class="header-title">{CONFIG['TITLE']}</div>
                    <div class="header-subtitle">
                        {CONFIG['SUBTITLE_LINE1']}<br>
                        {CONFIG['SUBTITLE_LINE2']}
                    </div>
                    <div class="header-dev">
                        {CONFIG['DEVELOPERS']} | {CONFIG['SUPERVISOR']}<br>
                        {CONFIG['DATE']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            # Explicitly use the correct keyword or file for the right logo
            a_path = find_image_path(CONFIG['RIGHT_LOGO_QUERY'])
            if a_path: st.image(a_path, use_container_width=True)

        # --- Initialize session state ---
        if 'results' not in st.session_state: st.session_state.results = None
        if 'figure' not in st.session_state: st.session_state.figure = None
        if 'figure_residuals' not in st.session_state: st.session_state.figure_residuals = None
        if 'dataframe' not in st.session_state: st.session_state.dataframe = None
        if 'mtr_info' not in st.session_state: st.session_state.mtr_info = None
        if 'mtr_r2_used' not in st.session_state: st.session_state.mtr_r2_used = None
        if 'validation_warnings' not in st.session_state: st.session_state.validation_warnings = []
        if 'input_params' not in st.session_state: st.session_state.input_params = {}

        # --- SIDEBAR INPUTS ---
        with st.sidebar:
            st.title("⚙️ Configuration")

            with st.expander("📄 About & Help", expanded=False):
                st.markdown("""
                **DST Horner Analyst**
                
                A professional tool for interpreting Drill Stem Test (DST) pressure buildup data using the Horner approximation method.
                
                **Features:**
                * Automatic Middle Time Region (MTR) detection
                * Calculation of Permeability, Skin, and Flow Efficiency
                * Quality control checks
                
                **Authors:** Bilal Rabah & Omar Yilmaz
                """)

            st.markdown("---")
            st.header("1. Data Input")

            example_choice = st.selectbox(
                "Load Example Dataset",
                ["None", "Typical Well", "Damaged Well", "Stimulated Well"],
                help="Load pre-configured example data"
            )

            if example_choice == "Typical Well":
                default_h, default_Qo, default_mu = 10.0, 135.0, 1.5
                default_data = "5, 965\n10, 1215\n15, 1405\n20, 1590\n25, 1685\n30, 1725\n35, 1740\n40, 1753\n45, 1765"
            elif example_choice == "Damaged Well":
                default_h, default_Qo, default_mu = 15.0, 85.0, 2.0
                default_data = "10, 1200\n20, 1450\n30, 1600\n40, 1700\n50, 1760\n60, 1800\n70, 1825\n80, 1840"
            elif example_choice == "Stimulated Well":
                default_h, default_Qo, default_mu = 12.0, 250.0, 1.2
                default_data = "5, 1100\n10, 1300\n15, 1425\n20, 1510\n25, 1565\n30, 1600\n35, 1620\n40, 1635"
            else:
                default_h, default_Qo, default_mu = 10.0, 135.0, 1.5
                default_data = "5, 965\n10, 1215\n15, 1405\n20, 1590\n25, 1685\n30, 1725\n35, 1740\n40, 1753\n45, 1765"

            with st.form(key='input_form'):
                dt_unit = st.radio("Time Unit for Δt", ("minutes", "hours"), horizontal=True)
                data_text = st.text_area("Shut-in Data (Δt, Pwsf)", value=default_data, height=150, help="Paste data here. Format: Time, Pressure")

                st.markdown("---")
                st.header("2. Reservoir Parameters")

                col1, col2 = st.columns(2)
                with col1:
                    h = st.number_input("h (ft)", value=default_h, format="%.2f", help="Pay Thickness")
                    Qo = st.number_input("Qo (bbl/d)", value=default_Qo, format="%.2f", help="Oil Flow Rate")
                    mu_o = st.number_input("μo (cp)", value=default_mu, format="%.2f", help="Oil Viscosity")
                    Bo = st.number_input("Bo (RB/STB)", value=1.15, format="%.3f", help="Formation Volume Factor")
                with col2:
                    rw = st.number_input("rw (ft)", value=0.333, format="%.3f", help="Wellbore Radius")
                    phi = st.number_input("φ (porosity)", value=0.10, format="%.3f", help="Porosity (fraction)")
                    Ct = st.number_input("Ct (psi⁻¹)", value=8.4e-6, format="%.2e", help="Total Compressibility")
                    pwf_final = st.number_input("Pwf (psi)", value=350.0, format="%.1f", help="Final Flowing Pressure")

                st.markdown("---")
                st.header("3. Test Settings")
                tp = st.number_input("Producing Time tp (min)", value=60.0, format="%.1f")

                with st.expander("Advanced Settings"):
                    mtr_sensitivity = st.slider("MTR Detection Sensitivity", 0.950, 1.000, 0.995, 0.001, format="%.3f", help="Higher values require straighter lines")
                    m_override = st.number_input("Override Slope (m)", value=0.0, format="%.2f")
                    pi_override = st.number_input("Override Pi (psi)", value=0.0, format="%.1f")
                    k_override = st.number_input("Override Permeability (k)", value=0.0, format="%.2f")
                    S_override = st.number_input("Override Skin (S)", value=0.0, format="%.2f")

                submitted = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

        if submitted:
            with st.spinner("🔍 Analyzing pressure data..."):
                st.session_state.input_params = {
                    'h': h, 'Qo': Qo, 'mu_o': mu_o, 'Bo': Bo, 'rw': rw,
                    'phi': phi, 'Ct': Ct, 'pwf_final': pwf_final, 'tp': tp, 'dt_unit': dt_unit
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

        if st.session_state.validation_warnings:
            with st.expander("⚠️ Data Quality Warnings", expanded=True):
                for warning in st.session_state.validation_warnings: st.markdown(f"- {warning}")

        # --- MAIN CONTENT ---
        if st.session_state.results:
            # Use Full Width Container for everything now

            st.markdown("### 📈 Analysis Results")
            results = st.session_state.results
            mtr_info = st.session_state.mtr_info

            # SINGLE COLUMN METRICS (As requested)
            # We use a white container and list them clearly
            st.markdown('<div class="result-container">', unsafe_allow_html=True)

            # To make it a "column" but still look good, we can use st.columns(1) effectively
            # or just list them. But standard st.metric in one column takes a lot of vertical space.
            # The user asked for "make this a coloume". This usually means vertical stacking.

            st.metric("Slope 'm'", format_metric(results['m'], "psi/cycle", ":.2f"), help="Horner semi-log slope")
            st.metric("Initial Pressure (p*)", format_metric(results['pi'], "psi", ":.1f"), help="Extrapolated pressure at infinite shut-in time")

            if results['k'] is not None: st.metric("Permeability (k)", format_metric(results['k'], "md", ":.2f"))
            if results['kh_calc'] is not None: st.metric("Flow Capacity (kh)", format_metric(results['kh_calc'], "md-ft", ":.1f"))

            if results['transmissibility_calc'] is not None:
                st.metric("Transmissibility", format_metric(results['transmissibility_calc'], "md-ft/cp", ":.1f"))

            st.metric("Skin Factor (S)", format_metric(results['S'], "", ":.2f"))
            st.metric("Flow Efficiency (FE)", format_metric(results['FE'], "", ":.3f"))
            st.metric("Damage Ratio (DR)", format_metric(results['DR'], "", ":.3f"))
            st.metric("Productivity Index (PI)", format_metric(results['PI'], "bbl/d/psi", ":.3f"))
            st.metric("Radius of Inv. (ri)", format_metric(results['ri'], "ft", ":.1f"))

            r2_display = f"{results['r_squared']:.4f}" if results['r_squared'] is not None else "N/A"
            st.metric("Fit Quality (R²)", r2_display)

            st.markdown('</div>', unsafe_allow_html=True)

            if mtr_info:
                r2_info = st.session_state.mtr_r2_used
                st.info(f"**Auto-MTR:** {mtr_info['num_points']} points used. Range: {mtr_info['end_dt_orig']:.2f} to {mtr_info['start_dt_orig']:.2f} {st.session_state.input_params.get('dt_unit', 'min')}")
            elif m_override > 0:
                st.info(f"**Manual Override:** Analysis forced with m = {m_override}")

            st.subheader("📋 Interpretation")
            if results['S'] is not None:
                if results['S'] < -3: skin_interp, color = "✅ Highly stimulated well (Excellent)", "success"
                elif results['S'] < 0: skin_interp, color = "✅ Stimulated well (Good)", "success"
                elif results['S'] < 3: skin_interp, color = "✅ Undamaged well (Normal)", "info"
                elif results['S'] < 10: skin_interp, color = "⚠️ Damaged well (Moderate)", "warning"
                else: skin_interp, color = "❌ Severely damaged well (High)", "error"

                msg = f"**Skin Factor ({results['S']:.2f}):** {skin_interp}"
                if color == "success": st.success(msg)
                elif color == "warning": st.warning(msg)
                elif color == "error": st.error(msg)
                else: st.info(msg)

            st.markdown("---")
            st.subheader("📥 Export Reports")
            if st.session_state.input_params:
                # Text Report
                report_text = export_results_to_txt(results, mtr_info, st.session_state.input_params)
                b64_report = base64.b64encode(report_text.encode()).decode()
                href_report = f'<a href="data:text/plain;base64,{b64_report}" download="DST_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt" style="text-decoration:none; background-color:#f0f2f6; padding:8px; border-radius:5px; display:block; text-align:center; margin-bottom:10px;">📄 Download Text Report</a>'
                st.markdown(href_report, unsafe_allow_html=True)

                # Excel Report
                excel_data = generate_excel_download(results, mtr_info, st.session_state.input_params, st.session_state.dataframe)
                if excel_data:
                    b64_excel = base64.b64encode(excel_data).decode()
                    href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="DST_Analysis_{datetime.now().strftime("%Y%m%d")}.xlsx" style="text-decoration:none; background-color:#28a745; color:white; padding:8px; border-radius:5px; display:block; text-align:center;">📊 Download Excel Report</a>'
                    st.markdown(href_excel, unsafe_allow_html=True)

                with st.expander("📋 Quick Copy"):
                    copy_to_clipboard_button(report_text, "Copy Report")

            # --- FULL WIDTH PLOTS BELOW RESULTS ---
            st.markdown("---")
            st.markdown("### 📊 Analysis Plots")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Horner Plot", "📉 Residuals", "📥 Data Table", "⚙️ Methodology", "🧪 Formulas"])

            with tab1:
                if st.session_state.figure:
                    st.pyplot(st.session_state.figure, dpi=300)
                    st.markdown(get_plot_download_link(st.session_state.figure, "Horner_Plot.png"), unsafe_allow_html=True)

            with tab2:
                if st.session_state.figure_residuals:
                    st.pyplot(st.session_state.figure_residuals, dpi=300)
                    st.markdown(get_plot_download_link(st.session_state.figure_residuals, "Residuals_Plot.png"), unsafe_allow_html=True)
                    st.info("Residuals show the deviation of data points from the straight-line fit. Random scatter indicates a good fit.")

            with tab3:
                if st.session_state.dataframe is not None:
                    st.subheader("Processed Data")
                    df = st.session_state.dataframe.copy()
                    mtr_rows = st.session_state.mtr_info['used_rows'] if st.session_state.mtr_info else []
                    df['MTR'] = ["✅" if i in df.index and mtr_rows and i in mtr_rows else "" for i in df.index]
                    cols = ['dt', 'pwsf', 'horner_time', 'log_horner_time', 'predicted_pwsf', 'residual', 'MTR']
                    df_display = df[[c for c in cols if c in df.columns]]
                    st.dataframe(df_display.style.format({'dt':'{:.2f}','pwsf':'{:.1f}','horner_time':'{:.2f}','log_horner_time':'{:.3f}','predicted_pwsf':'{:.1f}','residual':'{:.2f}'}), use_container_width=True)
                    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            with tab4:
                st.markdown("""
                ### Analysis Methodology
                
                **1. Horner Time**
                The analysis uses the superposition time function (Horner time) to linearize the pressure buildup equation:
                $HT = (t_p + \Delta t) / \Delta t$
                
                **2. Smart MTR Detection**
                This application uses an advanced algorithm to identify the **Middle Time Region (MTR)**—the straight-line portion of the semi-log plot representing infinite-acting radial flow.
                * **Scanning:** Iterates through all valid sub-segments of the data.
                * **Criteria:** Looks for the *longest* continuous segment with a linearity coefficient ($R^2$) > 0.995.
                * **Fallback:** If data is noisy, it gracefully degrades to finding the best available statistical fit.
                
                **3. Parameter Estimation**
                * **Permeability (k):** Calculated from the slope ($m$) of the MTR.
                * **Skin (S):** Calculated using the intercept ($p_{1hr}$) at Horner Time = 1.
                """)

            with tab5:
                st.subheader("Key Formulas")
                st.markdown("**Slope (m):**")
                st.latex(r"m = \frac{p_{ws_2} - p_{ws_1}}{\log(HT_2) - \log(HT_1)}")
                st.markdown("**Horner Equation:**")
                st.latex(r"p_{ws} = p_i - m \log\left(\frac{t_p + \Delta t}{\Delta t}\right)")
                st.markdown("**Permeability (k):**")
                st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")
                st.markdown("**Skin Factor (S):**")
                st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
                st.markdown("**Pressure Drop (Skin):**")
                st.latex(r"\Delta P_{skin} = \frac{141.2 \cdot Q_o \cdot \mu_o \cdot B_o}{k \cdot h} \cdot S")
                st.markdown("**Flow Efficiency (FE):**")
                st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
                st.markdown("**Damage Ratio (DR):**")
                st.latex(r"DR = \frac{1}{FE} = \frac{p_i - p_{wf}}{p_i - p_{wf} - \Delta p_{skin}}")
                st.markdown("**Productivity Index (PI):**")
                st.latex(r"PI = \frac{Q_o}{p_i - p_{wf}}")
                st.markdown("**Radius of Investigation (ri):**")
                st.latex(r"r_i = \sqrt{\frac{k \cdot t_p}{57600 \cdot \phi \cdot \mu_o \cdot C_t}}")
        else:
            # Empty state
            st.info("👈 Please load data and click **Run Analysis** in the sidebar to begin.")

            # Show a placeholder image or guide
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                ### Welcome to DST Horner Analyst
                
                This tool helps you interpret pressure buildup tests quickly and accurately.
                
                **Getting Started:**
                1.  Select an **Example Dataset** in the sidebar OR paste your own data.
                2.  Enter the **Reservoir Parameters** (pay thickness, porosity, etc.).
                3.  Click **Run Analysis**.
                
                The tool will automatically detect the straight line, calculate permeability/skin, and generate a professional report.
                """)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check your input data and try again.")

if __name__ == "__main__":
    main()
