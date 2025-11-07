"""
Enhanced Interactive DST Horner Plot Analyst (Smart Auto MTR Detection)
Professional web application for Drill Stem Test analysis
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64

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
</style>
""", unsafe_allow_html=True)

# --- NEW "SMART" Auto MTR detection function ---
def find_best_mtr(df, min_points=3, slope_stability_threshold=0.20): # <-- CHANGED to 0.20 as requested
    """
    Automatically find the *final* straight line segment (MTR)
    by finding the most stable, high-R-squared line from the *end* of the dataset.

    Returns:
        best_fit_df: The data slice used for best regression.
        best_regression: linregress result for best fit.
    """
    best_r_squared = -1
    best_regression = None
    best_fit_df = None
    best_n_points = 0

    n = len(df)

    # Store previous slope to check for stability
    previous_slope = None

    # We only check slices that *end* at the last data point.
    for num_regression_points in range(min_points, n + 1):
        slice_index = max(0, n - num_regression_points)
        fit_df_loop = df.iloc[slice_index:].copy()

        if len(fit_df_loop) < 2:
            continue

        try:
            regression = linregress(fit_df_loop['log_horner_time'].values, fit_df_loop['pwsf'].values)
            r_squared = regression.rvalue ** 2
            current_slope = abs(regression.slope)

            # --- SMART STABILITY CHECK ---
            # If this is the first valid line, save it
            if previous_slope is None:
                previous_slope = current_slope
                best_r_squared = r_squared
                best_regression = regression
                best_fit_df = fit_df_loop
                best_n_points = num_regression_points
                continue

            # Check how much the slope changed by adding one more point
            slope_change_percent = abs(current_slope - previous_slope) / previous_slope

            # Check how much the R-squared changed
            r_squared_change = r_squared - best_r_squared

            # --- *** NEW LOGIC *** ---
            # We prefer a new line IF:
            # 1. It is significantly "straighter" (R² got better)
            # OR
            # 2. It is still *very* straight (R² > 0.99) AND the slope is stable.

            if r_squared_change > 0.0001: # Case 1: Significantly straighter
                best_r_squared = r_squared
                best_regression = regression
                best_fit_df = fit_df_loop
                best_n_points = num_regression_points
                previous_slope = current_slope

            elif r_squared > 0.99 and slope_change_percent < slope_stability_threshold: # Case 2: Still very straight AND slope is stable
                # This line is also good and is longer, so we accept it
                # This handles cases where R² drops slightly (e.g. 0.9999 -> 0.9995)
                best_r_squared = r_squared
                best_regression = regression
                best_fit_df = fit_df_loop
                best_n_points = num_regression_points
                previous_slope = current_slope

            else:
                # This new point (e.g., dt=20) breaks the stability.
                # EITHER the R² dropped too much (e.g., below 0.99)
                # OR the R² is high BUT the slope was unstable.
                # So, we STOP and keep the *previous* stable line.
                break

        except Exception:
            continue

    if best_regression is None:
        return None, None

    return best_fit_df, best_regression

# --- Main Analysis Function ---
def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text):
    """
    Performs the complete DST analysis based on user inputs.
    """

    # Input validation
    if any(param <= 0 for param in [h, Qo, mu_o, Bo, rw, phi, Ct, tp]):
        st.error("All parameters (except Pwf) must be positive values.")
        return None, None, None, None

    if len(data_text.strip()) == 0:
        st.error("Please enter DST data into the text area.")
        return None, None, None, None

    # Clear previous results from session state
    st.session_state.results = None
    st.session_state.figure = None
    st.session_state.dataframe = None
    st.session_state.mtr_info = None

    tp_hr = tp / 60.0  # Convert flow time to hours for Skin equation

    # --- 1. Parse DST Data ---
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
            return None, None, None, None
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}. Please check the format (e.g., '5, 965').")
        return None, None, None, None

    delta_t = df['dt'].values
    pwsf = df['pwsf'].values

    # --- 2. Calculate Horner Time ---
    horner_time = (tp + delta_t) / delta_t
    log_horner_time = np.log10(horner_time)

    df['horner_time'] = horner_time
    df['log_horner_time'] = log_horner_time

    # --- 3. Auto-detect MTR, Perform Linear Regression ---
    # We set min_points to 3, the minimum for a stable regression.
    fit_df, regression = find_best_mtr(df, min_points=3)

    if fit_df is None or regression is None:
        st.error("Automatic straight-line detection failed. The data may be too noisy.")
        return None, None, None, None

    m = abs(regression.slope)  # m is positive psi/cycle
    intercept = regression.intercept
    r_squared = regression.rvalue ** 2

    # --- 4. Calculate Reservoir Properties ---
    try:
        k = (162.6 * (Qo * mu_o * Bo)) / (m * h)
        pi = intercept
        log_term = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
        S = 1.151 * (((pi - pwf_final) / m) - log_term + 3.23)
        dP_skin = (141.2 * (Qo * mu_o * Bo) / (k * h)) * S
        FE = (pi - pwf_final - dP_skin) / (pi - pwf_final)
        ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))
    except Exception as e:
        st.error(f"Error in reservoir property calculation: {str(e)}")
        return None, None, None, None

    results = {
        'm': m,
        'pi': pi,
        'k': k,
        'S': S,
        'FE': FE,
        'ri': ri,
        'r_squared': r_squared,
        'dP_skin': dP_skin
    }

    mtr_info = {
        'num_points': len(fit_df),
        'start_dt': float(fit_df['dt'].iloc[0]),
        'end_dt': float(fit_df['dt'].iloc[-1]),
        'used_rows': fit_df.index.tolist()
    }


    # --- 5. Create the Plot (Matplotlib) ---
    # Use a professional 'seaborn' style for a cleaner look
    # *** CHANGED: Switched from 'darkgrid' to 'whitegrid' for a cleaner, more obvious look ***
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Plot Customizations for Professional Look ---

    # 1. Plot all data points as blue circles with a black edge
    ax.scatter(horner_time, pwsf,
               s=50,                  # Set size
               color='C0',            # 'C0' is the default seaborn blue
               edgecolor='black',     # Add a black outline
               # *** NEW: Add a subtle line width to the edge ***
               linewidths=0.5,
               label='All DST Data',  # Add label for legend
               zorder=5)              # Ensure points are on top of grid

    # *** NEW: Plot MTR points in a different color (red) on top ***
    ax.scatter(fit_df['horner_time'], fit_df['pwsf'],
               # *** CHANGED: Made MTR points slightly larger ***
               s=70,
               color='C3',            # 'C3' is seaborn red
               edgecolor='black',     # Add a black outline
               # *** NEW: Add a bolder line width to MTR points ***
               linewidths=1.0,
               label=f"Auto-Detected MTR (n={mtr_info['num_points']})", # Add MTR label
               zorder=6)              # Ensure these are on top

    # 2. Set X-axis to match log paper
    ax.set_xscale('log')
    ax.invert_xaxis()
    # *** CHANGED: Force x-axis to go all the way to 1 ***
    ax.set_xlim(left=np.max(horner_time) * 1.5, right=1.0) # End at Horner Time = 1

    # 3. Calculate and plot the solid black regression line
    # *** CHANGED: We extend it from the max horner time to 1 (log(1)=0) ***
    x_line_log = np.array([np.log10(np.max(horner_time) * 1.5), 0]) # End at log(1) = 0
    y_line = intercept + regression.slope * x_line_log
    ax.plot(10 ** x_line_log, y_line, 'k-', # 'k-' is a solid black line
            label=f'MTR (m = {m:.2f}, R² = {r_squared:.3f})',
            zorder=4,
            # *** CHANGED: Made MTR line bolder ***
            linewidth=2.5)

    # 4. Add the extrapolated pressure (pi) line back
    ax.axhline(pi, color='green', linestyle='--',
               label=f'Extrapolated $p_i$ = {pi:.1f} psi',
               # *** CHANGED: Made pi line bolder ***
               linewidth=2.5)

    # 5. Set Y-axis to start at 0 and have correct ticks
    top_limit = max(pi * 1.05, np.max(pwsf) * 1.05)
    ax.set_ylim(bottom=0, top=top_limit)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100)) # Major ticks every 100
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))  # Minor ticks every 20

    # *** NEW: Move Y-axis to the right side ***
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # 6. Add gridlines (the style 'seaborn-whitegrid' handles this, but we ensure minor are on)
    # *** UPDATED: Made minor gridlines slightly bolder/more visible ***
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)

    # 7. Set labels, title, and legend
    ax.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12)
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12)
    ax.set_title('Horner Plot Analysis', fontsize=16, fontweight='bold')
    # *** CHANGED: Added 'frameon=True' to make legend easier to read ***
    ax.legend(loc='lower left', frameon=True)

    # 8. Remove the top and left plot borders for a cleaner look
    # *** CHANGED: Now remove 'left' spine instead of 'right' ***
    ax.spines[['top', 'left']].set_visible(False)
    # *** NEW: Keep the 'right' spine since the axis is there ***
    ax.spines['right'].set_visible(True)

    # --- 9. *** REMOVED ALL ANNOTATIONS FOR A CLEANER LOOK *** ---

    plt.tight_layout()

    return results, fig, df, mtr_info

def get_table_download_link(df):
    """Generate a link to download the processed data as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv">Download processed data as CSV</a>'
    return href

# --- Main Application UI ---
def main():
    st.markdown('<h1 class="main-header">Interactive DST Horner Plot Analyst 📈</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    A professional web application for Drill Stem Test (DST) analysis using the Horner method. 
    This tool automates the calculation of key reservoir properties from pressure buildup data,
    now with **smart automatic straight-line region (MTR) detection**.
    """)

    # --- Initialize session state ---
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'figure' not in st.session_state:
        st.session_state.figure = None
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    if 'mtr_info' not in st.session_state:
        st.session_state.mtr_info = None

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("📊 Input Parameters")
        with st.form(key='input_form'):
            st.subheader("Reservoir & Fluid Properties")
            col1, col2 = st.columns(2)
            with col1:
                # *** ADDED step=1.0 ***
                h = st.number_input("Pay Thickness, h (ft)", value=10.0, min_value=0.1, format="%.2f", step=1.0)
                # *** ADDED step=1.0 ***
                Qo = st.number_input("Flow Rate, Qo (bbl/d)", value=135.0, min_value=0.1, format="%.2f", step=1.0)
                # *** ADDED step=0.1 ***
                mu_o = st.number_input("Viscosity, μo (cp)", value=1.5, min_value=0.1, format="%.2f", step=0.1)
                # *** ADDED step=0.01 ***
                Bo = st.number_input("FVF, Bo (RB/STB)", value=1.15, min_value=0.1, format="%.3f", step=0.01)
            with col2:
                # *** ADDED step=0.01 ***
                rw = st.number_input("Wellbore Radius, rw (ft)", value=0.333, min_value=0.01, format="%.3f", step=0.01)
                # *** ADDED step=0.01 ***
                phi = st.number_input("Porosity, φ", value=0.10, min_value=0.01, max_value=0.5, format="%.3f", step=0.01)
                # *** ADDED step=1e-7 ***
                Ct = st.number_input("Compressibility, Ct (psi⁻¹)", value=8.4e-6, format="%.2e", step=1e-7)
                # *** ADDED step=1.0 ***
                pwf_final = st.number_input("Final Flow P, Pwf (psi)", value=350.0, min_value=0.0, format="%.1f", step=1.0)

            st.subheader("DST Test Parameters")
            # *** ADDED step=1.0 as requested ***
            tp = st.number_input("Total Flow Time, tp (min)", value=60.0, min_value=0.1, format="%.1f",
                                help="The *total* duration of the flow period (tp) in minutes. This is used for all calculations.",
                                step=1.0)

            st.subheader("Pressure Buildup Data")
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
                "Shut-in Data (Δt [min], Pwsf [psi])",
                value=default_data,
                height=200,
                help="Enter one 'time, pressure' pair per line. Example: '5, 965'"
            )

            # --- REMOVED SLIDER FOR REGRESSION SETTINGS ---
            # The app is now fully automatic.

            submitted = st.form_submit_button("🚀 Run Analysis")

    # --- Perform analysis when form is submitted ---
    if submitted:
        with st.spinner("Auto-detecting MTR and performing analysis..."):
            results, figure, dataframe, mtr_info = perform_analysis(
                h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text
            )
            if results is not None:
                st.session_state.results = results
                st.session_state.figure = figure
                st.session_state.dataframe = dataframe
                st.session_state.mtr_info = mtr_info
                st.success("Analysis completed successfully!")

    # --- Main panel with results and plots ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("📈 Analysis Results")

        if st.session_state.results:
            results = st.session_state.results
            mtr_info = st.session_state.mtr_info

            # Metrics
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.metric("Horner Slope 'm'", f"{results['m']:.2f} psi/cycle")
            st.metric("Initial Reservoir Pressure, pᵢ", f"{results['pi']:.1f} psi")
            st.metric("Formation Permeability, k", f"{results['k']:.2f} md")
            st.metric("Skin Factor, S", f"{results['S']:.2f}")
            st.metric("Pressure Drop (Skin), ΔP_skin", f"{results['dP_skin']:.1f} psi")
            st.metric("Flow Efficiency, FE", f"{results['FE']:.3f}")
            st.metric("Radius of Investigation, rᵢ", f"{results['ri']:.1f} ft")
            st.metric("Regression R-squared", f"{results['r_squared']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

            # MTR Info
            if mtr_info:
                st.info(f"**Auto-MTR Successful:** Found best line (R²={results['r_squared']:.4f}) using the **last {mtr_info['num_points']} points** (from Δt = {mtr_info['start_dt']} to {mtr_info['end_dt']} min).")

            # Interpretation
            st.subheader("📋 Interpretation")
            if results['S'] < -3:
                skin_interpretation = "Highly stimulated well (excellent)"
            elif results['S'] < 0:
                skin_interpretation = "Stimulated well (good)"
            elif results['S'] < 3:
                skin_interpretation = "Undamaged well"
            elif results['S'] < 10:
                skin_interpretation = "Damaged well"
            else:
                skin_interpretation = "Severely damaged well"
            st.write(f"**Skin Factor Interpretation:** {skin_interpretation}")

            if results['FE'] > 1.0:
                fe_interpretation = "Well is stimulated"
            elif results['FE'] > 0.8:
                fe_interpretation = "Good flow efficiency"
            elif results['FE'] > 0.5:
                fe_interpretation = "Moderate damage"
            else:
                fe_interpretation = "Poor flow efficiency"
            st.write(f"**Flow Efficiency:** {fe_interpretation}")

        else:
            st.info("👈 Enter parameters and click 'Run Analysis' to see results")

    with col2:
        st.header("Analysis Outputs")
        tab1, tab2, tab3 = st.tabs(["📊 Horner Plot", "📥 Data Table", "🧪 Formulas"])
        with tab1:
            if st.session_state.figure:
                # *** CHANGED: Increased dpi to 300 for maximum quality ***
                st.pyplot(st.session_state.figure, dpi=300)
            else:
                st.info("The Horner plot will appear here after analysis")
        with tab2:
            if st.session_state.dataframe is not None:
                st.subheader("Processed Data")
                df = st.session_state.dataframe.copy()
                mtr_rows = st.session_state.mtr_info['used_rows'] if st.session_state.mtr_info else []
                # Add MTR marker
                df['MTR'] = ["✅" if i in mtr_rows else "" for i in df.index]
                st.dataframe(df.style.format({
                    'dt': '{:.0f}',
                    'pwsf': '{:.1f}',
                    'horner_time': '{:.2f}',
                    'log_horner_time': '{:.3f}'
                }))
                st.markdown("---")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            else:
                st.info("The processed data table will appear here")
        with tab3:
            st.subheader("Key Formulas")
            st.latex(r"p_{ws} = p_i - m \log\left(\frac{t_p + \Delta t}{\Delta t}\right)")
            st.caption("Horner Equation (m = slope)")

            st.latex(r"m = \frac{P_{ws_1} - P_{ws_{10}}}{\log(10) - \log(1)}")
            st.caption("Horner Slope (Manual Plot Reading)")

            st.latex(r"m = \frac{\Delta P}{\Delta \log(\text{Horner Time})} \quad \text{(Linear Regression)}")
            st.caption("Horner Slope (Automated in this App)")

            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")
            st.caption("Permeability (k)")

            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
            st.caption("Skin Factor (S)")

            st.latex(r"(\Delta P_{Skin}) = 141.2 \left( \frac{Q_o \cdot \mu_o \cdot B_o}{k \cdot h} \right) S")
            st.caption("Pressure Drop (Skin)")


            st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
            st.caption("Flow Efficiency (FE)")

    st.markdown("---")
    st.markdown(
        "**DST Horner Plot Analyst** • Built with Python 🐍 and Streamlit • "
        "For professional petroleum engineering analysis"
    )

if __name__ == "__main__":
    main()
