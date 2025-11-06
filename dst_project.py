"""
Enhanced Interactive DST Horner Plot Analyst
Professional web application for Drill Stem Test analysis
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64

# --- Page configuration ---
st.set_page_config(
    page_title="DST Horner Analyst",
    page_icon="🛢️",
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

# --- Main Analysis Function ---
def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text, num_regression_points):
    """
    Performs the complete DST analysis based on user inputs.

    This function takes all reservoir parameters and the raw data text,
    parses the data, performs linear regression, calculates all key
    petroleum engineering metrics, and returns the results, plot, and data.
    """

    # Input validation
    if any(param <= 0 for param in [h, Qo, mu_o, Bo, rw, phi, Ct, tp]):
        st.error("All parameters (except Pwf) must be positive values.")
        return None, None, None

    if len(data_text.strip()) == 0:
        st.error("Please enter DST data into the text area.")
        return None, None, None

    # Clear previous results from session state
    st.session_state.results = None
    st.session_state.figure = None
    st.session_state.dataframe = None

    tp_hr = tp / 60.0  # Convert flow time to hours for Skin equation

    # --- 1. Parse DST Data ---
    try:
        data_io = io.StringIO(data_text)
        df = pd.read_csv(
            data_io,
            sep=r'[,\s]+',  # Allow comma or space as separator
            engine='python',
            header=None,
            names=['dt', 'pwsf']
        )
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Check for minimum data points
        if len(df) < 3:
            st.error(f"Please enter at least 3 valid data points (you have {len(df)}).")
            return None, None, None

    except Exception as e:
        st.error(f"Error parsing data: {str(e)}. Please check the format (e.g., '5, 965').")
        return None, None, None

    delta_t = df['dt'].values
    pwsf = df['pwsf'].values

    # --- 2. Calculate Horner Time ---
    horner_time = (tp + delta_t) / delta_t
    log_horner_time = np.log10(horner_time)

    df['horner_time'] = horner_time
    df['log_horner_time'] = log_horner_time

    # --- 3. Perform Linear Regression (Horner Slope 'm') ---
    # Use the number of points selected by the user
    slice_index = max(0, len(df) - num_regression_points)
    fit_df = df.iloc[slice_index:].copy()

    if len(fit_df) < 2:
        st.error(f"Not enough data points ({len(fit_df)}) for regression. Need at least 2.")
        return None, None, None

    try:
        # Use Scipy's linregress to find the slope and intercept
        regression = linregress(fit_df['log_horner_time'].values, fit_df['pwsf'].values)
        m = abs(regression.slope) # m is positive psi/cycle
        intercept = regression.intercept
        r_squared = regression.rvalue ** 2
    except Exception as e:
        st.error(f"Regression failed: {str(e)}")
        return None, None, None

    # --- 4. Calculate Reservoir Properties ---
    try:
        k = (162.6 * (Qo * mu_o * Bo)) / (m * h)  # Permeability
        pi = intercept  # Initial Pressure (from intercept at log(t)=0)
        log_term = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
        S = 1.151 * (((pi - pwf_final) / m) - log_term + 3.23)  # Skin Factor
        dP_skin = (141.2 * (Qo * mu_o * Bo) / (k * h)) * S  # Pressure Drop due to Skin
        FE = (pi - pwf_final - dP_skin) / (pi - pwf_final)  # Flow Efficiency
        ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))  # Radius of Investigation
    except Exception as e:
        st.error(f"Error in reservoir property calculation: {str(e)}")
        return None, None, None

    # --- 5. Store Results ---
    results = {
        'm': m,
        'pi': pi,
        'k': k,
        'S': S,
        'FE': FE,
        'ri': ri,
        'r_squared': r_squared
    }

    # --- 6. Create the Plot (Matplotlib) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all data points
    ax.scatter(horner_time, pwsf, label='All DST Data Points', color='blue', zorder=5, alpha=0.7)

    # Highlight MTR points used for regression
    ax.scatter(fit_df['horner_time'], fit_df['pwsf'], color='red', s=100,
               label=f'MTR Data (n={len(fit_df)})', zorder=6)

    # Plot the regression line
    # Extend line from pi (log_horner_time = 0) to max log_horner_time
    x_line_log = np.array([0, np.max(log_horner_time)])
    y_line = intercept + regression.slope * x_line_log
    ax.plot(10 ** x_line_log, y_line, 'r--',
            label=f'MTR Regression (m = {m:.2f} psi/cycle, R² = {r_squared:.3f})',
            zorder=4, linewidth=2)

    # Plot the extrapolated initial pressure line
    ax.axhline(pi, color='green', linestyle=':',
               label=f'Extrapolated $p_i$ = {pi:.1f} psi', linewidth=2)

    # Plot formatting
    ax.set_xscale('log')
    ax.invert_xaxis() # Standard for Horner plots
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title('DST Horner Plot Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12)
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12)
    ax.legend()
    plt.tight_layout()

    return results, fig, df

def get_table_download_link(df):
    """Generate a link to download the processed data as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv">Download processed data as CSV</a>'
    return href

# --- Main Application UI ---
def main():
    st.markdown('<h1 class="main-header">Interactive DST Horner Plot Analyst 🛢️</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    A professional web application for Drill Stem Test (DST) analysis using the Horner method. 
    This tool automates the calculation of key reservoir properties from pressure buildup data,
    based on the principles from your lecture notes.
    """)

    # --- Initialize session state ---
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'figure' not in st.session_state:
        st.session_state.figure = None
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("📊 Input Parameters")

        with st.form(key='input_form'):
            st.subheader("Reservoir & Fluid Properties")
            col1, col2 = st.columns(2)

            with col1:
                h = st.number_input("Pay Thickness, h (ft)", value=10.0, min_value=0.1, format="%.2f")
                Qo = st.number_input("Flow Rate, Qo (bbl/d)", value=135.0, min_value=0.1, format="%.2f")
                mu_o = st.number_input("Viscosity, μo (cp)", value=1.5, min_value=0.1, format="%.2f")
                Bo = st.number_input("FVF, Bo (RB/STB)", value=1.15, min_value=0.1, format="%.3f")

            with col2:
                rw = st.number_input("Wellbore Radius, rw (ft)", value=0.333, min_value=0.01, format="%.3f")
                phi = st.number_input("Porosity, φ", value=0.10, min_value=0.01, max_value=0.5, format="%.3f")
                Ct = st.number_input("Compressibility, Ct (psi⁻¹)", value=8.4e-6, format="%.2e")
                pwf_final = st.number_input("Final Flow P, Pwf (psi)", value=350.0, min_value=0.0, format="%.1f")

            st.subheader("DST Test Parameters")

            # --- Default values set to match lecture's final answer ---
            tp = st.number_input("Total Flow Time, tp (min)", value=60.0, min_value=0.1, format="%.1f")

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

            st.subheader("Regression Settings")
            num_regression_points = st.slider(
                "Points for MTR Regression",
                min_value=2,
                max_value=10,
                value=4, # Defaulted to 4 to match lecture
                help="Number of data points (from the end of the list) to use for the straight-line fit."
            )

            st.caption("Note: `tp=60` and `Points=4` are set to match the lecture's final answer of m ≈ 372.")

            submitted = st.form_submit_button("🚀 Run Analysis")

    # --- Perform analysis when form is submitted ---
    if submitted:
        with st.spinner("Performing DST analysis..."):
            results, figure, dataframe = perform_analysis(
                h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text, num_regression_points
            )

            if results is not None:
                st.session_state.results = results
                st.session_state.figure = figure
                st.session_state.dataframe = dataframe
                st.success("Analysis completed successfully!")

    # --- Main panel with results and plots ---
    col1, col2 = st.columns([1, 1.2]) # Create two columns for results and plot tabs

    with col1:
        st.header("📈 Analysis Results")

        if st.session_state.results:
            results = st.session_state.results

            # Create metrics with better formatting
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.metric("Horner Slope 'm'", f"{results['m']:.2f} psi/cycle")
            st.metric("Initial Reservoir Pressure, pᵢ", f"{results['pi']:.1f} psi")
            st.metric("Formation Permeability, k", f"{results['k']:.2f} md")
            st.metric("Skin Factor, S", f"{results['S']:.2f}")
            st.metric("Flow Efficiency, FE", f"{results['FE']:.3f}")
            st.metric("Radius of Investigation, rᵢ", f"{results['ri']:.1f} ft")
            st.metric("Regression R-squared", f"{results['r_squared']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

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

        # Create tabs for Plot, Data, and Formulas
        tab1, tab2, tab3 = st.tabs(["📊 Horner Plot", "📥 Data Table", "🧪 Formulas"])

        with tab1:
            if st.session_state.figure:
                st.pyplot(st.session_state.figure)
            else:
                st.info("The Horner plot will appear here after analysis")

        with tab2:
            if st.session_state.dataframe is not None:
                st.subheader("Processed Data")
                st.dataframe(st.session_state.dataframe.style.format({
                    'dt': '{:.0f}',
                    'pwsf': '{:.1f}',
                    'horner_time': '{:.2f}',
                    'log_horner_time': '{:.3f}'
                }))
                st.markdown("---")
                st.markdown(get_table_download_link(st.session_state.dataframe),
                            unsafe_allow_html=True)
            else:
                st.info("The processed data table will appear here")

        with tab3:
            st.subheader("Key Formulas")

            st.latex(r"p_{ws} = p_i - m \log\left(\frac{t_p + \Delta t}{\Delta t}\right)")
            st.caption("Horner Equation (m = slope)")

            # --- THIS IS THE MODIFIED SECTION ---
            st.latex(r"m = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{k \cdot h}")
            st.caption("Horner Slope (from Permeability)")

            # --- THIS IS THE NEWLY ADDED EQUATION ---
            st.latex(r"m = \frac{P_{ws_1} - P_{ws_{10}}}{\log(10) - \log(1)}")
            st.caption("Horner Slope (Manual Plot Reading, per your image)")
            # --- END OF MODIFICATION ---

            st.latex(r"k = \frac{162.6 \cdot Q_o \cdot \mu_o \cdot B_o}{m \cdot h}")
            st.caption("Permeability (k)")

            st.latex(r"S = 1.151 \left[ \left(\frac{p_i - p_{wf}}{m}\right) - \log\left(\frac{k \cdot t_{p(hr)}}{\phi \cdot \mu_o \cdot C_t \cdot r_w^2}\right) + 3.23 \right]")
            st.caption("Skin Factor (S)")

            st.latex(r"FE = \frac{p_i - p_{wf} - \Delta p_{skin}}{p_i - p_{wf}}")
            st.caption("Flow Efficiency (FE)")

    # Footer
    st.markdown("---")
    st.markdown(
        "**DST Horner Plot Analyst** • Built with Python 🐍 and Streamlit • "
        "For professional petroleum engineering analysis"
    )

if __name__ == "__main__":
    main()

