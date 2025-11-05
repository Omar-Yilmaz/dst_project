"""
Enhanced Interactive DST Horner Plot Analyst
Professional web application for Drill Stem Test analysis
- English User Interface
- Added Tabs for Plot, Data Table, and Formulas
- Fixed regression slider bug
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import io
import base64

# Page configuration
st.set_page_config(
    page_title="DST Horner Analyst",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Target Streamlit's metric value for custom style */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    /* Ensure LTR for number inputs */
    input[type="number"] {
        text-align: left;
        direction: ltr;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text, num_regression_points):
    """
    Enhanced analysis function with better error handling.
    Returns: results (dict), figure (matplotlib), dataframe (pandas)
    """

    # Input validation
    if any(param <= 0 for param in [h, Qo, mu_o, Bo, rw, phi, Ct, tp]):
        st.error("All parameters must be positive values.")
        return None, None, None

    if len(data_text.strip()) == 0:
        st.error("Please enter DST data.")
        return None, None, None

    tp_hr = tp / 60.0  # Convert flow time to hours

    # Parse DST Data with enhanced error handling
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
            return None, None, None

    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        return None, None, None

    delta_t = df['dt'].values
    pwsf = df['pwsf'].values

    # Calculate Horner Time
    horner_time = (tp + delta_t) / delta_t
    log_horner_time = np.log10(horner_time)

    df['Horner_Time'] = horner_time
    df['Log_Horner_Time'] = log_horner_time
    df = df.round(3) # Round data for clean display

    # Perform Linear Regression with dynamic point selection
    slice_index = max(0, len(df) - num_regression_points)
    fit_df = df.iloc[slice_index:].copy()

    if len(fit_df) < 2:
        st.error("Not enough data points for regression.")
        return None, None, None

    try:
        regression = linregress(fit_df['Log_Horner_Time'].values, fit_df['pwsf'].values)
        m = abs(regression.slope)
        intercept = regression.intercept
        r_squared = regression.rvalue ** 2
    except Exception as e:
        st.error(f"Regression failed: {str(e)}")
        return None, None, None

    # Calculate Reservoir Properties
    try:
        k = (162.6 * (Qo * mu_o * Bo)) / (m * h)  # Permeability
        pi = intercept  # Initial Pressure
        log_term = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
        S = 1.151 * (((pi - pwf_final) / m) - log_term + 3.23)  # Skin Factor
        dP_skin = (141.2 * (Qo * mu_o * Bo) / (k * h)) * S  # Pressure Drop due to Skin
        FE = (pi - pwf_final - dP_skin) / (pi - pwf_final)  # Flow Efficiency
        ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct))  # Radius of Investigation
    except Exception as e:
        st.error(f"Error in reservoir property calculation: {str(e)}")
        return None, None, None

    # Store results
    results = {
        'm': m,
        'pi': pi,
        'k': k,
        'S': S,
        'FE': FE,
        'ri': ri,
        'r_squared': r_squared
    }

    # Create enhanced plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Main data
    ax.scatter(horner_time, pwsf, label='All DST Data Points', color='blue', zorder=5, alpha=0.7)
    ax.scatter(fit_df['Horner_Time'], fit_df['pwsf'], color='red', s=100,
               label=f'MTR Data (n={len(fit_df)})', zorder=6)

    # Regression line
    x_line_log = np.linspace(0, np.max(log_horner_time), 100) # 0 = log(1)
    y_line = intercept + regression.slope * x_line_log
    ax.plot(10 ** x_line_log, y_line, 'r--',
            label=f'MTR Regression (m = {m:.2f} psi/cycle, R² = {r_squared:.3f})',
            zorder=4, linewidth=2)

    # Initial pressure line
    ax.axhline(pi, color='green', linestyle=':',
               label=f'Extrapolated $p_i$ = {pi:.1f} psi', linewidth=2)

    # Plot formatting
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title('DST Horner Plot Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Horner Time (tp + Δt) / Δt', fontsize=12)
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12)
    ax.legend(loc='best')

    plt.tight_layout()

    return results, fig, df

def get_table_download_link(df):
    """Generate a link to download the processed data as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv">📥 Download Processed Data (CSV)</a>'
    return href

# Main App
def main():
    st.markdown('<h1 class="main-header">Interactive DST Horner Plot Analyst 🛢️</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    A professional web application for Drill Stem Test (DST) analysis using the Horner method.
    This tool automates the calculation of key reservoir properties from pressure buildup data.
    """)

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'figure' not in st.session_state:
        st.session_state.figure = None
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None

    # Sidebar inputs
    with st.sidebar:
        st.header("📊 1. Input Parameters")

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
            tp = st.number_input("Total Flow Time, tp (min)", value=65.0, min_value=0.1, format="%.1f")

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
                min_value=3,
                max_value=10,
                value=5,
                help="Select the number of final data points to use for the regression line."
            )

            submitted = st.form_submit_button("🚀 Run Analysis")

    # Perform analysis when form is submitted
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

    # Display results
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("📈 2. Analysis Results")

        if st.session_state.results:
            results = st.session_state.results

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
            st.subheader("📋 Engineering Interpretation")
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
                fe_interpretation = "Well is stimulated (excellent efficiency)"
            elif results['FE'] > 0.8:
                fe_interpretation = "Good flow efficiency"
            elif results['FE'] > 0.5:
                fe_interpretation = "Moderate damage"
            else:
                fe_interpretation = "Poor flow efficiency (high damage)"

            st.write(f"**Flow Efficiency Interpretation:** {fe_interpretation}")

        else:
            st.info("👈 Enter parameters and click 'Run Analysis' to see results")

    # --- Output Tabs (Plot, Data, Formulas) ---
    st.header("📊 3. Detailed Outputs")
    tabs = st.tabs(["📊 Horner Plot", "📥 Data Table", "🧪 Formulas Used"])

    with tabs[0]:
        st.subheader("Horner Plot")
        if st.session_state.figure:
            st.pyplot(st.session_state.figure)
        else:
            st.info("The plot will appear here after analysis")

    with tabs[1]:
        st.subheader("Processed Data")
        if st.session_state.dataframe is not None:
            st.dataframe(st.session_state.dataframe)
            st.markdown(get_table_download_link(st.session_state.dataframe),
                        unsafe_allow_html=True)
        else:
            st.info("The data table will appear here after analysis")

    with tabs[2]:
        st.subheader("Key Formulas Used (from Lecture)")
        st.markdown("These are the core equations this analysis is based on:")

        st.latex(r'''
        \text{1. Horner Time:} \quad \frac{t_p + \Delta t}{\Delta t}
        ''')

        st.latex(r'''
        \text{2. Horner Line Equation:} \quad P_{ws} = p_i - m \log \left( \frac{t_p + \Delta t}{\Delta t} \right)
        ''')

        st.latex(r'''
        \text{3. Permeability (k):} \quad k = \frac{162.6 \cdot Q_o \mu_o B_o}{m \cdot h}
        ''')

        st.latex(r'''
        \text{4. Skin Factor (S):} \quad S = 1.151 \left[ \frac{p_i - p_{wf}}{m} - \log \left( \frac{k \cdot t_{p,hr}}{\phi \mu_o C_t r_w^2} \right) + 3.23 \right]
        ''')

        st.latex(r'''
        \text{5. Flow Efficiency (FE):} \quad FE = \frac{p_i - p_{wf} - \Delta P_{skin}}{p_i - p_{wf}}
        ''')

        st.latex(r'''
        \text{6. Radius of Investigation (r_i):} \quad r_i = \sqrt{\frac{k \cdot t_p}{5.76 \times 10^4 \cdot \phi \mu_o C_t}}
        ''')


    # Footer
    st.markdown("---")
    st.markdown(
        "**DST Horner Plot Analyst** • Built with Python 🐍 and Streamlit • "
        "For professional petroleum engineering analysis"
    )

if __name__ == "__main__":
    main()