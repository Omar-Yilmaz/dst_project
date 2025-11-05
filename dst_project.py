"""
Interactive DST Horner Plot Analyst: Pure Regression Model
This version strictly calculates the Horner slope (m) and Pi (Pi_calc) using linear regression
on the chosen Middle Time Region (MTR) without any manual overrides.
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
    page_title="DST Horner Analyst: Pure Math",
    page_icon="🧪",
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
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def perform_analysis(h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text):
    """Calculates all parameters strictly using mathematical regression."""
    
    # Input validation
    if any(param <= 0 for param in [h, Qo, mu_o, Bo, rw, phi, Ct]):
        st.error("All physical parameters must be positive values.")
        return None, None, None
    
    if len(data_text.strip()) == 0:
        st.error("Please enter DST data.")
        return None, None, None
        
    # Clear previous results
    st.session_state.results = None
    st.session_state.figure = None
    st.session_state.dataframe = None

    # Flow time used in Skin/FE/ri formulas (tp_hr)
    tp_hr = tp / 60.0  

    # Parse DST Data
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

    # Calculate Horner Time for plotting (using the implied tp=60 min for consistency with lecture's X-axis spacing)
    horner_time_plot = (60.0 + delta_t) / delta_t
    log_horner_time = np.log10(horner_time_plot)
    
    df['horner_time'] = horner_time_plot
    df['log_horner_time'] = log_horner_time

    # Perform Linear Regression 
    num_regression_points = st.session_state.num_reg_points 
    slice_index = max(0, len(df) - num_regression_points)
    fit_df = df.iloc[slice_index:].copy()
    
    if len(fit_df) < 2:
        st.error("Not enough data points for regression.")
        return None, None, None

    try:
        # Calculate regression slope (m) and intercept (Pi)
        regression = linregress(fit_df['log_horner_time'].values, fit_df['pwsf'].values)
        m = abs(regression.slope)
        pi = regression.intercept
        r_squared = regression.rvalue ** 2
    except Exception as e:
        st.error(f"Regression failed: {str(e)}")
        return None, None, None

    # --- FINAL CALCULATIONS (Using Calculated m and Pi) ---
    try:
        k = (162.6 * (Qo * mu_o * Bo)) / (m * h)  # Permeability
        
        # Skin Factor (S) - Uses calculated Pi and user's tp (65 min)
        log_term = np.log10((k * tp_hr) / (phi * mu_o * Ct * (rw ** 2)))
        S = 1.151 * (((pi - pwf_final) / m) - log_term + 3.23) 
        
        # Pressure Drop due to Skin (dP_skin)
        dP_skin = (141.2 * (Qo * mu_o * Bo) / (k * h)) * S
        
        # Flow Efficiency (FE)
        FE = (pi - pwf_final - dP_skin) / (pi - pwf_final)
        
        # Radius of Investigation (ri)
        ri = np.sqrt((k * tp) / (5.76e4 * phi * mu_o * Ct)) 

    except Exception as e:
        st.error(f"Error in reservoir property calculation: {str(e)}")
        return None, None, None

    # Store results
    results = {
        'm_final': m,
        'pi_final': pi,
        'k': k,
        'S': S,
        'FE': FE,
        'ri': ri,
        'r_squared': r_squared,
        'dP_skin': dP_skin
    }

    # Create enhanced plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Main data
    ax.scatter(horner_time_plot, pwsf, label='All DST Data Points', color='blue', zorder=5, alpha=0.7)
    ax.scatter(fit_df['horner_time'], fit_df['pwsf'], color='red', s=100, 
               label=f'MTR Data (n={len(fit_df)})', zorder=6)
    
    # Plotting the FINAL line based on the calculated 'm'
    x_line_log = np.array([0, np.max(log_horner_time)]) 
    y_line_final = pi - m * x_line_log # y = pi - m*X 
    ax.plot(10 ** x_line_log, y_line_final, 'r--', 
            label=f'Regression Slope (m = {m:.2f} psi/cycle)', 
            zorder=4, linewidth=2)
    
    # Initial pressure line
    ax.axhline(pi, color='green', linestyle=':', 
               label=f'Calculated $P_i$ = {pi:.1f} psi', linewidth=2)
    
    # Plot formatting
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title('DST Horner Plot Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Horner Time (60 min + Δt) / Δt', fontsize=12)
    ax.set_ylabel('Shut-in Pressure (Pwsf), psi', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    return results, fig, df

def get_table_download_link(df):
    """Generate a link to download the processed data as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dst_analysis_results.csv">Download processed data as CSV</a>'
    return href

# Main App
def main():
    st.markdown('<h1 class="main-header">Interactive DST Horner Plot Analyst 🧪</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("A professional web application for Drill Stem Test (DST) analysis using the Horner method. This version calculates the slope strictly via **linear regression (the equation)**.")

    # --- Initialize session state ---
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'figure' not in st.session_state:
        st.session_state.figure = None
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    if 'num_reg_points' not in st.session_state:
        st.session_state.num_reg_points = 4 

    # Sidebar inputs
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
            # tp is the only flow time input, used for Skin/ri formulas
            tp = st.number_input("Total Flow Time, tp (min)", value=65.0, min_value=0.1, format="%.1f", 
                                 help="Used in Skin/Ri formulas (65 min in lecture notes).")
            
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

            # --- REGRESSION SETTINGS ---
            st.subheader("Regression Settings")
            st.session_state.num_reg_points = st.slider(
                "Points for MTR Regression", 
                min_value=3, 
                max_value=9, 
                value=4, 
                help="Number of points at the end of the dataset used for the least-squares fit."
            )
            
            submitted = st.form_submit_button("🚀 Run Analysis")

    # Perform analysis when form is submitted
    if submitted:
        with st.spinner("Performing DST analysis..."):
            results, figure, dataframe = perform_analysis(
                h, Qo, mu_o, Bo, rw, phi, Ct, pwf_final, tp, data_text
            )
            
            if results is not None:
                st.session_state.results = results
                st.session_state.figure = figure
                st.session_state.dataframe = dataframe
                st.success("Analysis completed successfully!")

    # Display results
    tab1, tab2, tab3 = st.tabs(["📊 Results & Plot", "📝 Formulas (Theory)", "📥 Data Table"])

    with tab1:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.header("Analysis Results")
            if st.session_state.results:
                results = st.session_state.results
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"**Regression Slope $m$:** **${results['m_final']:.2f}$ psi/cycle**")
                st.metric("Initial Reservoir Pressure, pᵢ", f"{results['pi_final']:.1f} psi")
                st.metric("Formation Permeability, k", f"{results['k']:.2f} md")
                st.metric("Skin Factor, S", f"{results['S']:.2f}")
                st.metric("Flow Efficiency, FE", f"{results['FE']:.3f}")
                st.metric("Radius of Investigation, rᵢ", f"{results['ri']:.1f} ft")
                st.markdown(f"**$R^2$ of Fit:** {results['r_squared']:.4f}")
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
                st.write(f"**Flow Efficiency:** {'Well is stimulated' if results['FE'] > 1.0 else 'Well is damaged'}")
                
            else:
                st.info("👈 Enter parameters and click 'Run Analysis' to see results")

        with col2:
            st.header("Horner Plot")
            if st.session_state.figure:
                st.pyplot(st.session_state.figure)
            else:
                st.info("The Horner plot will appear here after analysis")

    with tab2:
        st.header("Theoretical Basis: Equations Used")
        
        st.subheader("1. Permeability ($k$)")
        st.markdown(f"$$k = 162.6 \\frac{{Q_o \\mu_o B_o}}{{h \\cdot m}}$$")
        
        st.subheader("2. Skin Factor ($S$)")
        st.markdown("The formula uses the input $t_p$ value (converted to hours for the log term):")
        st.latex(f"""
        S = 1.151\\left[ \\frac{{P_i - P_{{wf}}}}{{m}} - \\log_{{10}}\\left(\\frac{{k \\cdot t_p}}{{ \\phi \\mu_o C_t r_{{w}}^{{2}} }}\\right) + 3.23 \\right]
        """)
        
        st.subheader("3. Flow Efficiency ($FE$)")
        st.latex(f"""
        FE = \\frac{{(P_i - P_{{wf}}) - \\Delta P_{{skin}}}}{{(P_i - P_{{wf}})}}
        """)

    with tab3:
        st.header("Processed Data Table")
        if st.session_state.dataframe is not None:
            st.dataframe(st.session_state.dataframe)
            st.markdown(get_table_download_link(st.session_state.dataframe), unsafe_allow_html=True)
        else:
            st.info("Run the analysis to see the processed data (Δt, Pwsf, Horner Time).")

    # Footer
    st.markdown("---")
    st.markdown(
        "**DST Horner Plot Analyst** • Built with Python 🐍 and Streamlit"
    )

if __name__ == "__main__":
    main()
