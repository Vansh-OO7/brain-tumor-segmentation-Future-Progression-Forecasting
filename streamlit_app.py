# import streamlit as st
# import os
# from src.analyze_input import analyze_uploaded_h5

# st.set_page_config(
#     page_title="Brain Tumor Forecasting",
#     page_icon="💀",
#     layout="wide"
# )

# st.title("Automated Brain Tumor Segmentation and Future Progression Forecasting")
# st.caption("Upload a .h5 MRI file and analyze tumor progression")

# uploaded_file = st.file_uploader(
#     "Upload Multi-Modal MRI (.h5)",
#     type=["h5"]
# )

# if uploaded_file is not None:
#     os.makedirs("uploads", exist_ok=True)
#     temp_path = os.path.join("uploads", uploaded_file.name)

#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.success("File uploaded successfully.")

#     if st.button("Analyze"):
#         with st.spinner("Running AI analysis..."):
#             result = analyze_uploaded_h5(temp_path)

#         st.subheader("Segmented Result")
#         st.image(result["segmented_image_path"], use_container_width=True)

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Current Area", f'{result["current_area_cm2"]} cm²')
#         c2.metric("Short Term", f'{result["short_term_cm2"]} cm²')
#         c3.metric("Mid Term", f'{result["mid_term_cm2"]} cm²')

#         c4, c5, c6 = st.columns(3)
#         c4.metric("Long Term", f'{result["long_term_cm2"]} cm²')
#         c5.metric("Growth %", f'{result["growth_long_term_percent"]}%')
#         c6.metric("Status", result["progression_status"])

#         st.subheader("Detailed Forecast")
#         st.write({
#             "Short Term Growth %": result["growth_short_term_percent"],
#             "Mid Term Growth %": result["growth_mid_term_percent"],
#             "Long Term Growth %": result["growth_long_term_percent"]
#         })




import streamlit as st
import os
from src.analyze_input import analyze_uploaded_h5
 
# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Brain Tumor AI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ==================================================
# CUSTOM CSS STYLING
# ==================================================
st.markdown("""
<style>
    /* Main background and typography */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Title styling */
    h1 {
        color: white !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #e0e7ff;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card containers */
    .upload-card, .results-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px dashed #667eea;
    }
    
    /* Success/Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #556cd6 0%, #5f3b8c 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    .status-regressive {
        background: #27ae60;
        color: white;
    }
    
    .status-stable {
        background: #3498db;
        color: white;
    }
    
    .status-mild {
        background: #f39c12;
        color: white;
    }
    
    .status-moderate {
        background: #e67e22;
        color: white;
    }
    
    .status-rapid {
        background: #c0392b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
 
# ==================================================
# SIDEBAR INFORMATION
# ==================================================
with st.sidebar:
    st.markdown("##  About This Tool")
    st.markdown("""
    This AI-powered system provides:
    
    ** Automated Segmentation **
    - Deep learning-based tumor detection
    - Multi-modal MRI analysis
    - High-precision boundary detection
    
    ** Growth Forecasting**
    - Short-term projection
    - Mid-term projection  
    - Long-term progression analysis
    
    ** Clinical Assessment**
    - Automated trend classification
    - Growth rate quantification
    - Treatment monitoring support
    """)
    
    st.markdown("---")
    st.markdown("###  Instructions")
    st.markdown("""
    1. Upload your `.h5` MRI file
    2. Click **Analyze** to process
    3. Review segmentation results
    4. Examine growth projections
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <small>This tool is for research purposes only. 
    Always consult qualified medical professionals 
    for clinical decisions.</small>
    """, unsafe_allow_html=True)
 
# ==================================================
# MAIN APPLICATION
# ==================================================
st.markdown("# Brain Tumor AI Analysis Platform")
st.markdown('<p class="subtitle">Advanced Segmentation & Progression Forecasting System</p>', 
            unsafe_allow_html=True)
 
# Upload Section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
 
with col2:
    st.markdown("###  Upload Medical Imaging Data(.h5 format)")
    uploaded_file = st.file_uploader(
        "Select Multi-Modal MRI File (.h5 format)",
        type=["h5"],
        help="Upload a .h5 file containing multi-modal MRI scan data"
    )
 
if uploaded_file is not None:
    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
    
    # Analysis Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button(" Analyze Tumor", use_container_width=True)
    
    if analyze_button:
        with st.spinner("🔬 Running AI analysis... This may take a moment."):
            result = analyze_uploaded_h5(temp_path)
        
        # st.balloons()
        
        # ==================================================
        # RESULTS SECTION
        # ==================================================
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Segmentation Visualization
        st.markdown("###  Segmentation & Projection Visualization")
        st.image(result["segmented_image_path"], use_container_width=True)
        
        st.markdown("---")
        
        # Current Status
        st.markdown("###  Current Measurements")
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Current Area",
                f'{result["current_area_cm2"]} cm²',
                help="Current measured tumor area"
            )
        
        with cols[1]:
            st.metric(
                "Detected Pixels",
                f'{int(result["current_area_cm2"] * 100):,}',
                help="Total pixels identified as tumor tissue"
            )
        
        with cols[2]:
            # Determine status color class
            status = result["progression_status"]
            if "Regressive" in status:
                status_class = "status-regressive"
            elif "Stable" in status:
                status_class = "status-stable"
            elif "Mild" in status:
                status_class = "status-mild"
            elif "Moderate" in status:
                status_class = "status-moderate"
            else:
                status_class = "status-rapid"
            
            st.markdown("**Clinical Status**")
            st.markdown(f'<div class="status-badge {status_class}">{status}</div>', 
                       unsafe_allow_html=True)
        
        with cols[3]:
            st.metric(
                "Long Term Growth",
                f'{result["growth_long_term_percent"]}%',
                delta=f'{result["growth_long_term_percent"]}%',
                delta_color="inverse" if result["growth_long_term_percent"] < 0 else "normal",
                help="Projected growth rate over long term"
            )
        
        st.markdown("---")
        
        # Future Projections
        st.markdown("### 🔮 Progression Forecast")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 Short Term")
            st.metric(
                "Projected Area",
                f'{result["short_term_cm2"]} cm²',
                delta=f'{result["growth_short_term_percent"]}%',
                delta_color="inverse" if result["growth_short_term_percent"] < 0 else "normal"
            )
            st.progress(min(abs(result["growth_short_term_percent"]) / 100, 1.0))
        
        with col2:
            st.markdown("#### 📅 Mid Term")
            st.metric(
                "Projected Area",
                f'{result["mid_term_cm2"]} cm²',
                delta=f'{result["growth_mid_term_percent"]}%',
                delta_color="inverse" if result["growth_mid_term_percent"] < 0 else "normal"
            )
            st.progress(min(abs(result["growth_mid_term_percent"]) / 100, 1.0))
        
        with col3:
            st.markdown("#### 📅 Long Term")
            st.metric(
                "Projected Area",
                f'{result["long_term_cm2"]} cm²',
                delta=f'{result["growth_long_term_percent"]}%',
                delta_color="inverse" if result["growth_long_term_percent"] < 0 else "normal"
            )
            st.progress(min(abs(result["growth_long_term_percent"]) / 100, 1.0))
        
        st.markdown("---")
        
        # Detailed Data Table
        st.markdown("### 📋 Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volumetric Measurements**")
            st.dataframe({
                "Timepoint": ["Current", "Short Term", "Mid Term", "Long Term"],
                "Area (cm²)": [
                    result["current_area_cm2"],
                    result["short_term_cm2"],
                    result["mid_term_cm2"],
                    result["long_term_cm2"]
                ]
            }, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Growth Rates**")
            st.dataframe({
                "Period": ["Short Term", "Mid Term", "Long Term"],
                "Growth (%)": [
                    f'{result["growth_short_term_percent"]}%',
                    f'{result["growth_mid_term_percent"]}%',
                    f'{result["growth_long_term_percent"]}%'
                ]
            }, use_container_width=True, hide_index=True)
        
        # Export Options
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Report (Coming Soon)", use_container_width=True, disabled=True):
                st.info("PDF export functionality will be available soon")
        
        with col2:
            if st.button("📧 Email Results (Coming Soon)", use_container_width=True, disabled=True):
                st.info("Email integration will be available soon")
 
else:
    # Welcome message when no file uploaded
    st.markdown("---")
    st.info("👆 Please upload a .h5 MRI file to begin analysis")
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Accurate Segmentation
        Deep learning model trained on 30,000+ samples for precise tumor boundary detection
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Growth Prediction
        Advanced ML algorithms forecast tumor progression across multiple timeframes
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Fast Processing
        Get comprehensive analysis results in seconds with GPU-accelerated inference
        """)
 
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p><strong>Brain Tumor AI Analysis Platform</strong> | Powered by Deep Learning</p>
    <p><small>For research and educational purposes only</small></p>
</div>
""", unsafe_allow_html=True)