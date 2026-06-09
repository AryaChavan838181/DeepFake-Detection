"""
DFDetective - Deepfake Detection Dashboard
Premium dark-themed deepfake detector with Consolas-style typography
"""
import streamlit as st
import numpy as np
import time
import os
import tempfile
import io
from datetime import datetime

# Initialize Predictor lazily
@st.cache_resource
def get_predictor():
    from utils.real_predictor import RealPredictor
    return RealPredictor()

from utils.pdf_generator import build_pdf_report

# Page config
st.set_page_config(
    page_title="DFDetective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING & THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Consolas:wght@400;700&display=swap');
    
    :root {
        --primary-yellow: #FFD700;
        --bg-black: #0E1117;
        --card-bg: #1A1C24;
        --border-color: #2D3139;
        --text-white: #E0E0E0;
        --text-muted: #A0A0A0;
    }

    /* Global styles */
    .stApp {
        background-color: var(--bg-black);
        color: var(--text-white);
    }
    
    * {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    }

    /* Hide Streamlit elements */
    header[data-testid="stHeader"], footer {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* SIDEBAR: Dark Background */
    [data-testid="stSidebar"] {
        background-color: var(--card-bg) !important;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: var(--text-white) !important;
    }

    /* Main Container Width */
    [data-testid="stMainBlockContainer"] {
        max-width: 60vw !important;
        margin: auto !important;
        padding-top: 2rem !important;
    }

    /* Typography */
    h1 { font-size: 2.8rem !important; text-align: center; color: var(--primary-yellow) !important; font-weight: 700 !important; margin-bottom: 0 !important; }
    h3 { font-size: 1.0rem !important; color: var(--text-muted) !important; font-weight: 400 !important; text-align: center; letter-spacing: 0.2em; margin-top: 0.5rem !important; margin-bottom: 1.5rem !important; }
    
    .centered-text {
        text-align: center !important;
        color: var(--text-white);
        margin-bottom: 2rem;
        font-size: 0.85rem !important;
    }

    /* DROP BOX RESTRUCTURE - 40vw x 30vh */
    [data-testid="stFileUploadDropzone"] {
        background-color: var(--card-bg) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 8px !important;
        width: 40vw !important;
        height: 30vh !important;
        margin: 0 auto !important;
    }

    /* Vertical stack for inner uploader elements */
    [data-testid="stFileUploadDropzone"] section {
        display: flex !important;
        flex-direction: column !important; 
        align-items: center !important;
        justify-content: center !important;
        gap: 15px !important;
        width: 100% !important;
        height: 100% !important;
    }

    /* STATUS BOX (Replacement for Uploader) */
    .status-box {
        width: 40vw;
        height: 30vh;
        background-color: var(--card-bg);
        border: 2px solid var(--primary-yellow);
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        text-align: center;
    }
    .status-header {
        color: var(--primary-yellow);
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .file-name {
        color: var(--text-white);
        font-size: 1rem;
        word-break: break-all;
        padding: 0 20px;
    }

    /* START ANALYZING BUTTON: Center align to page */
    .btn-container {
        width: 100%;
        display: flex !important;
        justify-content: center !important;
        margin-top: 2.5rem;
    }

    .stButton {
        display: flex !important;
        justify-content: center !important;
    }

    /* Button Styling */
    .stButton > button {
        background-color: var(--primary-yellow) !important;
        color: var(--bg-black) !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.8rem 4rem !important;
        text-transform: uppercase;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: scale(1.02);
    }

    /* Verdict Cards */
    .verdict-card { padding: 1.5rem; border-radius: 4px; margin-bottom: 1.5rem; border: 1px solid; text-align: center; }
    .verdict-fake { border-color: #FF4B4B; background-color: rgba(255, 75, 75, 0.05); }
    .verdict-real { border-color: #00FFAB; background-color: rgba(0, 255, 171, 0.05); }

</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP LOGIC
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### SYSTEM OVERVIEW")
        st.markdown("""
        DFDetective is a production-grade engine.
        
        **ENGINE V5.0.0**
        - IMAGE ENSEMBLE: ACTIVE
        - VIDEO ENSEMBLE: ACTIVE
        - AUDIO ENSEMBLE: ACTIVE
        """)
        st.markdown("---")
        st.markdown("### DETECTION LOGS")
        st.metric("TOTAL SCANS (Today)", "342")

    # Header
    st.markdown("<h1>DFDETECTIVE</h1>", unsafe_allow_html=True)
    st.markdown("<h3>FORENSIC DEEPFAKE DETECTION ENGINE</h3>", unsafe_allow_html=True)
    st.markdown("<p class='centered-text'>upload your media for deepfake detection.</p>", unsafe_allow_html=True)

    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    if not st.session_state.analyzed:
        uploaded_file = st.file_uploader(
            "Upload", 
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mp3", "wav"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            # HIDE UPLOADER
            st.markdown("<style>[data-testid='stFileUploadDropzone'] { display: none !important; }</style>", unsafe_allow_html=True)
            
            # SHOW STATUS BOX
            st.markdown(f"""
                <div class="status-box">
                    <div class="status-header">FILE UPLOADED</div>
                    <div class="file-name">{uploaded_file.name}</div>
                </div>
            """, unsafe_allow_html=True)

            # Analyze Button - Centered on page
            st.markdown('<div class="btn-container">', unsafe_allow_html=True)
            if st.button("START ANALYSING"):
                with st.spinner("INITIATING FORENSIC SCAN..."):
                    
                    # --- REAL PREDICTION LOGIC ---
                    try:
                        import os
                        
                        # Calculate file type extension
                        ext = uploaded_file.name.split('.')[-1].lower()
                        
                        # Needs physical file for some video/audio operations
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, f"temp_upload.{ext}")
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Initialize Model
                        predictor = get_predictor()
                        
                        # Analyze media
                        analysis_results = predictor.analyze_media(temp_path, ext)
                        
                        st.session_state.result = analysis_results
                        st.session_state.analyzed = True
                        st.session_state.file_info = {'name': uploaded_file.name, 'type': ext}
                        
                    except Exception as e:
                        st.error(f"Error initializing analysis: {e}")
                        
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Results View
        res = st.session_state.result
        is_deepfake = res['is_deepfake']
        conf = int(res['confidence'] * 100)
        file_info = st.session_state.file_info
        
        st.markdown(f"### Report for {file_info['name']}")
        st.markdown(f"**Primary Warning:** {res.get('flag_reason', '')}")
        
        if is_deepfake:
            st.markdown(f"<div class='verdict-card verdict-fake'><h2 style='color:#FF4B4B;'>[ ALERT ] DEEPFAKE DETECTED</h2><p>{conf}% Synthetic Probability</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='verdict-card verdict-real'><h2 style='color:#00FFAB;'>[ VERIFIED ] AUTHENTIC CONTENT</h2><p>{conf}% Authentic Confidence</p></div>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### Forensic Evidence")
        
        # DISPLAY REAL PLOTS IN PREVIEW
        for plot in res.get('plots', []):
            st.image(plot['stream'], caption=plot['title'])
            
        st.markdown("---")
        
        # GENERATE PDF
        pdf_buffer = io.BytesIO()
        build_pdf_report(file_info, res, pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="DOWNLOAD FULL PDF REPORT",
                data=pdf_bytes,
                file_name=f"DFDetective_Report_{file_info['name']}.pdf",
                mime="application/pdf"
            )
        
        with col2:
            if st.button("NEW SCAN"):
                st.session_state.analyzed = False
                st.session_state.result = None
                st.session_state.file_info = None
                st.rerun()

if __name__ == "__main__":
    main()
