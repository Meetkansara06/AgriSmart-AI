import os
import tempfile
import streamlit as st
from PIL import Image

# Import model prediction and precautions modules
from model.predict import predict_with_confidence
from src.precautions import get_precautions

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AgriSmart AI — Crop Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS — Rich Modern & Farmer-Friendly Aesthetic
# ==============================================================================
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Hero Header Section */
        .hero-container {
            background: linear-gradient(135deg, #0d3b25 0%, #1e5631 50%, #2e7d32 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            color: #FFFFFF;
            box-shadow: 0 10px 25px rgba(13, 59, 37, 0.2);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .hero-subtitle {
            font-size: 1.15rem;
            color: #C8E6C9;
            font-weight: 400;
            max-width: 800px;
            margin-bottom: 1.2rem;
        }

        .hero-badges {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .badge-pill {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #E8F5E9;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        /* Card Container Styling */
        .custom-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.8rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .custom-card:hover {
            box-shadow: 0 8px 25px rgba(46, 125, 50, 0.08);
        }

        .card-header-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #1B4D3E;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Status Pills */
        .status-pill-healthy {
            background-color: #E8F5E9;
            color: #1B5E20;
            border: 1.5px solid #A5D6A7;
            padding: 0.5rem 1.2rem;
            border-radius: 30px;
            font-weight: 600;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .status-pill-diseased {
            background-color: #FFEBEE;
            color: #C62828;
            border: 1.5px solid #EF9A9A;
            padding: 0.5rem 1.2rem;
            border-radius: 30px;
            font-weight: 600;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        /* Metric Box */
        .metric-callout {
            background: #F4F9F4;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border: 1px solid #C8E6C9;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2E7D32;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #558B2F;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }

        /* Precautions List Styling */
        .precaution-item {
            background: #F8FAF8;
            border-left: 4px solid #2E7D32;
            padding: 1rem 1.2rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 0.75rem;
            font-size: 0.98rem;
            color: #2D3748;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .precaution-num {
            background: #2E7D32;
            color: #FFFFFF;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 2px;
        }

        /* Hide Streamlit Menu Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Hero Header Banner
# ==============================================================================
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌱 AgriSmart AI</div>
        <div class="hero-subtitle">
            Advanced Deep Learning Crop Leaf Disease Detection & Actionable Farmer Advice
        </div>
        <div class="hero-badges">
            <span class="badge-pill">⚡ MobileNetV2 Deep Learning</span>
            <span class="badge-pill">🔍 18 Plant Categories</span>
            <span class="badge-pill">🛡️ Instant Precautions</span>
            <span class="badge-pill">🌿 Farmer-Friendly UI</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# Disease Detection Section (Day 2 Shell & Inference)
# ==============================================================================
st.markdown('<div class="card-header-title">📸 Upload Plant Leaf Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drag and drop or select a leaf image for disease diagnosis...",
    type=["jpg", "jpeg", "png"],
    help="Supported file formats: JPG, JPEG, PNG (Max size: 200MB)"
)

if uploaded_file is not None:
    # Save uploaded file temporarily for TensorFlow prediction pipeline
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    try:
        image = Image.open(uploaded_file)
        
        # Display Columns: Left = Uploaded Image, Right = AI Diagnosis Report
        col1, col2 = st.columns([1, 1.1])

        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Crop Leaf Specimen", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            with st.spinner("🔬 Running AgriSmart Neural Network Analysis..."):
                raw_label, confidence = predict_with_confidence(temp_path)
                precaution_info = get_precautions(raw_label)

            is_healthy = precaution_info["status"] == "Healthy"
            status_class = "status-pill-healthy" if is_healthy else "status-pill-diseased"
            status_icon = "🟢" if is_healthy else "⚠️"
            status_text = "HEALTHY CROP" if is_healthy else "DISEASE DETECTED"

            st.markdown(
                f'<div class="custom-card">'
                f'<div style="font-size: 0.9rem; color: #718096; font-weight: 600; text-transform: uppercase;">'
                f'Crop Category: <span style="color: #2E7D32;">{precaution_info["crop"]}</span></div>'
                f'<h2 style="color: #1A202C; margin-top: 0.2rem; margin-bottom: 0.8rem; font-size: 1.8rem;">'
                f'{precaution_info["disease"]}</h2>'
                f'<div style="margin-bottom: 1.2rem;">'
                f'<span class="{status_class}">{status_icon} {status_text}</span></div>'
                f'<div class="metric-callout">'
                f'<div class="metric-label">AI Diagnostic Confidence</div>'
                f'<div class="metric-value">{confidence * 100:.2f}%</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

            # Native Streamlit progress bar callout
            st.progress(float(confidence))

        # Precautions and Actionable Plan Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="card-header-title">🛡️ Recommended Precautions & Treatment Plan for {precaution_info["disease"]}</div>', unsafe_allow_html=True)
        
        precaution_items = "".join([
            f'<div class="precaution-item"><div class="precaution-num">{i}</div><div>{item}</div></div>'
            for i, item in enumerate(precaution_info["precautions"], 1)
        ])
        
        st.markdown(f'<div class="custom-card">{precaution_items}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ An error occurred during image processing or disease prediction: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("👆 Please upload a plant leaf image above to view instant disease diagnosis, confidence metrics, and farmer precautions.")

# ==============================================================================
# Future Day 3 Expansion Hooks
# ==============================================================================
# TODO: Day 3 - Integrate Weather (src/weather.py)
# TODO: Day 3 - Integrate Irrigation (src/irrigation.py)
# TODO: Day 3 - Integrate Sustainability Score (src/sustainability.py)
# TODO: Day 3 - Integrate Crop Recommendation & Sidebar Input Parameters
