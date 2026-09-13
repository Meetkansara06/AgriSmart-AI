import os
import tempfile
import streamlit as st
from PIL import Image

# Import model prediction module
from model.predict import predict_with_confidence

# Import src modules created by teammates
from src.precautions import get_precautions
from src.weather import get_weather
from src.irrigation import irrigation_advice
from src.sustainability import sustainability_score

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AgriSmart AI — Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ==============================================================================
# Sidebar: UI Theme & Farm Parameters Configuration
# ==============================================================================
st.sidebar.markdown("## 🎨 App Theme")
theme_choice = st.sidebar.radio("Select Interface Theme", ["☀️ Light Theme", "🌙 Dark Theme"], index=0)
is_dark = "Dark" in theme_choice

# Define Theme Colors
if is_dark:
    bg_color = "#0F172A"
    sidebar_bg = "#1E293B"
    card_bg = "#1E293B"
    card_border = "#334155"
    text_primary = "#F8FAFC"
    text_secondary = "#94A3B8"
    input_bg = "#334155"
    input_text = "#FFFFFF"
    input_border = "#475569"
    uploader_bg = "#1E293B"
    uploader_border = "#475569"
    uploader_btn_bg = "#334155"
    alert_bg = "#064E3B"
    alert_text = "#ECFDF5"
    alert_border = "#059669"
    header_color = "#4ADE80"
    sub_header_color = "#86EFAC"
    weather_box_bg = "#1E293B"
    weather_box_border = "#334155"
    weather_val_color = "#4ADE80"
    weather_lbl_color = "#86EFAC"
    precaution_bg = "#1E293B"
    precaution_border = "#22C55E"
    precaution_num_bg = "#16A34A"
    status_healthy_bg = "#064E3B"
    status_healthy_text = "#6EE7B7"
    status_healthy_border = "#047857"
    status_diseased_bg = "#7F1D1D"
    status_diseased_text = "#FCA5A5"
    status_diseased_border = "#B91C1C"
    advice_ok_bg = "#064E3B"
    advice_ok_border = "#059669"
    advice_ok_header = "#6EE7B7"
    advice_warn_bg = "#451A03"
    advice_warn_border = "#D97706"
    advice_warn_header = "#FDE68A"
else:
    bg_color = "#FFFFFF"
    sidebar_bg = "#F8FAFC"
    card_bg = "#FFFFFF"
    card_border = "#E2E8F0"
    text_primary = "#1A202C"
    text_secondary = "#718096"
    input_bg = "#FFFFFF"
    input_text = "#1A202C"
    input_border = "#CBD5E1"
    uploader_bg = "#F8FAFC"
    uploader_border = "#94A3B8"
    uploader_btn_bg = "#E2E8F0"
    alert_bg = "#E8F5E9"
    alert_text = "#1B5E20"
    alert_border = "#A5D6A7"
    header_color = "#1B4D3E"
    sub_header_color = "#2E7D32"
    weather_box_bg = "#F4F9F4"
    weather_box_border = "#C8E6C9"
    weather_val_color = "#2E7D32"
    weather_lbl_color = "#558B2F"
    precaution_bg = "#F8FAF8"
    precaution_border = "#2E7D32"
    precaution_num_bg = "#2E7D32"
    status_healthy_bg = "#E8F5E9"
    status_healthy_text = "#1B5E20"
    status_healthy_border = "#A5D6A7"
    status_diseased_bg = "#FFEBEE"
    status_diseased_text = "#C62828"
    status_diseased_border = "#EF9A9A"
    advice_ok_bg = "#E8F5E9"
    advice_ok_border = "#2E7D32"
    advice_ok_header = "#1B4D3E"
    advice_warn_bg = "#FFF8E1"
    advice_warn_border = "#F57F17"
    advice_warn_header = "#1B4D3E"

# ==============================================================================
# Custom CSS Styling — Modern Glassmorphism & Dynamic Light/Dark Theme
# ==============================================================================
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Outfit', sans-serif;
            background-color: {bg_color} !important;
            color: {text_primary} !important;
        }}

        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
        }}

        /* Specific sidebar headers, labels, and text elements */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span:not([data-baseweb="select"] span),
        [data-testid="stSidebar"] div[class*="stCaption"] {{
            color: {text_primary} !important;
        }}

        /* Sidebar Input Fields - High Contrast Visibility in Both Themes */
        [data-testid="stSidebar"] input {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            border: 1px solid {input_border} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] div {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border-color: {input_border} !important;
            border-radius: 8px !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] div[role="button"] {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            border-color: {input_border} !important;
            border-radius: 8px !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            font-weight: 600 !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] svg {{
            fill: {input_text} !important;
        }}

        /* Number Input Buttons (+/-) */
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border-color: {input_border} !important;
        }}

        [data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] svg,
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] svg {{
            fill: {input_text} !important;
            color: {input_text} !important;
        }}

        /* File Uploader Container & Dropzone High-Contrast Styling */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {uploader_bg} !important;
            border: 2px dashed {uploader_border} !important;
            border-radius: 12px !important;
        }}

        [data-testid="stFileUploaderDropzone"] * {{
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
        }}

        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {uploader_btn_bg} !important;
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
            border: 1px solid {uploader_border} !important;
            border-radius: 8px !important;
        }}

        /* Top Header Strip - Transparent Seamless Theme Integration */
        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        header {{
            background-color: transparent !important;
            background: transparent !important;
        }}
        header[data-testid="stHeader"] *,
        [data-testid="stHeader"] * {{
            color: {text_primary} !important;
        }}

        /* Alert / st.info Box High-Contrast Styling */
        [data-testid="stAlert"], div[class*="stAlert"] {{
            background-color: {alert_bg} !important;
            color: {alert_text} !important;
            border: 1px solid {alert_border} !important;
            border-radius: 12px !important;
        }}

        [data-testid="stAlert"] *, div[class*="stAlert"] * {{
            color: {alert_text} !important;
            -webkit-text-fill-color: {alert_text} !important;
        }}

        /* Hero Header */
        .hero-container {{
            background: linear-gradient(135deg, #0d3b25 0%, #1e5631 50%, #2e7d32 100%);
            padding: 2rem 2.2rem;
            border-radius: 18px;
            color: #FFFFFF !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.8rem;
        }}
        
        .hero-title {{
            font-size: 2.4rem;
            font-weight: 700;
            color: #FFFFFF !important;
            margin-bottom: 0.3rem;
        }}
        
        .hero-subtitle {{
            font-size: 1.1rem;
            color: #C8E6C9 !important;
            font-weight: 400;
            margin-bottom: 1rem;
        }}

        .hero-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .badge-pill {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #E8F5E9 !important;
            padding: 0.3rem 0.85rem;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 500;
        }}

        /* Cards & Section Headers */
        .custom-card {{
            background: {card_bg} !important;
            border-radius: 16px;
            padding: 1.6rem;
            border: 1px solid {card_border} !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.2rem;
            color: {text_primary} !important;
        }}

        .section-header {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {header_color} !important;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Status Badges */
        .status-pill-healthy {{
            background-color: {status_healthy_bg} !important;
            color: {status_healthy_text} !important;
            border: 1.5px solid {status_healthy_border} !important;
            padding: 0.45rem 1.1rem;
            border-radius: 30px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .status-pill-diseased {{
            background-color: {status_diseased_bg} !important;
            color: {status_diseased_text} !important;
            border: 1.5px solid {status_diseased_border} !important;
            padding: 0.45rem 1.1rem;
            border-radius: 30px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        /* Weather Metric Boxes */
        .weather-box {{
            background: {weather_box_bg} !important;
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid {weather_box_border} !important;
            text-align: center;
        }}

        .weather-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {weather_val_color} !important;
        }}

        .weather-label {{
            font-size: 0.85rem;
            color: {weather_lbl_color} !important;
            font-weight: 600;
            text-transform: uppercase;
        }}

        /* Sustainability Score Badge Colors */
        .score-callout {{
            background: linear-gradient(135deg, #1e5631 0%, #2e7d32 100%);
            color: #FFFFFF !important;
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
        }}

        .precaution-item {{
            background: {precaution_bg} !important;
            border-left: 4px solid {precaution_border} !important;
            padding: 0.9rem 1.1rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
            color: {text_primary} !important;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}

        .precaution-num {{
            background: {precaution_num_bg} !important;
            color: #FFFFFF !important;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 2px;
        }}

        /* Navigation Tabs Styling */
        button[data-baseweb="tab"] {{
            color: {text_secondary} !important;
            font-weight: 500;
        }}
        button[aria-selected="true"] {{
            color: {header_color} !important;
            font-weight: 700 !important;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# Sidebar: Farm Parameters Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("## 🚜 Farm Parameters")
st.sidebar.caption("Configure field location, crop stage, soil, and nutrient levels.")

st.sidebar.markdown("### 📍 Location & Climate")
lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=23.03, step=0.01, help="e.g., 23.03 for Ahmedabad")
lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=72.58, step=0.01, help="e.g., 72.58 for Ahmedabad")

st.sidebar.markdown("### 🌱 Crop & Field Setup")
crop_stage = st.sidebar.selectbox("Crop Growth Stage", ["seedling", "growing", "mature"], index=1)
soil_type = st.sidebar.selectbox("Soil Type", ["loamy", "sandy", "clay"], index=0)

st.sidebar.markdown("### 💧 Resource Usage")
water_used_liters = st.sidebar.number_input("Water Used (Liters)", min_value=0.0, value=500.0, step=50.0)
fertilizer_level = st.sidebar.slider("Fertilizer Usage Level (0–100)", min_value=0.0, max_value=100.0, value=30.0, step=5.0)

st.sidebar.markdown("### 🧪 Soil Nutrients & Environment")
n_val = st.sidebar.number_input("Nitrogen (N, mg/kg)", min_value=0, max_value=300, value=90)
p_val = st.sidebar.number_input("Phosphorus (P, mg/kg)", min_value=0, max_value=300, value=42)
k_val = st.sidebar.number_input("Potassium (K, mg/kg)", min_value=0, max_value=300, value=43)
humidity = st.sidebar.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
soil_ph = st.sidebar.slider("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# ==============================================================================
# Hero Header Banner
# ==============================================================================
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌱 AgriSmart AI — Smart Farming Assistant</div>
        <div class="hero-subtitle">
            AI-Driven Crop Disease Diagnosis, Weather Intelligence, Smart Irrigation & Sustainability Scoring
        </div>
        <div class="hero-badges">
            <span class="badge-pill">🔬 Disease Detection</span>
            <span class="badge-pill">🌤️ Live Weather</span>
            <span class="badge-pill">💧 Smart Irrigation</span>
            <span class="badge-pill">📊 Sustainability Score</span>
            <span class="badge-pill">🌾 Crop Recommendation</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# Navigation Tabs for Main Sections
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 1. Disease Detection",
    "🌤️ 2. Weather & Irrigation",
    "📊 3. Sustainability Score",
    "🌾 4. Crop Recommendation"
])

# Initialize Session State for Disease Detection status
if "disease_detected" not in st.session_state:
    st.session_state["disease_detected"] = False

# ==============================================================================
# Section 1: Disease Detection & Precautions
# ==============================================================================
with tab1:
    st.markdown('<div class="section-header">📸 Upload Crop Leaf Image</div>', unsafe_allow_html=True)
    st.write("Upload a clear, close-up image of the affected plant leaf for AI disease diagnosis and treatment steps.")

    uploaded_file = st.file_uploader(
        label="Select a leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        try:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1.1])

            with col1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Crop Leaf Specimen", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                with st.spinner("🔬 Running AgriSmart AI Neural Network Inference..."):
                    raw_label, confidence = predict_with_confidence(temp_path)
                    precaution_info = get_precautions(raw_label)

                is_healthy = precaution_info["status"] == "Healthy"
                st.session_state["disease_detected"] = not is_healthy
                
                status_class = "status-pill-healthy" if is_healthy else "status-pill-diseased"
                status_icon = "🟢" if is_healthy else "⚠️"
                status_text = "HEALTHY CROP" if is_healthy else "DISEASE DETECTED"

                st.markdown(
                    f'<div class="custom-card">'
                    f'<div style="font-size: 0.85rem; color: {text_secondary}; font-weight: 600; text-transform: uppercase;">'
                    f'Crop Category: <span style="color: {header_color};">{precaution_info["crop"]}</span></div>'
                    f'<h2 style="color: {text_primary}; margin-top: 0.2rem; margin-bottom: 0.8rem; font-size: 1.7rem;">'
                    f'{precaution_info["disease"]}</h2>'
                    f'<div style="margin-bottom: 1rem;">'
                    f'<span class="{status_class}">{status_icon} {status_text}</span></div>'
                    f'<div class="weather-box">'
                    f'<div class="weather-label">AI Model Confidence</div>'
                    f'<div class="weather-value">{confidence * 100:.2f}%</div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                st.progress(float(confidence))

            # Actionable Precautions List
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">🛡️ Recommended Precautions & Action Plan for {precaution_info["disease"]}</div>', unsafe_allow_html=True)
            
            precaution_items = "".join([
                f'<div class="precaution-item"><div class="precaution-num">{i}</div><div>{item}</div></div>'
                for i, item in enumerate(precaution_info["precautions"], 1)
            ])
            st.markdown(f'<div class="custom-card">{precaution_items}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during disease prediction: {str(e)}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    else:
        st.info("👆 Upload a crop leaf image above to receive instant disease classification and precautions.")

# ==============================================================================
# Section 2: Weather & Irrigation
# ==============================================================================
with tab2:
    st.markdown('<div class="section-header">🌤️ Live Weather Forecast & Smart Irrigation Advisor</div>', unsafe_allow_html=True)
    st.write(f"Fetching location forecast for **Latitude: {lat}**, **Longitude: {lon}** via Open-Meteo API.")

    try:
        weather_data = get_weather(lat, lon)
        rain_tomorrow = weather_data["rain_tomorrow_mm"]
        max_t = weather_data["max_temp"]
        min_t = weather_data["min_temp"]

        # Weather Metrics Cards
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown(
                f'<div class="weather-box">'
                f'<div class="weather-label">🌧️ Rain Tomorrow</div>'
                f'<div class="weather-value">{rain_tomorrow} <span style="font-size: 1rem;">mm</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_w2:
            st.markdown(
                f'<div class="weather-box">'
                f'<div class="weather-label">🌡️ Max Temperature</div>'
                f'<div class="weather-value">{max_t} <span style="font-size: 1rem;">°C</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_w3:
            st.markdown(
                f'<div class="weather-box">'
                f'<div class="weather-label">❄️ Min Temperature</div>'
                f'<div class="weather-value">{min_t} <span style="font-size: 1rem;">°C</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Irrigation Recommendation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">💧 Irrigation Decision</div>', unsafe_allow_html=True)

        advice = irrigation_advice(rain_tomorrow, crop_stage, soil_type)
        
        # Color coding advice based on rain/action
        is_delay_or_skip = "Delay" in advice or "Skip" in advice
        advice_bg = advice_ok_bg if is_delay_or_skip else advice_warn_bg
        advice_border = advice_ok_border if is_delay_or_skip else advice_warn_border
        advice_hdr = advice_ok_header if is_delay_or_skip else advice_warn_header

        st.markdown(
            f'<div class="custom-card" style="background-color: {advice_bg} !important; border-left: 5px solid {advice_border} !important;">'
            f'<h3 style="margin-top: 0; color: {advice_hdr} !important;">💧 Field Irrigation Guidance</h3>'
            f'<p style="font-size: 1.1rem; font-weight: 500; color: {text_primary} !important; margin-bottom: 0.5rem;">{advice}</p>'
            f'<div style="font-size: 0.85rem; color: {text_secondary} !important;">'
            f'Parameters used: <b>{rain_tomorrow}mm</b> expected rain • <b>{crop_stage.capitalize()}</b> stage • <b>{soil_type.capitalize()}</b> soil'
            f'</div></div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"⚠️ Could not retrieve live weather forecast: {str(e)}")

# ==============================================================================
# Section 3: Sustainability Score
# ==============================================================================
with tab3:
    st.markdown('<div class="section-header">📊 Farm Sustainability Score & Resource Rating</div>', unsafe_allow_html=True)
    st.write("Calculates an explainable 0–100 sustainability index based on water efficiency, crop disease status, and fertilizer usage.")

    # Calculate sustainability score from src/sustainability.py
    sust_result = sustainability_score(
        water_used_liters=water_used_liters,
        max_water=1000.0,  # Reference maximum baseline
        disease_detected=st.session_state["disease_detected"],
        fertilizer_level=fertilizer_level
    )

    col_s1, col_s2 = st.columns([1, 1.2])

    with col_s1:
        badge_colors = {
            "Platinum": "🏆 Platinum",
            "Green Guardian": "🛡️ Green Guardian",
            "Needs Improvement": "📈 Needs Improvement",
            "At Risk": "⚠️ At Risk"
        }
        badge_display = badge_colors.get(sust_result["badge"], "🌱 " + sust_result["badge"])

        st.markdown(
            f'<div class="score-callout">'
            f'<div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Overall Sustainability Index</div>'
            f'<div style="font-size: 3.2rem; font-weight: 700; margin: 0.4rem 0;">{sust_result["score"]} / 100</div>'
            f'<div style="font-size: 1.1rem; font-weight: 600; background: rgba(255,255,255,0.2); padding: 0.4rem 1rem; border-radius: 20px; display: inline-block;">'
            f'{badge_display}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_s2:
        st.markdown(
            f'<div class="custom-card">'
            f'<h4 style="margin-top: 0; color: {header_color} !important;">Component Breakdown</h4>'
            f'<div style="margin-bottom: 0.8rem;">'
            f'💧 <b>Water Efficiency Score:</b> {sust_result["water_score"]:.1f}%</div>'
            f'<div style="margin-bottom: 0.8rem;">'
            f'🌿 <b>Crop Health Score:</b> {sust_result["health_score"]:.1f}% '
            f'<i>({"No Disease Detected" if not st.session_state["disease_detected"] else "Disease Alert On Field"})</i></div>'
            f'<div style="margin-bottom: 1rem;">'
            f'🧪 <b>Resource/Fertilizer Efficiency:</b> {sust_result["resource_score"]:.1f}%</div>'
            f'<hr style="border: 0; border-top: 1px solid {card_border}; margin: 0.8rem 0;">'
            f'<div style="color: {sub_header_color} !important; font-weight: 600;">💡 Key Recommendation:</div>'
            f'<div style="color: {text_secondary} !important; font-size: 0.95rem; margin-top: 0.2rem;">{sust_result["suggestion"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ==============================================================================
# Section 4: Crop Recommendation
# ==============================================================================
with tab4:
    st.markdown('<div class="section-header">🌾 Smart Crop Recommendation Engine</div>', unsafe_allow_html=True)
    st.write("Agronomic matching based on soil NPK levels, soil pH, humidity, and soil type.")

    # Rule-based agronomic crop matching
    crop_profiles = [
        {"name": "Rice / Paddy", "n": 90, "p": 40, "k": 40, "ph_range": (5.5, 7.0), "soil": "clay", "icon": "🌾"},
        {"name": "Corn (Maize)", "n": 80, "p": 45, "k": 40, "ph_range": (5.8, 7.2), "soil": "loamy", "icon": "🌽"},
        {"name": "Wheat", "n": 50, "p": 25, "k": 25, "ph_range": (6.0, 7.5), "soil": "loamy", "icon": "🌾"},
        {"name": "Cotton", "n": 120, "p": 45, "k": 45, "ph_range": (6.0, 8.0), "soil": "loamy", "icon": "☁️"},
        {"name": "Tomato", "n": 100, "p": 50, "k": 50, "ph_range": (6.0, 6.8), "soil": "loamy", "icon": "🍅"},
        {"name": "Potato", "n": 90, "p": 50, "k": 50, "ph_range": (5.0, 6.5), "soil": "sandy", "icon": "🥔"},
        {"name": "Bell Pepper", "n": 80, "p": 40, "k": 40, "ph_range": (6.0, 7.0), "soil": "loamy", "icon": "🫑"},
        {"name": "Apple", "n": 60, "p": 30, "k": 30, "ph_range": (5.5, 6.8), "soil": "loamy", "icon": "🍎"},
        {"name": "Grape", "n": 40, "p": 30, "k": 30, "ph_range": (5.5, 7.0), "soil": "sandy", "icon": "🍇"},
    ]

    # Calculate match score for each crop profile
    scores = []
    for c in crop_profiles:
        # Distance penalty for NPK
        npk_diff = abs(c["n"] - n_val) + abs(c["p"] - p_val) + abs(c["k"] - k_val)
        
        # pH suitability check
        ph_ok = c["ph_range"][0] <= soil_ph <= c["ph_range"][1]
        ph_bonus = 20 if ph_ok else 0
        
        # Soil type match
        soil_bonus = 15 if c["soil"] == soil_type else 0

        match_score = max(0.0, 100.0 - (npk_diff * 0.4) + ph_bonus + soil_bonus)
        scores.append((match_score, c))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_match = scores[0]

    col_cr1, col_cr2 = st.columns([1, 1.1])

    with col_cr1:
        st.markdown(
            f'<div class="custom-card" style="border-left: 5px solid {precaution_border} !important;">'
            f'<div style="font-size: 0.85rem; color: {sub_header_color} !important; font-weight: 700; text-transform: uppercase;">'
            f'⭐ Top Recommended Crop</div>'
            f'<h2 style="color: {header_color} !important; margin: 0.4rem 0;">{best_match[1]["icon"]} {best_match[1]["name"]}</h2>'
            f'<div style="font-size: 1.1rem; font-weight: 600; color: {weather_val_color} !important; margin-bottom: 0.8rem;">'
            f'Agronomic Suitability: {min(99.0, best_match[0]):.1f}%</div>'
            f'<div style="font-size: 0.9rem; color: {text_secondary} !important;">'
            f'Matches your field parameters: <b>N={n_val}, P={p_val}, K={k_val}</b>, Soil pH <b>{soil_ph}</b>, and <b>{soil_type.capitalize()}</b> soil.'
            f'</div></div>',
            unsafe_allow_html=True
        )

    with col_cr2:
        st.markdown(f'<div class="custom-card"><h4 style="margin-top:0; color:{header_color} !important;">Alternative Suitable Crops</h4>', unsafe_allow_html=True)
        for score_val, crop_data in scores[1:4]:
            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">'
                f'<div>{crop_data["icon"]} <b>{crop_data["name"]}</b></div>'
                f'<div style="color: {weather_val_color} !important; font-weight: 600;">{min(98.0, score_val):.1f}% Match</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
