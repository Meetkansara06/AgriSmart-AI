import os
import tempfile
import streamlit as st
from PIL import Image

# Import model prediction module
from model.predict import predict_with_confidence
from model.crop_predict import predict_crop

# Import src modules created by teammates
from src.precautions import get_precautions
from src.weather import get_weather
from src.irrigation import irrigation_advice
from src.sustainability import sustainability_score
from src.farmer_assistant import FarmerAssistantError, get_farmer_response

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
theme_choice = st.sidebar.radio("Select Interface Theme", ["☀️ Light Theme", "🌙 Dark Theme"], index=0, horizontal=True)
is_dark = "Dark" in theme_choice

# Define Theme Colors
if is_dark:
    bg_color = "#0F172A"
    sidebar_bg = "#1E293B"
    card_bg = "#1E293B"
    card_border = "#334155"
    text_primary = "#F8FAFC"
    text_secondary = "#CBD5E1"
    input_bg = "#1E293B"
    input_text = "#F8FAFC"
    input_border = "#64748B"
    uploader_bg = "#1E293B"
    uploader_border = "#64748B"
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
    text_secondary = "#4A5568"
    input_bg = "#FFFFFF"
    input_text = "#1A202C"
    input_border = "#94A3B8"
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

        /* Remove top rainbow decoration bar completely */
        [data-testid="stDecoration"],
        div[data-testid="stDecoration"],
        div[class*="stDecoration"] {{
            display: none !important;
            height: 0px !important;
            visibility: hidden !important;
        }}

        /* Force top header bar to be transparent in both Light and Dark themes */
        header,
        [data-testid="stHeader"],
        div[data-testid="stHeader"],
        header[data-testid="stHeader"],
        [data-testid="stAppHeader"],
        div[data-testid="stAppHeader"],
        [class*="stAppHeader"],
        .stAppHeader,
        [class*="stAppToolbar"],
        .stAppToolbar {{
            background-color: transparent !important;
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }}

        header *,
        [data-testid="stHeader"] *,
        [class*="stAppHeader"] * {{
            color: {text_primary} !important;
        }}

        /* Sidebar Container */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
        }}

        /* Sidebar Section Headings with Accent Underlines */
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {{
            color: {header_color} !important;
            -webkit-text-fill-color: {header_color} !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            margin-top: 1.3rem !important;
            margin-bottom: 0.8rem !important;
            padding-bottom: 0.4rem !important;
            border-bottom: 2.5px solid {header_color} !important;
            letter-spacing: 0.3px !important;
        }}

        /* Segmented Horizontal Pill Toggle for Theme Switcher */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            background-color: {card_bg} !important;
            border: 1.5px solid {input_border} !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 6px !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {{
            flex: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
            margin: 0 !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer !important;
            font-size: 0.88rem !important;
            border: 1px solid transparent !important;
            background: transparent !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
            background-color: {card_border} !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
            background-color: {header_color} !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        }}

        /* All Widget Labels (Main Page & Sidebar) — Clean plain text without any boxes or borders */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] div,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label *,
        [data-testid="stSidebar"] p {{
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            border: none !important;
            border-radius: 0px !important;
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        /* Tooltip Icons & Inner Label Elements */
        [data-testid="stWidgetLabel"] svg,
        [data-testid="stTooltipIcon"],
        [data-testid="stTooltipIcon"] * {{
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }}

        /* High-Contrast Visible 2px Borders for ALL Input Fields across Light & Dark Themes */
        [data-baseweb="input"],
        div[data-baseweb="select"],
        [data-baseweb="textarea"],
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stTextArea"] [data-baseweb="textarea"],
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] div[data-baseweb="select"] {{
            background-color: {input_bg} !important;
            background: {input_bg} !important;
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            border: 2px solid {input_border} !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06) !important;
        }}

        /* Clean inner text input box inside BaseWeb container */
        [data-baseweb="base-input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div[role="combobox"] {{
            border: none !important;
            border-radius: 0px !important;
            box-shadow: none !important;
            background: transparent !important;
            background-color: transparent !important;
        }}

        /* Text Input & Text Area Inner Element Padding */
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        textarea {{
            background-color: transparent !important;
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            font-weight: 600 !important;
            padding: 0.4rem 0.8rem !important;
        }}

        /* Selectbox Alignment & Text Styling — Perfectly Centered Without Padding Shift */
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div[role="combobox"],
        div[data-baseweb="select"] div[role="button"] {{
            min-height: 40px !important;
            display: flex !important;
            align-items: center !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }}

        div[data-baseweb="select"] input {{
            padding: 0 !important;
            margin: 0 !important;
            height: auto !important;
        }}

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] p,
        div[data-baseweb="select"] div,
        [data-testid="stSelectbox"] span,
        [data-testid="stSelectbox"] p {{
            background-color: transparent !important;
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            font-weight: 600 !important;
            line-height: normal !important;
        }}

        /* Number Input Step Controls (- / +) — High contrast bordered buttons */
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {{
            background-color: {uploader_btn_bg} !important;
            background: {uploader_btn_bg} !important;
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            border: 1.5px solid {input_border} !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }}

        button[data-testid="stNumberInputStepDown"]:hover,
        button[data-testid="stNumberInputStepUp"]:hover {{
            background-color: {card_border} !important;
            background: {card_border} !important;
        }}

        button[data-testid="stNumberInputStepDown"] svg,
        button[data-testid="stNumberInputStepUp"] svg,
        div[data-baseweb="select"] svg,
        [data-testid="stSelectbox"] svg {{
            fill: {input_text} !important;
            color: {input_text} !important;
        }}

        /* Dropdown Popover Menus & Expanded Options Lists (Light & Dark Theme) */
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        div[role="listbox"] {{
            background-color: {card_bg} !important;
            border: 1.5px solid {card_border} !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15) !important;
        }}

        [data-baseweb="popover"] li,
        [data-baseweb="menu"] li,
        ul[role="listbox"] li,
        div[role="option"],
        li[role="option"],
        div[data-baseweb="menu-item"] {{
            background-color: {card_bg} !important;
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}

        [data-baseweb="popover"] li *,
        [data-baseweb="menu"] li *,
        ul[role="listbox"] li *,
        div[role="option"] *,
        li[role="option"] * {{
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
        }}

        [data-baseweb="popover"] li:hover,
        [data-baseweb="menu"] li:hover,
        ul[role="listbox"] li:hover,
        li[role="option"]:hover,
        div[role="option"]:hover,
        li[aria-selected="true"],
        div[role="option"][aria-selected="true"] {{
            background-color: {uploader_btn_bg} !important;
            color: {header_color} !important;
            -webkit-text-fill-color: {header_color} !important;
        }}

        [data-baseweb="popover"] li:hover *,
        [data-baseweb="menu"] li:hover *,
        ul[role="listbox"] li:hover *,
        li[role="option"]:hover *,
        div[role="option"]:hover *,
        li[aria-selected="true"] *,
        div[role="option"][aria-selected="true"] * {{
            color: {header_color} !important;
            -webkit-text-fill-color: {header_color} !important;
        }}

        /* Hide ONLY the moving red tab highlight bar without hiding tab scroll buttons */
        [data-baseweb="tab-highlight"],
        div[data-baseweb="tab-highlight"],
        [data-testid="stTabHighlight"] {{
            display: none !important;
            height: 0px !important;
            width: 0px !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }}

        /* Universal Navigation Tabs Styling — Works across Streamlit DOM variations */
        [data-baseweb="tab-list"],
        div[data-baseweb="tab-list"],
        div[role="tablist"] {{
            background-color: transparent !important;
            background: transparent !important;
            border-bottom: 2px solid {card_border} !important;
            gap: 2px !important;
            overflow-x: auto !important;
            scrollbar-width: none !important;
        }}

        [data-baseweb="tab"],
        [role="tab"],
        button[data-baseweb="tab"],
        div[data-baseweb="tab"],
        div[role="tab"],
        [data-testid="stTab"] {{
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            padding: 0.5rem 0.75rem !important;
            box-shadow: none !important;
            outline: none !important;
            white-space: nowrap !important;
            transition: all 0.2s ease-in-out !important;
        }}

        [data-baseweb="tab"] *,
        [data-baseweb="tab"] p,
        [data-baseweb="tab"] div,
        [data-baseweb="tab"] span,
        [role="tab"] *,
        [role="tab"] p,
        [role="tab"] div,
        [role="tab"] span,
        button[data-baseweb="tab"] *,
        div[role="tab"] *,
        [data-testid="stTab"] * {{
            color: {text_secondary} !important;
            -webkit-text-fill-color: {text_secondary} !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            opacity: 1 !important;
        }}

        [data-baseweb="tab"]:hover *,
        [role="tab"]:hover *,
        button[data-baseweb="tab"]:hover *,
        div[role="tab"]:hover *,
        [data-testid="stTab"]:hover * {{
            color: {header_color} !important;
            -webkit-text-fill-color: {header_color} !important;
        }}

        [aria-selected="true"][role="tab"],
        [aria-selected="true"][data-baseweb="tab"],
        button[data-baseweb="tab"][aria-selected="true"],
        div[role="tab"][aria-selected="true"],
        [data-testid="stTab"][aria-selected="true"] {{
            border-bottom: 3px solid {header_color} !important;
            background-color: transparent !important;
            background: transparent !important;
        }}

        [aria-selected="true"][role="tab"] *,
        [aria-selected="true"][data-baseweb="tab"] *,
        button[data-baseweb="tab"][aria-selected="true"] *,
        div[role="tab"][aria-selected="true"] *,
        [data-testid="stTab"][aria-selected="true"] * {{
            color: {header_color} !important;
            -webkit-text-fill-color: {header_color} !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}

        /* Tab Scroll Arrows & Controls (< and >) — Always functional and crisp */
        [data-baseweb="tab-list"] button,
        div[data-baseweb="tab-list"] button[aria-label],
        div[role="tablist"] button,
        div[data-baseweb="tab-scroll-right"],
        div[data-baseweb="tab-scroll-left"] {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: {card_bg} !important;
            background: {card_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {card_border} !important;
            border-radius: 6px !important;
            padding: 4px 8px !important;
            margin: 0 2px !important;
            opacity: 1 !important;
            visibility: visible !important;
            cursor: pointer !important;
        }}

        div[data-baseweb="tab-border"] *,
        [data-baseweb="tab-list"] button svg,
        div[data-baseweb="tab-list"] button[aria-label] svg,
        div[role="tablist"] button svg {{
            fill: {text_primary} !important;
            color: {text_primary} !important;
        }}

        /* Streamlit General Buttons (e.g. Clear Chat, Recommend Crop) */
        .stButton > button,
        button[kind="secondary"],
        button[kind="primary"],
        [data-testid="stForm"] button {{
            background-color: {uploader_btn_bg} !important;
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.45rem 1.2rem !important;
        }}

        /* Streamlit Chat Messages & Bubbles High-Contrast Styling */
        [data-testid="stChatMessage"] {{
            background-color: {card_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 12px !important;
            color: {text_primary} !important;
            padding: 1rem !important;
            margin-bottom: 0.8rem !important;
        }}

        [data-testid="stChatMessage"] *,
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] td,
        [data-testid="stChatMessage"] th {{
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
        }}

        /* Chat Input Field Styling */
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {{
            background-color: {input_bg} !important;
            border: 1px solid {input_border} !important;
            border-radius: 12px !important;
        }}

        [data-testid="stChatInput"] textarea {{
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
        }}

        [data-testid="stChatInput"] button {{
            color: {header_color} !important;
            fill: {header_color} !important;
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

        /* Navigation Tabs Styling - High Visibility */
        button[data-baseweb="tab"],
        [data-baseweb="tab-list"] button,
        button[data-baseweb="tab"] div,
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {{
            color: {text_secondary} !important;
            font-weight: 600 !important;
        }}
        button[aria-selected="true"],
        button[aria-selected="true"] div,
        button[aria-selected="true"] p,
        button[aria-selected="true"] span {{
            color: {header_color} !important;
            font-weight: 700 !important;
        }}

        /* File Uploader Container & Dropzone High-Contrast Styling */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploaderDropzone"] {{
            background-color: {uploader_bg} !important;
            border: 2px dashed {uploader_border} !important;
            border-radius: 12px !important;
        }}

        [data-testid="stFileUploaderDropzone"] *,
        section[data-testid="stFileUploaderDropzone"] * {{
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
        }}

        [data-testid="stFileUploaderDropzone"] button,
        section[data-testid="stFileUploaderDropzone"] button {{
            background-color: {uploader_btn_bg} !important;
            color: {text_primary} !important;
            -webkit-text-fill-color: {text_primary} !important;
            border: 1px solid {uploader_border} !important;
            border-radius: 8px !important;
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 1. Disease Detection",
    "🌤️ 2. Weather & Irrigation",
    "📊 3. Sustainability Score",
    "🌾 4. Crop Recommendation",
    "🤖 5. Farmer Assistant"
])

# Initialize Session State for Disease Detection status
if "disease_detected" not in st.session_state:
    st.session_state["disease_detected"] = False
if "agrismart_context" not in st.session_state:
    st.session_state["agrismart_context"] = {
        "disease": "Unavailable",
        "weather": "Unavailable",
        "irrigation": "Unavailable",
        "crop_recommendation": "Unavailable",
        "sustainability": "Unavailable",
    }
if "farmer_messages" not in st.session_state:
    st.session_state["farmer_messages"] = []

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

                st.session_state["agrismart_context"]["disease"] = {
                    "crop": precaution_info["crop"],
                    "disease": precaution_info["disease"],
                    "model_label": raw_label,
                    "confidence": round(confidence * 100, 2),
                    "precautions": precaution_info["precautions"],
                }

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
        st.session_state["agrismart_context"]["weather"] = {
            "rain_tomorrow_mm": rain_tomorrow,
            "max_temperature_c": max_t,
            "min_temperature_c": min_t,
        }
        st.session_state["agrismart_context"]["irrigation"] = advice
        
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
        st.session_state["agrismart_context"]["weather"] = "Unavailable"
        st.session_state["agrismart_context"]["irrigation"] = "Unavailable"
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
    st.session_state["agrismart_context"]["sustainability"] = sust_result

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
    st.write("Enter soil and weather measurements to receive an offline Random Forest crop recommendation.")

    with st.form("crop_recommendation_form"):
        crop_col1, crop_col2 = st.columns(2)
        with crop_col1:
            crop_n = st.number_input("Nitrogen (N)", min_value=0.0, value=float(n_val), step=1.0)
            crop_p = st.number_input("Phosphorus (P)", min_value=0.0, value=float(p_val), step=1.0)
            crop_k = st.number_input("Potassium (K)", min_value=0.0, value=float(k_val), step=1.0)
            crop_temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
        with crop_col2:
            crop_humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(humidity), step=0.1)
            crop_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=float(soil_ph), step=0.1)
            crop_rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, step=1.0)

        recommend_crop = st.form_submit_button("Recommend Crop")

    if recommend_crop:
        try:
            recommended_crop = predict_crop(
                N=crop_n,
                P=crop_p,
                K=crop_k,
                temperature=crop_temperature,
                humidity=crop_humidity,
                ph=crop_ph,
                rainfall=crop_rainfall,
            )
            st.session_state["agrismart_context"]["crop_recommendation"] = {
                "recommended_crop": recommended_crop,
                "inputs": {
                    "N": crop_n,
                    "P": crop_p,
                    "K": crop_k,
                    "temperature": crop_temperature,
                    "humidity": crop_humidity,
                    "ph": crop_ph,
                    "rainfall": crop_rainfall,
                },
            }
            st.success(f"Recommended crop: {recommended_crop.title()}")
            st.caption("This recommendation uses the trained Random Forest model and the seven entered measurements.")
        except (FileNotFoundError, ValueError, TypeError) as error:
            st.session_state["agrismart_context"]["crop_recommendation"] = "Unavailable"
            st.error(f"Crop recommendation unavailable: {error}")

# ==============================================================================\
# Section 5: Farmer Assistant\
# ==============================================================================\
with tab5:
    st.markdown('<div class="section-header">🤖 Farmer Assistant</div>', unsafe_allow_html=True)
    st.write("Ask questions about the recommendations currently produced by AgriSmart AI.")

    language = st.selectbox("Assistant language", ["English", "Gujarati"], key="farmer_language")
    if st.button("Clear Chat", key="clear_farmer_chat"):
        st.session_state["farmer_messages"] = []

    for message in st.session_state["farmer_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about your AgriSmart results...")
    if question:
        st.session_state["farmer_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        api_key = os.getenv("GEMINI_API_KEY")
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", api_key)
        except Exception:
            pass

        try:
            answer = get_farmer_response(
                question=question,
                agrismart_context=st.session_state["agrismart_context"],
                language=language,
                api_key=api_key,
                conversation_history=st.session_state["farmer_messages"][:-1],
            )
        except (FarmerAssistantError, ValueError) as error:
            answer = str(error)
        except Exception:
            answer = "The Farmer Assistant is temporarily unavailable. Please try again later."

        st.session_state["farmer_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
