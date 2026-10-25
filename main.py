import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import io
from datetime import datetime

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Plant Disease Recognition | AI-Powered",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== CUSTOM CSS - MODERN UI ==============
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Root variables */
    :root {
        --primary: #0d9488;
        --primary-dark: #0f766e;
        --primary-light: #5eead4;
        --accent: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text: #f8fafc;
        --text-muted: #94a3b8;
        --success: #22c55e;
        --border: #334155;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(165deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(94, 234, 212, 0.15);
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    /* Force all main content text to be visible on dark background */
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li, [data-testid="stMarkdown"] span,
    [data-testid="stMarkdown"] label, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #e2e8f0 !important;
    }
    [data-testid="stMarkdown"] strong { color: #f8fafc !important; }
    [data-testid="stSelectbox"] label, [data-testid="stFileUploader"] label { color: #e2e8f0 !important; }
    
    /* Headers - use solid teal so they are always visible */
    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: #5eead4 !important;
    }
    
    /* Custom cards */
    .feature-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(94, 234, 212, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(13, 148, 136, 0.2);
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.2) 0%, rgba(94, 234, 212, 0.05) 100%);
        border: 1px solid rgba(94, 234, 212, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #5eead4;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Hero section - solid colors for visibility */
    .hero-title {
        font-size: 2.75rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        color: #5eead4 !important;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #cbd5e1 !important;
        text-align: center !important;
        max-width: 640px;
        margin: 0 auto 2rem !important;
        width: 100%;
        display: block;
    }
    
    /* Upload zone enhancement */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed rgba(94, 234, 212, 0.4);
        border-radius: 16px;
        padding: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.5) !important;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 12px;
        background: #334155;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.5rem 0 1rem;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #0d9488, #5eead4);
        border-radius: 999px;
        transition: width 0.5s ease;
    }
    
    /* Prediction result card */
    .result-card {
        background: linear-gradient(145deg, rgba(34, 197, 94, 0.15) 0%, rgba(13, 148, 136, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Tips box - ensure text is visible */
    .tips-box {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #f1f5f9 !important;
    }
    .tips-box strong { color: #fcd34d !important; }
    
    /* Encyclopedia and general content boxes */
    .encyclopedia-card, .page-intro {
        color: #e2e8f0 !important;
    }
    .page-intro { margin-bottom: 1.5rem; }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #334155;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============== DISEASE INFO ==============
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_info = {
    'Apple___Apple_scab': {
        'description': 'Apple Scab is a fungal disease that affects the leaves and fruit of apple trees.',
        'cause': 'It is caused by the fungus *Venturia inaequalis*.',
        'prevention': 'Prevent by removing fallen leaves, using fungicides, and selecting resistant varieties.'
    },
    'Apple___Black_rot': {
        'description': 'Black Rot affects apple trees, causing lesions on the fruit and branches.',
        'cause': 'Caused by the fungus *Glomerella cingulata*.',
        'prevention': 'Prune infected branches and use fungicides during the growing season.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar-Apple Rust causes orange, rust-like spots on apple leaves.',
        'cause': 'Caused by the fungus *Gymnosporangium juniper-virginianae*.',
        'prevention': 'Remove cedar trees near apple trees and apply fungicides in spring.'
    },
    'Apple___healthy': {
        'description': 'The apple tree is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the tree.'
    },
    'Blueberry___healthy': {
        'description': 'The blueberry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Maintain proper watering and soil conditions.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Powdery mildew causes white fungal growth on leaves, stems, and buds of cherry trees.',
        'cause': 'Caused by fungi in the *Erysiphaceae* family.',
        'prevention': 'Use fungicides and prune the tree to improve air circulation.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'The cherry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the tree.'
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': 'Cercospora leaf spot causes gray to brown lesions on maize leaves.',
        'cause': 'Caused by the fungus *Cercospora zeae-maydis*.',
        'prevention': 'Use resistant varieties, crop rotation, and fungicides.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Cercospora leaf spot causes gray to brown lesions on maize leaves.',
        'cause': 'Caused by the fungus *Cercospora zeae-maydis*.',
        'prevention': 'Use resistant varieties, crop rotation, and fungicides.'
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': 'Cercospora leaf spot causes gray to brown lesions on maize leaves.',
        'cause': 'Caused by the fungus *Cercospora zeae-maydis*.',
        'prevention': 'Use resistant varieties, crop rotation, and fungicides.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust causes reddish-brown pustules on maize leaves.',
        'cause': 'Caused by the fungus *Puccinia sorghi*.',
        'prevention': 'Use resistant varieties and apply fungicides.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight causes long, gray lesions on maize leaves.',
        'cause': 'Caused by the fungus *Exserohilum turcicum*.',
        'prevention': 'Use resistant varieties and apply fungicides.'
    },
    'Corn_(maize)___healthy': {
        'description': 'The corn plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Grape___Black_rot': {
        'description': 'Black rot causes dark lesions and shriveling of grape berries.',
        'cause': 'Caused by the fungus *Guignardia bidwellii*.',
        'prevention': 'Prune infected areas and use fungicides.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca causes brown lesions on grape leaves and a decline in plant health.',
        'cause': 'Caused by fungal pathogens including *Phaeomoniella chlamydospora*.',
        'prevention': 'Prune infected vines and avoid wounding during pruning.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Leaf blight causes dark lesions with yellow margins on grape leaves.',
        'cause': 'Caused by the fungus *Isariopsis caprea*.',
        'prevention': 'Use fungicides and remove infected leaves.'
    },
    'Grape___healthy': {
        'description': 'The grape plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Huanglongbing causes yellowing of leaves and stunted growth in citrus plants.',
        'cause': 'Caused by the bacterium *Candidatus Liberibacter*.',
        'prevention': 'Use disease-free stock and control citrus psyllids.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial spot causes lesions on peach leaves and fruit.',
        'cause': 'Caused by the bacterium *Xanthomonas campestris*.',
        'prevention': 'Prune infected branches and apply copper-based fungicides.'
    },
    'Peach___healthy': {
        'description': 'The peach tree is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the tree.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot causes dark lesions on pepper leaves and fruit.',
        'cause': 'Caused by the bacterium *Xanthomonas campestris*.',
        'prevention': 'Use resistant varieties and remove infected plants.'
    },
    'Pepper,_bell___healthy': {
        'description': 'The bell pepper plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight causes dark spots on potato leaves with concentric rings.',
        'cause': 'Caused by the fungus *Alternaria solani*.',
        'prevention': 'Use resistant varieties and rotate crops.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight causes dark, water-soaked lesions on potato leaves.',
        'cause': 'Caused by the fungus *Phytophthora infestans*.',
        'prevention': 'Use fungicides and remove infected plants.'
    },
    'Potato___healthy': {
        'description': 'The potato plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Raspberry___healthy': {
        'description': 'The raspberry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Maintain proper watering and soil conditions.'
    },
    'Soybean___healthy': {
        'description': 'The soybean plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Powdery mildew causes white fungal growth on squash leaves.',
        'cause': 'Caused by fungi in the *Erysiphaceae* family.',
        'prevention': 'Use fungicides and improve air circulation around the plant.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Leaf scorch causes browning of the edges of strawberry leaves.',
        'cause': 'Caused by various environmental stress factors, including high temperatures.',
        'prevention': 'Water regularly and ensure proper soil nutrition.'
    },
    'Strawberry___healthy': {
        'description': 'The strawberry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot causes dark lesions on tomato leaves and fruit.',
        'cause': 'Caused by the bacterium *Xanthomonas vesicatoria*.',
        'prevention': 'Use resistant varieties and remove infected plants.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight causes dark, concentric lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Alternaria solani*.',
        'prevention': 'Use fungicides and remove infected plants.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight causes dark, water-soaked lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Phytophthora infestans*.',
        'prevention': 'Use fungicides and remove infected plants.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold causes yellowing and mold growth on the undersides of tomato leaves.',
        'cause': 'Caused by the fungus *Cladosporium fulvum*.',
        'prevention': 'Improve air circulation and use fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Septoria leaf spot causes dark, circular lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Septoria lycopersici*.',
        'prevention': 'Use fungicides and rotate crops.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Two-spotted spider mite damage causes stippling and webbing on tomato leaves.',
        'cause': 'Caused by the pest *Tetranychus urticae*.',
        'prevention': 'Use miticides, increase humidity, and introduce beneficial predators.'
    },
    'Tomato___Target_Spot': {
        'description': 'Target spot causes dark, concentric ring lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Corynespora cassiicola*.',
        'prevention': 'Use fungicides and remove infected plant debris.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Yellow leaf curl virus causes curling and yellowing of tomato leaves.',
        'cause': 'Caused by begomoviruses, spread by whiteflies.',
        'prevention': 'Use resistant varieties and control whitefly populations.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Tomato mosaic virus causes mottling and distortion of leaves and fruit.',
        'cause': 'Caused by Tobacco mosaic virus (TMV).',
        'prevention': 'Use virus-free seeds, sanitize tools, and avoid smoking near plants.'
    },
    'Tomato___healthy': {
        'description': 'The tomato plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
}


def get_disease_info(name):
    """Get disease info with flexible key lookup (handles spaces vs underscores)."""
    if name in disease_info:
        return disease_info[name]
    alt = name.replace(' ', '_')
    return disease_info.get(alt)


# Extended info for Disease Encyclopedia (symptoms, severity, treatment, affected_parts)
def _ek(symptoms, severity, treatment, affected_parts):
    return {"symptoms": symptoms, "severity": severity, "treatment": treatment, "affected_parts": affected_parts}

disease_extra = {
    "Apple___Apple_scab": _ek("Olive-green spots on leaves turning dark, cracked scabs on fruit.", "Moderate", "Apply sulfur or fungicide from green tip through petal fall; remove fallen leaves.", "Leaves, fruit"),
    "Apple___Black_rot": _ek("Brown cankers on branches; dark rot with concentric rings on fruit.", "High", "Prune cankers; apply fungicides at bloom and pre-harvest.", "Fruit, branches, leaves"),
    "Apple___Cedar_apple_rust": _ek("Bright orange spots on leaves and fruit; yellow spots on upper leaf surface.", "Moderate", "Remove nearby junipers; apply fungicide in early spring.", "Leaves, fruit"),
    "Apple___healthy": _ek("No lesions, spots, or discoloration; normal green foliage.", "N/A", "Maintain good nutrition and irrigation; monitor for pests.", "N/A"),
    "Blueberry___healthy": _ek("Vibrant green leaves; no spots or wilting.", "N/A", "Routine care; avoid overhead irrigation to reduce disease risk.", "N/A"),
    "Cherry_(including_sour)___Powdery_mildew": _ek("White powdery coating on leaves, shoots, and fruit.", "Moderate", "Fungicides; improve air flow; resistant varieties.", "Leaves, shoots, fruit"),
    "Cherry_(including_sour)___healthy": _ek("Healthy green foliage; no powdery or spotted areas.", "N/A", "Standard care and pruning.", "N/A"),
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": _ek("Gray to tan elongated lesions on leaves.", "Moderate", "Fungicide; crop rotation; resistant hybrids.", "Leaves"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": _ek("Gray to tan elongated lesions on leaves.", "Moderate", "Fungicide; crop rotation; resistant hybrids.", "Leaves"),
    "Corn_(maize)___Common_rust_": _ek("Reddish-brown pustules on upper and lower leaf surfaces.", "Moderate", "Fungicide; early planting; resistant varieties.", "Leaves"),
    "Corn_(maize)___Northern_Leaf_Blight": _ek("Long, elliptical gray-green lesions; cigar-shaped.", "High", "Fungicide; tillage; resistant hybrids.", "Leaves"),
    "Corn_(maize)___healthy": _ek("Green leaves without lesions or pustules.", "N/A", "Crop rotation; balanced fertility.", "N/A"),
    "Grape___Black_rot": _ek("Brown spots on leaves; shriveled, black mummified berries.", "High", "Fungicide from pre-bloom to 4–6 weeks after bloom; remove mummies.", "Leaves, fruit"),
    "Grape___Esca_(Black_Measles)": _ek("Tiger-stripe pattern on leaves; brown wood and decline.", "High", "Avoid wounding; remove infected vines; no cure once established.", "Leaves, trunk, canes"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": _ek("Dark spots with yellow halos on leaves.", "Moderate", "Fungicide; remove infected leaves.", "Leaves"),
    "Grape___healthy": _ek("Normal green leaves and healthy fruit clusters.", "N/A", "Good canopy management and sanitation.", "N/A"),
    "Orange___Haunglongbing_(Citrus_greening)": _ek("Yellow mottling; misshapen fruit; twig dieback.", "High", "No cure; remove infected trees; control psyllids; use clean stock.", "Leaves, fruit, whole tree"),
    "Peach___Bacterial_spot": _ek("Water-soaked spots on leaves and fruit; cracks on fruit.", "Moderate", "Copper sprays; avoid overhead irrigation; resistant varieties.", "Leaves, fruit"),
    "Peach___healthy": _ek("Green, spot-free leaves and normal fruit.", "N/A", "Prune for air flow; monitor for pests.", "N/A"),
    "Pepper,_bell___Bacterial_spot": _ek("Small, dark, water-soaked spots on leaves and fruit.", "Moderate", "Copper-based bactericide; drip irrigation; resistant cultivars.", "Leaves, fruit"),
    "Pepper,_bell___healthy": _ek("Healthy green foliage and fruit.", "N/A", "Rotate crops; avoid wet foliage.", "N/A"),
    "Potato___Early_blight": _ek("Concentric ring lesions on leaves; target-board pattern.", "Moderate", "Fungicide; remove volunteer plants; resistant varieties.", "Leaves, stems, tubers"),
    "Potato___Late_blight": _ek("Water-soaked lesions; white mold in humid conditions.", "High", "Fungicide; destroy culls; use certified seed.", "Leaves, stems, tubers"),
    "Potato___healthy": _ek("No lesions or yellowing on foliage.", "N/A", "Crop rotation; certified seed.", "N/A"),
    "Raspberry___healthy": _ek("Normal green leaves; no spots or mold.", "N/A", "Sanitation; good spacing.", "N/A"),
    "Soybean___healthy": _ek("Green, healthy foliage.", "N/A", "Crop rotation; scout for pests.", "N/A"),
    "Squash___Powdery_mildew": _ek("White powdery patches on leaves; yellowing and death of leaves.", "Moderate", "Fungicide; resistant varieties; reduce humidity.", "Leaves"),
    "Strawberry___Leaf_scorch": _ek("Purple or brown spots on leaves; scorched appearance.", "Low", "Fungicide; remove old leaves; improve air flow.", "Leaves"),
    "Strawberry___healthy": _ek("Green, unscorched leaves.", "N/A", "Good irrigation and fertility.", "N/A"),
    "Tomato___Bacterial_spot": _ek("Small, dark, greasy spots on leaves and fruit.", "Moderate", "Copper spray; avoid overhead watering; resistant varieties.", "Leaves, fruit"),
    "Tomato___Early_blight": _ek("Dark concentric rings on older leaves; stem cankers.", "Moderate", "Fungicide; stake plants; remove infected tissue.", "Leaves, stems, fruit"),
    "Tomato___Late_blight": _ek("Water-soaked lesions; white mold in wet weather.", "High", "Fungicide; remove infected plants; avoid wet foliage.", "Leaves, stems, fruit"),
    "Tomato___Leaf_Mold": _ek("Yellow patches on top; olive-green mold underneath leaves.", "Moderate", "Fungicide; reduce humidity; improve ventilation.", "Leaves"),
    "Tomato___Septoria_leaf_spot": _ek("Small circular spots with gray centers on leaves.", "Moderate", "Fungicide; remove lower leaves; rotate crops.", "Leaves"),
    "Tomato___Spider_mites Two-spotted_spider_mite": _ek("Stippling, webbing, and bronzing of leaves.", "Moderate", "Miticide; increase humidity; predatory mites.", "Leaves"),
    "Tomato___Target_Spot": _ek("Dark spots with concentric rings; yellow halo.", "Moderate", "Fungicide; remove debris; avoid overhead irrigation.", "Leaves, fruit"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": _ek("Curled, yellowed leaves; stunted growth.", "High", "Control whiteflies; use resistant varieties; remove infected plants.", "Leaves, whole plant"),
    "Tomato___Tomato_mosaic_virus": _ek("Mottling, distortion, and stunting of leaves and fruit.", "Moderate", "Virus-free seed; sanitize tools; control aphids.", "Leaves, fruit"),
    "Tomato___healthy": _ek("Normal green foliage and fruit.", "N/A", "Routine care; scout for pests and diseases.", "N/A"),
}


# ============== MODEL ==============
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")


def model_prediction(image_path):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr, verbose=0)
    return predictions[0]


# ============== SESSION STATE ==============
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("### 🌿 Plant Disease AI")
    st.markdown("---")
    app_mode = st.selectbox(
        "**Navigate**",
        ["Home", "Disease Recognition", "Disease Encyclopedia", "About"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div class='stat-box'><span class='stat-number'>38</span><br><span class='stat-label'>Disease Classes</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Upload a leaf image to get an instant AI diagnosis.")


# ============== HOME ==============
if app_mode == "Home":
    st.markdown("""
    <div style='width:100%; text-align:center; margin-bottom:0.5rem'>
        <p class='hero-title'>Plant Disease Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='width:100%; text-align:center; margin-bottom:2rem'>
        <p class='hero-subtitle'>AI-powered leaf analysis to identify diseases quickly. Protect your crops with accurate, instant diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-box'>
            <span class='stat-number'>38</span>
            <span class='stat-label'>Disease Classes</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-box'>
            <span class='stat-number'>14</span>
            <span class='stat-label'>Plant Types</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-box'>
            <span class='stat-number'>AI</span>
            <span class='stat-label'>Deep Learning</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_img, col_text = st.columns([1, 1])
    with col_img:
        st.image("th.jpg", use_container_width=True)
    with col_text:
        st.markdown("""
        <div class='feature-card'>
            <h4 style='color:#5eead4; margin-top:0'>Why use this system?</h4>
            <ul style='color:#cbd5e1; line-height:1.8'>
                <li><strong>Instant results</strong> — Get a diagnosis in seconds</li>
                <li><strong>Actionable advice</strong> — Causes and prevention for each disease</li>
                <li><strong>Wide coverage</strong> — Apples, tomatoes, potatoes, grapes, and more</li>
                <li><strong>Easy to use</strong> — Just upload a clear photo of the leaf</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='feature-card'>
            <strong style='color:#5eead4'>1. Upload</strong><br>
            <span style='color:#94a3b8'>Choose a clear image of a plant leaf (JPG, PNG).</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='feature-card'>
            <strong style='color:#5eead4'>2. Analyze</strong><br>
            <span style='color:#94a3b8'>Our AI model processes the image and runs inference.</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='feature-card'>
            <strong style='color:#5eead4'>3. Results</strong><br>
            <span style='color:#94a3b8'>See the predicted disease, confidence, and prevention tips.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='app-footer'>Plant Disease Recognition System — Built with Streamlit & TensorFlow</div>", unsafe_allow_html=True)


# ============== DISEASE RECOGNITION ==============
elif app_mode == "Disease Recognition":
    st.markdown("### 🔬 Disease Recognition")
    st.markdown(
        "<p style='color:#cbd5e1 !important; margin-bottom:1rem'>"
        "Upload a leaf image to get an AI-powered diagnosis and prevention tips.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class='tips-box'>
        <strong style='color:#fcd34d !important'>📷 Tips for best results:</strong>
        <span style='color:#f1f5f9 !important'> Use a clear, well-lit photo of a single leaf; avoid shadows and blur.
        Supported formats: JPG, PNG. Our model supports 14 plant types and 38 disease classes.</span>
    </div>
    """, unsafe_allow_html=True)

    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        if test_image is not None:
            st.image(test_image, caption="Uploaded image", use_container_width=True)

    with col_result:
        if test_image is not None:
            if st.button("🔍 Analyze image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(test_image.getvalue())
                            path = tmp.name
                        preds = model_prediction(path)

                        top3_idx = np.argsort(preds)[-3:][::-1]
                        result_index = top3_idx[0]
                        confidence = float(preds[result_index])
                        predicted_disease = CLASS_NAMES[result_index]

                        st.success("Analysis complete")
                        st.markdown(f"**Prediction:** {predicted_disease.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                        st.progress(min(1.0, confidence))

                        st.markdown("**Top 3 predictions**")
                        for i, idx in enumerate(top3_idx, 1):
                            pct = preds[idx] * 100
                            st.caption(f"{i}. {CLASS_NAMES[idx].replace('_', ' ').title()} — {pct:.1f}%")

                        info = get_disease_info(predicted_disease)
                        if info:
                            with st.expander("Description, cause & prevention"):
                                st.markdown(f"**Description:** {info['description']}")
                                st.markdown(f"**Cause:** {info['cause']}")
                                st.markdown(f"**Prevention:** {info['prevention']}")

                        report_lines = [
                            f"Plant Disease Recognition — Report",
                            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            f"Prediction: {predicted_disease}",
                            f"Confidence: {confidence*100:.1f}%",
                            "",
                            "Top 3:",
                        ]
                        for idx in top3_idx:
                            report_lines.append(f"  - {CLASS_NAMES[idx]}: {preds[idx]*100:.1f}%")
                        if info:
                            report_lines.extend(["", "Description: " + info["description"], "Prevention: " + info["prevention"]])
                        report_text = "\n".join(report_lines)
                        st.download_button("📥 Download report", report_text, file_name="disease_report.txt", mime="text/plain", use_container_width=True)

                        st.session_state.prediction_history.insert(0, {
                            "disease": predicted_disease,
                            "confidence": confidence,
                            "time": datetime.now().strftime("%H:%M"),
                        })
                        st.session_state.prediction_history = st.session_state.prediction_history[:10]
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("**Recent predictions (this session)**")
        for h in st.session_state.prediction_history[:5]:
            st.caption(f"• {h['disease'].replace('_', ' ').title()} — {h['confidence']*100:.1f}% ({h['time']})")

    st.markdown("<div class='app-footer'>Plant Disease Recognition System — Built with Streamlit & TensorFlow</div>", unsafe_allow_html=True)


# ============== DISEASE ENCYCLOPEDIA ==============
elif app_mode == "Disease Encyclopedia":
    st.markdown("### 📚 Disease Encyclopedia")
    st.markdown(
        "<p class='page-intro' style='color:#cbd5e1 !important; margin-bottom:1.5rem'>"
        "Browse all supported plant and disease classes. Each entry includes description, symptoms, cause, severity, "
        "affected plant parts, treatment, and prevention so you can identify and manage diseases effectively.</p>",
        unsafe_allow_html=True,
    )

    # Filter by plant type
    plant_types = sorted(set(c.split("___")[0].strip() for c in CLASS_NAMES))
    selected_plant = st.selectbox(
        "Filter by plant type",
        options=["All plants"] + plant_types,
        index=0,
    )
    if selected_plant == "All plants":
        filtered_classes = CLASS_NAMES
    else:
        filtered_classes = [c for c in CLASS_NAMES if c.startswith(selected_plant + "___")]

    chosen = st.selectbox(
        "Select disease or condition",
        options=filtered_classes,
        format_func=lambda x: x.replace("_", " ").replace("  ", " ").title(),
        index=0,
    )

    info = get_disease_info(chosen)
    extra = disease_extra.get(chosen) or disease_extra.get(chosen.replace(" ", "_"))

    if info:
        title = chosen.replace("_", " ").replace("  ", " ").title()
        severity = (extra or {}).get("severity", "—")
        symptoms = (extra or {}).get("symptoms", "—")
        treatment = (extra or {}).get("treatment", "—")
        affected = (extra or {}).get("affected_parts", "—")

        st.markdown(f"""
        <div class='feature-card' style='margin-top:1rem'>
            <h3 style='color:#5eead4 !important; margin-top:0'>{title}</h3>
            <p style='color:#94a3b8; font-size:0.9rem; margin-bottom:1.25rem'>
                <strong style='color:#f59e0b'>Severity:</strong> <span style='color:#e2e8f0'>{severity}</span> &nbsp;|&nbsp;
                <strong style='color:#f59e0b'>Affected parts:</strong> <span style='color:#e2e8f0'>{affected}</span>
            </p>
            <h4 style='color:#5eead4 !important; font-size:1rem'>Description</h4>
            <p style='color:#e2e8f0; line-height:1.6'>{info['description']}</p>
            <h4 style='color:#5eead4 !important; font-size:1rem'>Symptoms</h4>
            <p style='color:#e2e8f0; line-height:1.6'>{symptoms}</p>
            <h4 style='color:#5eead4 !important; font-size:1rem'>Cause</h4>
            <p style='color:#e2e8f0; line-height:1.6'>{info['cause']}</p>
            <h4 style='color:#5eead4 !important; font-size:1rem'>Treatment</h4>
            <p style='color:#e2e8f0; line-height:1.6'>{treatment}</p>
            <h4 style='color:#5eead4 !important; font-size:1rem'>Prevention</h4>
            <p style='color:#e2e8f0; line-height:1.6'>{info['prevention']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No detailed information available for this class.")

    st.markdown("<div class='app-footer'>Plant Disease Recognition System</div>", unsafe_allow_html=True)


# ============== ABOUT ==============
else:
    st.markdown("### About")
    st.markdown("""
    <div class='feature-card'>
        <h4 style='color:#5eead4'>About the dataset</h4>
        <p style='color:#cbd5e1'>
        This system uses a dataset of healthy and diseased crop leaf images, categorized into 
        <strong>38 classes</strong> across <strong>14 plant types</strong> (e.g. Apple, Tomato, Potato, Grape, Corn).
        The model is trained to recognize both healthy leaves and common diseases such as blight, rust, 
        powdery mildew, and bacterial spot.
        </p>
        <h4 style='color:#5eead4'>Technology</h4>
        <p style='color:#cbd5e1'>
        Built with <strong>TensorFlow/Keras</strong> for the deep learning model and <strong>Streamlit</strong> 
        for the web interface. Images are resized to 224×224 and fed into the trained model for inference.
        </p>
        <p style='color:#94a3b8'>
        Dataset source: 
        <a href='https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset' target='_blank' style='color:#5eead4'>Kaggle — New Plant Diseases Dataset</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='app-footer'>Plant Disease Recognition System — Built with Streamlit & TensorFlow</div>", unsafe_allow_html=True)
