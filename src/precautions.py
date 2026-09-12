"""
Precautions and recommended actions for crop leaf diseases.
Provides actionable advice for farmers based on predicted disease labels.
"""

PRECAUTIONS = {
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "crop": "Apple",
        "status": "Diseased",
        "precautions": [
            "Remove and destroy infected fallen leaves during autumn.",
            "Apply approved fungicides (e.g., captan or mancozeb) early in spring.",
            "Prune tree canopy to improve air circulation and sunlight penetration.",
            "Avoid overhead watering to keep foliage dry."
        ]
    },
    "Apple___Black_rot": {
        "disease": "Black Rot",
        "crop": "Apple",
        "status": "Diseased",
        "precautions": [
            "Prune out dead, diseased, or cankered wood and burn/dispose of prunings.",
            "Remove mummified fruits remaining on the tree or ground.",
            "Apply copper-based fungicides during late dormant stage.",
            "Maintain overall tree health through proper fertilization and watering."
        ]
    },
    "Apple___healthy": {
        "disease": "Healthy",
        "crop": "Apple",
        "status": "Healthy",
        "precautions": [
            "Plant shows no signs of disease. Continue routine maintenance.",
            "Ensure regular balanced fertilization and appropriate irrigation.",
            "Monitor periodically for early signs of pests or disease."
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Gray Leaf Spot",
        "crop": "Corn (Maize)",
        "status": "Diseased",
        "precautions": [
            "Use disease-resistant corn hybrids where available.",
            "Practice crop rotation with non-host crops like soybeans or alfalfa.",
            "Till crop residue after harvest to accelerate decomposition of fungi.",
            "Apply foliar fungicides if disease appears before tasseling."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "disease": "Common Rust",
        "crop": "Corn (Maize)",
        "status": "Diseased",
        "precautions": [
            "Plant resistant corn hybrids for effective management.",
            "Apply foliar fungicides if infection starts early in the season.",
            "Avoid high nitrogen applications that promote overly succulent growth."
        ]
    },
    "Corn_(maize)___healthy": {
        "disease": "Healthy",
        "crop": "Corn (Maize)",
        "status": "Healthy",
        "precautions": [
            "Crop is healthy! Maintain current irrigation and nutrient management.",
            "Inspect fields regularly during high humidity periods."
        ]
    },
    "Grape___Black_rot": {
        "disease": "Black Rot",
        "crop": "Grape",
        "status": "Diseased",
        "precautions": [
            "Prune canes and destroy mummified berries from previous season.",
            "Apply protective fungicides starting from bud break to bloom.",
            "Ensure canopy management for max ventilation and sunlight."
        ]
    },
    "Grape___healthy": {
        "disease": "Healthy",
        "crop": "Grape",
        "status": "Healthy",
        "precautions": [
            "Vines are healthy. Keep up standard canopy management and trellising.",
            "Ensure proper soil drainage."
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "disease": "Bacterial Spot",
        "crop": "Bell Pepper",
        "status": "Diseased",
        "precautions": [
            "Use certified disease-free seeds and transplants.",
            "Apply copper spray mixed with mancozeb to manage bacterial spread.",
            "Avoid working in fields when plants are wet.",
            "Rotate crops with non-solanaceous plants for 2–3 years."
        ]
    },
    "Pepper,_bell___healthy": {
        "disease": "Healthy",
        "crop": "Bell Pepper",
        "status": "Healthy",
        "precautions": [
            "Plants are healthy! Continue balanced drip irrigation.",
            "Monitor for sucking pests like aphids or thrips."
        ]
    },
    "Potato___Early_blight": {
        "disease": "Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "precautions": [
            "Apply protective fungicides (e.g., chlorothalonil or mancozeb).",
            "Maintain optimal plant vigor with adequate nitrogen and water.",
            "Rotate crops for at least 2 years out of potatoes or tomatoes.",
            "Destroy crop residue post-harvest."
        ]
    },
    "Potato___Late_blight": {
        "disease": "Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "precautions": [
            "Apply systemic fungicides (e.g., metalaxyl or cymoxanil) immediately.",
            "Destroy infected plants to prevent rapid fungal spore dispersal.",
            "Avoid overhead irrigation and keep foliage dry.",
            "Store harvested tubers in cool, dry, well-ventilated space."
        ]
    },
    "Potato___healthy": {
        "disease": "Healthy",
        "crop": "Potato",
        "status": "Healthy",
        "precautions": [
            "Potato foliage is healthy. Continue routine hilling and watering.",
            "Monitor soil moisture levels to prevent tuber rot."
        ]
    },
    "Tomato___Bacterial_spot": {
        "disease": "Bacterial Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "precautions": [
            "Apply copper-based bactericides early in the disease onset.",
            "Avoid overhead watering; use drip irrigation at plant base.",
            "Sanitize tools and stakes between uses.",
            "Remove heavily infected lower leaves."
        ]
    },
    "Tomato___Early_blight": {
        "disease": "Early Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "precautions": [
            "Mulch around plant base to prevent soil splash onto lower leaves.",
            "Prune lower branches close to the ground for air circulation.",
            "Spray approved fungicides at first sign of target-spot lesions.",
            "Practice 3-year crop rotation."
        ]
    },
    "Tomato___Late_blight": {
        "disease": "Late Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "precautions": [
            "Remove and bag infected plants immediately to stop spore spread.",
            "Apply preventative copper or chlorothalonil fungicides during humid weather.",
            "Ensure wide plant spacing for fast leaf drying.",
            "Avoid planting near potatoes."
        ]
    },
    "Tomato___Leaf_Mold": {
        "disease": "Leaf Mold",
        "crop": "Tomato",
        "status": "Diseased",
        "precautions": [
            "Increase greenhouse/field ventilation to lower relative humidity below 85%.",
            "Water at the root zone early in the day.",
            "Apply copper or sulfur fungicides to protect upper leaves.",
            "Remove and destroy severely affected leaves."
        ]
    },
    "Tomato___healthy": {
        "disease": "Healthy",
        "crop": "Tomato",
        "status": "Healthy",
        "precautions": [
            "Plants are healthy! Continue regular staking, pruning, and watering.",
            "Maintain consistent soil moisture to prevent blossom end rot."
        ]
    }
}


def get_precautions(label: str) -> dict:
    """
    Retrieve precaution information for a given class label.
    
    Args:
        label (str): The raw predicted class label from the model.
        
    Returns:
        dict: Precaution details including formatted disease name, crop name,
              status, and list of recommended precautions.
    """
    if label in PRECAUTIONS:
        return PRECAUTIONS[label]
    
    # Fallback for unknown labels
    formatted_name = label.replace("___", " - ").replace("_", " ")
    return {
        "disease": formatted_name,
        "crop": "General Crop",
        "status": "Unknown",
        "precautions": [
            "Consult a local agricultural extension worker for precise diagnosis.",
            "Isolate the affected plant to prevent potential spread to neighboring crops.",
            "Ensure proper field hygiene and avoid excessive moisture on leaves."
        ]
    }
