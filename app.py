import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import markdown
import bleach
import re
from werkzeug.utils import secure_filename

# Import our enhanced weather utilities
from utils.weather_utils import get_comprehensive_weather_context, WeatherService

app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
WARDROBE_FOLDER = 'static/wardrobe'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WARDROBE_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# --- Session Storage ---
sessions = {}

# --- Gemini API setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    print("WARNING: GEMINI_API_KEY not found.")

# --- Initialize Weather Service ---
weather_service = WeatherService()

# --- Global variables for models ---
models = {}
reverse_label_mappings = {}

# --- Load classification models from TensorFlow ---
def load_classification_models():
    """Load the TensorFlow classification models"""
    print("Loading classification models...")
    
    # Define the model attributes we want to load
    model_attributes = ['articleType', 'baseColour', 'usage', 'season']
    
    global models, reverse_label_mappings
    models = {}
    reverse_label_mappings = {}
    
    try:
        # Load each model and its corresponding label mapping from saved_models directory
        for attribute in model_attributes:
            model_path = os.path.join('saved_models', f'model_{attribute}.keras')
            mapping_path = os.path.join('saved_models', f'map_{attribute}.npy')
            
            if os.path.exists(model_path) and os.path.exists(mapping_path):
                # Load the model
                models[attribute] = tf.keras.models.load_model(model_path)
                # Load the label mapping
                reverse_label_mappings[attribute] = np.load(mapping_path, allow_pickle=True).item()
                print(f"Successfully loaded model for {attribute}")
            else:
                print(f"Warning: Could not find model files for {attribute}")
                # Fall back to simulation for this attribute
                models[attribute] = None
                
        return len(models) > 0
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# --- Preprocess image for model input ---
def preprocess_image(image_bytes):
    """Preprocess image bytes for model input"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize to the expected input dimensions (224x224 for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Real classification function using loaded models
def classify_image_bytes(image_bytes, filename="uploaded_image"):
    """Classify clothing item in image using TensorFlow models"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            print("Failed to preprocess image")
            return fallback_classification()
            
        results = {}
        
        # Use the models to classify each attribute
        for attribute, model in models.items():
            if model is not None:
                # Run prediction
                prediction = model.predict(processed_image, verbose=0)
                predicted_index = np.argmax(prediction[0])
                
                # Get the label from the reverse mapping
                current_reverse_map = reverse_label_mappings[attribute]
                results[attribute] = current_reverse_map.get(predicted_index, "Unknown")
            else:
                # Use fallback if model is not available
                results[attribute] = fallback_value_for_attribute(attribute)
        
        return results
    except Exception as e:
        print(f"Error classifying image: {e}")
        return fallback_classification()

# Fallback classification (used when real classification fails)
def fallback_classification():
    """Fallback classification when real models fail"""
    import random
    
    # Try to make an educated guess based on filename
    # If we can't, use these defaults
    article_types = ["Tshirts", "Jeans", "Shirts", "Casual Shoes", "Skirts", "Heels", "Jackets", "Dresses"]
    colors = ["Blue", "Black", "White", "Red", "Green", "Yellow", "Purple", "Brown"]
    usages = ["Casual", "Formal", "Party", "Sports"]
    seasons = ["Summer", "Winter", "Spring", "Fall"]
    
    print("WARNING: Using fallback classification since real classification failed")
    return {
        'articleType': random.choice(article_types),
        'baseColour': random.choice(colors),
        'usage': random.choice(usages),
        'season': random.choice(seasons)
    }

def fallback_value_for_attribute(attribute):
    """Get a fallback value for a specific attribute"""
    import random
    
    if attribute == 'articleType':
        return random.choice(["Tshirts", "Jeans", "Shirts", "Casual Shoes", "Skirts", "Heels", "Jackets", "Dresses"])
    elif attribute == 'baseColour':
        return random.choice(["Blue", "Black", "White", "Red", "Green", "Yellow", "Purple", "Brown"])
    elif attribute == 'usage':
        return random.choice(["Casual", "Formal", "Party", "Sports"])
    elif attribute == 'season':
        return random.choice(["Summer", "Winter", "Spring", "Fall"])
    else:
        return "Unknown"

# --- Dynamic Wardrobe Generation ---
def get_wardrobe():
    """Dynamically generate wardrobe from images in the wardrobe folder"""
    wardrobe = []
    item_id = 1000
    
    # Get all image files in the wardrobe folder
    for filename in os.listdir(WARDROBE_FOLDER):
        if not allowed_file(filename):
            continue
            
        # Extract metadata from filename or classify the image
        # Format: color_type_usage.jpg (e.g., blue_tshirt_casual.jpg)
        name_parts = os.path.splitext(filename)[0].split('_')
        
        if len(name_parts) >= 2:
            # If filename has at least color and type (e.g., blue_tshirt.jpg)
            color = name_parts[0].capitalize()
            category = name_parts[1].capitalize()
            # Optional usage part
            usage = name_parts[2].capitalize() if len(name_parts) > 2 else "Casual"
            # Generate a readable description
            description = f"{color} {category}"
        else:
            # If filename doesn't follow the convention, use real classification
            try:
                # Try to classify the image
                with open(os.path.join(WARDROBE_FOLDER, filename), 'rb') as f:
                    image_bytes = f.read()
                    classification = classify_image_bytes(image_bytes, filename)
                    color = classification['baseColour']
                    category = classification['articleType']
                    usage = classification['usage']
                    description = f"{color} {category}"
            except Exception as e:
                print(f"Error classifying wardrobe image {filename}: {e}")
                # Use filename as fallback
                color = "Unknown"
                category = filename.split('.')[0].capitalize()
                usage = "Casual"
                description = category
        
        wardrobe.append({
            "id": item_id,
            "category": category,
            "color": color,
            "usage": usage,
            "description": description,
            "image": filename
        })
        item_id += 1
    
    # Sort the wardrobe by category and color for better organization
    wardrobe.sort(key=lambda x: (x['category'], x['color']))
    
    return wardrobe

def format_wardrobe_for_prompt():
    """Format wardrobe items for chatbot prompt"""
    wardrobe = get_wardrobe()
    if not wardrobe: 
        return "Wardrobe is empty."
    items_str_list = [f"Item(id={item.get('id', 'N/A')}, description='{item.get('description', 'Unknown')}', color='{item.get('color', 'N/A')}')" for item in wardrobe]
    return f"Current Wardrobe Items: [{', '.join(items_str_list)}]"

def get_wardrobe_item_by_id(item_id):
    """Get wardrobe item details by ID"""
    wardrobe = get_wardrobe()
    for item in wardrobe:
        if item.get('id') == item_id:
            return item
    return None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_seasonal_appropriateness(item_classification, current_season, weather_context):
    """Check if clothing item is appropriate for current season and weather"""
    item_season = item_classification.get('season', 'Unknown')
    item_usage = item_classification.get('usage', 'Unknown')
    item_type = item_classification.get('articleType', 'Unknown')
    
    # Weather-based appropriateness
    essential_items = weather_context['recommendations']['essential_items']
    avoid_items = weather_context['recommendations']['avoid_items']
    
    appropriateness = {
        'seasonal_match': item_season.lower() == current_season.lower(),
        'weather_appropriate': True,
        'recommendations': []
    }
    
    # Check if item type should be avoided in current weather
    item_type_lower = item_type.lower()
    for avoid_item in avoid_items:
        if avoid_item.lower() in item_type_lower:
            appropriateness['weather_appropriate'] = False
            appropriateness['recommendations'].append(f"Not recommended for current weather conditions")
            break
    
    # Check if item is in essential items
    for essential_item in essential_items:
        if essential_item.lower() in item_type_lower:
            appropriateness['recommendations'].append(f"Perfect for current weather!")
            break
    
    return appropriateness

def format_weather_context_for_prompt(weather_context):
    """Format weather context for AI prompt"""
    weather = weather_context['weather']
    season = weather_context['season']
    recommendations = weather_context['recommendations']
    
    context_parts = []
    
    # Current weather info
    context_parts.append(f"Current Weather: {weather['temperature']}°C ({weather['condition']}) in {weather['location']}")
    context_parts.append(f"Feels like: {weather['feels_like']}°C, Humidity: {weather['humidity']}%")
    context_parts.append(f"Current Season: {season}")
    
    # Weather recommendations
    if recommendations['essential_items']:
        context_parts.append(f"Essential items for this weather: {', '.join(recommendations['essential_items'])}")
    
    if recommendations['avoid_items']:
        context_parts.append(f"Items to avoid: {', '.join(recommendations['avoid_items'])}")
    
    if recommendations['color_recommendations']:
        context_parts.append(f"Recommended colors: {', '.join(recommendations['color_recommendations'])}")
    
    if recommendations['fabric_recommendations']:
        context_parts.append(f"Recommended fabrics: {', '.join(recommendations['fabric_recommendations'])}")
    
    if recommendations['layering_tips']:
        context_parts.append(f"Layering tips: {'; '.join(recommendations['layering_tips'])}")
    
    return "\n".join(context_parts)

# --- Process chat with Gemini AI and Weather Integration ---
def process_chat(message, session_id, uploaded_images=None):
    """Process chat message with snappy, fun Flair personality and limited suggestions"""
    chat_history = sessions.get(session_id, [])
    weather_context = get_comprehensive_weather_context(message)

    # Personality & Response Rules
    prompt_parts = [
        "You're Flair — a snappy, stylish AI fashion buddy.",
        "Speak casually, like texting a stylish friend 😎.",
        "",  # spacer
        "Response Rules:",
        "1. Give JUST ONE outfit suggestion by default.",
        "2. Only offer more if the user explicitly asks.",
        "3. Use markdown + emojis; keep it short and sweet.",
    ]

    # Add user's wardrobe
    wardrobe_string = format_wardrobe_for_prompt()
    prompt_parts.append(f"User's wardrobe: {wardrobe_string}")

    # Add weather context
    weather_prompt = format_weather_context_for_prompt(weather_context)
    prompt_parts.append(f"Weather Context:\n{weather_prompt}")

    # Include any uploaded images context
    if uploaded_images:
        if len(uploaded_images) == 1:
            item = uploaded_images[0]
            cls = item['classification']
            desc = f"{cls.get('baseColour','Unknown')} {cls.get('articleType','Item')}"
            prompt_parts.append(f"User showed an image of: {desc}")
        else:
            items_desc = [f"{img['classification'].get('baseColour','?')} {img['classification'].get('articleType','Item')}" for img in uploaded_images]
            prompt_parts.append(f"User showed images of: {', '.join(items_desc)}")

    # Add recent conversation
    if chat_history:
        prompt_parts.append("Previous convo:")
        for msg in chat_history[-6:]:
            role = "User" if msg['role']=='user' else "Flair"
            prompt_parts.append(f"{role}: {msg['content']}")

    # Add current message
    prompt_parts.append(f"User: {message}")
    prompt_parts.append("Flair:")

    # Generate with Gemini
    full_prompt = "\n".join(prompt_parts)
    try:
        resp = gemini_model.generate_content(full_prompt)
        ai_text = resp.text.strip()

        # Save to history
        sessions.setdefault(session_id, []).extend([
            {'role':'user','content':message},
            {'role':'assistant','content':ai_text}
        ])

        # Convert markdown to HTML
        html = bleach.clean(markdown.markdown(ai_text), tags=['p','em','strong','ul','ol','li','code'], strip=True)
        result = {'text': ai_text, 'html': html}
        # Attach images if any
        images = add_image_references(ai_text, html)
        result.update(images)
        result['weather_context'] = weather_context
        return result
    except Exception as e:
        print(f"AI error: {e}")
        return {'text': "Oops! Something went wrong.", 'html': '<p>Oops! Something went wrong.</p>', 'images':[], 'weather_context':weather_context}


def add_image_references(text, html):
    """Add image references to the response"""
    # Look for item ID references in the response
    images = []
    response = {"text": text, "html": html, "images": []}
    
    # Get current wardrobe
    wardrobe = get_wardrobe()
    
    # Look for item IDs in the text
    for item in wardrobe:
        item_id = item.get('id')
        # Check if the item ID is mentioned in the response
        # Match different formats: id=1000, ID: 1000, id: 1000
        if f"id={item_id}" in text or f"(ID: {item_id})" in text or f"(id: {item_id})" in text:
            images.append({
                "id": item_id,
                "description": item.get('description'),
                "image_path": f"/static/wardrobe/{item.get('image')}"
            })
    
    if images:
        response["images"] = images
    
    return response

# --- Routes ---
@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/style-history')
def style_history():
    """Serve the style history page"""
    return render_template('style_history.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with weather integration"""
    data = request.json
    session_id = data.get('session_id', 'default_session')
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    response = process_chat(message, session_id)
    return jsonify(response)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle single image upload and classification with weather context"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    session_id = request.form.get('session_id', 'default_session')
    message = request.form.get('message', 'What do you think of this item?')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Classify image
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
            classification = classify_image_bytes(image_bytes, filename)
        
        # Create uploaded image info
        uploaded_image = {
            'path': f"/static/uploads/{filename}",
            'classification': classification
        }
        
        # Process chat with image context
        response = process_chat(message, session_id, [uploaded_image])
        
        # Add image data
        response['uploaded_image'] = uploaded_image
        
        return jsonify(response)
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/upload/multiple', methods=['POST'])
def upload_multiple_images():
    """Handle multiple image uploads and classification with weather context"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files')
    session_id = request.form.get('session_id', 'default_session')
    message = request.form.get('message', 'What do you think of these items?')
    
    if len(files) == 0 or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    uploaded_images = []
    
    for file in files:
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Classify image
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
                classification = classify_image_bytes(image_bytes, filename)
            
            # Add to uploaded images
            uploaded_images.append({
                'path': f"/static/uploads/{filename}",
                'classification': classification
            })
    
    if not uploaded_images:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Process chat with images context
    response = process_chat(message, session_id, uploaded_images)
    
    # Add uploaded images data
    response['uploaded_images'] = uploaded_images
    
    return jsonify(response)

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get current weather for a location"""
    location = request.args.get('location', 'New York, NY')
    
    try:
        weather_data = weather_service.get_current_weather(location)
        season = weather_service.get_current_season(location)
        recommendations = weather_service.get_weather_based_recommendations(weather_data, season)
        
        return jsonify({
            'weather': weather_data,
            'season': season,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return jsonify({'error': 'Unable to fetch weather data'}), 500

@app.route('/static/wardrobe/<path:filename>')
def serve_wardrobe_image(filename):
    """Serve wardrobe images"""
    return send_from_directory(WARDROBE_FOLDER, filename)

@app.route('/static/uploads/<path:filename>')
def serve_uploaded_image(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    return send_from_directory('static/css', filename)

@app.route('/api/wardrobe')
def get_wardrobe_api():
    """API endpoint to get wardrobe items"""
    return jsonify(get_wardrobe())

@app.route('/wardrobe')
def wardrobe_page():
    """Wardrobe management page"""
    return render_template('wardrobe.html')

@app.route('/api/wardrobe/add', methods=['POST'])
def add_to_wardrobe():
    """Add item to wardrobe"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    color = request.form.get('color', 'Unknown')
    category = request.form.get('category', 'Unknown')
    usage = request.form.get('usage', 'Casual')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create filename based on metadata: color_category_usage.jpg
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        new_filename = f"{color.lower()}_{category.lower()}_{usage.lower()}.{original_ext}"
        
        # Save file
        file_path = os.path.join(WARDROBE_FOLDER, new_filename)
        file.save(file_path)
        
        # Return new wardrobe item
        return jsonify({
            'success': True,
            'message': 'Item added to wardrobe',
            'item': {
                'id': 9999,  # Placeholder, real ID will be assigned when loaded
                'category': category,
                'color': color,
                'usage': usage,
                'description': f"{color} {category}",
                'image': new_filename
            }
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    # Create theme.css if it doesn't exist
    theme_css_path = 'static/css/theme.css'
    if not os.path.exists(theme_css_path):
        # Create a basic theme CSS file with brown and white colors
        with open(theme_css_path, 'w') as f:
            f.write("""
/* Basic brown and white theme for Attierly */
:root {
    --primary-color: #8B4513;        /* Rich brown */
    --primary-light: #A0522D;        /* Lighter brown */
    --primary-dark: #5D2906;         /* Darker brown */
    --secondary-color: #F5F5DC;      /* Beige/off-white */
    --text-on-primary: #FFFFFF;      /* White text on brown */
    --text-primary: #3E2723;         /* Dark brown text */
    --text-secondary: #795548;       /* Medium brown text */
    --background-color: #FFF8E1;     /* Light cream background */
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--background-color);
    color: var(--text-primary);
}

.navbar {
    background-color: var(--primary-color);
}

.navbar-brand {
    color: var(--text-on-primary) !important;
}

.btn-primary {
    background-color: var(--primary-color);
    border-color: var(--primary-color);
}

.btn-primary:hover {
    background-color: var(--primary-light);
    border-color: var(--primary-light);
}
""")
    
    # Ensure wardrobe image directory exists
    os.makedirs(WARDROBE_FOLDER, exist_ok=True)
    
    # Load classification models
    load_classification_models()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)