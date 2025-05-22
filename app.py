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

# Import the new outfit visualizer
from utils.outfit_visualizer import OutfitVisualizer

# Import virtual try-on
from utils.virtual_tryon import virtual_tryon

app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
WARDROBE_FOLDER = 'static/wardrobe'
USER_PHOTOS_FOLDER = 'static/user_photos'
TRYON_RESULTS_FOLDER = 'static/tryon_results'
DATASET_PATH = '/Users/gvssriram/Desktop/projects-internship/Flair_POC/fashion-dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WARDROBE_FOLDER, exist_ok=True)
os.makedirs(USER_PHOTOS_FOLDER, exist_ok=True)
os.makedirs(TRYON_RESULTS_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# --- Session Storage ---
sessions = {}
user_photos = {}  # Store user photos for try-on

# --- Gemini API setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    print("WARNING: GEMINI_API_KEY not found.")

# --- Initialize Services ---
weather_service = WeatherService()
outfit_visualizer = OutfitVisualizer(DATASET_PATH, WARDROBE_FOLDER)

# --- Global variables for models ---
models = {}
reverse_label_mappings = {}

# --- Load classification models from TensorFlow ---
def load_classification_models():
    """Load the TensorFlow classification models"""
    print("Loading classification models...")
    
    model_attributes = ['articleType', 'baseColour', 'usage', 'season']
    
    global models, reverse_label_mappings
    models = {}
    reverse_label_mappings = {}
    
    try:
        for attribute in model_attributes:
            model_path = os.path.join('saved_models', f'model_{attribute}.keras')
            mapping_path = os.path.join('saved_models', f'map_{attribute}.npy')
            
            if os.path.exists(model_path) and os.path.exists(mapping_path):
                models[attribute] = tf.keras.models.load_model(model_path)
                reverse_label_mappings[attribute] = np.load(mapping_path, allow_pickle=True).item()
                print(f"Successfully loaded model for {attribute}")
            else:
                print(f"Warning: Could not find model files for {attribute}")
                models[attribute] = None
                
        return len(models) > 0
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# --- Image Processing Functions ---
def preprocess_image(image_bytes):
    """Preprocess image bytes for model input"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image_bytes(image_bytes, filename="uploaded_image"):
    """Classify clothing item in image using TensorFlow models"""
    try:
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            print("Failed to preprocess image")
            return fallback_classification()
            
        results = {}
        
        for attribute, model in models.items():
            if model is not None:
                prediction = model.predict(processed_image, verbose=0)
                predicted_index = np.argmax(prediction[0])
                current_reverse_map = reverse_label_mappings[attribute]
                results[attribute] = current_reverse_map.get(predicted_index, "Unknown")
            else:
                results[attribute] = fallback_value_for_attribute(attribute)
        
        return results
    except Exception as e:
        print(f"Error classifying image: {e}")
        return fallback_classification()

def fallback_classification():
    """Fallback classification when real models fail"""
    import random
    
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
    
    fallback_values = {
        'articleType': ["Tshirts", "Jeans", "Shirts", "Casual Shoes", "Skirts", "Heels", "Jackets", "Dresses"],
        'baseColour': ["Blue", "Black", "White", "Red", "Green", "Yellow", "Purple", "Brown"],
        'usage': ["Casual", "Formal", "Party", "Sports"],
        'season': ["Summer", "Winter", "Spring", "Fall"]
    }
    
    return random.choice(fallback_values.get(attribute, ["Unknown"]))

# --- Wardrobe Management ---
def get_wardrobe():
    """Dynamically generate wardrobe from images in the wardrobe folder"""
    wardrobe = []
    item_id = 1000
    
    for filename in os.listdir(WARDROBE_FOLDER):
        if not allowed_file(filename):
            continue
            
        name_parts = os.path.splitext(filename)[0].split('_')
        
        if len(name_parts) >= 2:
            color = name_parts[0].capitalize()
            category = name_parts[1].capitalize()
            usage = name_parts[2].capitalize() if len(name_parts) > 2 else "Casual"
            description = f"{color} {category}"
        else:
            try:
                with open(os.path.join(WARDROBE_FOLDER, filename), 'rb') as f:
                    image_bytes = f.read()
                    classification = classify_image_bytes(image_bytes, filename)
                    color = classification['baseColour']
                    category = classification['articleType']
                    usage = classification['usage']
                    description = f"{color} {category}"
            except Exception as e:
                print(f"Error classifying wardrobe image {filename}: {e}")
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
    
    wardrobe.sort(key=lambda x: (x['category'], x['color']))
    return wardrobe

def format_wardrobe_for_prompt():
    """Format wardrobe items for chatbot prompt"""
    wardrobe = get_wardrobe()
    if not wardrobe: 
        return "Wardrobe is empty."
    items_str_list = [f"Item(id={item.get('id', 'N/A')}, description='{item.get('description', 'Unknown')}', color='{item.get('color', 'N/A')}')" for item in wardrobe]
    return f"Current Wardrobe Items: [{', '.join(items_str_list)}]"

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_weather_context_for_prompt(weather_context):
    """Format weather context for AI prompt"""
    weather = weather_context['weather']
    season = weather_context['season']
    recommendations = weather_context['recommendations']
    
    context_parts = []
    context_parts.append(f"Current Weather: {weather['temperature']}°C ({weather['condition']}) in {weather['location']}")
    context_parts.append(f"Feels like: {weather['feels_like']}°C, Humidity: {weather['humidity']}%")
    context_parts.append(f"Current Season: {season}")
    
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

# --- Chat Processing ---
def process_chat(message, session_id, uploaded_images=None):
    """Process chat message with snappy, fun Flair personality and outfit visualization"""
    chat_history = sessions.get(session_id, [])
    weather_context = get_comprehensive_weather_context(message)

    # Personality & Response Rules
    prompt_parts = [
        "You're Flair — a snappy, stylish AI fashion buddy.",
        "Speak casually, like texting a stylish friend 😎.",
        "",
        "Response Rules:",
        "1. Give JUST ONE outfit suggestion by default.",
        "2. Only offer more if the user explicitly asks.",
        "3. Use markdown + emojis; keep it short and sweet.",
        "4. When suggesting outfits, be specific about colors and items.",
        "5. Format outfit suggestions clearly like: 'Outfit: Red tee, blue jeans, black jacket'",
        "6. IMPORTANT: After suggesting an outfit, check what items are missing from the user's wardrobe and mention they need to purchase those items.",
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
        
        # Get outfit visualization data
        outfit_data = outfit_visualizer.get_outfit_visualization_data(ai_text)
        
        # Enhance AI response with purchase recommendations
        enhanced_ai_text = enhance_response_with_purchase_info(ai_text, outfit_data)
        enhanced_html = bleach.clean(markdown.markdown(enhanced_ai_text), tags=['p','em','strong','ul','ol','li','code'], strip=True)
        
        result = {
            'text': enhanced_ai_text, 
            'html': enhanced_html,
            'weather_context': weather_context,
            'outfit_visualization': outfit_data
        }
        
        # Attach wardrobe images if any
        images = add_image_references(ai_text, html)
        result.update(images)
        
        return result
    except Exception as e:
        print(f"AI error: {e}")
        return {
            'text': "Oops! Something went wrong.", 
            'html': '<p>Oops! Something went wrong.</p>', 
            'images':[], 
            'weather_context':weather_context,
            'outfit_visualization': {'outfits': [], 'has_outfits': False}
        }

def enhance_response_with_purchase_info(ai_text, outfit_data):
    """Enhance AI response with purchase recommendations for missing items"""
    if not outfit_data or not outfit_data.get('has_outfits'):
        return ai_text
    
    enhanced_text = ai_text
    
    # Check each outfit for missing items (only process first outfit to avoid repetition)
    outfit = outfit_data.get('outfits', [{}])[0]
    missing_items = []
    suggested_items = []
    wardrobe_items = []
    
    for item_data in outfit.get('items_with_images', []):
        item_desc = f"{item_data['item']['color']} {item_data['item']['type']}".replace(' Any', '').strip()
        
        if item_data['source'] == 'placeholder':
            # Item not found anywhere - definitely need to buy
            missing_items.append(item_desc.lower())
        elif item_data['source'] == 'dataset':
            # Item found in dataset but not in wardrobe - suggest purchase
            suggested_items.append(item_desc.lower())
        elif item_data['source'] == 'wardrobe':
            # Item found in user's wardrobe - great!
            wardrobe_items.append(item_desc.lower())
    
    # Add purchase recommendations if there are missing or suggested items
    if missing_items or suggested_items:
        enhanced_text += "\n\n🛍️ **Shopping Update:**\n"
        
        if wardrobe_items:
            enhanced_text += f"✅ You already have: {', '.join(wardrobe_items)}\n"
        
        if suggested_items:
            enhanced_text += f"🛒 Consider buying: {', '.join(suggested_items)}\n"
        
        if missing_items:
            enhanced_text += f"❗ Definitely need: {', '.join(missing_items)}\n"
        
        # Add encouraging shopping message
        if missing_items:
            enhanced_text += "\nTime for some shopping! Those items will complete your look perfectly! 💫"
        elif suggested_items:
            enhanced_text += "\nA little shopping trip would make this outfit even better! ✨"
    else:
        # All items found in wardrobe
        enhanced_text += "\n\n✨ **Perfect!** You already have everything for this look in your wardrobe! 👏"
    
    return enhanced_text

def add_image_references(text, html):
    """Add image references to the response"""
    images = []
    response = {"text": text, "html": html, "images": []}
    
    wardrobe = get_wardrobe()
    
    for item in wardrobe:
        item_id = item.get('id')
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

@app.route('/wardrobe')
def wardrobe_page():
    """Wardrobe management page"""
    return render_template('wardrobe.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with weather integration and outfit visualization"""
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
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
            classification = classify_image_bytes(image_bytes, filename)
        
        uploaded_image = {
            'path': f"/static/uploads/{filename}",
            'classification': classification
        }
        
        response = process_chat(message, session_id, [uploaded_image])
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
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
                classification = classify_image_bytes(image_bytes, filename)
            
            uploaded_images.append({
                'path': f"/static/uploads/{filename}",
                'classification': classification
            })
    
    if not uploaded_images:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    response = process_chat(message, session_id, uploaded_images)
    response['uploaded_images'] = uploaded_images
    
    return jsonify(response)

# --- Virtual Try-On Routes ---
@app.route('/api/upload/user-photo', methods=['POST'])
def upload_user_photo():
    """Upload user photo for virtual try-on"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    session_id = request.form.get('session_id', 'default_session')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = f"user_{session_id}_{secure_filename(file.filename)}"
        file_path = os.path.join(USER_PHOTOS_FOLDER, filename)
        file.save(file_path)
        
        # Store user photo path in session
        user_photos[session_id] = file_path
        
        return jsonify({
            'success': True,
            'message': 'User photo uploaded successfully',
            'photo_path': f"/static/user_photos/{filename}"
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon_api():
    """Virtual try-on API endpoint"""
    data = request.json
    session_id = data.get('session_id', 'default_session')
    outfit_items = data.get('outfit_items', [])
    selected_indices = data.get('selected_items', None)
    
    # Check if user has uploaded a photo
    if session_id not in user_photos:
        return jsonify({'error': 'No user photo uploaded. Please upload your photo first.'}), 400
    
    user_photo_path = user_photos[session_id]
    
    if not os.path.exists(user_photo_path):
        return jsonify({'error': 'User photo not found. Please upload again.'}), 400
    
    try:
        # Perform virtual try-on
        result_path = virtual_tryon.try_on_outfit(user_photo_path, outfit_items, selected_indices)
        
        if result_path:
            # Convert result to base64 for web display
            result_base64 = virtual_tryon.image_to_base64(result_path)
            
            if result_base64:
                return jsonify({
                    'success': True,
                    'result_image': result_base64,
                    'message': 'Virtual try-on completed successfully'
                })
            else:
                return jsonify({'error': 'Failed to process try-on result'}), 500
        else:
            return jsonify({'error': 'Virtual try-on failed. Please try again.'}), 500
            
    except Exception as e:
        print(f"Virtual try-on error: {e}")
        return jsonify({'error': f'Virtual try-on failed: {str(e)}'}), 500

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

@app.route('/api/wardrobe')
def get_wardrobe_api():
    """API endpoint to get wardrobe items"""
    return jsonify(get_wardrobe())

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
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        new_filename = f"{color.lower()}_{category.lower()}_{usage.lower()}.{original_ext}"
        
        file_path = os.path.join(WARDROBE_FOLDER, new_filename)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'message': 'Item added to wardrobe',
            'item': {
                'id': 9999,
                'category': category,
                'color': color,
                'usage': usage,
                'description': f"{color} {category}",
                'image': new_filename
            }
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

# --- Static File Serving ---
@app.route('/static/wardrobe/<path:filename>')
def serve_wardrobe_image(filename):
    """Serve wardrobe images"""
    return send_from_directory(WARDROBE_FOLDER, filename)

@app.route('/static/uploads/<path:filename>')
def serve_uploaded_image(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/user_photos/<path:filename>')
def serve_user_photo(filename):
    """Serve user photos"""
    return send_from_directory(USER_PHOTOS_FOLDER, filename)

@app.route('/static/tryon_results/<path:filename>')
def serve_tryon_result(filename):
    """Serve try-on result images"""
    return send_from_directory(TRYON_RESULTS_FOLDER, filename)

@app.route('/static/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    return send_from_directory('static/css', filename)

@app.route('/dataset/images/<path:filename>')
def serve_dataset_image(filename):
    """Serve dataset images"""
    dataset_images_path = os.path.join(DATASET_PATH, 'images')
    return send_from_directory(dataset_images_path, filename)

@app.route('/static/placeholder/<int:width>/<int:height>')
def serve_placeholder(width, height):
    """Serve placeholder images"""
    from PIL import Image, ImageDraw
    
    # Create a simple placeholder image
    img = Image.new('RGB', (width, height), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    text = f"{width}x{height}"
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill='#666666')
    
    # Save to memory
    output = io.BytesIO()
    img.save(output, format='JPEG')
    output.seek(0)
    
    from flask import Response
    return Response(output.getvalue(), mimetype='image/jpeg')

if __name__ == '__main__':
    # Create theme.css if it doesn't exist
    theme_css_path = 'static/css/theme.css'
    if not os.path.exists(theme_css_path):
        with open(theme_css_path, 'w') as f:
            f.write("""
/* Basic brown and white theme for Attierly */
:root {
    --primary-color: #8B4513;
    --primary-light: #A0522D;
    --primary-dark: #5D2906;
    --secondary-color: #F5F5DC;
    --text-on-primary: #FFFFFF;
    --text-primary: #3E2723;
    --text-secondary: #795548;
    --background-color: #FFF8E1;
    --border-color: #DEB887;
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
    
    # Ensure directories exist
    os.makedirs(WARDROBE_FOLDER, exist_ok=True)
    os.makedirs(USER_PHOTOS_FOLDER, exist_ok=True)
    os.makedirs(TRYON_RESULTS_FOLDER, exist_ok=True)
    
    # Load classification models
    load_classification_models()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)