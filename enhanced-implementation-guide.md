# Attierly Enhanced Implementation Guide

This guide explains how to implement the enhanced version of Attierly with multiple image upload support, custom messaging, and a rich brown & white color theme.

## New Features

1. **Multiple Image Upload**
   - Users can upload multiple images at once
   - Custom messaging for uploaded images
   - Visual preview before uploading

2. **Rich Brown & White Theme**
   - Elegant brown and cream color scheme
   - Consistent styling across all pages
   - Enhanced visual hierarchy

3. **Custom Image Upload Messages**
   - Users can add specific questions or comments with their image uploads
   - Context is preserved in AI responses

4. **Dynamic Wardrobe Detection**
   - Automatically detects clothing items from the wardrobe folder
   - Extracts metadata from filename patterns
   - Organizes items by category for easy filtering

5. **Style History Page**
   - Keeps track of past outfit recommendations
   - Shows image references for wardrobe items
   - Organizes by date and occasion

## Implementation Steps

### Option 1: Use the Setup Script (Recommended)

The easiest way to implement all these changes is using the provided setup script:

1. Save the setup script (`setup.sh`) to your computer
2. Make it executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the script:
   ```bash
   ./setup.sh
   ```
4. Follow the post-installation instructions to add your Gemini API key
5. Start the application:
   ```bash
   cd attierly
   python app.py
   ```

### Option 2: Manual Implementation

If you prefer to implement the changes manually:

1. Create the project directory structure:
   ```bash
   mkdir -p attierly/static/uploads attierly/static/wardrobe attierly/static/css attierly/templates
   ```

2. Copy the enhanced files:
   - `app.py` from the "app.py (with Multiple Image Upload Support)" artifact to `attierly/app.py`
   - The CSS theme to `attierly/static/css/theme.css`
   - The enhanced index.html to `attierly/templates/index.html`
   - The style_history.html template to `attierly/templates/style_history.html`
   - The wardrobe.html template to `attierly/templates/wardrobe.html`

3. Create sample wardrobe items:
   - Add SVG placeholders or actual clothing images to `attierly/static/wardrobe`
   - Use the naming convention: `color_category_usage.jpg` (e.g., `blue_tshirts_casual.jpg`)

4. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

5. Install dependencies:
   ```bash
   pip install flask python-dotenv google-generativeai pillow tensorflow markdown bleach werkzeug
   ```

6. Run the application:
   ```bash
   cd attierly
   python app.py
   ```

## File Structure Explanation

- `app.py`: Flask application with enhanced multiple image upload support
- `static/css/theme.css`: Rich brown & white color theme
- `templates/index.html`: Chat interface with multiple image upload
- `templates/wardrobe.html`: Wardrobe management page
- `templates/style_history.html`: Style history tracking page
- `static/wardrobe/`: Directory containing wardrobe items
- `static/uploads/`: Directory for user-uploaded images

## API Endpoints

- `/`: Main chat interface
- `/wardrobe`: Wardrobe management page
- `/style-history`: Style history page
- `/api/chat`: Chat endpoint for text messages
- `/api/upload/multiple`: Endpoint for multiple image uploads
- `/api/wardrobe`: Get wardrobe items
- `/api/wardrobe/add`: Add item to wardrobe

## Key Code Changes

### 1. Multiple Image Upload

The new index.html adds a modal for uploading multiple images with a custom message:

```javascript
function handleImageUpload(files, customMessage) {
    // Create FormData for multiple file upload
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    formData.append('session_id', sessionId);
    formData.append('message', customMessage || "What do you think of these items?");
    
    // Send request to the backend
    fetch('/api/upload/multiple', {
        method: 'POST',
        body: formData
    })
    // ...
}
```

### 2. Custom Brown & White Theme

The theme.css file defines a rich color palette using CSS variables:

```css
:root {
    --primary-color: #8B4513;        /* Rich brown */
    --primary-light: #A0522D;        /* Lighter brown */
    --primary-dark: #5D2906;         /* Darker brown */
    --secondary-color: #F5F5DC;      /* Beige/off-white */
    --text-on-primary: #FFFFFF;      /* White text on brown */
    --text-primary: #3E2723;         /* Dark brown text */
    /* ... */
}
```

### 3. Backend Multiple Image Support

The app.py file adds a new endpoint for handling multiple images:

```python
@app.route('/api/upload/multiple', methods=['POST'])
def upload_multiple_images():
    """Handle multiple image uploads and classification"""
    files = request.files.getlist('files')
    session_id = request.form.get('session_id', 'default_session')
    message = request.form.get('message', 'What do you think of these items?')
    
    uploaded_images = []
    
    for file in files:
        # Process each image...
        uploaded_images.append({
            'path': f"/static/uploads/{filename}",
            'classification': classification
        })
    
    # Process chat with all images context
    response = process_chat(message, session_id, uploaded_images)
    response['uploaded_images'] = uploaded_images
    
    return jsonify(response)
```

## Integration with Classification Models

To integrate your actual TensorFlow classification models:

1. Replace the simulated `classify_image_bytes` function in `app.py`:

```python
def classify_image_bytes(image_bytes, filename="uploaded_image"):
    """Classify clothing item in image using actual models"""
    try:
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return {}
            
        results = {}
        for attribute, model in models.items():
            prediction = model.predict(processed_image, verbose=0)
            current_reverse_map = reverse_label_mappings[attribute]
            predicted_index = np.argmax(prediction[0])
            results[attribute] = current_reverse_map.get(predicted_index, "Unknown")
            
        return results
    except Exception as e:
        print(f"Error classifying image: {e}")
        return {}
```

2. Update the `load_classification_models` function to load your models.

## User Experience Flow

1. **Chat Interface**:
   - Type messages directly to chat with Flair
   - Click the image upload button to upload multiple images
   - Add a custom message with uploaded images
   - View AI responses with formatted markdown and image references

2. **Wardrobe Management**:
   - Browse all wardrobe items with filtering options
   - Add new items with detailed metadata
   - Drag and drop image uploads

3. **Style History**:
   - View past outfit recommendations
   - See visual references for wardrobe items
   - Track purchase recommendations

## Troubleshooting

- **Missing API key**: Check that you've added your Gemini API key to the `.env` file
- **Image upload errors**: Check folder permissions for the uploads directory
- **Missing wardrobe items**: Make sure files in the wardrobe folder follow the naming convention

## Next Steps for Enhancement

1. **Database Integration**:
   - Replace file-based storage with a database
   - Add user accounts and authentication
   - Support multiple wardrobes per user

2. **Advanced AI Features**:
   - Add outfit scoring and ranking
   - Implement seasonal recommendations
   - Create style personality profiles

3. **Mobile Optimization**:
   - Create a responsive mobile app version
   - Add camera capture for real-time styling
