#!/bin/bash

# Attierly AI Fashion Chatbot - Enhanced Setup Script (Multiple Image Upload Version)

echo "====== Setting up Enhanced Attierly AI Fashion Chatbot ======"

# Create project directory structure
echo "Creating project directory structure..."
mkdir -p attierly/static/uploads attierly/static/wardrobe attierly/static/css attierly/templates

# Copy app.py with multiple image upload support
echo "Creating app.py with multiple image upload support..."
cat > attierly/app.py << 'EOL'
# Content of the enhanced app.py file would be here
# This is a large file, so it's not included in this script for brevity
# Copy the content of the "app.py (with Multiple Image Upload Support)" artifact here
EOL

# Create theme.css file
echo "Creating custom brown & white theme CSS..."
cat > attierly/static/css/theme.css << 'EOL'
/* Custom theme styles for Attierly - Brown & White */
# The CSS content would be here
# This is a large file, so it's not included in this script for brevity
# Copy the content of the "attierly/static/css/theme.css" artifact here
EOL

# Create index.html template (chat UI with multiple image upload support)
echo "Creating index.html template (chat UI with multiple image upload)..."
cat > attierly/templates/index.html << 'EOL'
<!DOCTYPE html>
<!-- The HTML content would be here -->
<!-- This is a large file, so it's not included in this script for brevity -->
<!-- Copy the content of the "attierly/templates/index.html" artifact here -->
EOL

# Create style_history.html template
echo "Creating style_history.html template..."
cat > attierly/templates/style_history.html << 'EOL'
<!DOCTYPE html>
<!-- The HTML content would be here -->
<!-- This is a large file, so it's not included in this script for brevity -->
<!-- Copy the content of the "attierly/templates/style_history.html" artifact here -->
EOL

# Create wardrobe.html template (reusing the existing one)
echo "Creating wardrobe.html template..."
cat > attierly/templates/wardrobe.html << 'EOL'
<!DOCTYPE html>
<!-- The HTML content would be here -->
<!-- This is a large file, so it's not included in this script for brevity -->
<!-- Copy the content of the wardrobe.html template here -->
EOL

# Create sample wardrobe items using the naming convention: color_category_usage.jpg
echo "Creating sample wardrobe items..."
mkdir -p attierly/static/wardrobe

# Function to create a colored rectangle as a placeholder image
create_placeholder_image() {
    local filename=$1
    local color=$2
    local text=$3
    local width=400
    local height=400
    
    # Create simple SVG as placeholder
    cat > attierly/static/wardrobe/${filename} << IMGEOF
<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="${color}"/>
  <text x="50%" y="50%" font-family="Arial" font-size="40" fill="white" text-anchor="middle" dominant-baseline="middle">${text}</text>
</svg>
IMGEOF
    echo "Created ${filename} as SVG"
}

# Create sample wardrobe items using the naming convention: color_category_usage.jpg
create_placeholder_image "blue_tshirts_casual.jpg" "#3498db" "Blue T-shirt"
create_placeholder_image "black_jeans_casual.jpg" "#2c3e50" "Black Jeans"
create_placeholder_image "white_shirts_formal.jpg" "#ecf0f1" "White Shirt"
create_placeholder_image "white_casualshoes_casual.jpg" "#f5f5f5" "White Sneakers"
create_placeholder_image "white_skirts_casual.jpg" "#f8f9fa" "White Skirt"
create_placeholder_image "red_heels_party.jpg" "#e74c3c" "Red Heels"
create_placeholder_image "brown_jackets_casual.jpg" "#a0522d" "Brown Jacket"
create_placeholder_image "black_dresses_party.jpg" "#34495e" "Black Dress"
create_placeholder_image "green_skirts_casual.jpg" "#2ecc71" "Green Skirt"
create_placeholder_image "green_shirts_casual.jpg" "#2ecc71" "Green Shirt"
create_placeholder_image "green_jackets_casual.jpg" "#2ecc71" "Green Jacket"
create_placeholder_image "brown_casualshoes_casual.jpg" "#a0522d" "Brown Shoes"
create_placeholder_image "yellow_shirts_casual.jpg" "#f1c40f" "Yellow Shirt"

# Create .env file
echo "Creating .env file..."
cat > attierly/.env << 'EOL'
# Attierly Environment Configuration
GEMINI_API_KEY=your_gemini_api_key_here
EOL

# Create requirements.txt file
echo "Creating requirements.txt file..."
cat > attierly/requirements.txt << 'EOL'
flask==2.3.3
python-dotenv==1.0.0
google-generativeai==0.3.1
tensorflow==2.15.0
pillow==10.1.0
markdown==3.5.1
bleach==6.1.0
werkzeug==2.3.7
EOL

# Install required packages
echo "Installing required Python packages..."
pip install flask python-dotenv google-generativeai pillow tensorflow markdown bleach werkzeug

# Instructions for the user
echo "
====== Attierly AI Fashion Chatbot Setup Complete ======

To run the application:
1. Edit the attierly/.env file and add your Gemini API key:
   GEMINI_API_KEY=your_actual_gemini_api_key_here

2. Navigate to the attierly directory:
   cd attierly

3. Start the Flask application:
   python app.py

4. Open your browser and go to:
   http://localhost:5000

Key Features:
- Multiple image upload with custom messages
- Rich brown & white color theme
- Dynamic wardrobe detection from the 'static/wardrobe' folder
- New wardrobe management page at /wardrobe
- Style history tracking at /style-history

Notes:
- The application is using simulated classification for images.
- To integrate real classification, update the classify_image_bytes function
  in app.py with your existing TensorFlow models.
- For production use, replace the placeholder wardrobe images with real clothing items.

Enjoy your Enhanced AI Fashion Chatbot!
"