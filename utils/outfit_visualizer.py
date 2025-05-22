import pandas as pd
import os
import re
from typing import Dict, List, Optional
import random

class OutfitVisualizer:
    """Simple outfit visualization for demo - no complex mappings"""
    
    def __init__(self, dataset_path: str, wardrobe_folder: str):
        self.dataset_path = dataset_path
        self.wardrobe_folder = wardrobe_folder
        self.styles_csv_path = os.path.join(dataset_path, 'styles.csv')
        self.dataset_images_path = os.path.join(dataset_path, 'images')
        
        # Load dataset
        self.dataset_df = self._load_dataset()
        
        # Hardcoded color mappings for quick demo
        self.color_mappings = {
            'red': ['Red', 'Maroon', 'Burgundy', 'Pink'],
            'blue': ['Blue', 'Navy Blue', 'Light Blue', 'Dark Blue', 'Navy'],
            'black': ['Black', 'Dark Grey', 'Charcoal'],
            'white': ['White', 'Off White', 'Cream', 'Ivory'],
            'green': ['Green', 'Dark Green', 'Olive', 'Forest Green'],
            'brown': ['Brown', 'Tan', 'Beige', 'Khaki'],
            'grey': ['Grey', 'Gray', 'Light Grey', 'Dark Grey'],
            'yellow': ['Yellow', 'Gold', 'Mustard'],
            'purple': ['Purple', 'Violet', 'Lavender'],
            'orange': ['Orange', 'Coral'],
            'pink': ['Pink', 'Rose', 'Magenta']
        }
        
        # Hardcoded article type mappings
        self.article_mappings = {
            'tshirt': ['Tshirts', 'Shirts'],
            't-shirt': ['Tshirts', 'Shirts'],
            'tee': ['Tshirts'],
            'shirt': ['Shirts', 'Tshirts'],
            'trouser': ['Jeans', 'Trousers', 'Track Pants'],
            'trousers': ['Jeans', 'Trousers', 'Track Pants'],
            'jeans': ['Jeans'],
            'pants': ['Jeans', 'Trousers', 'Track Pants'],
            'jacket': ['Jackets', 'Blazers'],
            'coat': ['Jackets'],
            'dress': ['Dresses'],
            'skirt': ['Skirts'],
            'shoes': ['Casual Shoes', 'Formal Shoes', 'Sports Shoes'],
            'sneakers': ['Casual Shoes', 'Sports Shoes'],
            'boots': ['Boots'],
            'sandals': ['Sandals', 'Flip Flops'],
            'heels': ['Heels']
        }
        
    def _load_dataset(self) -> Optional[pd.DataFrame]:
        """Load the fashion dataset"""
        try:
            if os.path.exists(self.styles_csv_path):
                df = pd.read_csv(self.styles_csv_path, on_bad_lines='skip')
                df['id'] = df['id'].astype(str)
                print(f"Loaded dataset with {len(df)} items")
                return df
            else:
                print(f"Dataset not found at {self.styles_csv_path}")
                return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def extract_outfit_items(self, ai_response: str) -> List[Dict]:
        """Extract clothing items with simple regex patterns"""
        print(f"DEBUG: Parsing AI response: {ai_response}")
        
        # Remove emojis and clean text
        clean_text = re.sub(r'[🔴💙🖤👕👖🧥⚫⚪🔵🟡🟠🟢🟣🟤😍😎]', '', ai_response)
        
        # Look for "Outfit:" sections
        outfit_sections = re.split(r'(?i)outfit\s*[:\-]?', clean_text)
        
        all_outfits = []
        
        for i, section in enumerate(outfit_sections):
            if not section.strip():
                continue
                
            items = self._extract_items_simple(section)
            
            if items:
                all_outfits.append({
                    'outfit_id': i,
                    'title': f"Outfit {i}" if i > 0 else "Recommended Outfit",
                    'items': items,
                    'description': section.strip()[:150]
                })
        
        # If no "Outfit:" found, parse the whole response
        if not all_outfits:
            items = self._extract_items_simple(clean_text)
            if items:
                all_outfits.append({
                    'outfit_id': 0,
                    'title': "Recommended Outfit",
                    'items': items,
                    'description': clean_text[:150]
                })
        
        print(f"DEBUG: Extracted {len(all_outfits)} outfits")
        return all_outfits
    
    def _extract_items_simple(self, text: str) -> List[Dict]:
        """Simple item extraction with direct patterns"""
        items = []
        
        # Simple patterns that match your AI's format
        patterns = [
            r'(\w+)\s+([Tt]-shirt|[Tt]shirt)',  # "Red T-shirt"
            r'(\w+)\s+(trouser|trousers)',       # "blue trousers"
            r'(\w+)\s+(jacket|coat)',            # "black jacket"
            r'(\w+)\s+(jeans?)',                 # "blue jeans"
            r'(\w+)\s+(shirt)',                  # "white shirt"
            r'(\w+)\s+(dress)',                  # "red dress"
            r'(\w+)\s+(shoes?|sneakers?|boots?|heels?)', # footwear
            r'(\w+)\s+(skirt)',                  # "black skirt"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                color_word = match.group(1).lower()
                item_word = match.group(2).lower()
                
                print(f"DEBUG: Found {color_word} {item_word}")
                
                # Map to dataset values
                dataset_color = self._map_color(color_word)
                dataset_article = self._map_article(item_word)
                
                if dataset_article:  # Only add if we can map the article
                    items.append({
                        'color': dataset_color or 'Any',
                        'type': dataset_article,
                        'usage': 'Casual',  # Default for demo
                        'season': 'Summer'  # Default for demo
                    })
                    print(f"DEBUG: Mapped to {dataset_color} {dataset_article}")
        
        return items
    
    def _map_color(self, color_word: str) -> Optional[str]:
        """Map color word to dataset color"""
        color_lower = color_word.lower()
        
        # Direct mapping first
        if color_lower in self.color_mappings:
            possible_colors = self.color_mappings[color_lower]
            
            # Check which colors exist in dataset
            if self.dataset_df is not None:
                for color in possible_colors:
                    if any(self.dataset_df['baseColour'].str.contains(color, case=False, na=False)):
                        return color
            
            # Fallback to first option
            return possible_colors[0]
        
        # Try partial matching
        for key, values in self.color_mappings.items():
            if key in color_lower or color_lower in key:
                return values[0]
        
        return None
    
    def _map_article(self, item_word: str) -> Optional[str]:
        """Map item word to dataset article type"""
        item_lower = item_word.lower().replace('-', '')
        
        # Direct mapping
        if item_lower in self.article_mappings:
            possible_articles = self.article_mappings[item_lower]
            
            # Check which articles exist in dataset
            if self.dataset_df is not None:
                for article in possible_articles:
                    if any(self.dataset_df['articleType'].str.contains(article, case=False, na=False)):
                        return article
            
            # Fallback to first option
            return possible_articles[0]
        
        # Try partial matching
        for key, values in self.article_mappings.items():
            if key in item_lower or item_lower in key:
                return values[0]
        
        return None
    
    def find_matching_items(self, outfit_items: List[Dict]) -> List[Dict]:
        """Find images for outfit items"""
        outfits_with_images = []
        
        for outfit in outfit_items:
            items_with_images = []
            
            for item in outfit['items']:
                print(f"DEBUG: Finding image for {item}")
                
                # Try wardrobe first
                wardrobe_match = self._find_in_wardrobe(item)
                if wardrobe_match:
                    items_with_images.append({
                        'item': item,
                        'image_path': wardrobe_match['image_path'],
                        'source': 'wardrobe',
                        'details': wardrobe_match
                    })
                    continue
                
                # Try dataset
                dataset_match = self._find_in_dataset(item)
                if dataset_match:
                    items_with_images.append({
                        'item': item,
                        'image_path': dataset_match['image_path'],
                        'source': 'dataset',
                        'details': dataset_match
                    })
                    continue
                
                # Placeholder
                items_with_images.append({
                    'item': item,
                    'image_path': '/static/placeholder/150/200',
                    'source': 'placeholder',
                    'details': {'description': f"{item['color']} {item['type']}"}
                })
            
            outfits_with_images.append({
                **outfit,
                'items_with_images': items_with_images
            })
        
        return outfits_with_images
    
    def _find_in_wardrobe(self, item: Dict) -> Optional[Dict]:
        """Find item in wardrobe folder"""
        try:
            if not os.path.exists(self.wardrobe_folder):
                return None
            
            files = [f for f in os.listdir(self.wardrobe_folder) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            
            color_lower = item['color'].lower() if item['color'] != 'Any' else ''
            type_lower = item['type'].lower()
            
            # Look for matching filename
            for filename in files:
                filename_lower = filename.lower()
                
                # Check if both color and type are in filename
                color_match = not color_lower or color_lower in filename_lower
                type_match = any(word in filename_lower for word in [type_lower, type_lower.replace('s', '')])
                
                if color_match and type_match:
                    return {
                        'image_path': f"/static/wardrobe/{filename}",
                        'description': f"{item['color']} {item['type']}",
                        'filename': filename
                    }
            
            return None
        except Exception as e:
            print(f"Error in wardrobe search: {e}")
            return None
    
    def _find_in_dataset(self, item: Dict) -> Optional[Dict]:
        """Find item in dataset"""
        if self.dataset_df is None:
            return None
        
        try:
            df = self.dataset_df.copy()
            
            # Filter by article type
            if item['type']:
                df = df[df['articleType'].str.contains(item['type'], case=False, na=False)]
                print(f"DEBUG: After article filter ({item['type']}): {len(df)} items")
            
            # Filter by color if specified
            if item['color'] and item['color'] != 'Any':
                color_df = df[df['baseColour'].str.contains(item['color'], case=False, na=False)]
                if not color_df.empty:
                    df = color_df
                    print(f"DEBUG: After color filter ({item['color']}): {len(df)} items")
            
            if not df.empty:
                # Pick random item
                selected = df.sample(n=1).iloc[0]
                item_id = str(selected['id'])
                
                # Check if image exists
                image_path = os.path.join(self.dataset_images_path, f"{item_id}.jpg")
                if os.path.exists(image_path):
                    print(f"DEBUG: Found dataset item {item_id}")
                    return {
                        'image_path': f"/dataset/images/{item_id}.jpg",
                        'description': f"{selected['baseColour']} {selected['articleType']}",
                        'id': item_id,
                        'details': selected.to_dict()
                    }
            
            return None
            
        except Exception as e:
            print(f"Error in dataset search: {e}")
            return None
    
    def get_outfit_visualization_data(self, ai_response: str) -> Dict:
        """Main method - simple and direct"""
        try:
            print("DEBUG: Starting outfit visualization")
            
            # Extract items
            outfit_items = self.extract_outfit_items(ai_response)
            if not outfit_items:
                return {'outfits': [], 'has_outfits': False}
            
            # Find images
            outfits_with_images = self.find_matching_items(outfit_items)
            
            # Filter valid outfits
            valid_outfits = [o for o in outfits_with_images if o['items_with_images']]
            
            return {
                'outfits': valid_outfits,
                'has_outfits': len(valid_outfits) > 0,
                'total_outfits': len(valid_outfits)
            }
            
        except Exception as e:
            print(f"ERROR in outfit visualization: {e}")
            import traceback
            traceback.print_exc()
            return {'outfits': [], 'has_outfits': False}