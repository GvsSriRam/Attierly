import os
import cv2
import numpy as np
from gradio_client import Client, file
from typing import Dict, List, Optional
import tempfile
import requests
from PIL import Image
import base64
import io

class VirtualTryOn:
    """Virtual Try-On using IDM-VTON"""
    
    def __init__(self):
        self.client = None
        self.temp_dir = tempfile.mkdtemp()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the IDM-VTON client"""
        try:
            self.client = Client("yisol/IDM-VTON")
            print("IDM-VTON client initialized successfully")
        except Exception as e:
            print(f"Error initializing IDM-VTON client: {e}")
            self.client = None
    
    def process_user_image(self, user_image_path: str) -> str:
        """Process and prepare user image for try-on"""
        try:
            # Load and resize image if needed
            img = cv2.imread(user_image_path)
            if img is None:
                raise ValueError("Could not load user image")
            
            # Resize to standard dimensions for better try-on results
            height, width = img.shape[:2]
            
            # IDM-VTON works best with 768x1024 or similar aspect ratios
            target_height = 1024
            target_width = int(width * (target_height / height))
            
            if target_width > 768:
                target_width = 768
                target_height = int(height * (target_width / width))
            
            img_resized = cv2.resize(img, (target_width, target_height))
            
            # Save processed image
            processed_path = os.path.join(self.temp_dir, "processed_user.png")
            cv2.imwrite(processed_path, img_resized)
            
            return processed_path
            
        except Exception as e:
            print(f"Error processing user image: {e}")
            return user_image_path
    
    def download_garment_image(self, image_url: str, item_info: Dict) -> Optional[str]:
        """Download garment image from URL"""
        try:
            # Handle different URL formats
            if image_url.startswith('/static/'):
                # Local file
                local_path = image_url.replace('/static/', 'static/')
                if os.path.exists(local_path):
                    return local_path
            elif image_url.startswith('/dataset/'):
                # Dataset file
                dataset_path = image_url.replace('/dataset/', '')
                full_path = os.path.join('/Users/gvssriram/Desktop/projects-internship/Flair_POC/fashion-dataset/', dataset_path)
                if os.path.exists(full_path):
                    return full_path
            elif image_url.startswith('http'):
                # Remote URL
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    temp_path = os.path.join(self.temp_dir, f"garment_{hash(image_url)}.jpg")
                    with open(temp_path, 'wb') as f:
                        f.write(response.content)
                    return temp_path
            
            return None
            
        except Exception as e:
            print(f"Error downloading garment image {image_url}: {e}")
            return None
    
    def prepare_garment_image(self, garment_path: str, item_info: Dict) -> str:
        """Prepare garment image for try-on"""
        try:
            img = cv2.imread(garment_path)
            if img is None:
                raise ValueError("Could not load garment image")
            
            # Resize garment image to reasonable size
            height, width = img.shape[:2]
            max_size = 512
            
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                else:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                
                img = cv2.resize(img, (new_width, new_height))
            
            # Save prepared garment
            prepared_path = os.path.join(self.temp_dir, f"garment_{item_info.get('type', 'item')}.png")
            cv2.imwrite(prepared_path, img)
            
            return prepared_path
            
        except Exception as e:
            print(f"Error preparing garment image: {e}")
            return garment_path
    
    def try_on_single_item(self, user_image_path: str, garment_path: str, item_info: Dict) -> Optional[str]:
        """Try on a single garment item"""
        if not self.client:
            print("IDM-VTON client not available")
            return None
        
        try:
            # Prepare images
            processed_user = self.process_user_image(user_image_path)
            prepared_garment = self.prepare_garment_image(garment_path, item_info)
            
            # Create garment description
            garment_description = f"{item_info.get('color', '')} {item_info.get('type', 'clothing item')}".strip()
            
            print(f"Trying on: {garment_description}")
            
            # Call IDM-VTON API
            result = self.client.predict(
                dict={"background": file(processed_user), "layers": [], "composite": None},
                garm_img=file(prepared_garment),
                garment_des=garment_description,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            
            if result and len(result) > 0 and result[0]:
                # Expand result to original aspect ratio
                result_image = self.expand_to_original_aspect_ratio(result[0], user_image_path)
                
                # Save result
                result_path = os.path.join(self.temp_dir, f"tryon_result_{hash(garment_path)}.png")
                cv2.imwrite(result_path, result_image)
                
                return result_path
            
            return None
            
        except Exception as e:
            print(f"Error in try-on process: {e}")
            return None
    
    def try_on_outfit(self, user_image_path: str, outfit_items: List[Dict], selected_items: List[int] = None) -> Optional[str]:
        """Try on multiple items (layered approach)"""
        if not outfit_items:
            return None
        
        # If no selection specified, use all items
        if selected_items is None:
            selected_items = list(range(len(outfit_items)))
        
        current_result = user_image_path
        
        # Sort items by try-on priority (bottoms first, then tops, then outerwear)
        priority_order = {'bottom': 1, 'dress': 1, 'top': 2, 'outerwear': 3, 'footwear': 4, 'accessories': 5}
        
        selected_outfit_items = [outfit_items[i] for i in selected_items if i < len(outfit_items)]
        selected_outfit_items.sort(key=lambda x: priority_order.get(x['item'].get('category', 'other'), 999))
        
        for item_data in selected_outfit_items:
            item_info = item_data['item']
            image_path = item_data['image_path']
            
            # Skip placeholder images
            if image_path == '/static/placeholder/150/200':
                continue
            
            # Download/prepare garment
            garment_path = self.download_garment_image(image_path, item_info)
            if not garment_path:
                continue
            
            # Try on this item
            try_on_result = self.try_on_single_item(current_result, garment_path, item_info)
            if try_on_result:
                current_result = try_on_result
                print(f"Successfully tried on {item_info.get('type', 'item')}")
            else:
                print(f"Failed to try on {item_info.get('type', 'item')}")
        
        return current_result if current_result != user_image_path else None
    
    def expand_to_original_aspect_ratio(self, result_path: str, original_path: str) -> np.ndarray:
        """Expand result image to match original aspect ratio"""
        try:
            result_img = cv2.imread(result_path)
            original_img = cv2.imread(original_path)
            
            if result_img is None or original_img is None:
                return result_img if result_img is not None else original_img
            
            original_h, original_w = original_img.shape[:2]
            result_img_resized = cv2.resize(result_img, (original_w, original_h))
            
            return result_img_resized
            
        except Exception as e:
            print(f"Error expanding image: {e}")
            return cv2.imread(result_path) if os.path.exists(result_path) else None
    
    def image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 for web display"""
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.temp_dir = tempfile.mkdtemp()
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

# Global instance
virtual_tryon = VirtualTryOn()