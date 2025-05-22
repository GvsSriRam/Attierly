import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import math

# --- Configuration ---
DATASET_ROOT = '/Users/gvssriram/Desktop/projects-internship/Flair_POC/fashion-dataset'
# Define all columns we want to train models for
ALL_LABEL_COLUMNS = ['articleType', 'baseColour', 'season', 'usage']

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32 # Reduced batch size slightly, adjust based on memory
NUM_EPOCHS_FINE_TUNE = 2
LEARNING_RATE = 0.0001
FINE_TUNE_LR_MULTIPLIER = 0.1
FINE_TUNE_AT_LAYER = 100
VALIDATION_SPLIT = 0.2

# --- Custom Data Sequence Class (Modified for Single Label) ---
class MyntraSequence(Sequence):
    # Now takes a single label_column string
    def __init__(self, dataset_root, label_column, batch_size, target_size=(224, 224), subset='train', validation_split=0.2):
        self.dataset_root = dataset_root
        self.label_column = label_column # Store the single label column name
        self.batch_size = batch_size
        self.target_size = target_size
        self.subset = subset.lower()

        self.images_dir = os.path.join(dataset_root, 'images')
        self.styles_path = os.path.join(dataset_root, 'styles.csv')

        # Verify paths
        if not os.path.isdir(self.dataset_root): raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")
        if not os.path.isdir(self.images_dir): raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isfile(self.styles_path): raise FileNotFoundError(f"Styles CSV file not found: {self.styles_path}")

        # Load and prepare metadata for the specific label column
        self.dataframe, self.label_mapping, self.reverse_label_mapping, self.num_classes = self._load_and_prepare_metadata(validation_split)

        self.image_ids = self.dataframe.index.tolist()

        if not self.image_ids:
             raise ValueError(f"Sequence for subset '{self.subset}' and label '{self.label_column}' resulted in zero valid images/labels.")

        print(f"Initialized sequence for '{self.label_column}' - subset '{self.subset}' with {len(self.image_ids)} samples.")
        self.on_epoch_end()

    def _load_and_prepare_metadata(self, validation_split):
        try:
            df = pd.read_csv(self.styles_path, on_bad_lines='skip') # Handle potential bad lines
        except Exception as e:
            raise IOError(f"Error reading styles CSV '{self.styles_path}': {e}")

        required_cols = ['id', self.label_column]
        if not all(col in df.columns for col in required_cols):
             missing_cols = [col for col in required_cols if col not in df.columns]
             raise ValueError(f"Missing required columns in styles.csv for label '{self.label_column}'. Need: {required_cols}. Missing: {missing_cols}")

        df['id'] = df['id'].astype(str)
        df = df.dropna(subset=[self.label_column]) # Drop NA only for the current label column
        print(f"Rows for '{self.label_column}' after dropping NA: {len(df)}")
        if len(df) == 0: raise ValueError(f"No data for '{self.label_column}' after dropping NA.")

        # Add image path and check existence
        df['image_path'] = df['id'].apply(lambda x: os.path.join(self.images_dir, x + '.jpg'))
        df = df[df['image_path'].apply(os.path.exists)]
        print(f"Rows for '{self.label_column}' after checking image existence: {len(df)}")
        if len(df) == 0: raise ValueError(f"No existing images found for label '{self.label_column}'.")

        df = df.set_index('id')

        # Create numerical label mapping for the SINGLE column
        unique_labels = sorted(df[self.label_column].unique())
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        reverse_label_mapping = {i: label for label, i in label_mapping.items()}
        num_classes = len(unique_labels)
        print(f"Found {num_classes} unique labels in column '{self.label_column}'.")

        df['label_index'] = df[self.label_column].map(label_mapping) # Single label index column

        # Split data
        ids = df.index
        if len(ids) < 2: raise ValueError("Not enough data points to perform train/validation split.")
        stratify_col = 'label_index'
        try:
            train_ids, val_ids = train_test_split(ids, test_size=validation_split, random_state=42, stratify=df[stratify_col])
        except ValueError as e:
             print(f"Warning: Could not stratify split on '{stratify_col}' for '{self.label_column}' ({e}). Splitting without stratification.")
             train_ids, val_ids = train_test_split(ids, test_size=validation_split, random_state=42)

        if self.subset == 'train':
            df_subset = df.loc[train_ids]
        elif self.subset == 'validation':
            df_subset = df.loc[val_ids]
        else:
            raise ValueError("subset must be 'train' or 'validation'")

        return df_subset, label_mapping, reverse_label_mapping, num_classes

    def __len__(self):
        return math.ceil(len(self.image_ids) / self.batch_size)

    def __getitem__(self, index):
        if not self.image_ids: return np.array([]), np.array([])

        batch_start_index = index * self.batch_size
        batch_end_index = min((index + 1) * self.batch_size, len(self.image_ids))
        batch_ids = self.image_ids[batch_start_index:batch_end_index]

        batch_x = []
        batch_y = [] # Now a simple list

        batch_df = self.dataframe.loc[batch_ids]

        for image_id, row in batch_df.iterrows():
            full_img_path = row['image_path']
            try:
                img = load_img(full_img_path, target_size=self.target_size, color_mode='rgb')
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                batch_x.append(img_array)

                label_index = row['label_index'] # Get the single label index
                label = to_categorical(label_index, num_classes=self.num_classes)
                batch_y.append(label)

            except Exception as e:
                print(f"Warning: Error loading/processing image {full_img_path} (ID: {image_id}) for label '{self.label_column}': {e}")
                if batch_x: batch_x.pop() # Remove corresponding x if y fails
                continue

        if not batch_x: return np.array([]), np.array([])

        return np.array(batch_x), np.array(batch_y) # Return simple numpy arrays

    def on_epoch_end(self):
        # Optional: Shuffle data between epochs
        # np.random.shuffle(self.image_ids)
        pass

    # Methods to get mappings for the single label
    def get_label_mapping(self): return self.label_mapping
    def get_reverse_label_mapping(self): return self.reverse_label_mapping
    def get_num_classes(self): return self.num_classes

# --- Main Training Loop ---
if __name__ == '__main__':
    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Training separate models for: {ALL_LABEL_COLUMNS}")

    # Loop through each label column to train a model
    for current_label_col in ALL_LABEL_COLUMNS:
        print(f"\n{'='*20} Training Model for: {current_label_col} {'='*20}")

        # --- Instantiate Data Sequences for the current label ---
        print("\nCreating data sequences...")
        try:
            train_sequence = MyntraSequence(DATASET_ROOT, current_label_col, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), 'train', VALIDATION_SPLIT)
            validation_sequence = MyntraSequence(DATASET_ROOT, current_label_col, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), 'validation', VALIDATION_SPLIT)

            num_classes = train_sequence.get_num_classes()
            reverse_label_mapping = train_sequence.get_reverse_label_mapping()

            if len(train_sequence) == 0 or len(validation_sequence) == 0:
                 raise ValueError("FATAL: One or both data sequences are empty after initialization.")

        except Exception as e:
            print(f"FATAL ERROR creating data sequences for '{current_label_col}': {e}")
            print(f"Skipping training for '{current_label_col}'.")
            continue # Skip to the next label column

        # --- Build Single-Output Model ---
        print("\nBuilding single-output model...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        base_model.trainable = False # Freeze base model initially

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='shared_dense')(x)
        # Single output layer - name it simply 'output' or specific like 'output_...'
        output_layer = Dense(num_classes, activation='softmax', name=f'output_{current_label_col}')(x)

        model = Model(inputs=base_model.input, outputs=output_layer)

        # --- Compile for initial training ---
        # Simple compile for single output
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("\n--- Model Summary (Head Frozen) ---")
        model.summary(line_length=120)

        # --- Fine-tuning Phase ---
        print("\n--- Setting up for Fine-tuning ---")
        base_model.trainable = True
        if FINE_TUNE_AT_LAYER > 0:
            for layer in base_model.layers[:FINE_TUNE_AT_LAYER]: layer.trainable = False
            print(f"Fine-tuning from layer {FINE_TUNE_AT_LAYER} onwards.")
        else:
            print("Fine-tuning all layers of the base model.")

        # Re-compile with lower learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * FINE_TUNE_LR_MULTIPLIER),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("\n--- Model Summary (Fine-tuning Enabled) ---")
        model.summary(line_length=120)

        # --- Start Training ---
        print(f"\n--- Starting Training ({NUM_EPOCHS_FINE_TUNE} epochs) for '{current_label_col}' ---")
        history = model.fit(
            train_sequence,
            validation_data=validation_sequence,
            epochs=NUM_EPOCHS_FINE_TUNE
        )

        # --- Save Model and Label Mapping ---
        # Ensure saved_models directory exists
        os.makedirs('saved_models', exist_ok=True)
        
        # Save with names indicating the attribute in saved_models directory
        model_save_path = os.path.join('saved_models', f'model_{current_label_col}.keras')
        label_map_save_path = os.path.join('saved_models', f'map_{current_label_col}.npy')

        print(f"\nSaving model for '{current_label_col}' to {model_save_path}...")
        model.save(model_save_path)

        print(f"Saving label mapping for '{current_label_col}' to {label_map_save_path}...")
        np.save(label_map_save_path, reverse_label_mapping)

        print(f"\nTraining complete for '{current_label_col}'. Artifacts saved.")
        # Optional: Clear session if memory is an issue between loops
        # tf.keras.backend.clear_session()

    print(f"\n{'='*20} All Training Finished {'='*20}")