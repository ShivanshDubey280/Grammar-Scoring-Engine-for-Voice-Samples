import sys
import os
import pickle
from utils.feature_extraction import extract_audio_features, extract_text_features, combine_features

# Hardcoded paths
model_path = r"C:\Users\shiva\OneDrive\Desktop\Project\models\grammar_model.pkl"
audio_dir = r"C:\Users\shiva\OneDrive\Desktop\Project\data\audio"

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Get audio file name from command line
filename = sys.argv[1]
filepath = os.path.join(audio_dir, filename)

# Extract features
audio_feat = extract_audio_features(filepath)
text_feat = extract_text_features("This is a sample transcription of the audio.")  # Dummy text
features = combine_features(audio_feat, text_feat)

# Predict
pred = model.predict([features])[0]
print(f"Predicted grammar score for {filename}: {pred}")