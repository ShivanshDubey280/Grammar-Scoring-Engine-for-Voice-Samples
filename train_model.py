import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from utils.feature_extraction import extract_audio_features, extract_text_features, combine_features
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


# Load data directly using full path
df = pd.read_csv("./data/train_metadata.csv")

X, y = [], []

for _, row in df.iterrows():
    audio_filepath = os.path.join("./data/audio", row['filename'])
    audio_feat = extract_audio_features(audio_filepath)

    # Dummy transcription for now
    text_feat = extract_text_features("This is a sample transcription of the audio.")

    features = combine_features(audio_feat, text_feat)
    X.append(features)
    y.append(row['score'])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model to full path
os.makedirs(r"C:\Users\shiva\OneDrive\Desktop\Project\models", exist_ok=True)
with open(r"C:\Users\shiva\OneDrive\Desktop\Project\models\grammar_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to C:\\Users\\shiva\\OneDrive\\Desktop\\Project\\models\\grammar_model.pkl")
