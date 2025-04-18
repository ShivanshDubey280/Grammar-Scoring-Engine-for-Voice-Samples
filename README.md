# Grammar Scoring Engine for Voice Samples 🎙️

This project is a **Grammar Scoring Engine** designed to evaluate spoken audio samples and assign a grammar score on a scale of 1 to 10. It utilizes basic audio processing and natural language processing (NLP) techniques to extract features and apply a machine learning model to predict the grammar quality of the given audio sample.

---

## ⚙️ Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/grammar-scoring-engine.git
    cd grammar-scoring-engine
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare data**:
    - Place your audio files in `data/audio/`.
    - Create a `train_metadata.csv` file inside the `data/` folder with the following format:
      ```
      filename,score
      sample1.wav,7
      sample2.wav,5
      ...
      ```

4. **Train the model**:
    ```bash
    python train_model.py
    ```

5. **Predict grammar score for new audio**:
    ```bash
    python predict.py sample1.wav
    ```

---

## 🧠 How It Works

- **Feature Extraction**:
  - Extracts basic **audio features** like duration and loudness using `librosa`.
  - Extracts **textual features** from a dummy or transcribed sentence using NLTK (e.g., POS tag ratio).
  - Combines audio and text features into a final feature vector.

- **Model Training**:
  - Uses a simple **Logistic Regression** model (can be upgraded to any other ML algorithm).
  - Trained on the CSV dataset that maps audio filenames to grammar scores.

- **Prediction**:
  - For a given `.wav` file, features are extracted and passed to the trained model to output a score between 1 and 10.

---

## 🗒️ Notes

- You’ll need to **provide or generate accurate transcriptions** for better grammar scoring accuracy.
- You can upgrade the model or feature set to improve performance.
- All code is modular and easy to extend.

---

## 💡 Future Enhancements

- Integrate speech-to-text (e.g., Google Speech API or Whisper) for real transcriptions.
- Use deep learning (e.g., CNN, RNN) for better audio understanding.
- Create a web interface using Flask or Streamlit.

---

## 👤 Author
<a href="https://www.github.com/shivanshdubey280"><src="ShivanshDubey"></a>
