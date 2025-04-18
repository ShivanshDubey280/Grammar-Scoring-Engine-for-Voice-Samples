#utils/feature_extraction.py
import librosa
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_audio_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def extract_text_features(transcription):
    tokens = nltk.word_tokenize(transcription)
    pos_tags = nltk.pos_tag(tokens)
    num_nouns = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    num_verbs = sum(1 for word, tag in pos_tags if tag.startswith('VB'))
    return np.array([len(tokens), num_nouns, num_verbs])

def combine_features(audio_features, text_features):
    return np.concatenate((audio_features, text_features))