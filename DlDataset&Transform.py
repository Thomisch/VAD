import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# 📌 Chemins d'entrée et sortie
LIBRISPEECH_PATH = "LibriSpeech/train-clean-100/"  # Dossier contenant les fichiers FLAC
OUTPUT_PATH = "processed_data/"  # Dossier des fichiers traités
VOICE_PATH = os.path.join(OUTPUT_PATH, "voix")
SILENCE_PATH = os.path.join(OUTPUT_PATH, "silence")

# 📌 Paramètres audio
TARGET_SR = 16000  # Fréquence cible 16kHz
MIN_PHRASE_DURATION = 2.0  # Durée minimale d'une phrase (sec)
MAX_PHRASE_DURATION = 10.0  # Durée max d'une phrase (sec)
SILENCE_THRESHOLD = 20  # Seuil en dB pour détecter le silence
PAUSE_DURATION = 0.5  # Durée min d'un silence pour marquer la fin d'une phrase (sec)

# ✅ Création des dossiers de sortie
os.makedirs(VOICE_PATH, exist_ok=True)
os.makedirs(SILENCE_PATH, exist_ok=True)

def extract_phrases(file_path):
    """Détecte les phrases en fonction du silence et extrait des fichiers audio."""
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)  # Chargement audio
    
    # Détection des silences (retourne les intervalles de voix détectées)
    intervals = librosa.effects.split(y, top_db=SILENCE_THRESHOLD)

    phrases = []
    for start, end in intervals:
        duration = (end - start) / sr  # Convertir en secondes
        if MIN_PHRASE_DURATION <= duration <= MAX_PHRASE_DURATION:
            phrases.append(y[start:end])

    return phrases

# 🔄 Traitement des fichiers
flac_files = [os.path.join(root, f) for root, _, files in os.walk(LIBRISPEECH_PATH) for f in files if f.endswith(".flac")]

for file_path in tqdm(flac_files[:200]):  # Limite à 200 fichiers pour éviter surcharge
    phrases = extract_phrases(file_path)

    for i, phrase in enumerate(phrases):
        # Sauvegarde les extraits de phrase détectés
        sf.write(f"{VOICE_PATH}/phrase_{os.path.basename(file_path).replace('.flac', '')}_{i}.wav", phrase, TARGET_SR)

print("✅ Extraction des phrases terminée !")
