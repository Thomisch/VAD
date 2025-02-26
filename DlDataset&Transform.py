import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
import gdown
import tarfile

# 📌 Chemins d'entrée et sortie
LIBRISPEECH_PATH = "LibriSpeech/train-clean-100/"  # Dossier contenant les fichiers FLAC
WORKING_PATH = "temp_wav/"  # Dossier temporaire pour les fichiers WAV
OUTPUT_PATH = "processed_data/"  # Dossier des fichiers extraits
VOICE_PATH = os.path.join(OUTPUT_PATH, "voix")

# 📌 Paramètres audio
TARGET_SR = 16000  # Fréquence cible 16kHz
MIN_PHRASE_DURATION = 2.0  # Durée minimale d'une phrase (sec)
MAX_PHRASE_DURATION = 10.0  # Durée max d'une phrase (sec)
SILENCE_THRESHOLD = 20  # Seuil en dB pour détecter le silence

# ✅ Création des dossiers
os.makedirs(WORKING_PATH, exist_ok=True)  # Dossier temporaire pour WAV
os.makedirs(VOICE_PATH, exist_ok=True)  # Dossier final pour les phrases

def download_librispeech():
    """Télécharge et extrait le dataset LibriSpeech train-clean-100"""
    url = 'https://drive.google.com/uc?id=1g0EdL4AFKMkkFIt1RE1V_1lmAqDizjr4'  # Lien vers le dataset train-clean-100
    output = 'librispeech_train_clean_100.tar.gz'  # Fichier compressé
    if not os.path.exists(LIBRISPEECH_PATH):  # Si le dataset n'est pas déjà téléchargé
        print("Téléchargement de LibriSpeech train-clean-100...")
        gdown.download(url, output, quiet=False)
        
        # Décompression du fichier téléchargé
        print("Décompression du dataset...")
        with tarfile.open(output, "r:gz") as tar:
            tar.extractall(path="LibriSpeech")
        
        # Supprimer l'archive une fois décompressée
        os.remove(output)
    else:
        print("Le dataset est déjà présent.")

def convert_flac_to_wav(flac_path, wav_path):
    """Convertit un fichier FLAC en WAV 16kHz mono"""
    y, sr = librosa.load(flac_path, sr=TARGET_SR, mono=True)
    sf.write(wav_path, y, TARGET_SR)

def extract_phrases(wav_path):
    """Détecte et extrait des phrases basées sur le silence"""
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    intervals = librosa.effects.split(y, top_db=SILENCE_THRESHOLD)
    
    phrases = []
    for start, end in intervals:
        duration = (end - start) / sr
        if MIN_PHRASE_DURATION <= duration <= MAX_PHRASE_DURATION:
            phrases.append(y[start:end])
    
    return phrases

# 📥 Télécharger le dataset LibriSpeech si nécessaire
download_librispeech()

# 🔄 Sélection de 200 fichiers FLAC
flac_files = [os.path.join(root, f) for root, _, files in os.walk(LIBRISPEECH_PATH) for f in files if f.endswith(".flac")]
flac_files = flac_files[:200]  # Limiter à 200 fichiers pour la gestion de l'espace

for flac_file in tqdm(flac_files, desc="Processing Audio"):
    # Étape 1 : Convertir FLAC → WAV temporaire
    wav_temp = os.path.join(WORKING_PATH, os.path.basename(flac_file).replace(".flac", ".wav"))
    convert_flac_to_wav(flac_file, wav_temp)

    # Étape 2 : Extraire les phrases et les sauvegarder
    phrases = extract_phrases(wav_temp)
    for i, phrase in enumerate(phrases):
        phrase_path = os.path.join(VOICE_PATH, f"phrase_{os.path.basename(flac_file).replace('.flac', '')}_{i}.wav")
        sf.write(phrase_path, phrase, TARGET_SR)

    # Étape 3 : Supprimer le fichier WAV temporaire pour économiser l’espace
    os.remove(wav_temp)

# 🔥 Suppression du dossier temporaire (sécurité)
shutil.rmtree(WORKING_PATH, ignore_errors=True)

print("✅ Extraction terminée ! Les fichiers sont dans 'processed_data/voix/'")
