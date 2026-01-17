# -*- coding: utf-8 -*-
"""
TEST RAPIDE - Vérification de l'installation
Compatible Python 3.13 + TensorFlow 2.20
Lancez ce script pour vérifier que tout est bien installé
"""

import sys

print("="*60)
print("TEST D'INSTALLATION - RECONNAISSANCE LANGUE DES SIGNES")
print("="*60)
print()

# Test 1: Python version
print("[TEST 1/7] Version de Python...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major == 3 and python_version.minor >= 8:
    print("   ✓ Version Python OK")
    if python_version.minor >= 13:
        print("   ℹ️  Python 3.13 détecté - Utilise TensorFlow 2.20+")
else:
    print("   ⚠️ Attention: Python 3.8+ recommandé")
print()

# Test 2: TensorFlow
print("[TEST 2/7] TensorFlow...")
try:
    import tensorflow as tf
    print(f"   Version: {tf.__version__}")
    print("   ✓ TensorFlow installé")
except ImportError:
    print("   ✗ TensorFlow non installé!")
    if python_version.minor >= 13:
        print("   → pip install tensorflow==2.20.0")
    else:
        print("   → pip install tensorflow==2.12.0")
print()

# Test 3: Keras
print("[TEST 3/7] Keras...")
try:
    import keras
    print(f"   Version: {keras.__version__}")
    print("   ✓ Keras installé")
except ImportError:
    try:
        from tensorflow import keras
        print(f"   Version: {keras.__version__} (via TensorFlow)")
        print("   ✓ Keras disponible")
    except ImportError:
        print("   ✗ Keras non installé!")
        if python_version.minor >= 13:
            print("   → pip install keras==3.8.0")
        else:
            print("   → pip install keras==2.12.0")
print()

# Test 4: OpenCV
print("[TEST 4/7] OpenCV...")
try:
    import cv2
    print(f"   Version: {cv2.__version__}")
    print("   ✓ OpenCV installé")
except ImportError:
    print("   ✗ OpenCV non installé!")
    print("   → pip install opencv-python")
print()

# Test 5: NumPy
print("[TEST 5/7] NumPy...")
try:
    import numpy as np
    print(f"   Version: {np.__version__}")
    print("   ✓ NumPy installé")
except ImportError:
    print("   ✗ NumPy non installé!")
    print("   → pip install numpy")
print()

# Test 6: Fichiers du modèle
print("[TEST 6/7] Fichiers du modèle...")
import os

files_needed = ['model-bw.json', 'model-bw.h5']
all_present = True

for file in files_needed:
    if os.path.exists(file):
        print(f"   ✓ {file} trouvé")
    else:
        print(f"   ✗ {file} manquant!")
        all_present = False

if not all_present:
    print("   → Assurez-vous que les fichiers du modèle sont dans ce dossier")
print()

# Test 7: Caméra
print("[TEST 7/7] Test de la caméra...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   ✓ Caméra accessible")
        ret, frame = cap.read()
        if ret:
            print("   ✓ Capture d'image fonctionnelle")
        cap.release()
    else:
        print("   ⚠️ Impossible d'accéder à la caméra")
        print("   → Vérifiez que votre webcam est connectée")
        print("   → Fermez Zoom, Teams ou autres applications utilisant la webcam")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
print()

# Test du modèle
print("[TEST BONUS] Chargement du modèle...")
try:
    # Import compatible avec toutes les versions
    try:
        from tensorflow import keras
        from tensorflow.keras.models import model_from_json
    except ImportError:
        from keras.models import model_from_json
    
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights("model-bw.h5")
    print("   ✓ Modèle chargé avec succès!")
    print()
    
    # Afficher l'architecture
    print("Architecture du modèle:")
    print("-" * 60)
    try:
        loaded_model.summary()
    except:
        print("   (Résumé du modèle non disponible)")
    
except FileNotFoundError:
    print("   ✗ Fichiers du modèle non trouvés!")
    print("   → Vérifiez que model-bw.json et model-bw.h5 sont dans ce dossier")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
print()

print("="*60)
print("RÉSUMÉ DU TEST")
print("="*60)
print()
print("Si tous les tests sont ✓, vous êtes prêt pour la démo!")
print("Si certains tests ont échoué (✗), installez les bibliothèques manquantes.")
print()

# Recommandation basée sur la version de Python
if python_version.minor >= 13:
    print("VOTRE SYSTÈME: Python 3.13+")
    print("→ Utilisez: python demo_signes_updated.py")
    print()
    print("Installation recommandée:")
    print("pip install tensorflow==2.20.0 keras==3.8.0 opencv-python numpy")
else:
    print("VOTRE SYSTÈME: Python 3.8-3.12")
    print("→ Utilisez: python demo_signes.py")
    print()
    print("Installation recommandée:")
    print("pip install tensorflow==2.12.0 keras==2.12.0 opencv-python numpy")

print("="*60)
