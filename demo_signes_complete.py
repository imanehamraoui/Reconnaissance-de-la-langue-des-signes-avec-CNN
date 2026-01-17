# -*- coding: utf-8 -*-
"""
Reconnaissance de la langue des signes - Version Complète avec Actions
Compatible Python 3.13 + TensorFlow 2.20 + Keras 3.x
Reconstruit le modèle manuellement pour compatibilité totale
"""

import numpy as np
import cv2
import operator
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("="*60)
print("SYSTÈME DE RECONNAISSANCE DE LA LANGUE DES SIGNES")
print("="*60)

# Reconstruction manuelle du modèle
print("\n[1/3] Reconstruction du modèle...")
try:
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(29, activation='softmax')
    ])
    
    print("✓ Architecture du modèle créée")
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Modèle compilé")
    model.load_weights("model-bw.h5")
    print("✓ Poids chargés avec succès!")
    print(f"  Architecture: {len(model.layers)} couches")
    
    loaded_model = model
    
except FileNotFoundError:
    print("✗ Erreur: Le fichier model-bw.h5 n'a pas été trouvé!")
    print("\nAssurez-vous que model-bw.h5 est dans le même dossier que ce script.")
    input("\nAppuyez sur Entrée pour quitter...")
    exit(1)
except Exception as e:
    print(f"✗ Erreur lors du chargement: {e}")
    input("\nAppuyez sur Entrée pour quitter...")
    exit(1)

# Dictionnaires de catégories
categorie_nombres = {
    0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 
    5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE'
}

categorie_alphabet = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M'
}

categorie_actions = {
    0: 'Creer Dossier',
    1: 'Coppier Fichier', 
    2: 'Jeu',
    3: 'Ecrire Fichier'
}

print("\n[2/3] Initialisation de la caméra...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Erreur: Impossible d'accéder à la caméra")
    print("\nVérifiez que:")
    print("  - Votre webcam est connectée")
    print("  - Aucune autre application n'utilise la webcam (Zoom, Teams...)")
    input("\nAppuyez sur Entrée pour quitter...")
    exit(1)

print("✓ Caméra initialisée!")

print("\n[3/3] Démarrage de la détection...")
print("\nCONSIGNES:")
print("- Placez votre main dans le cadre bleu")
print("- Appuyez sur '1' pour mode NOMBRES (0-9)")
print("- Appuyez sur '2' pour mode ALPHABET (A-M)")
print("- Appuyez sur '3' pour mode ACTIONS (gestes spéciaux)")
print("- Appuyez sur 'ESC' pour quitter")
print("\n" + "="*60 + "\n")

mode = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la caméra")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Définir la région d'intérêt (ROI)
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])
        
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 2)
        
        roi = frame[y1:y2, x1:x2]
        
        # Prétraitement de l'image
        roi_processed = cv2.resize(roi, (64, 64))
        roi_processed = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2GRAY)
        _, roi_processed = cv2.threshold(roi_processed, 120, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Image traitee", roi_processed)
        
        roi_normalized = roi_processed.astype('float32') / 255.0
        
        # Prédiction
        result = loaded_model.predict(roi_normalized.reshape(1, 64, 64, 1), verbose=0)
        
        # Affichage selon le mode
        if mode == "1":
            # Mode NOMBRES
            prediction_nombres = {
                'ZERO': result[0][0], 'ONE': result[0][1], 'TWO': result[0][2],
                'THREE': result[0][3], 'FOUR': result[0][4], 'FIVE': result[0][5],
                'SIX': result[0][6], 'SEVEN': result[0][7], 'EIGHT': result[0][8],
                'NINE': result[0][9]
            }
            prediction_sorted = sorted(prediction_nombres.items(), 
                                       key=operator.itemgetter(1), reverse=True)
            
            detected_sign = prediction_sorted[0][0]
            confidence = prediction_sorted[0][1] * 100
            
            cv2.putText(frame, "MODE: NOMBRES (0-9)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Signe detecte: {detected_sign}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Confiance: {confidence:.1f}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif mode == "2":
            # Mode ALPHABET
            prediction_alphabet = {
                'A': result[0][0], 'B': result[0][1], 'C': result[0][2],
                'D': result[0][3], 'E': result[0][4], 'F': result[0][5],
                'G': result[0][6], 'H': result[0][7], 'I': result[0][8],
                'J': result[0][9], 'K': result[0][10], 'L': result[0][11],
                'M': result[0][12]
            }
            prediction_sorted = sorted(prediction_alphabet.items(), 
                                       key=operator.itemgetter(1), reverse=True)
            
            detected_sign = prediction_sorted[0][0]
            confidence = prediction_sorted[0][1] * 100
            
            cv2.putText(frame, "MODE: ALPHABET (A-M)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Lettre detectee: {detected_sign}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Confiance: {confidence:.1f}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif mode == "3":
            # Mode ACTIONS
            prediction_actions = {
                'Creer Dossier': result[0][13],
                'Coppier Fichier': result[0][14],
                'Jeu': result[0][15],
                'Ecrire Fichier': result[0][16]
            }
            prediction_sorted = sorted(prediction_actions.items(), 
                                       key=operator.itemgetter(1), reverse=True)
            
            detected_sign = prediction_sorted[0][0]
            confidence = prediction_sorted[0][1] * 100
            
            cv2.putText(frame, "MODE: ACTIONS SPECIALES", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Action: {detected_sign}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Confiance: {confidence:.1f}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Afficher le top 3
            y_offset = 150
            for i in range(min(3, len(prediction_sorted))):
                action, conf = prediction_sorted[i]
                cv2.putText(frame, f"{i+1}. {action}: {conf*100:.1f}%", 
                           (10, y_offset + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            # Pas de mode sélectionné
            cv2.putText(frame, "Appuyez sur:", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "'1' = Nombres | '2' = Alphabet | '3' = Actions", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.imshow("Detection de la langue des signes", frame)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            mode = "1"
            print(">>> Mode NOMBRES activé")
        elif key == ord('2'):
            mode = "2"
            print(">>> Mode ALPHABET activé")
        elif key == ord('3'):
            mode = "3"
            print(">>> Mode ACTIONS activé")

except KeyboardInterrupt:
    print("\n\nInterruption par l'utilisateur...")
except Exception as e:
    print(f"\n\nErreur: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Programme terminé avec succès!")
