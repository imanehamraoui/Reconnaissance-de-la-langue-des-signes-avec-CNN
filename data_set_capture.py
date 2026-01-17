# -*- coding: utf-8 -*-
"""
Script de Capture du Dataset - Reconnaissance Langue des Signes
Permet de créer le dataset en capturant des images via webcam

Utilisation:
1. Lance le script
2. Fais le signe devant la caméra
3. Appuie sur la touche correspondante (0-9, A-Z, ou symboles)
4. L'image est sauvegardée automatiquement dans le bon dossier

Adapté pour chemins relatifs et structure moderne
"""

import cv2
import os

print("="*60)
print("📸 SCRIPT DE CAPTURE DU DATASET")
print("="*60)

# Configuration
MODE = 'train'  # 'train' ou 'test'
BASE_DIR = 'DataSet'

# Définir les classes
CLASSES = [
    # Chiffres
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # Lettres
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    # Actions spéciales
    'Creer_Dossier', 'Coppier_Fichier', 'Jeu', 'Musique',
    'Ouvrire_un_Site_Web', 'Video', 'Ecrire_Dans_un_Fichier',
    'PDF_ou_Word', 'fichier_signes'
]

# Création de la structure de dossiers
print(f"\n[1/3] Création de la structure de dossiers...")
print(f"Mode: {MODE}")
print(f"Répertoire de base: {BASE_DIR}")

# Créer les dossiers s'ils n'existent pas
for mode in ['train', 'test']:
    mode_dir = os.path.join(BASE_DIR, mode)
    if not os.path.exists(mode_dir):
        os.makedirs(mode_dir)
        print(f"✓ Créé: {mode_dir}/")
    
    for cls in CLASSES:
        class_dir = os.path.join(mode_dir, cls)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

print(f"✓ Structure de {len(CLASSES)} classes créée!")

# Initialiser la caméra
print("\n[2/3] Initialisation de la caméra...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Erreur: Impossible d'accéder à la caméra")
    exit(1)

print("✓ Caméra initialisée!")

# Définir le répertoire de travail
directory = os.path.join(BASE_DIR, MODE)

print("\n[3/3] Démarrage de la capture...")
print("\n📋 INSTRUCTIONS:")
print("- Fais le signe devant la caméra")
print("- Appuie sur la touche correspondante:")
print("  • 0-9 pour les chiffres")
print("  • A-Z pour les lettres")
print("  • * pour Copier_Fichier")
print("  • + pour Créer_Dossier")
print("  • - pour Jeu")
print("  • ESC pour quitter")
print("\n" + "="*60 + "\n")

# Boucle de capture
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture de la caméra")
        break
    
    # Effet miroir pour utilisation intuitive
    frame = cv2.flip(frame, 1)
    
    # Compter les images existantes pour quelques classes principales
    count = {
        '0': len([f for f in os.listdir(os.path.join(directory, '0')) if f.endswith('.jpg')]) if os.path.exists(os.path.join(directory, '0')) else 0,
        '1': len([f for f in os.listdir(os.path.join(directory, '1')) if f.endswith('.jpg')]) if os.path.exists(os.path.join(directory, '1')) else 0,
        '2': len([f for f in os.listdir(os.path.join(directory, '2')) if f.endswith('.jpg')]) if os.path.exists(os.path.join(directory, '2')) else 0,
        'A': len([f for f in os.listdir(os.path.join(directory, 'A')) if f.endswith('.jpg')]) if os.path.exists(os.path.join(directory, 'A')) else 0,
        'B': len([f for f in os.listdir(os.path.join(directory, 'B')) if f.endswith('.jpg')]) if os.path.exists(os.path.join(directory, 'B')) else 0,
    }
    
    # Afficher les informations sur l'écran
    cv2.putText(frame, f"MODE: {MODE.upper()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "COMPTEUR D'IMAGES:", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"0: {count['0']}  1: {count['1']}  2: {count['2']}", (10, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"A: {count['A']}  B: {count['B']}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "Appuyez sur ESC pour quitter", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Définir la région d'intérêt (ROI)
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    
    # Dessiner le cadre ROI
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 2)
    
    # Extraire la ROI
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (64, 64))
    
    # Prétraitement: conversion en niveaux de gris et seuillage
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY)
    
    # Afficher les images
    cv2.imshow("Capture Dataset - Frame Principale", frame)
    cv2.imshow("ROI Pretraitee (64x64)", roi_thresh)
    
    # Gestion des touches
    key = cv2.waitKey(10) & 0xFF
    
    # ESC pour quitter
    if key == 27:
        print("\n>>> Arrêt de la capture...")
        break
    
    # Capturer pour les chiffres (0-9)
    if key >= ord('0') and key <= ord('9'):
        cls = chr(key)
        filepath = os.path.join(directory, cls, f'{count.get(cls, 0)}.jpg')
        cv2.imwrite(filepath, roi_thresh)
        print(f"✓ Image sauvegardée: {cls}/ ({count.get(cls, 0) + 1} images)")
    
    # Capturer pour les lettres (A-Z)
    elif key >= ord('A') and key <= ord('Z'):
        cls = chr(key)
        cls_dir = os.path.join(directory, cls)
        if os.path.exists(cls_dir):
            num_images = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
            filepath = os.path.join(cls_dir, f'{num_images}.jpg')
            cv2.imwrite(filepath, roi_thresh)
            print(f"✓ Image sauvegardée: {cls}/ ({num_images + 1} images)")
    
    # Touches spéciales pour actions
    elif key == ord('*'):
        cls_dir = os.path.join(directory, 'Coppier_Fichier')
        num_images = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        cv2.imwrite(os.path.join(cls_dir, f'{num_images}.jpg'), roi_thresh)
        print(f"✓ Image sauvegardée: Coppier_Fichier/ ({num_images + 1} images)")
    
    elif key == ord('+'):
        cls_dir = os.path.join(directory, 'Creer_Dossier')
        num_images = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        cv2.imwrite(os.path.join(cls_dir, f'{num_images}.jpg'), roi_thresh)
        print(f"✓ Image sauvegardée: Creer_Dossier/ ({num_images + 1} images)")
    
    elif key == ord('-'):
        cls_dir = os.path.join(directory, 'Jeu')
        num_images = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        cv2.imwrite(os.path.join(cls_dir, f'{num_images}.jpg'), roi_thresh)
        print(f"✓ Image sauvegardée: Jeu/ ({num_images + 1} images)")

# Libération des ressources
cap.release()
cv2.destroyAllWindows()

print("\n✓ Capture terminée avec succès!")
print("="*60)
