# -*- coding: utf-8 -*-
"""
Script d'Entraînement du Modèle CNN - Reconnaissance Langue des Signes
Entraîne un réseau de neurones convolutif pour classifier les signes

Architecture:
- 2 blocs Conv2D + MaxPooling
- Flatten + 2 couches Dense
- Softmax pour classification multi-classes

Compatible avec TensorFlow 2.20 et Keras 3.x
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("="*60)
print("🧠 ENTRAÎNEMENT DU MODÈLE CNN")
print("="*60)

# Configuration
TRAIN_DIR = 'DataSet/train'
TEST_DIR = 'DataSet/test'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 5
EPOCHS = 10
NUM_CLASSES = 45  # Nombre total de classes dans le dataset

# Vérifier que les répertoires existent
print("\n[1/5] Vérification des répertoires...")
if not os.path.exists(TRAIN_DIR):
    print(f"✗ Erreur: {TRAIN_DIR} n'existe pas")
    print("Assurez-vous d'avoir créé le dataset avec data_set_capture.py")
    exit(1)
if not os.path.exists(TEST_DIR):
    print(f"✗ Erreur: {TEST_DIR} n'existe pas")
    exit(1)

print(f"✓ Répertoire train: {TRAIN_DIR}")
print(f"✓ Répertoire test: {TEST_DIR}")

# Étape 1: Construction de l'architecture CNN
print("\n[2/5] Construction du modèle CNN...")

model = keras.Sequential([
    # Premier bloc de convolution
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Deuxième bloc de convolution
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Aplatissement
    layers.Flatten(),
    
    # Couches entièrement connectées
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
], name='SignLanguageCNN')

# Afficher le résumé de l'architecture
print("\n📊 Architecture du modèle:")
model.summary()

# Compilation du modèle
print("\n[3/5] Compilation du modèle...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Modèle compilé (optimizer=adam, loss=categorical_crossentropy)")

# Étape 2: Préparation des données avec augmentation
print("\n[4/5] Préparation des générateurs de données...")

# Générateur pour les données d'entraînement (avec augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisation
    shear_range=0.2,          # Cisaillement
    zoom_range=0.2,           # Zoom
    horizontal_flip=True      # Retournement horizontal
)

# Générateur pour les données de test (sans augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des données d'entraînement
print("\nChargement du training set...")
training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

# Chargement des données de test
print("Chargement du test set...")
test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

print(f"\n✓ Training set: {training_set.samples} images")
print(f"✓ Test set: {test_set.samples} images")
print(f"✓ Nombre de classes détectées: {training_set.num_classes}")

# Étape 3: Entraînement du modèle
print("\n[5/5] Entraînement du modèle...")
print(f"Époques: {EPOCHS}")
print(f"Steps par époque: 1000")
print(f"Validation steps: 30")
print("\nCela peut prendre plusieurs minutes/heures selon votre GPU...")
print("="*60 + "\n")

# Entraîner le modèle
history = model.fit(
    training_set,
    steps_per_epoch=1000,
    epochs=EPOCHS,
    validation_data=test_set,
    validation_steps=30,
    verbose=1
)

# Afficher les résultats finaux
print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ!")
print("="*60)
print(f"\nRésultats finaux (époque {EPOCHS}):")
print(f"  • Précision training:   {history.history['accuracy'][-1]*100:.2f}%")
print(f"  • Précision validation: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"  • Loss training:        {history.history['loss'][-1]:.4f}")
print(f"  • Loss validation:      {history.history['val_loss'][-1]:.4f}")

# Sauvegarde du modèle
print("\n[6/6] Sauvegarde du modèle...")

# Sauvegarder l'architecture en JSON (compatibilité)
model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print("✓ Architecture sauvegardée: model-bw.json")

# Sauvegarder les poids
model.save_weights('model-bw.h5')
print("✓ Poids sauvegardés: model-bw.h5")

# Optionnel: Sauvegarder le modèle complet (format moderne)
model.save('model-complete.keras')
print("✓ Modèle complet sauvegardé: model-complete.keras")

print("\n" + "="*60)
print("🎉 ENTRAÎNEMENT RÉUSSI!")
print("="*60)
print("\nFichiers générés:")
print("  • model-bw.json   - Architecture du modèle")
print("  • model-bw.h5     - Poids du modèle")
print("  • model-complete.keras - Modèle complet (optionnel)")
print("\nVous pouvez maintenant utiliser ce modèle avec:")
print("  • demo_signes_WORKING.py")
print("  • app_interface_elegante.py")
print("="*60 + "\n")
