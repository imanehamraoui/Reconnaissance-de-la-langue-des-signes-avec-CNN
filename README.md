# 🧠 Reconnaissance de la Langue des Signes en Temps Réel avec CNN

## 📌 Description du projet
Ce projet vise à faciliter la communication entre les personnes sourdes et entendantes
grâce à un système de reconnaissance automatique de la langue des signes basé sur
le Deep Learning, plus précisément les réseaux de neurones convolutifs (CNN).

Le système permet la reconnaissance en temps réel :
- des chiffres de 0 à 9
- des lettres de A à M
- de certaines actions spéciales

---

## 👩‍💻 Réalisé par
- **Imane Hamraoui**
- **Maroua Lhassouani**

🎓 Encadré par : **Pr. Moumoun Lahcen**  
📅 Année : **2026**

---

## 🎯 Objectifs du projet
- Créer un dataset d’images de la langue des signes
- Développer un modèle CNN performant
- Reconnaître les signes en temps réel via webcam
- Déployer une application avec interface graphique

---

## 🏗️ Architecture du projet

### Scripts principaux
- `data_set_capture.py` : capture des images via webcam
- `training_model_updated.py` : entraînement du modèle CNN
- `app_interface_elegante.py` : application finale avec interface graphique

### Modèle entraîné
- `model-bw.h5` : poids du réseau
- `model-bw.json` : architecture du modèle

---

## 📊 Dataset
- 28 classes au total  
  - 10 chiffres (0–9)
  - 13 lettres (A–M)
  - 5 actions spéciales
- Images 64×64 pixels
- Niveaux de gris (grayscale)
- Prétraitement : seuillage binaire + normalisation

---

## 🧠 Architecture CNN
- Couches de convolution (3×3)
- MaxPooling (2×2)
- Flatten
- Dense (128 neurones)
- Couche de sortie avec Softmax

---

## 🖥️ Interface graphique
L’application a été développée avec **Tkinter** et **OpenCV** :
- Capture webcam en temps réel
- Zone ROI pour la main
- Affichage de la prédiction instantanée

---

## 🎥 Démonstration vidéo
Cliquez sur la vidéo ci-dessous pour voir le fonctionnement de l’interface :

👉 **[Voir la démonstration](demo/demo_interface.mp4)**

---

## 🛠️ Technologies utilisées
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Tkinter

---

## 🚀 Applications pratiques
- Accessibilité et inclusion
- Apprentissage de la langue des signes
- Services publics et santé
- Applications éducatives

---

## ✅ Conclusion
Ce projet démontre l’efficacité du Deep Learning dans la reconnaissance visuelle.
Il illustre comment l’intelligence artificielle peut transformer une problématique
réelle en une solution concrète et utile pour la société.
