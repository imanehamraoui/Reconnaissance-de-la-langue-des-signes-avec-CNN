# -*- coding: utf-8 -*-
"""
Interface Graphique Élégante - Reconnaissance Langue des Signes
Avec boutons pour Chiffres, Lettres et Actions Spéciales
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import operator

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance de la Langue des Signes")
        self.root.geometry("1300x750")
        self.root.configure(bg='#2C3E50')
        
        # Variables
        self.mode = None  # 'chiffres', 'lettres', ou 'actions'
        self.running = False
        self.cap = None
        self.model = None
        
        # Charger le modèle
        self.load_model()
        
        # Créer l'interface
        self.create_widgets()
        
        # Démarrer la caméra
        self.start_camera()
        
    def load_model(self):
        """Charge le modèle CNN"""
        try:
            self.model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(29, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.load_weights("model-bw.h5")
            print("✓ Modèle chargé avec succès!")
        except Exception as e:
            print(f"✗ Erreur de chargement du modèle: {e}")
            
    def create_widgets(self):
        """Crée l'interface graphique"""
        
        # ===== TITRE =====
        title_frame = tk.Frame(self.root, bg='#34495E', height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="🤚 Reconnaissance de la Langue des Signes 🤚",
            font=('Helvetica', 24, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        title_label.pack(pady=20)
        
        # ===== CONTENEUR PRINCIPAL =====
        main_container = tk.Frame(self.root, bg='#2C3E50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # ===== COLONNE GAUCHE - VIDÉO =====
        left_frame = tk.Frame(main_container, bg='#34495E', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_title = tk.Label(
            left_frame,
            text="📹 Flux Vidéo",
            font=('Helvetica', 16, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        video_title.pack(pady=10)
        
        self.video_label = tk.Label(left_frame, bg='#2C3E50')
        self.video_label.pack(padx=10, pady=10)
        
        # ===== COLONNE DROITE - CONTRÔLES & RÉSULTATS =====
        right_frame = tk.Frame(main_container, bg='#34495E', width=380, relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Section Mode
        mode_frame = tk.Frame(right_frame, bg='#34495E')
        mode_frame.pack(pady=20, padx=20, fill=tk.X)
        
        mode_label = tk.Label(
            mode_frame,
            text="🎯 Mode de Détection",
            font=('Helvetica', 14, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        mode_label.pack(pady=(0, 15))
        
        # Bouton Chiffres
        self.btn_chiffres = tk.Button(
            mode_frame,
            text="🔢 DÉTECTER LES CHIFFRES",
            font=('Helvetica', 11, 'bold'),
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            cursor='hand2',
            command=self.mode_chiffres
        )
        self.btn_chiffres.pack(fill=tk.X, pady=5, ipady=10)
        
        # Bouton Lettres
        self.btn_lettres = tk.Button(
            mode_frame,
            text="🔤 DÉTECTER LES LETTRES",
            font=('Helvetica', 11, 'bold'),
            bg='#9B59B6',
            fg='white',
            activebackground='#8E44AD',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            cursor='hand2',
            command=self.mode_lettres
        )
        self.btn_lettres.pack(fill=tk.X, pady=5, ipady=10)
        
        # Bouton Actions
        self.btn_actions = tk.Button(
            mode_frame,
            text="⚡ DÉTECTER LES ACTIONS",
            font=('Helvetica', 11, 'bold'),
            bg='#E67E22',
            fg='white',
            activebackground='#D35400',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            cursor='hand2',
            command=self.mode_actions
        )
        self.btn_actions.pack(fill=tk.X, pady=5, ipady=10)
        
        # Séparateur
        separator = ttk.Separator(right_frame, orient='horizontal')
        separator.pack(fill=tk.X, padx=20, pady=20)
        
        # Section État
        status_frame = tk.Frame(right_frame, bg='#34495E')
        status_frame.pack(pady=10, padx=20, fill=tk.X)
        
        status_title = tk.Label(
            status_frame,
            text="📊 État Actuel",
            font=('Helvetica', 14, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        status_title.pack(pady=(0, 10))
        
        self.status_label = tk.Label(
            status_frame,
            text="Mode: Aucun",
            font=('Helvetica', 12),
            bg='#2C3E50',
            fg='#ECF0F1',
            relief=tk.SUNKEN,
            bd=2
        )
        self.status_label.pack(fill=tk.X, ipady=5)
        
        # Section Résultats
        results_frame = tk.Frame(right_frame, bg='#34495E')
        results_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        results_title = tk.Label(
            results_frame,
            text="🎯 Prédictions",
            font=('Helvetica', 14, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        results_title.pack(pady=(0, 10))
        
        # Top 5 prédictions (4 pour actions)
        self.result_labels = []
        for i in range(4):
            frame = tk.Frame(results_frame, bg='#2C3E50', relief=tk.RAISED, bd=2)
            frame.pack(fill=tk.X, pady=3)
            
            label = tk.Label(
                frame,
                text=f"{i+1}. --- : 0.0%",
                font=('Helvetica', 10, 'bold' if i == 0 else 'normal'),
                bg='#2C3E50',
                fg='#1ABC9C' if i == 0 else '#ECF0F1',
                anchor='w'
            )
            label.pack(padx=10, pady=6, fill=tk.X)
            self.result_labels.append(label)
        
        # Bouton Quitter
        quit_btn = tk.Button(
            right_frame,
            text="❌ QUITTER",
            font=('Helvetica', 11, 'bold'),
            bg='#E74C3C',
            fg='white',
            activebackground='#C0392B',
            activeforeground='white',
            cursor='hand2',
            command=self.quit_app
        )
        quit_btn.pack(side=tk.BOTTOM, pady=20, padx=20, fill=tk.X, ipady=8)
        
    def start_camera(self):
        """Démarre la capture vidéo"""
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()
        
    def update_frame(self):
        """Met à jour le flux vidéo"""
        if not self.running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # ROI
            h, w = frame.shape[:2]
            x1 = int(0.5 * w)
            y1 = 10
            x2 = w - 10
            y2 = int(0.5 * w)
            
            # Dessiner le cadre
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (52, 152, 219), 3)
            
            # Extraire ROI
            roi = frame[y1:y2, x1:x2]
            
            # Prétraitement
            roi_processed = cv2.resize(roi, (64, 64))
            roi_processed = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2GRAY)
            _, roi_processed = cv2.threshold(roi_processed, 120, 255, cv2.THRESH_BINARY)
            roi_normalized = roi_processed.astype('float32') / 255.0
            
            # Prédiction si mode actif
            if self.mode and self.model:
                result = self.model.predict(roi_normalized.reshape(1, 64, 64, 1), verbose=0)
                self.update_predictions(result[0])
            
            # Afficher le mode
            if self.mode == 'chiffres':
                cv2.putText(frame, "MODE: CHIFFRES (0-9)", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 152, 219), 2)
            elif self.mode == 'lettres':
                cv2.putText(frame, "MODE: LETTRES (A-M)", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 89, 182), 2)
            elif self.mode == 'actions':
                cv2.putText(frame, "MODE: ACTIONS SPECIALES", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 126, 34), 2)
            
            # Convertir pour Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (800, 600))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)
        
    def update_predictions(self, predictions):
        """Met à jour l'affichage des prédictions"""
        if self.mode == 'chiffres':
            categories = {
                0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR',
                5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE'
            }
            pred_dict = {categories[i]: predictions[i] for i in range(10)}
            
        elif self.mode == 'lettres':
            categories = {
                0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
                5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                10: 'K', 11: 'L', 12: 'M'
            }
            pred_dict = {categories[i]: predictions[i] for i in range(13)}
            
        elif self.mode == 'actions':
            categories = {
                13: 'Creer Dossier',
                14: 'Coppier Fichier',
                15: 'Jeu',
                16: 'Ecrire Fichier'
            }
            pred_dict = {categories[i]: predictions[i] for i in categories.keys() if i < len(predictions)}
        
        sorted_pred = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)
        
        for i in range(min(4, len(sorted_pred))):
            sign, conf = sorted_pred[i]
            self.result_labels[i].config(
                text=f"{i+1}. {sign} : {conf*100:.1f}%",
                fg='#1ABC9C' if i == 0 else '#ECF0F1'
            )
        
        # Effacer les labels non utilisés
        for i in range(len(sorted_pred), 4):
            self.result_labels[i].config(text=f"{i+1}. --- : 0.0%", fg='#ECF0F1')
    
    def mode_chiffres(self):
        """Active le mode chiffres"""
        self.mode = 'chiffres'
        self.status_label.config(text="Mode: 🔢 CHIFFRES", fg='#3498DB')
        self.btn_chiffres.config(relief=tk.SUNKEN, bg='#2980B9')
        self.btn_lettres.config(relief=tk.RAISED, bg='#9B59B6')
        self.btn_actions.config(relief=tk.RAISED, bg='#E67E22')
        print(">>> Mode CHIFFRES activé")
        
    def mode_lettres(self):
        """Active le mode lettres"""
        self.mode = 'lettres'
        self.status_label.config(text="Mode: 🔤 LETTRES", fg='#9B59B6')
        self.btn_lettres.config(relief=tk.SUNKEN, bg='#8E44AD')
        self.btn_chiffres.config(relief=tk.RAISED, bg='#3498DB')
        self.btn_actions.config(relief=tk.RAISED, bg='#E67E22')
        print(">>> Mode LETTRES activé")
        
    def mode_actions(self):
        """Active le mode actions"""
        self.mode = 'actions'
        self.status_label.config(text="Mode: ⚡ ACTIONS", fg='#E67E22')
        self.btn_actions.config(relief=tk.SUNKEN, bg='#D35400')
        self.btn_chiffres.config(relief=tk.RAISED, bg='#3498DB')
        self.btn_lettres.config(relief=tk.RAISED, bg='#9B59B6')
        print(">>> Mode ACTIONS activé")
        
    def quit_app(self):
        """Ferme l'application proprement"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()

# Lancement de l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()
