"""
Module pour le prétraitement des images faciales
"""

import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class FacePreprocessor:
    """Classe pour le prétraitement des images faciales"""
    
    def __init__(self, target_size=(64, 64)):
        """
        Initialise le préprocesseur
        
        Args:
            target_size: Taille cible pour le redimensionnement
        """
        self.target_size = target_size
        self.label_encoder = None
        
    def preprocess_images(self, images):
        """
        Prétraite les images pour l'entraînement
        
        Args:
            images: Tableau d'images
            
        Returns:
            Images prétraitées
        """
        processed_images = []
        
        for img in images:
            # Normaliser les valeurs des pixels [0, 1]
            if img.max() > 1:
                img_normalized = img.astype('float32') / 255.0
            else:
                img_normalized = img.astype('float32')
            
            # S'assurer que l'image a 3 canaux pour la compatibilité
            if len(img_normalized.shape) == 3 and img_normalized.shape[2] == 1:
                img_normalized = np.repeat(img_normalized, 3, axis=2)
            
            processed_images.append(img_normalized)
        
        return np.array(processed_images)
    
    def encode_labels(self, labels):
        """
        Encode les labels en format one-hot
        
        Args:
            labels: Labels à encoder
            
        Returns:
            Labels encodés
        """
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        labels_one_hot = to_categorical(labels_encoded)
        
        return labels_one_hot, labels_encoded
    
    def detect_face(self, frame):
        """
        Détecte un visage dans une frame
        
        Args:
            frame: Image d'entrée
            
        Returns:
            Visage détecté et coordonnées
        """
        # Charger le classificateur en cascade pour la détection de visages
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Redimensionner et normaliser
            face_resized = cv2.resize(face_roi, self.target_size)
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=-1)
            face_expanded = np.repeat(face_expanded, 3, axis=2)
            face_expanded = np.expand_dims(face_expanded, axis=0)
            
            return face_expanded, (x, y, w, h)
        
        return None, None
    
    def prepare_for_prediction(self, image):
        """
        Prépare une image unique pour la prédiction
        
        Args:
            image: Image à prétraiter
            
        Returns:
            Image prétraitée
        """
        # Normaliser
        if image.max() > 1:
            image_normalized = image.astype('float32') / 255.0
        else:
            image_normalized = image.astype('float32')
        
        # Ajouter la dimension du batch
        if len(image_normalized.shape) == 3:
            image_expanded = np.expand_dims(image_normalized, axis=0)
        else:
            image_expanded = image_normalized
            
        return image_expanded