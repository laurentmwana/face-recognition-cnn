"""
Module pour la reconnaissance faciale en temps réel avec caméra
"""

import cv2
import numpy as np
import time

class LiveCameraRecognizer:
    """Classe pour la reconnaissance faciale en temps réel"""
    
    def __init__(self, model, preprocessor, target_names):
        """
        Initialise le reconnaisseur en temps réel
        
        Args:
            model: Modèle entraîné
            preprocessor: Préprocesseur avec label_encoder
            target_names: Noms des personnes
        """
        self.model = model
        self.preprocessor = preprocessor
        self.target_names = target_names
        self.camera = None
        
    def start_camera(self, camera_id=0):
        """
        Démarre la caméra
        
        Args:
            camera_id: ID de la caméra (0 par défaut)
        """
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            raise Exception(f"Impossible d'ouvrir la caméra {camera_id}")
        
        print(f"Caméra {camera_id} démarrée avec succès")
        return self.camera
    
    def run_recognition(self, confidence_threshold=0.7):
        """
        Lance la reconnaissance faciale en temps réel
        
        Args:
            confidence_threshold: Seuil de confiance minimum
        """
        if self.camera is None:
            self.start_camera()
        
        print("\nDémarrage de la reconnaissance faciale...")
        print("Appuyez sur 'q' pour quitter")
        print("Appuyez sur 's' pour sauvegarder une image")
        
        while True:
            # Lire une frame
            ret, frame = self.camera.read()
            if not ret:
                print("Erreur de lecture de la caméra")
                break
            
            # Détecter et prédire
            result_frame = self.process_frame(frame, confidence_threshold)
            
            # Afficher la frame
            cv2.imshow('Reconnaissance Faciale - Labeya', result_frame)
            
            # Gérer les touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Arrêt de la reconnaissance...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image sauvegardée: {filename}")
        
        # Nettoyage
        self.release_camera()
        cv2.destroyAllWindows()
    
    def process_frame(self, frame, confidence_threshold=0.7):
        """
        Traite une frame unique
        
        Args:
            frame: Frame d'entrée
            confidence_threshold: Seuil de confiance
            
        Returns:
            Frame avec annotations
        """
        # Faire une copie pour l'affichage
        display_frame = frame.copy()
        
        # Détecter le visage
        face, coords = self.preprocessor.detect_face(frame)
        
        if face is not None and coords is not None:
            x, y, w, h = coords
            
            # Faire la prédiction
            predictions = self.model.predict(face, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Récupérer le nom
            if self.preprocessor.label_encoder is not None:
                person_name = self.target_names[self.preprocessor.label_encoder.inverse_transform([predicted_class])[0]]
            else:
                person_name = self.target_names[predicted_class]
            
            # Déterminer la couleur en fonction de la confiance
            if confidence > confidence_threshold:
                color = (0, 255, 0)  # Vert
                label = f"{person_name}: {confidence:.2%}"
            else:
                color = (0, 165, 255)  # Orange
                label = f"Inconnu: {confidence:.2%}"
            
            # Dessiner le rectangle et le texte
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Afficher les informations
            cv2.putText(display_frame, f"Confiance: {confidence:.2%}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Appuyez sur 'q' pour quitter", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def release_camera(self):
        """Libère la caméra"""
        if self.camera:
            self.camera.release()
            print("Caméra libérée")