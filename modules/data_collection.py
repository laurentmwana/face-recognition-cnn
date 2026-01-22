"""
Module pour la collecte et le téléchargement des données
"""

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """Classe pour collecter et préparer le dataset"""
    
    def __init__(self, min_faces_per_person=20, resize=0.4):
        """
        Initialise le collecteur de données
        
        Args:
            min_faces_per_person: Nombre minimum d'images par personne
            resize: Facteur de redimensionnement
        """
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize
        self.data = None
        self.labels = None
        self.target_names = None
        
    def load_lfw_dataset(self):
        """
        Télécharge et charge le dataset LFW
        """
        print("Téléchargement du dataset LFW...")
        
        # Télécharger le dataset
        lfw_people = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person, 
            resize=self.resize,
            funneled=True,
            color=False
        )
        
        # Extraire les données
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.images.reshape((n_samples, h, w, 1))
        y = lfw_people.target
        target_names = lfw_people.target_names
        
        print(f"Dataset chargé: {len(X)} images, {len(target_names)} personnes")
        print(f"Taille des images: {h}x{w} pixels")
        
        self.data = X
        self.labels = y
        self.target_names = target_names
        
        return X, y, target_names
    
    def split_data(self, test_size=0.25, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test
        
        Args:
            test_size: Proportion pour le test set
            random_state: Seed pour la reproductibilité
        """
        if self.data is None:
            raise ValueError("Veuillez d'abord charger le dataset")
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        print(f"Train set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, X_test, y_train, y_test