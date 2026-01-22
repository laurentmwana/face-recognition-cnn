"""
Module pour l'entraînement du modèle
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

class ModelTrainer:
    """Classe pour gérer l'entraînement du modèle"""
    
    def __init__(self, model):
        """
        Initialise le formateur de modèle
        
        Args:
            model: Modèle Keras à entraîner
        """
        self.model = model
        self.history = None
        
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, validation_split=0.2):
        """
        Entraîne le modèle
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation
            epochs: Nombre d'époques
            batch_size: Taille du batch
            validation_split: Proportion pour la validation
            
        Returns:
            Historique d'entraînement
        """
        print("Début de l'entraînement...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        start_time = time.time()
        
        # Entraînement
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\nTemps d'entraînement: {training_time:.2f} secondes")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test
        
        Args:
            X_test: Données de test
            y_test: Labels de test
            
        Returns:
            Score de test
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné ou chargé")
            
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self, save_path=None):
        """
        Affiche les courbes d'apprentissage
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        """
        if self.history is None:
            print("Aucun historique d'entraînement disponible")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Courbe de précision
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Précision du modèle')
        axes[0].set_xlabel('Époque')
        axes[0].set_ylabel('Précision')
        axes[0].legend()
        axes[0].grid(True)
        
        # Courbe de perte
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Perte du modèle')
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel('Perte')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graphique sauvegardé sous: {save_path}")
        
        plt.show()