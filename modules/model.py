"""
Module pour la définition du modèle CNN
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class FaceRecognitionModel:
    """Classe pour créer le modèle CNN de reconnaissance faciale"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        """
        Initialise le modèle
        
        Args:
            input_shape: Forme des images d'entrée
        """
        self.input_shape = input_shape
        self.model = None
        
    def build_simple_cnn(self, num_classes):
        """
        Construit un CNN simple pour la reconnaissance faciale
        
        Args:
            num_classes: Nombre de classes (personnes) à reconnaître
            
        Returns:
            Modèle Keras compilé
        """
        model = Sequential([
            # Première couche convolutive
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Deuxième couche convolutive
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Troisième couche convolutive
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Couches fully connected
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Couche de sortie
            Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_model(self, num_classes):
        """
        Construit et compile le modèle
        
        Args:
            num_classes: Nombre de classes (personnes) à reconnaître
            
        Returns:
            Modèle compilé
        """
        model = self.build_simple_cnn(num_classes)
        
        # Compiler le modèle
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Afficher le résumé
        model.summary()
        
        self.model = model
        return model
    
    def save_model(self, filepath='face_recognition_model.h5'):
        """
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin pour sauvegarder le modèle
        """
        if self.model:
            self.model.save(filepath)
            print(f"Modèle sauvegardé sous: {filepath}")
    
    def load_model(self, filepath='face_recognition_model.h5'):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin du modèle à charger
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Modèle chargé depuis: {filepath}")
        return self.model