# Face Recognition with CNN (Real-Time Camera)

## Description

Ce projet implémente un système de **reconnaissance faciale en temps réel** à l’aide des **réseaux de neurones convolutifs profonds (CNN)**.  
L’application démarre automatiquement la **caméra**, détecte les visages et identifie les personnes entraînées à partir d’un modèle CNN.

Ce projet a été réalisé dans un cadre académique pour l’étude et l’implémentation pratique de la reconnaissance faciale.

---

## Objectifs du projet

- Comprendre le fonctionnement des **CNN appliqués aux images**
- Implémenter un modèle de reconnaissance faciale en Python
- Utiliser la **vision par ordinateur (OpenCV)** pour le traitement vidéo
- Réaliser une application fonctionnelle en **temps réel**

---

## Technologies utilisées

- **Python 3**
- **TensorFlow / Keras** (CNN)
- **OpenCV** (caméra et détection des visages)
- **NumPy**
- **Dataset LFW / dataset personnalisé**

---

## Structure du projet

```

face-recognition-cnn/
│
├── env/                     # Environnement virtuel (non versionné)
├── dataset/                 # Dataset des visages (non versionné)
├── main.py                  # Script principal
├── face_recognition_model.h5# Modèle entraîné (optionnel)
├── requirements.txt         # Dépendances
├── .gitignore
└── README.md

```

---

## Installation

### Cloner le projet

```bash
git clone https://github.com/votre-username/face-recognition-cnn.git
cd face-recognition-cnn
```

### Créer et activer l’environnement virtuel (Windows)

```bash
python -m venv env
env\Scripts\Activate.ps1
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Dataset

Le dataset doit être organisé de la manière suivante :

```
dataset/
├── personne1/
│   ├── img1.jpg
│   ├── img2.jpg
├── personne2/
│   ├── img1.jpg
│   ├── img2.jpg
```

Chaque dossier correspond à **une personne (une classe)**.

---

## Lancement de l’application

Pour démarrer l’application et ouvrir la caméra :

```bash
python main.py
```

- La caméra démarre automatiquement
- Les visages sont détectés
- Le nom prédit est affiché en temps réel
- Appuyer sur **q** pour quitter

---

## Fonctionnement

1. Prétraitement des images (redimensionnement, normalisation)
2. Entraînement d’un **CNN**
3. Détection des visages avec **Haar Cascade**
4. Prédiction de l’identité à partir du modèle entraîné
5. Affichage du résultat en temps réel via la caméra

---

## Limites

- Sensible à la lumière et à l’angle du visage
- Performances dépendantes de la taille du dataset
- Modèle CNN simple (peut être amélioré)

---

## Améliorations possibles

- Utilisation de **MTCNN** ou **Dlib**
- Modèles pré-entraînés (VGGFace, FaceNet)
- Sauvegarde des embeddings
- Interface graphique (Tkinter / PyQt)

---

## Aspects éthiques

La reconnaissance faciale implique des questions de **vie privée et de protection des données personnelles**.
Ce projet est destiné **uniquement à des fins pédagogiques**.
