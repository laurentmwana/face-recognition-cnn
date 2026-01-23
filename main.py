"""
Système de Reconnaissance Faciale avec CNN
Auteur : Groupe 9, Master 1 IA et Data Science
Université de Kinshasa, République Démocratique du Congo
Année académique 2024-2025
"""

import cv2
import numpy as np
import os
import urllib.request
import time
from datetime import datetime
import sys

class SystemeReconnaissanceFaciale:
    """
    Classe principale du système de reconnaissance faciale
    Implémente les fonctionnalités décrites dans le document
    """
    
    def __init__(self):
        """Initialisation du système"""
        self.version_opencv = cv2.__version__
        self.version_numpy = np.__version__
        
        # Chemins des fichiers
        self.chemin_haarcascade = "haarcascade_frontalface_default.xml"
        self.dossier_reference = "visages_reference"
        self.dossier_sauvegardes = "captures"
        
        # Paramètres du système
        self.taille_visage = (100, 100)  # Taille standard pour la reconnaissance
        self.seuil_reconnaissance = 0.5  # Seuil de similarité minimum
        self.nombre_references = 10      # Nombre de visages de référence à capturer
        
        # Initialiser les composants
        self.initialiser_dossiers()
        self.telecharger_classificateur()
        self.charger_classificateur()
        
        # Base de données des visages de référence
        self.visages_reference = []
        self.identifiants_reference = []
        
        print("=" * 60)
        print("SYSTÈME DE RECONNAISSANCE FACIALE")
        print("=" * 60)
        print(f"OpenCV Version: {self.version_opencv}")
        print(f"NumPy Version: {self.version_numpy}")
        print(f"Classificateur: {self.chemin_haarcascade}")
        print("=" * 60)
    
    def initialiser_dossiers(self):
        """Créer les dossiers nécessaires au fonctionnement du système"""
        for dossier in [self.dossier_reference, self.dossier_sauvegardes]:
            if not os.path.exists(dossier):
                os.makedirs(dossier)
                print(f"Dossier créé: {dossier}")
    
    def telecharger_classificateur(self):
        """Télécharger automatiquement le classificateur Haarcascade s'il n'existe pas"""
        if not os.path.exists(self.chemin_haarcascade):
            print("Téléchargement du classificateur Haarcascade...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                urllib.request.urlretrieve(url, self.chemin_haarcascade)
                print(f"Classificateur téléchargé: {self.chemin_haarcascade}")
            except Exception as e:
                print(f"Erreur de téléchargement: {e}")
                print("Veuillez télécharger manuellement le fichier Haarcascade.")
                sys.exit(1)
    
    def charger_classificateur(self):
        """Charger le classificateur Haarcascade pour la détection de visages"""
        try:
            self.detecteur_visage = cv2.CascadeClassifier(self.chemin_haarcascade)
            if self.detecteur_visage.empty():
                raise ValueError("Impossible de charger le classificateur")
            print("Classificateur Haarcascade chargé avec succès")
        except Exception as e:
            print(f"Erreur de chargement du classificateur: {e}")
            sys.exit(1)
    
    def afficher_menu_principal(self):
        """Afficher le menu principal interactif"""
        print("\n" + "=" * 60)
        print("MENU PRINCIPAL - SYSTÈME DE RECONNAISSANCE FACIALE")
        print("=" * 60)
        print("1. Détection faciale simple")
        print("2. Reconnaissance faciale (avec entraînement)")
        print("3. Tester la caméra")
        print("4. Visualiser les visages de référence")
        print("5. Quitter")
        print("=" * 60)
    
    def tester_camera(self):
        """Tester le fonctionnement de la caméra web"""
        print("\nTest de la caméra en cours...")
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Erreur: Impossible d'accéder à la caméra")
            return False
        
        ret, frame = camera.read()
        camera.release()
        
        if ret:
            print("✓ Caméra fonctionnelle")
            print(f"Résolution: {frame.shape[1]}x{frame.shape[0]} pixels")
            cv2.imshow("Test Caméra", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            return True
        else:
            print("✗ Problème avec la caméra")
            return False
    
    def detecter_visages(self, image):
        """
        Détecter les visages dans une image
        Utilise le classificateur Haarcascade avec les paramètres optimisés
        """
        # Convertir en niveaux de gris pour la détection
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Appliquer la détection de visages avec les paramètres spécifiés
        visages = self.detecteur_visage.detectMultiScale(
            gris,
            scaleFactor=1.1,      # Facteur de réduction d'échelle
            minNeighbors=5,       # Nombre minimum de voisins
            minSize=(30, 30),     # Taille minimale des visages
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return visages
    
    def preprocess_visage(self, image, region):
        """
        Prétraiter un visage détecté
        Redimensionne et normalise l'image
        """
        x, y, w, h = region
        
        # Extraire la région du visage
        visage = image[y:y+h, x:x+w]
        
        # Redimensionner à la taille standard
        visage = cv2.resize(visage, self.taille_visage)
        
        # Convertir en niveaux de gris
        visage = cv2.cvtColor(visage, cv2.COLOR_BGR2GRAY)
        
        # Normaliser les valeurs des pixels
        visage = cv2.equalizeHist(visage)
        
        return visage
    
    def calculer_similarite(self, visage1, visage2):
        """
        Calculer la similarité entre deux visages
        Utilise la méthode Template Matching (corrélation normalisée)
        """
        # Appliquer Template Matching
        resultat = cv2.matchTemplate(visage1, visage2, cv2.TM_CCOEFF_NORMED)
        
        # Récupérer le score maximum de similarité
        similarite = np.max(resultat)
        
        return similarite
    
    def mode_detection_simple(self):
        """
        Mode 1: Détection faciale simple
        Détecte les visages sans effectuer de reconnaissance
        """
        print("\n" + "=" * 60)
        print("MODE DÉTECTION FACIALE SIMPLE")
        print("=" * 60)
        print("Instructions:")
        print("- 's' : Sauvegarder la capture courante")
        print("- 'q' : Quitter le mode")
        print("=" * 60)
        
        # Initialiser la caméra
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        compteur_detections = 0
        compteur_sauvegardes = 0
        
        while True:
            # Capturer une frame
            ret, frame = camera.read()
            if not ret:
                print("Erreur de capture vidéo")
                break
            
            # Détecter les visages
            visages = self.detecter_visages(frame)
            compteur_detections = len(visages)
            
            # Annoter les visages détectés
            for (x, y, w, h) in visages:
                # Dessiner un rectangle vert autour du visage
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Ajouter un label
                cv2.putText(frame, f"Visage", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Afficher le compteur de détections
            cv2.putText(frame, f"Visages: {compteur_detections}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Afficher les instructions
            cv2.putText(frame, "'s': Sauvegarder  'q': Quitter", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Afficher la frame
            cv2.imshow("Detection Faciale Simple", frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and compteur_detections > 0:
                # Sauvegarder la capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chemin_sauvegarde = os.path.join(self.dossier_sauvegardes, 
                                               f"detection_{timestamp}.jpg")
                cv2.imwrite(chemin_sauvegarde, frame)
                compteur_sauvegardes += 1
                print(f"Capture sauvegardée: {chemin_sauvegarde}")
        
        # Libérer les ressources
        camera.release()
        cv2.destroyAllWindows()
        
        print(f"\nRésumé du mode détection:")
        print(f"- Nombre total de captures sauvegardées: {compteur_sauvegardes}")
        print("Retour au menu principal...")
    
    def capturer_visages_reference(self):
        """
        Capturer des visages pour la base de référence
        Phase 1 du système de reconnaissance
        """
        print("\n" + "=" * 60)
        print("CAPTURE DES VISAGES DE RÉFÉRENCE")
        print("=" * 60)
        print("Instructions:")
        print("- Positionnez-vous face à la caméra")
        print("- 'c' : Capturer un visage")
        print("- 'q' : Terminer la capture")
        print(f"Objectif: {self.nombre_references} visages de référence")
        print("=" * 60)
        
        # Réinitialiser la base de référence
        self.visages_reference = []
        self.identifiants_reference = []
        
        # Initialiser la caméra
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        compteur_captures = 0
        
        while compteur_captures < self.nombre_references:
            # Capturer une frame
            ret, frame = camera.read()
            if not ret:
                print("Erreur de capture vidéo")
                break
            
            # Détecter les visages
            visages = self.detecter_visages(frame)
            
            if len(visages) == 1:
                # Un seul visage détecté
                x, y, w, h = visages[0]
                
                # Dessiner un rectangle bleu autour du visage
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Ajouter un label
                cv2.putText(frame, f"Visage detecte", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Afficher le compteur de captures
            cv2.putText(frame, f"Captures: {compteur_captures}/{self.nombre_references}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Afficher les instructions
            cv2.putText(frame, "'c': Capturer  'q': Quitter", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Afficher la frame
            cv2.imshow("Capture Visages Reference", frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c') and len(visages) == 1:
                # Capturer le visage
                visage_preprocess = self.preprocess_visage(frame, visages[0])
                
                # Sauvegarder le visage
                chemin_visage = os.path.join(self.dossier_reference, 
                                           f"reference_{compteur_captures:02d}.jpg")
                cv2.imwrite(chemin_visage, visage_preprocess)
                
                # Ajouter à la base de référence
                self.visages_reference.append(visage_preprocess)
                self.identifiants_reference.append(f"Personne_{compteur_captures}")
                
                compteur_captures += 1
                print(f"Visage capturé: {compteur_captures}/{self.nombre_references}")
                
                # Afficher un feedback visuel
                cv2.putText(frame, "VISAGE CAPTURE!", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Capture Visages Reference", frame)
                cv2.waitKey(500)
        
        # Libérer les ressources
        camera.release()
        cv2.destroyAllWindows()
        
        if len(self.visages_reference) > 0:
            print(f"\nCapture terminée: {len(self.visages_reference)} visages de référence")
            return True
        else:
            print("\nAucun visage capturé")
            return False
    
    def mode_reconnaissance(self):
        """
        Mode 2: Reconnaissance faciale avec entraînement
        Phase 2 du système de reconnaissance
        """
        print("\n" + "=" * 60)
        print("MODE RECONNAISSANCE FACIALE")
        print("=" * 60)
        
        # Phase 1: Capturer les visages de référence
        print("\n--- PHASE 1: CAPTURE DES VISAGES DE RÉFÉRENCE ---")
        if not self.capturer_visages_reference():
            print("Échec de la capture des visages de référence")
            return
        
        # Phase 2: Reconnaissance en temps réel
        print("\n--- PHASE 2: RECONNAISSANCE EN TEMPS RÉEL ---")
        print("Instructions:")
        print("- 'q' : Quitter le mode reconnaissance")
        print("=" * 60)
        
        # Initialiser la caméra
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        stats_reconnaissance = {
            'total_visages': 0,
            'reconnus': 0,
            'inconnus': 0,
            'temps_traitement': []
        }
        
        while True:
            # Mesurer le temps de traitement
            debut_temps = time.time()
            
            # Capturer une frame
            ret, frame = camera.read()
            if not ret:
                print("Erreur de capture vidéo")
                break
            
            # Détecter les visages
            visages = self.detecter_visages(frame)
            stats_reconnaissance['total_visages'] += len(visages)
            
            # Traiter chaque visage détecté
            for (x, y, w, h) in visages:
                # Prétraiter le visage
                visage_courant = self.preprocess_visage(frame, (x, y, w, h))
                
                # Initialiser les variables de reconnaissance
                meilleur_score = 0
                meilleure_correspondance = -1
                
                # Comparer avec chaque visage de référence
                for idx, visage_ref in enumerate(self.visages_reference):
                    similarite = self.calculer_similarite(visage_ref, visage_courant)
                    
                    if similarite > meilleur_score:
                        meilleur_score = similarite
                        meilleure_correspondance = idx
                
                # Décision de reconnaissance
                if meilleur_score > self.seuil_reconnaissance:
                    # Visage reconnu
                    identifiant = self.identifiants_reference[meilleure_correspondance]
                    couleur = (0, 255, 0)  # Vert
                    label = f"{identifiant} ({meilleur_score:.2%})"
                    stats_reconnaissance['reconnus'] += 1
                else:
                    # Visage non reconnu
                    couleur = (0, 0, 255)  # Rouge
                    label = f"Inconnu ({meilleur_score:.2%})"
                    stats_reconnaissance['inconnus'] += 1
                
                # Dessiner l'annotation
                cv2.rectangle(frame, (x, y), (x+w, y+h), couleur, 2)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
            
            # Calculer le temps de traitement
            temps_traitement = (time.time() - debut_temps) * 1000  # en ms
            stats_reconnaissance['temps_traitement'].append(temps_traitement)
            
            # Afficher les statistiques
            fps = 1000 / temps_traitement if temps_traitement > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Reconnus: {stats_reconnaissance['reconnus']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Inconnus: {stats_reconnaissance['inconnus']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Afficher les instructions
            cv2.putText(frame, "'q': Quitter", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Afficher la frame
            cv2.imshow("Reconnaissance Faciale", frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Libérer les ressources
        camera.release()
        cv2.destroyAllWindows()
        
        # Afficher les statistiques finales
        self.afficher_statistiques(stats_reconnaissance)
    
    def afficher_statistiques(self, stats):
        """Afficher les statistiques de performance du système"""
        print("\n" + "=" * 60)
        print("STATISTIQUES DE PERFORMANCE")
        print("=" * 60)
        
        if stats['total_visages'] > 0:
            taux_reconnaissance = (stats['reconnus'] / stats['total_visages']) * 100
        else:
            taux_reconnaissance = 0
        
        if stats['temps_traitement']:
            temps_moyen = np.mean(stats['temps_traitement'])
        else:
            temps_moyen = 0
        
        print(f"Visages détectés: {stats['total_visages']}")
        print(f"Visages reconnus: {stats['reconnus']}")
        print(f"Visages inconnus: {stats['inconnus']}")
        print(f"Taux de reconnaissance: {taux_reconnaissance:.1f}%")
        print(f"Temps de traitement moyen: {temps_moyen:.1f} ms/frame")
        print(f"FPS moyen: {1000/temps_moyen:.1f}" if temps_moyen > 0 else "FPS: N/A")
        print("=" * 60)
        
        # Sauvegarder les statistiques dans un fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin_stats = os.path.join(self.dossier_sauvegardes, f"stats_{timestamp}.txt")
        
        with open(chemin_stats, 'w') as f:
            f.write("STATISTIQUES DE RECONNAISSANCE FACIALE\n")
            f.write("=" * 40 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Visages de référence: {len(self.visages_reference)}\n")
            f.write(f"Visages détectés: {stats['total_visages']}\n")
            f.write(f"Visages reconnus: {stats['reconnus']}\n")
            f.write(f"Visages inconnus: {stats['inconnus']}\n")
            f.write(f"Taux de reconnaissance: {taux_reconnaissance:.1f}%\n")
            f.write(f"Temps de traitement moyen: {temps_moyen:.1f} ms\n")
            f.write(f"Seuil de reconnaissance: {self.seuil_reconnaissance}\n")
        
        print(f"Statistiques sauvegardées: {chemin_stats}")
    
    def visualiser_references(self):
        """Visualiser les visages de référence capturés"""
        print("\n" + "=" * 60)
        print("VISUALISATION DES VISAGES DE RÉFÉRENCE")
        print("=" * 60)
        
        if len(self.visages_reference) == 0:
            print("Aucun visage de référence disponible")
            print("Veuillez d'abord exécuter le mode reconnaissance")
            return
        
        # Créer une image composite avec tous les visages de référence
        nb_visages = len(self.visages_reference)
        nb_colonnes = 5
        nb_lignes = (nb_visages + nb_colonnes - 1) // nb_colonnes
        
        # Créer une image vide
        hauteur_cellule = self.taille_visage[0] + 40  # +40 pour le texte
        largeur_cellule = self.taille_visage[1] + 20
        composite = np.zeros((hauteur_cellule * nb_lignes, 
                            largeur_cellule * nb_colonnes, 3), dtype=np.uint8)
        
        for idx, visage in enumerate(self.visages_reference):
            # Calculer la position dans la grille
            ligne = idx // nb_colonnes
            colonne = idx % nb_colonnes
            
            # Convertir le visage en BGR pour l'affichage
            visage_bgr = cv2.cvtColor(visage, cv2.COLOR_GRAY2BGR)
            
            # Position dans l'image composite
            y_debut = ligne * hauteur_cellule
            x_debut = colonne * largeur_cellule
            
            # Ajouter le visage
            composite[y_debut:y_debut+self.taille_visage[0], 
                     x_debut:x_debut+self.taille_visage[1]] = visage_bgr
            
            # Ajouter l'étiquette
            cv2.putText(composite, f"Personne_{idx}", 
                       (x_debut, y_debut + self.taille_visage[0] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Afficher l'image composite
        cv2.imshow("Visages de Reference", composite)
        print(f"Affichage de {nb_visages} visages de référence")
        print("Appuyez sur une touche pour continuer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def executer(self):
        """Méthode principale pour exécuter le système"""
        print("\nInitialisation du système de reconnaissance faciale...")
        
        # Tester la caméra
        if not self.tester_camera():
            print("Erreur: La caméra n'est pas accessible")
            return
        
        # Boucle principale du menu
        while True:
            self.afficher_menu_principal()
            
            try:
                choix = input("\nVotre choix (1-5): ").strip()
                
                if choix == '1':
                    self.mode_detection_simple()
                elif choix == '2':
                    self.mode_reconnaissance()
                elif choix == '3':
                    self.tester_camera()
                elif choix == '4':
                    self.visualiser_references()
                elif choix == '5':
                    print("\n" + "=" * 60)
                    print("Fermeture du système de reconnaissance faciale")
                    print("Merci d'avoir utilisé notre système!")
                    print("=" * 60)
                    break
                else:
                    print("Choix invalide. Veuillez entrer un nombre entre 1 et 5.")
            
            except KeyboardInterrupt:
                print("\n\nInterruption par l'utilisateur")
                break
            except Exception as e:
                print(f"\nErreur: {e}")
                print("Retour au menu principal...")

# Fonction principale pour lancer le système
def main():
    """Point d'entrée principal du programme"""
    try:
        systeme = SystemeReconnaissanceFaciale()
        systeme.executer()
    except Exception as e:
        print(f"\nErreur critique: {e}")
        print("Arrêt du système.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("SYSTÈME DE RECONNAISSANCE FACIALE AVEC CNN")
    print("Travail Dirigé - Master 1 IA et Data Science")
    print("Université de Kinshasa, RDC - 2024-2025")
    print("=" * 60)
    main()