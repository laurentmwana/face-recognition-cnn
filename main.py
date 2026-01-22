"""
Reconnaissance Faciale - Version corrigée avec téléchargement du classificateur
Auteur : Labeya
"""

import sys
import os
import urllib.request
import cv2
import numpy as np

print("="*60)
print("SYSTÈME DE RECONNAISSANCE FACIALE")
print("Auteur : Labeya")
print("="*60)

def download_classifier():
    """Télécharge le classificateur si nécessaire"""
    classifier_filename = "haarcascade_frontalface_default.xml"
    
    if os.path.exists(classifier_filename):
        print(f"✓ Classificateur trouvé: {classifier_filename}")
        return classifier_filename
    
    print("Téléchargement du classificateur...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    try:
        urllib.request.urlretrieve(url, classifier_filename)
        print(f"✓ Classificateur téléchargé: {classifier_filename}")
        return classifier_filename
    except Exception as e:
        print(f"❌ Erreur de téléchargement: {e}")
        
        # Essayer un chemin local
        local_paths = [
            "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                print(f"✓ Classificateur trouvé à: {path}")
                return path
        
        return None

def check_camera():
    """Vérifie si la caméra fonctionne"""
    print("\nTest de la caméra...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra")
        return False
    
    print("✓ Caméra détectée")
    
    # Test de capture
    ret, frame = cap.read()
    if ret:
        print(f"✓ Image capturée: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True
    else:
        print("❌ Impossible de lire l'image")
        cap.release()
        return False

def simple_face_detection():
    """Détection faciale simple"""
    print("\n" + "="*60)
    print("DÉTECTION FACIALE")
    print("="*60)
    print("Instructions:")
    print("- Placez-vous face à la caméra")
    print("- Appuyez sur 'q' pour quitter")
    print("- Appuyez sur 's' pour sauvegarder une capture")
    print("="*60)
    
    # Télécharger ou trouver le classificateur
    classifier_path = download_classifier()
    if not classifier_path:
        print("❌ Impossible de trouver le classificateur")
        return
    
    # Charger le classificateur
    face_cascade = cv2.CascadeClassifier(classifier_path)
    if face_cascade.empty():
        print("❌ Erreur de chargement du classificateur")
        return
    
    print("✓ Classificateur chargé")
    
    # Ouvrir la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra")
        return
    
    print("\nDémarrage de la détection...")
    
    while True:
        # Lire une frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur de lecture")
            break
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Dessiner les rectangles autour des visages
        face_count = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Visage", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            face_count += 1
        
        # Afficher les informations
        info_y = 30
        cv2.putText(frame, f"Visages detectes: {face_count}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Appuyez sur 's' pour sauvegarder", (10, info_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Afficher la frame
        cv2.imshow('Detection Faciale - Labeya', frame)
        
        # Gérer les touches
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Capture sauvegardee: {filename}")
    
    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Detection terminee")

def face_recognition_with_training():
    """Reconnaissance faciale avec entraînement simple"""
    print("\n" + "="*60)
    print("RECONNAISSANCE FACIALE AVEC ENTRAÎNEMENT")
    print("="*60)
    
    # Télécharger ou trouver le classificateur
    classifier_path = download_classifier()
    if not classifier_path:
        print("❌ Impossible de trouver le classificateur")
        return
    
    # Charger le classificateur
    face_cascade = cv2.CascadeClassifier(classifier_path)
    if face_cascade.empty():
        print("❌ Erreur de chargement du classificateur")
        return
    
    print("✓ Classificateur charge")
    
    # Phase 1: Capture des visages de référence
    print("\n--- PHASE 1: CAPTURE DES VISAGES DE REFERENCE ---")
    print("Placez-vous face a la camera")
    print("Appuyez sur 'c' pour capturer, 'q' pour quitter")
    
    reference_faces = []
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la camera")
        return
    
    print("\nCapture en cours...")
    
    while len(reference_faces) < 10:  # Capturer 10 visages
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Afficher le compteur
        cv2.putText(frame, f"Visages captures: {len(reference_faces)}/10", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Appuyez sur 'c' pour capturer", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Capture des Visages', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and len(faces) > 0:
            # Capturer le premier visage détecté
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            reference_faces.append(face_resized)
            print(f"  Visage {len(reference_faces)}/10 capture")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(reference_faces) < 3:
        print("❌ Pas assez de visages captures")
        return
    
    print(f"\n✓ {len(reference_faces)} visages de reference captures")
    
    # Phase 2: Reconnaissance en temps réel
    print("\n--- PHASE 2: RECONNAISSANCE EN TEMPS REEL ---")
    print("Appuyez sur 'q' pour quitter")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Extraire le visage
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Comparer avec les visages de référence
            best_match_idx = -1
            best_similarity = 0
            
            for i, ref_face in enumerate(reference_faces):
                # Calcul de similarité simple (corrélation)
                similarity = cv2.matchTemplate(face_resized, ref_face, cv2.TM_CCOEFF_NORMED)[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            # Décision
            if best_similarity > 0.5:  # Seuil de similarité
                color = (0, 255, 0)  # Vert = reconnu
                label = f"Personne {best_match_idx+1} ({best_similarity:.0%})"
            else:
                color = (0, 0, 255)  # Rouge = inconnu
                label = "Inconnu"
            
            # Dessiner
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Afficher
        cv2.putText(frame, "Reconnaissance Faciale - Labeya", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Visages de reference: {len(reference_faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Reconnaissance Faciale', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Reconnaissance terminee")

def main_menu():
    """Menu principal"""
    print("\n" + "="*60)
    print("MENU PRINCIPAL")
    print("="*60)
    print("1. Detection faciale simple")
    print("2. Reconnaissance faciale (avec entrainement)")
    print("3. Tester la camera")
    print("4. Quitter")
    print("="*60)
    
    choice = input("Votre choix (1-4): ").strip()
    
    if choice == '1':
        simple_face_detection()
    elif choice == '2':
        face_recognition_with_training()
    elif choice == '3':
        if check_camera():
            print("✓ Camera fonctionnelle!")
        else:
            print("❌ Probleme avec la camera")
    elif choice == '4':
        print("\nAu revoir!")
        return False
    else:
        print("❌ Choix invalide")
    
    return True

def main():
    """Fonction principale"""
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Vérifier la caméra
    if not check_camera():
        print("\nAttention: Probleme avec la camera")
        print("Le programme peut continuer mais la detection ne fonctionnera pas.")
        response = input("Continuer quand meme? (o/n): ").strip().lower()
        if response != 'o':
            return
    
    # Boucle principale
    while True:
        try:
            if not main_menu():
                break
        except KeyboardInterrupt:
            print("\n\nProgramme interrompu")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Redemarrage du menu...")
    
    print("\nMerci d'avoir utilise le systeme de reconnaissance faciale!")
    print("Auteur: Labeya")

if __name__ == "__main__":
    main()
    input("\nAppuyez sur Entree pour quitter...")