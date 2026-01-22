import os

# Chemin du dossier contenant les fichiers txt
dossier = "feeds"  # Remplace par le chemin de ton dossier

# Nom du fichier de sortie
fichier_sortie = "rss_feeds.txt"

# Ouvrir le fichier de sortie en mode écriture
with open(fichier_sortie, 'w', encoding='utf-8') as outfile:
    # Parcourir tous les fichiers du dossier
    for nom_fichier in os.listdir(dossier):
        # Vérifier si c'est un fichier .txt
        if nom_fichier.endswith('.txt'):
            chemin_fichier = os.path.join(dossier, nom_fichier)
            
            # Lire et écrire le contenu
            with open(chemin_fichier, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

print(f"Concaténation terminée dans {fichier_sortie}")