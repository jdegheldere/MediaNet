import sqlite3


"""
Deuxième routine : 
La deuxième routine doit actualiser la BDD avec des nouvelles URL. donc elle doit :
-Vérifier que personne n'utilise la bdd (Plus tard)
-Se connecter à la BDD (créer la table URL si rien n'existe)
-Parser des txt pour ajouter des URL de flux RSS à la bdd
(Format de la table d'URLs (Title : feeds) |ID|URL|LastState|LastVisited|)
"""


"""
Additions futures:
-Ajouter de nouveaux URLs, attention aux connections concurrantes dans la bdd!!!
"""
#On va concaténer tus les flux RSS enun seul fichier por commencer
BDD_PATH = "databaseTest.db"
RSS_PATH = "txt_rss_test.txt"


def init_feeds():
    with sqlite3.connect(BDD_PATH) as conn:
        print("Creating table ...")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feeds (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        URL TEXT NOT NULL,
        LastState INTEGER DEFAULT 1,
        last_modified TEXT,
        etag TEXT
        )
        """)
        conn.commit()
        print("Done!")
        print("parsing RSS txt file ...")
        with open(RSS_PATH) as f:
            list_urls = f.readlines()
        print(f"Done!, extracted {len(list_urls)} URLs from file")
        print("Saving to database ...")
        for url in list_urls:
            conn.execute("INSERT INTO feeds (URL) VALUES (?)",(url,))
        conn.commit()
        print("feeds database fully initiated")

def display_titles():
    with sqlite3.connect(BDD_PATH) as conn:
        cur = conn.execute("SELECT DISTINCT Title FROM articles")
        print(len(cur.fetchall()))

def display_RSS_laststates():
    with sqlite3.connect(BDD_PATH) as conn:
        cur = conn.execute("SELECT URL, LastState from feeds WHERE LastState=0")
        print(cur.fetchall())

if __name__ == "__main__":
   init_feeds()