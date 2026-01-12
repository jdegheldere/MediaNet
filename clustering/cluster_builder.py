"""
Système de recherche et regroupement d'articles de presse similaires
Architecture hybride : Embeddings sémantiques + BM25 + Cache DB
"""

import sqlite3
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

# Installation requise :
# pip install sentence-transformers faiss-cpu rank-bm25 beautifulsoup4 requests feedparser

from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import feedparser
import requests
from bs4 import BeautifulSoup


@dataclass
class Article:
    """Représentation d'un article"""
    id: str
    title: str
    content: str
    url: str
    source: str
    published_date: Optional[datetime]
    embedding: Optional[np.ndarray] = None


class ArticleClusteringEngine:
    """Moteur de recherche et clustering d'articles de presse"""
    
    def __init__(self, db_path: str = "articles.db", 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            db_path: Chemin vers la base SQLite
            model_name: Modèle de sentence-transformers (optimisé pour le français)
        """
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension du modèle MiniLM
        
        # Index FAISS pour recherche vectorielle ultra-rapide
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        self.id_to_index = {}  # Mapping article_id -> position dans l'index
        self.index_to_id = {}  # Mapping inverse
        
        # BM25 pour recherche lexicale
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_ids = []
        
        self._init_database()
        self._load_cache()
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT UNIQUE,
                source TEXT,
                published_date TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_source ON articles(source)
        ''')
        
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_published ON articles(published_date)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_cache(self):
        """Charge les embeddings et index depuis la DB"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT id, title, content, embedding FROM articles WHERE embedding IS NOT NULL')
        
        embeddings_list = []
        idx = 0
        
        for row in c.fetchall():
            article_id, title, content, embedding_blob = row
            
            if embedding_blob:
                embedding = pickle.loads(embedding_blob)
                embeddings_list.append(embedding)
                
                self.id_to_index[article_id] = idx
                self.index_to_id[idx] = article_id
                
                # Pour BM25
                self.bm25_corpus.append(self._tokenize(title + " " + content))
                self.bm25_ids.append(article_id)
                
                idx += 1
        
        # Reconstruction de l'index FAISS
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            faiss.normalize_L2(embeddings_array)  # Normalisation pour cosine similarity
            self.index.add(embeddings_array)
            
            # Reconstruction BM25
            self.bm25 = BM25Okapi(self.bm25_corpus)
        
        conn.close()
        print(f"✓ Cache chargé : {len(embeddings_list)} articles indexés")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenization simple pour BM25"""
        return text.lower().split()
    
    def _generate_id(self, url: str) -> str:
        """Génère un ID unique basé sur l'URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Calcule l'embedding d'un texte"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')
    
    def add_article(self, title: str, content: str, url: str, 
                   source: str, published_date: Optional[datetime] = None) -> str:
        """
        Ajoute un article à la base et l'indexe
        
        Returns:
            article_id si ajouté, None si déjà existant
        """
        article_id = self._generate_id(url)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Vérifier si existe déjà
        c.execute('SELECT id FROM articles WHERE id = ?', (article_id,))
        if c.fetchone():
            conn.close()
            return None
        
        # Calculer l'embedding
        combined_text = f"{title} {content}"
        embedding = self._compute_embedding(combined_text)
        embedding_blob = pickle.dumps(embedding)
        
        # Insérer en DB
        c.execute('''
            INSERT INTO articles (id, title, content, url, source, published_date, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (article_id, title, content, url, source, 
              published_date.isoformat() if published_date else None,
              embedding_blob))
        
        conn.commit()
        conn.close()
        
        # Ajouter à l'index FAISS
        normalized_embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(normalized_embedding)
        
        current_index = self.index.ntotal
        self.index.add(normalized_embedding)
        self.id_to_index[article_id] = current_index
        self.index_to_id[current_index] = article_id
        
        # Ajouter à BM25
        tokens = self._tokenize(combined_text)
        self.bm25_corpus.append(tokens)
        self.bm25_ids.append(article_id)
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        return article_id
    
    def find_similar_articles(self, query_text: str, 
                            top_k: int = 50,
                            similarity_threshold: float = 0.5,
                            alpha: float = 0.7) -> List[Dict]:
        """
        Trouve les articles similaires à un texte donné
        
        Args:
            query_text: Texte de l'article source
            top_k: Nombre max de résultats
            similarity_threshold: Seuil de similarité (0-1)
            alpha: Poids embeddings vs BM25 (0.7 = 70% embeddings, 30% BM25)
        
        Returns:
            Liste de dicts avec id, title, url, source, similarity_score
        """
        if self.index.ntotal == 0:
            return []
        
        # 1. Recherche vectorielle (sémantique)
        query_embedding = self._compute_embedding(query_text).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
        
        # Scores sémantiques
        semantic_scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 = pas de résultat
                article_id = self.index_to_id[idx]
                semantic_scores[article_id] = float(dist)
        
        # 2. Recherche lexicale (BM25)
        lexical_scores = {}
        if self.bm25:
            query_tokens = self._tokenize(query_text)
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Normalisation des scores BM25
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
            for i, score in enumerate(bm25_scores):
                article_id = self.bm25_ids[i]
                lexical_scores[article_id] = score / max_bm25 if max_bm25 > 0 else 0
        
        # 3. Fusion des scores (hybride)
        all_ids = set(semantic_scores.keys()) | set(lexical_scores.keys())
        hybrid_scores = {}
        
        for article_id in all_ids:
            sem_score = semantic_scores.get(article_id, 0)
            lex_score = lexical_scores.get(article_id, 0)
            hybrid_scores[article_id] = alpha * sem_score + (1 - alpha) * lex_score
        
        # 4. Filtrage et tri
        filtered = [(aid, score) for aid, score in hybrid_scores.items() 
                   if score >= similarity_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Récupération des métadonnées
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for article_id, score in filtered[:top_k]:
            c.execute('''
                SELECT title, url, source, published_date 
                FROM articles WHERE id = ?
            ''', (article_id,))
            
            row = c.fetchone()
            if row:
                results.append({
                    'id': article_id,
                    'title': row[0],
                    'url': row[1],
                    'source': row[2],
                    'published_date': row[3],
                    'similarity_score': round(score, 3)
                })
        
        conn.close()
        return results
    
    def scrape_rss_feeds(self, feed_urls: List[str]) -> int:
        """
        Récupère et indexe les articles depuis des flux RSS
        
        Returns:
            Nombre d'articles ajoutés
        """
        added_count = 0
        
        for feed_url in feed_urls:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    title = entry.get('title', '')
                    url = entry.get('link', '')
                    summary = entry.get('summary', '')
                    
                    # Extraction de la date
                    pub_date = None
                    if 'published_parsed' in entry:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Extraction du contenu complet (si possible)
                    content = self._extract_article_content(url) or summary
                    
                    if title and url and content:
                        result = self.add_article(
                            title=title,
                            content=content,
                            url=url,
                            source=feed.feed.get('title', 'Unknown'),
                            published_date=pub_date
                        )
                        
                        if result:
                            added_count += 1
                            
            except Exception as e:
                print(f"Erreur lors du parsing de {feed_url}: {e}")
        
        return added_count
    
    def _extract_article_content(self, url: str, timeout: int = 5) -> Optional[str]:
        """Extrait le contenu textuel d'une page web"""
        try:
            response = requests.get(url, timeout=timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Suppression des scripts, styles, etc.
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Extraction du texte
            text = soup.get_text(separator=' ', strip=True)
            return text[:10000]  # Limite à 10k caractères
            
        except Exception:
            return None
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur la base"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM articles')
        total = c.fetchone()[0]
        
        c.execute('SELECT source, COUNT(*) FROM articles GROUP BY source')
        by_source = dict(c.fetchall())
        
        conn.close()
        
        return {
            'total_articles': total,
            'indexed_articles': self.index.ntotal,
            'sources': by_source
        }


# ===== EXEMPLE D'UTILISATION =====

if __name__ == "__main__":
    # Initialisation
    engine = ArticleClusteringEngine()
    
    # Exemple 1 : Ajout manuel d'articles
    engine.add_article(
        title="Intelligence artificielle : Claude 4 dévoilé par Anthropic",
        content="Anthropic a annoncé aujourd'hui le lancement de Claude 4, sa nouvelle génération de modèles d'IA...",
        url="https://example.com/article1",
        source="TechNews"
    )
    
    # Exemple 2 : Scraping de flux RSS (sources d'actualité francophones)
    rss_feeds = [
        'https://www.lemonde.fr/rss/une.xml',
        'https://www.lefigaro.fr/rss/figaro_actualites.xml',
        'https://www.liberation.fr/arc/outboundfeeds/rss-all/?outputType=xml'
    ]
    
    print("Scraping des flux RSS...")
    new_articles = engine.scrape_rss_feeds(rss_feeds)
    print(f"✓ {new_articles} nouveaux articles ajoutés")
    
    # Exemple 3 : Recherche d'articles similaires
    query = """
    L'intelligence artificielle générative transforme le paysage technologique.
    Les entreprises investissent massivement dans ces nouvelles technologies.
    """
    
    print("\nRecherche d'articles similaires...")
    similar = engine.find_similar_articles(query, top_k=10, similarity_threshold=0.4)
    
    print(f"\nTrouvé {len(similar)} articles similaires :\n")
    for article in similar:
        print(f"[{article['similarity_score']}] {article['title']}")
        print(f"  Source: {article['source']} | URL: {article['url']}\n")
    
    # Statistiques
    stats = engine.get_statistics()
    print(f"\n📊 Statistiques:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Articles indexés: {stats['indexed_articles']}")
    print(f"  Sources: {stats['sources']}")