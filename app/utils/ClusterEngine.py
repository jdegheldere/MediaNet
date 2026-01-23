"""
Moteur de recherche d'articles similaires - Version optimisée mémoire
Usage: from search_engine import ArticleSearchEngine
"""

import asyncio
import numpy as np
import pickle
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer


class ArticleSearchEngine:
    
    def __init__(self, fetcher, batch_size=1000):
        self.fetcher = fetcher
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.batch_size = batch_size  # Nombre d'articles chargés à la fois
    
    def _text_to_vector(self, text):
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.astype('float32')
    
    async def _count_articles_with_embedding(self):
        """Compte le nombre total d'articles avec embedding"""
        conn = await self.fetcher._get_conn()
        async with conn.execute(
            'SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL'
        ) as cur:
            row = await cur.fetchone()
            return row[0]
    
    async def _get_articles_batch(self, offset, limit):
        """Récupère un batch d'articles avec embedding"""
        conn = await self.fetcher._get_conn()
        async with conn.execute(
            'SELECT ID, Title, Article_URL, ID_RSS, Date, Description, embedding FROM articles WHERE embedding IS NOT NULL ORDER BY ID LIMIT ? OFFSET ?',
            (limit, offset)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]
    
    async def search(self, query_text, top_k=10, min_score=0.6, 
                    exclude_id=None, display=True, save_to=None):
        """
        Cherche les articles similaires (traitement par batch pour économiser la RAM)
        
        query_text: texte de recherche
        top_k: nombre max de résultats
        min_score: score minimum (0-1, recommandé 0.6)
        exclude_id: ID d'article à exclure (optionnel)
        display: afficher les résultats dans le terminal
        save_to: nom du fichier JSON pour sauvegarder (ex: "results.json")
        """
        query_vector = self._text_to_vector(query_text)
        
        total_articles = await self._count_articles_with_embedding()
        
        if total_articles == 0:
            print("⚠️ Aucun article avec embedding trouvé")
            return []
        
        print(f"🔍 Recherche dans {total_articles} articles (par batch de {self.batch_size})...")
        
        # Stocker seulement les meilleurs scores (top_k * 3 pour avoir de la marge)
        best_scores = []
        max_stored = top_k * 3
        
        offset = 0
        processed = 0
        
        while offset < total_articles:
            batch = await self._get_articles_batch(offset, self.batch_size)
            
            for article in batch:
                if exclude_id and article['ID'] == exclude_id:
                    continue
                
                article_vector = pickle.loads(article['embedding'])
                
                # Cosine similarity
                similarity = np.dot(query_vector, article_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(article_vector)
                )
                
                score = float(similarity)
                
                # Garder seulement si score >= min_score
                if score >= min_score:
                    # Stocker sans l'embedding (économie mémoire)
                    article_light = {
                        'ID': article['ID'],
                        'Title': article['Title'],
                        'Article_URL': article['Article_URL'],
                        'ID_RSS': article['ID_RSS'],
                        'Date': article['Date'],
                        'Description': article['Description']
                    }
                    best_scores.append((article_light, score))
                    
                    # Si on dépasse max_stored, trier et garder seulement les meilleurs
                    if len(best_scores) > max_stored:
                        best_scores.sort(key=lambda x: x[1], reverse=True)
                        best_scores = best_scores[:max_stored]
            
            processed += len(batch)
            offset += self.batch_size
            
            # Afficher progression tous les 5000 articles
            if processed % 5000 == 0 or processed >= total_articles:
                print(f"  Traité: {processed}/{total_articles} articles")
        
        # Tri final et limitation à top_k
        best_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for article, score in best_scores[:top_k]:
            results.append({
                'id': article['ID'],
                'title': article['Title'],
                'url': article['Article_URL'],
                'source': article['ID_RSS'],
                'date': article['Date'],
                'description': article['Description'],
                'similarity_score': round(score, 3)
            })
        
        if display:
            self._display_results(results, query_text)
        
        if save_to:
            self._save_results(results, save_to, query_text)
        
        return results
    
    async def search_by_id(self, article_id, top_k=10, min_score=0.6, 
                          display=True, save_to=None):
        """
        Cherche les articles similaires à un article existant
        
        article_id: ID de l'article de référence
        autres params: identiques à search()
        """
        articles = await self.fetcher.query_db(
            conditions={"ID": article_id},
            limit=1,
            order_by="ID DESC"
        )
        
        if not articles:
            print(f"⚠️ Article {article_id} introuvable")
            return []
        
        article = articles[0]
        query_text = f"{article['Title']} {article['Description']}"
        
        return await self.search(
            query_text, 
            top_k=top_k, 
            min_score=min_score,
            exclude_id=article_id,
            display=display,
            save_to=save_to
        )
    
    def _display_results(self, results, query):
        print(f"\n{'='*80}")
        print(f"🔍 Requête: {query[:100]}...")
        print(f"📊 {len(results)} résultats trouvés")
        print(f"{'='*80}\n")
        
        for i, art in enumerate(results, 1):
            print(f"{i}. [{art['similarity_score']}] {art['title']}")
            print(f"   Source: {art['source']} | Date: {art['date']}")
            print(f"   URL: {art['url']}")
            if art['description']:
                desc = art['description'][:150].replace('\n', ' ')
                print(f"   {desc}...")
            print()
    
    def _save_results(self, results, filename, query):
        output = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Résultats sauvegardés dans {filename}")


async def main():
    from AsyncFetcher import AsyncFetcher
    
    fetcher = AsyncFetcher()
    
    # batch_size=1000 = charge 1000 articles à la fois (ajustable)
    engine = ArticleSearchEngine(fetcher, batch_size=1000)
    
    await engine.search(
        query_text="Trump Iran ayatollah Khamenei",
        top_k=10,
        min_score=0.65,
        display=True,
        save_to="results_trump_iran.json"
    )


if __name__ == "__main__":
    asyncio.run(main())