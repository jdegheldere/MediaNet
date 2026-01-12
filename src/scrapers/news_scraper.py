import re
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import logging

from .base_scraper import BaseScraper
from .paywall_bypass import PaywallBypass

class NewsScraper(BaseScraper):
    """Scraper spécialisé pour les articles de presse"""
    
    def __init__(self):
        super().__init__()
        
    async def scrape_article(self, url: str, bypass_paywall: bool = True) -> Dict[str, Any]:
        """Scrape un article complet"""
        html = await self.fetch_html(url)
        
        if not html:
            raise Exception(f"Impossible de récupérer l'URL: {url}")
        
        # Détecter et contourner le paywall si nécessaire
        if bypass_paywall and PaywallBypass.detect_paywall(html, url):
            logging.info(f"Paywall détecté pour {url}, tentative de contournement...")
            accessible_text = PaywallBypass.bypass_paywall(html, url)
            
            if accessible_text and len(accessible_text) > 500:  # Seuil minimal
                # On a réussi à contourner
                html = self._reconstruct_html_with_text(html, accessible_text)
        
        # Extraire le contenu
        return await self.extract_content(html, url)
    
    async def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extrait le contenu structuré d'un article"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extraction des métadonnées
        metadata = self.extract_metadata(soup)
        
        # Extraction spécifique aux articles
        article_data = {
            'url': url,
            'title': self._extract_title(soup, metadata),
            'author': self._extract_authors(soup, metadata),
            'publication_date': self._extract_date(soup, metadata),
            'text': self._extract_article_text(soup),
            'html': html if len(html) < 10000 else None,  # Stocker si petit
            'summary': self._extract_summary(soup, metadata),
            'language': self._detect_language(html),
            'word_count': 0,
            'metadata': metadata,
            'images': self._extract_images(soup),
            'domain': url.split('/')[2]
        }
        
        # Calcul du nombre de mots
        article_data['word_count'] = len(article_data['text'].split())
        
        return article_data
    
    def _extract_title(self, soup: BeautifulSoup, metadata: Dict) -> str:
        """Extrait le titre de l'article"""
        # Priorité 1: Open Graph
        if 'og:title' in metadata:
            return metadata['og:title']
        
        # Priorité 2: JSON-LD
        if 'jsonld' in metadata and 'headline' in metadata['jsonld']:
            return metadata['jsonld']['headline']
        
        # Priorité 3: Balise title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Priorité 4: H1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return ""
    
    def _extract_authors(self, soup: BeautifulSoup, metadata: Dict) -> List[str]:
        """Extrait la/les auteur(s)"""
        authors = []
        
        # JSON-LD
        if 'jsonld' in metadata:
            jsonld = metadata['jsonld']
            if 'author' in jsonld:
                if isinstance(jsonld['author'], list):
                    for author in jsonld['author']:
                        if isinstance(author, dict) and 'name' in author:
                            authors.append(author['name'])
                        elif isinstance(author, str):
                            authors.append(author)
                elif isinstance(jsonld['author'], dict) and 'name' in jsonld['author']:
                    authors.append(jsonld['author']['name'])
        
        # Meta tags
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            'meta[name="twitter:creator"]',
            '[rel="author"]',
            '.author', '.byline', '.article-author'
        ]
        
        for selector in author_selectors:
            elements = soup.select(selector)
            for elem in elements:
                if elem.get('content'):
                    authors.append(elem['content'])
                elif elem.get_text(strip=True):
                    authors.append(elem.get_text(strip=True))
        
        # Nettoyage
        cleaned_authors = []
        for author in set(authors):
            if author and len(author) > 2 and not author.startswith('http'):
                cleaned_authors.append(author.strip())
        
        return cleaned_authors[:5]  # Limiter à 5 auteurs max
    
    def _extract_date(self, soup: BeautifulSoup, metadata: Dict) -> Optional[str]:
        """Extrait la date de publication"""
        date = None
        
        # JSON-LD
        if 'jsonld' in metadata:
            jsonld = metadata['jsonld']
            for date_field in ['datePublished', 'dateCreated', 'dateModified']:
                if date_field in jsonld:
                    date = jsonld[date_field]
                    break
        
        # Open Graph
        if not date and 'article:published_time' in metadata:
            date = metadata['article:published_time']
        
        # Meta tags
        if not date:
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publication-date"]',
                'meta[name="publish-date"]',
                'time[datetime]',
                '.date', '.publication-date', '.article-date'
            ]
            
            for selector in date_selectors:
                elem = soup.select_one(selector)
                if elem:
                    if elem.get('datetime'):
                        date = elem['datetime']
                        break
                    elif elem.get('content'):
                        date = elem['content']
                        break
                    elif elem.get_text(strip=True):
                        date = elem.get_text(strip=True)
                        break
        
        # Parser et formatter la date
        if date:
            try:
                # Essayer plusieurs formats
                for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        parsed = datetime.strptime(date[:19], fmt)
                        return parsed.isoformat()
                    except:
                        continue
            except:
                pass
        
        return None
    
    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """Extrait le texte principal de l'article"""
        # Stratégies d'extraction
        
        # 1. Chercher des balises article spécifiques
        article_selectors = [
            'article',
            'div.article-content',
            'div.post-content',
            'div.story-content',
            'div.entry-content',
            'main',
            '[role="main"]'
        ]
        
        for selector in article_selectors:
            article = soup.select_one(selector)
            if article and len(article.get_text(strip=True)) > 500:
                return self._clean_text(article)
        
        # 2. Algorithm de densité de texte
        paragraphs = soup.find_all('p')
        if len(paragraphs) > 5:
            # Sélectionner les paragraphes les plus longs
            long_paragraphs = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 100:  # Seuil de longueur
                    long_paragraphs.append(text)
            
            if long_paragraphs:
                return '\n\n'.join(long_paragraphs[:20])  # Limiter
        
        # 3. Fallback: tout le body
        return self._clean_text(soup.body or soup)
    
    def _clean_text(self, element) -> str:
        """Nettoie le texte d'un élément"""
        if not element:
            return ""
        
        # Supprimer les éléments inutiles
        for tag in element(['script', 'style', 'nav', 'footer', 'aside', 
                           'form', 'button', 'iframe', 'noscript']):
            tag.decompose()
        
        # Récupérer le texte avec des sauts de ligne
        text = element.get_text(separator='\n', strip=True)
        
        # Nettoyer les espaces multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _extract_summary(self, soup: BeautifulSoup, metadata: Dict) -> Optional[str]:
        """Extrait le résumé/description"""
        if 'og:description' in metadata:
            return metadata['og:description']
        
        if 'jsonld' in metadata and 'description' in metadata['jsonld']:
            return metadata['jsonld']['description']
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content']
        
        return None
    
    def _detect_language(self, html: str) -> str:
        """Détecte la langue de l'article"""
        # Simple détection basée sur les balises lang
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        lang = soup.find('html').get('lang', '')
        if lang:
            return lang[:2]
        
        # Détection par contenu (basique)
        text = soup.get_text()[:500]
        if ' le ' in text.lower() or ' la ' in text.lower() or ' de ' in text.lower():
            return 'fr'
        elif ' the ' in text.lower() or ' and ' in text.lower():
            return 'en'
        
        return 'fr'  # Par défaut
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extrait les images principales"""
        images = []
        
        # Images Open Graph
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            images.append(og_image['content'])
        
        # Images JSON-LD
        if soup.find('script', type='application/ld+json'):
            try:
                for script in soup.find_all('script', type='application/ld+json'):
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'image' in data:
                        if isinstance(data['image'], str):
                            images.append(data['image'])
                        elif isinstance(data['image'], list):
                            images.extend(data['image'][:3])
            except:
                pass
        
        # Images dans l'article
        img_selectors = ['article img', '.article-image', '.post-thumbnail']
        for selector in img_selectors:
            for img in soup.select(selector)[:5]:
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http'):
                    images.append(src)
        
        # Déduplication
        return list(set(images))[:10]
    
    def _reconstruct_html_with_text(self, original_html: str, text: str) -> str:
        """Reconstruit un HTML avec le texte contourné"""
        soup = BeautifulSoup(original_html, 'html.parser')
        
        # Créer un nouvel article
        article = soup.new_tag('article')
        article['class'] = 'extracted-content'
        
        # Ajouter le texte
        for paragraph in text.split('\n\n'):
            if paragraph.strip():
                p = soup.new_tag('p')
                p.string = paragraph.strip()
                article.append(p)
        
        # Remplacer le body
        if soup.body:
            soup.body.clear()
            soup.body.append(article)
        
        return str(soup)