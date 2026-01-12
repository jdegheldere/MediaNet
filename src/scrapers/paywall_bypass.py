import re
import json
from typing import Optional, Dict, Any
import logging

class PaywallBypass:
    """Techniques pour contourner les paywalls"""
    
    # Patterns de paywalls connus
    PAYWALL_PATTERNS = {
        'lemonde': r'paywall|abonnement|article réservé',
        'lefigaro': r'premium|abonné',
        'lesechos': r'premium|abonnement',
        'mediapart': r'abonnement',
        'nytimes': r'subscription|paywall',
        'generic': r's\'inscrire|se connecter|abonnement|premium'
    }
    
    # Stratégies de contournement par domaine
    DOMAIN_STRATEGIES = {
        'lemonde.fr': ['text_content', 'amp_version', 'reader_mode'],
        'lefigaro.fr': ['text_content', 'amp_version'],
        'lesechos.fr': ['text_content', 'api_search'],
        'mediapart.fr': ['limited_access'],  # Très difficile
        'nytimes.com': ['reader_mode', 'text_content']
    }
    
    @staticmethod
    def detect_paywall(html: str, url: str) -> bool:
        """Détecte si la page a un paywall"""
        domain = url.split('/')[2]
        
        # Vérifier les patterns spécifiques
        for pattern in PaywallBypass.PAYWALL_PATTERNS.values():
            if re.search(pattern, html, re.IGNORECASE):
                return True
                
        # Vérifier les sélecteurs CSS communs
        paywall_selectors = [
            '.paywall', '.subscription-required', '.premium-content',
            '[class*="paywall"]', '[class*="premium"]', '[class*="abonnement"]'
        ]
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        for selector in paywall_selectors:
            if soup.select(selector):
                return True
                
        return False
    
    @staticmethod
    def bypass_paywall(html: str, url: str, strategy: Optional[str] = None) -> Optional[str]:
        """
        Tente de contourner le paywall
        Retourne le texte accessible ou None
        """
        domain = url.split('/')[2]
        
        # Choix de la stratégie
        if not strategy:
            strategy = PaywallBypass.DOMAIN_STRATEGIES.get(domain, ['text_content'])[0]
        
        try:
            if strategy == 'text_content':
                return PaywallBypass._extract_accessible_text(html)
            elif strategy == 'amp_version':
                return PaywallBypass._try_amp_version(url)
            elif strategy == 'reader_mode':
                return PaywallBypass._reader_mode_extract(html)
            elif strategy == 'api_search':
                return PaywallBypass._search_cached_version(url)
            elif strategy == 'limited_access':
                # Pour les sites très restrictifs comme Mediapart
                # Nécessite parfois d'utiliser des APIs tierces ou web.archive
                return PaywallBypass._try_archive(url)
                
        except Exception as e:
            logging.error(f"Échec bypass pour {url}: {str(e)}")
            
        return None
    
    @staticmethod
    def _extract_accessible_text(html: str) -> str:
        """Extrait le texte accessible avant le paywall"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Supprimer les éléments indésirables
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'form']):
            tag.decompose()
            
        # Trouver le contenu principal
        main_content = soup.find('article') or \
                      soup.find('main') or \
                      soup.find('div', {'role': 'main'}) or \
                      soup.find('div', class_=re.compile(r'content|article|post'))
        
        if main_content:
            # Extraire uniquement les paragraphes avant un éventuel "lire la suite"
            paragraphs = []
            for p in main_content.find_all(['p', 'h1', 'h2', 'h3']):
                text = p.get_text(strip=True)
                if text and len(text) > 50:  # Éviter les petits éléments
                    if any(phrase in text.lower() for phrase in ['abonné', 'connectez-vous', 'premium']):
                        break
                    paragraphs.append(text)
            
            if paragraphs:
                return '\n\n'.join(paragraphs)
        
        # Fallback: tout le texte du body
        if soup.body:
            return soup.body.get_text(separator='\n', strip=True)
        
        return ''
    
    @staticmethod
    def _try_amp_version(url: str) -> Optional[str]:
        """Essaie la version AMP de l'article"""
        import aiohttp
        import asyncio
        
        amp_url = None
        if '?' in url:
            amp_url = url + '&amp=1'
        else:
            amp_url = url + '?amp=1'
        
        try:
            # Note: À implémenter avec une session async
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(amp_url) as response:
            #         if response.status == 200:
            #             html = await response.text()
            #             return PaywallBypass._extract_accessible_text(html)
            pass
        except:
            pass
        
        return None
    
    @staticmethod
    def _reader_mode_extract(html: str) -> str:
        """Simule un mode lecture simplifié"""
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Algorithm simplifié de mode lecture
        # 1. Calculer le score des éléments
        elements = soup.find_all(['p', 'div', 'article', 'section'])
        scored_elements = []
        
        for elem in elements:
            if elem.name == 'p' and elem.get_text(strip=True):
                text = elem.get_text(strip=True)
                score = len(text.split())
                
                # Bonus pour les paragraphes longs
                if score > 20:
                    score *= 1.5
                
                scored_elements.append((score, text))
        
        # 2. Prendre les meilleurs éléments
        scored_elements.sort(reverse=True, key=lambda x: x[0])
        top_elements = scored_elements[:15]  # Top 15 paragraphes
        
        # 3. Assembler le texte
        return '\n\n'.join([elem[1] for elem in top_elements])
    
    @staticmethod
    def _try_archive(url: str) -> Optional[str]:
        """Essaie de récupérer via archive.org"""
        # À implémenter: utiliser l'API web.archive.org
        # ou d'autres services de cache
        return None
    
    @staticmethod
    def _search_cached_version(url: str) -> Optional[str]:
        """Cherche une version en cache (Google, etc.)"""
        cache_urls = [
            f"https://webcache.googleusercontent.com/search?q=cache:{url}",
            f"http://archive.is/newest/{url}"
        ]
        # À implémenter
        return None