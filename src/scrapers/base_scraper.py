import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import time

class BaseScraper(ABC):
    """Classe mère pour tous les scrapers"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.timeout = 30
        self.max_retries = 3
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch_html(self, url: str, retry_count: int = 0) -> Optional[str]:
        """Récupère le HTML d'une URL avec retry"""
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 403 or response.status == 429:
                    # Rate limiting ou interdiction
                    if retry_count < self.max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Backoff exponentiel
                        return await self.fetch_html(url, retry_count + 1)
                    else:
                        logging.warning(f"Échec après {self.max_retries} tentatives: {url}")
                return None
        except Exception as e:
            logging.error(f"Erreur fetch_html pour {url}: {str(e)}")
            if retry_count < self.max_retries:
                await asyncio.sleep(1)
                return await self.fetch_html(url, retry_count + 1)
            return None
    
    @abstractmethod
    async def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Méthode abstraite pour extraire le contenu"""
        pass
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extrait les métadonnées standards (OpenGraph, JSON-LD)"""
        metadata = {}
        
        # Open Graph
        og_properties = ['title', 'description', 'type', 'url', 'image']
        for prop in og_properties:
            tag = soup.find('meta', property=f'og:{prop}')
            if tag:
                metadata[f'og:{prop}'] = tag.get('content')
        
        # JSON-LD (riche pour les articles)
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if '@type' in data and data['@type'] == 'NewsArticle':
                        metadata['jsonld'] = data
                        break
            except:
                continue
                
        return metadata