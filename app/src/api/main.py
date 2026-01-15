from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Union
import logging

#from scraper.news_scraper import NewsScraper
#from processors.text_processor import TextProcessor

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Media Analyzer API",
    description="API d'analyse d'articles médiatiques",
    version="1.0.0"
)

# Modèles de requêtes/réponses
class ArticleRequest(BaseModel):
    """Modèle pour la requête d'analyse d'article"""
    url: Optional[HttpUrl] = None
    title: Optional[str] = None
    text: Optional[str] = None
    html_content: Optional[str] = None
    bypass_paywall: bool = True
    full_analysis: bool = False

class ArticleResponse(BaseModel):
    """Modèle de base pour la réponse"""
    success: bool
    article_id: str
    status: str
    message: Optional[str] = None
    data: Optional[dict] = None

class ScrapedData(BaseModel):
    """Données scrapées d'un article"""
    url: str
    title: str
    author: Optional[List[str]] = None
    publication_date: Optional[str] = None
    text: str
    html: Optional[str] = None
    summary: Optional[str] = None
    language: str = "fr"
    word_count: int
    metadata: dict = {}
    images: List[str] = []

# Initialisation des composants
scraper = NewsScraper()
text_processor = TextProcessor()

@app.post("/analyze", response_model=ArticleResponse)
async def analyze_article(
    request: ArticleRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Point d'entrée principal pour analyser un article
    """
    try:
        # Validation: au moins un identifiant doit être fourni
        if not any([request.url, request.text, request.html_content]):
            raise HTTPException(
                status_code=400,
                detail="Fournissez au moins: URL, texte ou HTML"
            )
        
        article_id = generate_article_id()
        
        # Si URL fournie, on scrape
        if request.url:
            logger.info(f"Scraping de l'URL: {request.url}")
            
            scraped_data = await scraper.scrape_article(
                url=str(request.url),
                bypass_paywall=request.bypass_paywall
            )
            
            # Traitement du texte
            processed_text = text_processor.clean_text(scraped_data.text)
            scraped_data.text = processed_text
            
            # Si full_analysis est demandé, lancer l'analyse complète en background
            if request.full_analysis and background_tasks:
                background_tasks.add_task(
                    run_full_analysis,
                    article_id,
                    scraped_data
                )
                return ArticleResponse(
                    success=True,
                    article_id=article_id,
                    status="processing",
                    message="Analyse complète en cours"
                )
            
            return ArticleResponse(
                success=True,
                article_id=article_id,
                status="partial",
                data=scraped_data.dict()
            )
        
        # Si texte brut fourni
        elif request.text:
            processed_text = text_processor.clean_text(request.text)
            
            return ArticleResponse(
                success=True,
                article_id=article_id,
                status="ready",
                data={
                    "text": processed_text,
                    "word_count": len(processed_text.split())
                }
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{article_id}")
async def get_analysis_status(article_id: str):
    """Vérifier le statut d'une analyse"""
    # À implémenter: vérifier en base de données
    return {"status": "completed", "article_id": article_id}

def generate_article_id() -> str:
    """Génère un ID unique pour l'article"""
    import uuid
    import hashlib
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"ART_{timestamp}_{unique_id}"

async def run_full_analysis(article_id: str, article_data: ScrapedData):
    """Fonction pour lancer l'analyse complète en arrière-plan"""
    # À implémenter avec les autres modules
    logger.info(f"Début de l'analyse complète pour {article_id}")
    # ... analyse sémantique, fact-checking, etc.