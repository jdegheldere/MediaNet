import asyncio
import aiohttp
import feedparser
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RSSCollector:
    def __init__(self, db_path: str = 'rss_articles.db'):
        self.db_path = db_path
        self.init_database()
        # In-memory cache for feed metadata (minimal memory usage)
        self.feed_cache = {}
        
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    last_checked TIMESTAMP,
                    last_modified TEXT,
                    etag TEXT
                );
                
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    article_url TEXT UNIQUE NOT NULL,
                    published TIMESTAMP,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feed_id) REFERENCES feeds (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(article_url);
                CREATE INDEX IF NOT EXISTS idx_feeds_last_checked ON feeds(last_checked);
            """)
            conn.commit()
            
    def load_feed_urls(self, txt_files: List[str]) -> List[str]:
        """Load RSS feed URLs from text files"""
        urls = set()
        for file_path in txt_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        url = line.strip()
                        if url and not url.startswith('#'):
                            urls.add(url)
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}")
        return list(urls)
    

    
    def process_entries(self, conn, feed_url: str, entries: List, 
                    feed_id: int, etag: str = None, 
                    last_modified: str = None):
        """Process and store new entries from a feed"""
        # Remove the "with sqlite3.connect" line - use the passed connection
        new_articles = 0
        
        for entry in entries:
            try:
                # Extract article data
                title = entry.get('title', 'No title')
                summary = entry.get('summary', entry.get('description', ''))
                article_url = entry.get('link', '')
                published = entry.get('published_parsed') or entry.get('updated_parsed')
                
                if published:
                    published = datetime(*published[:6])
                
                # Check if article already exists
                cursor = conn.execute(
                    "SELECT id FROM articles WHERE article_url = ?",
                    (article_url,)
                )
                
                if not cursor.fetchone():
                    # Insert new article
                    conn.execute(
                        """INSERT INTO articles 
                        (feed_id, title, summary, article_url, published)
                        VALUES (?, ?, ?, ?, ?)""",
                        (feed_id, title[:500], summary[:2000], article_url, published)
                    )
                    new_articles += 1
                    
            except Exception as e:
                logger.error(f"Error processing entry: {e}")
                continue
        
        # Update feed metadata
        conn.execute(
            """UPDATE feeds 
            SET last_checked = ?, last_modified = ?, etag = ?
            WHERE url = ?""",
            (datetime.now(), last_modified, etag, feed_url)
        )
        
        # Don't commit here - let the caller handle it
        return new_articles
    
    async def process_feeds(self, feed_urls: List[str], 
                      batch_size: int = 10):
        """Process multiple feeds in batches to avoid resource exhaustion"""
        # Update or insert feed URLs in database (do this once for all URLs)
        with sqlite3.connect(self.db_path) as conn:
            for url in feed_urls:
                conn.execute(
                    "INSERT OR IGNORE INTO feeds (url, last_checked) VALUES (?, ?)",
                    (url, datetime.now() - timedelta(days=1))
                )
            conn.commit()
        
        total_new = 0
        total_batches = (len(feed_urls) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(feed_urls)} feeds in {total_batches} batches of {batch_size}")
        
        # Process feeds in batches
        for batch_num in range(0, len(feed_urls), batch_size):
            batch = feed_urls[batch_num:batch_num + batch_size]
            current_batch = (batch_num // batch_size) + 1
            
            logger.info(f"Processing batch {current_batch}/{total_batches}")
            
            # Fetch feeds in parallel for this batch only
            connector = aiohttp.TCPConnector(
                limit_per_host=3, 
                limit=batch_size,
                force_close=True
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [self.fetch_feed(session, url) for url in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results from this batch
            batch_new = await self.process_batch_results(results)
            total_new += batch_new
            
            logger.info(f"Batch {current_batch}/{total_batches} complete: {batch_new} new articles")
            
            # Small delay between batches to ensure resources are released
            if current_batch < total_batches:
                await asyncio.sleep(0.5)
        
        logger.info(f"Total processing complete: {total_new} new articles from {len(feed_urls)} feeds")
        return total_new

    async def process_batch_results(self, results: List) -> int:
        """Process results from a single batch of feeds"""
        total_new = 0
        
        # Collect all valid results first
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue
            
            if result:
                valid_results.append(result)
        
        if not valid_results:
            return 0
        
        # Process all results with a single database connection
        with sqlite3.connect(self.db_path) as conn:
            for result in valid_results:
                try:
                    # Get feed ID
                    cursor = conn.execute(
                        "SELECT id FROM feeds WHERE url = ?",
                        (result['url'],)
                    )
                    feed_row = cursor.fetchone()
                    
                    if not feed_row:
                        logger.error(f"Feed not found in database: {result['url']}")
                        continue
                    
                    feed_id = feed_row[0]
                    
                    new_articles = self.process_entries(
                        conn,  # ADD THIS - pass the connection
                        result['url'], 
                        result['entries'],
                        feed_id,
                        result.get('etag'),
                        result.get('last_modified')
                    )
                                        
                    if new_articles > 0:
                        logger.info(f"Added {new_articles} new articles from {result['url']}")
                        total_new += new_articles
                        
                except Exception as e:
                    logger.error(f"Error processing result for {result['url']}: {e}")
                    continue
            conn.commit()
        return total_new


    async def fetch_feed(self, session: aiohttp.ClientSession, 
                        feed_url: str) -> Optional[Dict]:
        """Fetch and parse a single RSS feed"""
        try:
            # Get feed metadata from DB
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT last_modified, etag FROM feeds WHERE url = ?",
                    (feed_url,)
                )
                result = cursor.fetchone()
                last_modified = result[0] if result else None
                etag = result[1] if result else None
            
            # Prepare headers for conditional GET
            headers = {}
            if last_modified:
                headers['If-Modified-Since'] = last_modified
            if etag:
                headers['If-None-Match'] = etag
            
            # Fetch feed
            async with session.get(feed_url, headers=headers, 
                                timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 304:  # Not Modified
                    logger.debug(f"Feed not modified: {feed_url}")
                    return None
                
                content = await response.read()
                etag = response.headers.get('ETag')
                last_modified = response.headers.get('Last-Modified')
                
                # Parse feed
                feed = feedparser.parse(content)
                
                if feed.bozo:  # Check for parsing errors
                    logger.warning(f"Error parsing feed {feed_url}: {feed.bozo_exception}")
                    return None
                
                return {
                    'url': feed_url,
                    'feed': feed,
                    'etag': etag,
                    'last_modified': last_modified,
                    'entries': feed.entries
                }
                
        except Exception as e:
            logger.error(f"Error fetching {feed_url}: {e}")
            return None
    
    async def run_continuously(self, txt_files: List[str], 
                             check_interval_minutes: int = 5):
        """Run the collector continuously"""
        logger.info("Starting RSS collector in continuous mode")
        
        while True:
            try:
                # Load feed URLs
                feed_urls = self.load_feed_urls(txt_files)
                logger.info(f"Loaded {len(feed_urls)} feed URLs")
                
                # Process feeds
                start_time = datetime.now()
                new_articles = await self.process_feeds(feed_urls)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"Cycle completed: {new_articles} new articles in {elapsed:.2f}s")
                
                # Wait before next cycle
                logger.info(f"Waiting {check_interval_minutes} minutes before next check")
                await asyncio.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying


class MemoryEfficientCollector(RSSCollector):
    """Enhanced collector with better memory management"""
    
    def __init__(self, db_path: str = 'rss_articles.db', 
                 cache_size: int = 1000):
        super().__init__(db_path)
        self.cache_size = cache_size
        self.url_cache = set()  # Cache of recently seen URLs
        
    def load_recent_urls(self, hours: int = 24):
        """Load recently seen URLs into memory cache"""
        cutoff = datetime.now() - timedelta(hours=hours)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT article_url FROM articles WHERE fetched_at > ?",
                (cutoff,)
            )
            urls = {row[0] for row in cursor.fetchall()}
            
            # Keep only cache_size most recent
            self.url_cache = set(list(urls)[-self.cache_size:])
    
    def is_url_cached(self, url: str) -> bool:
        """Check if URL is in memory cache"""
        return url in self.url_cache

def main():
    """Main entry point"""
    # Example usage
    txt_files = [
        'feeds/technology.txt',
        'feeds/politics.txt', 
        'feeds/business.txt'
    ]
    
    # Create collector and run
    collector = RSSCollector('rss_articles.db')
    
    # Run once (for testing)
    # asyncio.run(collector.process_feeds(collector.load_feed_urls(txt_files)))
    
    # Run continuously
    asyncio.run(collector.run_continuously(txt_files, check_interval_minutes=5))

if __name__ == "__main__":
    main()