import aiohttp
import asyncio
import aiosqlite
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import sys
import json
from io import BytesIO
from lxml import etree
from email.utils import parsedate_to_datetime
"""
TODO :
[ ] Autres accès a la BDD, en ecriture (ajout manuel d'articles, d'url rss,...)
[ ] gestion mémoire (backup,...)

"""
async def parse_feed_lxml(content):
    """Parse RSS/Atom feed using lxml (much faster than feedparser)"""
    loop = asyncio.get_event_loop()
    
    parser = etree.XMLParser(recover=True)
    tree = await loop.run_in_executor(None, etree.parse, BytesIO(content), parser)
    
    entries = []
    for item in tree.xpath('//item') + tree.xpath('//entry'):
        # Récupérer la date brute
        pub_date_str = item.findtext('pubDate') or item.findtext('published')
        
        # Parser la date
        published_parsed = None
        if pub_date_str:
            try:
                # Pour les dates RFC 2822 (RSS)
                dt = parsedate_to_datetime(pub_date_str)
                published_parsed = dt.timetuple()
            except:
                try:
                    # Pour les dates ISO 8601 (Atom)
                    dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    published_parsed = dt.timetuple()
                except:
                    pass
        
        entry = {
            'title': item.findtext('title', ''),
            'link': item.findtext('link', ''),
            'summary': item.findtext('description') or item.findtext('summary') or '',
            'published': pub_date_str,
            'published_parsed': published_parsed,  # Ajout du champ parsé
        }
        entries.append(entry)
    
    return {'entries': entries}


class AsyncFetcher:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path):
        if self._initialized:
            return
        self.config_path = config_path
        self._initialized = True
        self.session = None
        self.conn = None
        self.config = self._load_config()
        self._running = False
        self.logger = self._setup_logging()

    def _setup_logging(self): 
        logs_folder = self.config.get('logs_folder', '')   
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s() ] | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        file_handler = RotatingFileHandler(
            f'{logs_folder}/rss_fetcher.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        error_handler = RotatingFileHandler(
            f'{logs_folder}/rss_errors.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.INFO)
        error_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        
        return root_logger

    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = await aiohttp.ClientSession()
        return self.session
    
    async def _get_conn(self):
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.config.get('db_path'))
            self.conn.row_factory = aiosqlite.Row
        return self.conn
    
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _init_db(self, conn):
        #On initialise la table des feeds, a partir d'un fichier texte UNIQUE contenant toutes les URL
        self.logger.debug("Setting up Database ...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS feeds (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        URL TEXT UNIQUE NOT NULL,
        LastState INTEGER DEFAULT 1,
        last_modified TEXT,
        etag TEXT
        )
        """)
        await conn.commit()
        self.logger.debug("parsing RSS txt file ...")
        with open(self.config.get('rss_path')) as f:
            list_urls = f.readlines()
        self.logger.info(f"Done!, extracted {len(list_urls)} URLs from file")
        self.logger.debug("Saving to database ...")
        await conn.executemany("INSERT OR IGNORE INTO feeds (URL) VALUES (?)",[(url,) for url in list_urls])
        await conn.commit()
        self.logger.debug("feeds database fully initiated")

        #On initialise la table articles
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                ID_RSS INTEGER NOT NULL,
                Article_URL TEXT NOT NULL,
                Title TEXT NOT NULL,
                Description TEXT,
                Date INTEGER,
                FOREIGN KEY (ID_RSS) REFERENCES feeds(ID)
            )
        """)

        await self.create_indexes(conn)

    async def fetch_url(self, url, session):
        #timeSync0 = time.time()
        conn = await self._get_conn()
        ###GESTION DES HEADER, SAVOIR SI LE FLUX RSS A ETE MODIFIE
        cur = await conn.execute("SELECT last_modified, etag FROM feeds WHERE URL = ?", (url,))
        meta = await cur.fetchone()
        last_modified = meta[0] if meta else None
        etag = meta[1] if meta else None
        headers = {}
        if last_modified:
            headers['If-Modified-Since'] = last_modified
        if etag:
            headers['If-None-Match'] = etag

        #async with session.head(url, headers=headers) as response:
        #    if response.status == 304:  # Not Modified
        #        #logger.debug(f"Feed not modified: {url}")
        #        self.logger.info(f"RSS feed not modified : {url}")
        #        return "No new content in feed"
            
        async with session.get(url,headers=headers) as response:
            if response.status == 304:  # Not Modified
                #logger.debug(f"Feed not modified: {url}")
                self.logger.debug(f"RSS feed not modified : {url}")
                return {"message":"No new content in feed",
                        "num_articles":0}
            output = await response.read()
            etag = response.headers.get('ETag')
            last_modified = response.headers.get('Last-Modified')
            #Ici on implémente le parsing du XML de la page, sachant que la gestion des erreurs se fait dans les fonctions au dessus
            #feed = feedparser.parse(output)
            feed = await parse_feed_lxml(output)
            try:
                #await self.save_article(url, feed.entries, etag, last_modified)
                #await self.save_article(url, feed['entries'], etag, last_modified)
                num_articles, num_skipped = await self.save_articles_bulk(url, feed["entries"], etag, last_modified)
                await conn.execute("UPDATE feeds SET LastState=1 WHERE URL=?", (url,)) #Si tout a bien marché, le lastState est à 1
                await conn.commit()
                return {"message":"Articles saved successfuly",
                        "num_articles":num_articles,
                        "skipped":num_skipped}
            except Exception as e:
                await conn.execute("UPDATE feeds SET LastState=0 WHERE URL=?",(url,)) #Si tout a pas bien marché, le lastState est à 0
                await conn.commit() 
                self.logger.warning(f"An error occured while saving articles to DB {e}")
                return {"message":f"Exception occured while saving article to db : {e}",
                        "num_articles":0,
                        "skipped_error":len(feed['entries'])}
            #timeMeasure.timeSync += time.time() - timeSync0
            #return  output

    async def fetch_url_retry(self, url, session):
        max_retries = self.config.get('max_connection_retries', 4)
        for attempt in range(max_retries):
            try:
                response = await self.fetch_url(url, session)
                if response is None:
                    raise ValueError(f"Erreur, pas de réponse à l'addresse {url}")
                return response
            except (aiohttp.ClientError, ValueError) as e:
                if attempt == max_retries - 1:
                    #TODO : implémenter logger
                    self.logger.warning(f"Request at {url} failed after {max_retries} attempts ... {e}")
                    return None
                backoff = 2**attempt
                await asyncio.sleep(backoff)

    async def fetch_url_semaphore(self,url, semaphore, session):
        async with semaphore:
            self.logger.debug(f"Fetching URL {url}")
            response = await self.fetch_url_retry(url, session)
            if response:
                self.logger.debug(f"{url} Response from fetching feed : {response.get('message')}")
                return response
            return None

    async def save_article(self, feed_url ,entries, etag, last_modified):
        #Mise à jour des meta des URL
        conn = await self._get_conn()
        await conn.execute("UPDATE feeds SET etag=?, last_modified=? WHERE URL = ?", (etag, last_modified, feed_url))
        #Enregistrement de tout
        for entry in entries:
            title = entry.get('title', 'No title')
            cur = await conn.execute("SELECT 1 FROM articles WHERE Title=? LIMIT 1", (title,))
            exists = await cur.fetchone()
            if exists:
                self.logger.debug(f"Artcle {title} already exists in database, skipping...")
                continue
            self.logger.debug(f"Saved article data {title}")
            summary = entry.get('summary', entry.get('description', ''))
            article_url = entry.get('link', '')
            source = ''
            #self.logger.info(f"Article courant : RSS feed: {feed_url}, SOURCE : {source}")
            published = entry.get('published_parsed') or entry.get('updated_parsed')
            if published:
                published = datetime(*published[:6])
            try:
                await conn.execute("INSERT INTO articles (URL_RSS,Source,Article_URL,Title,Description,Date) VALUES (?,?,?,?,?,?)",(feed_url,source, article_url, title, summary, published))
            except Exception as e:
                raise Exception(f"Exception : {e} source : {(feed_url,source, article_url, title, summary, published)}")

    async def save_articles_bulk(self, feed_url, entries, etag, last_modified):
        conn = await self._get_conn()
        await conn.execute("UPDATE feeds SET etag=?, last_modified=? WHERE URL = ?", (etag, last_modified, feed_url))
        titles = [entry.get('title', 'No title') for entry in entries]
        placeholders = ','.join(['?']*len(titles))
        cur = await conn.execute(f"SELECT Title FROM articles WHERE Title in ({placeholders})", titles)
        existing_titles = {row['title'] for row in await cur.fetchall()}
        cur = await conn.execute("SELECT ID FROM feeds WHERE URL = ?", (feed_url,))
        row = await cur.fetchone()
        id_rss = row['ID'] if row else 99999
        #self.logger.info(id_rss)

        #Péparation du bulk d'articles àinsérer d'un coup
        to_insert = []
        for entry in entries:
            title = entry.get('title', 'No title')
            if title in existing_titles:
                self.logger.debug(f"Artcle {title} already exists in database, skipping...")
                continue
            self.logger.debug(f"Saved article data {title}")
            summary = entry.get('summary', entry.get('description', ''))
            article_url = entry.get('link', '')
            published = entry.get('published_parsed') or entry.get('updated_parsed')
            if published:
                published = datetime(*published[:6])
                published_ts = int(published.timestamp())
            else:
                published = datetime.now()
                published_ts = int(published.timestamp())
            to_insert.append((id_rss, article_url, title, summary, published_ts))
        self.logger.debug(f"Id RSS : {id_rss} | articles stored : {len(to_insert)} | Skipped duplicates: {len(entries) - len(to_insert)}")
        if to_insert:
            await conn.executemany("INSERT INTO articles (ID_RSS,Article_URL,Title,Description,Date) VALUES (?,?,?,?,?)",to_insert)
        await conn.commit()
        return len(to_insert), len(entries) - len(to_insert)

    async def process_batch(self, url_batch, batch_id):
        time0 = time.time()
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.config.get('semaphore_count', 10))
            tasks = [self.fetch_url_semaphore(url[0], semaphore, session) for url in url_batch]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            tot_articles = 0
            tot_skipped = 0
            tot_err = 0
            for response in responses:
                tot_articles += response.get('num_articles',0)
                tot_skipped += response.get('skipped',0)
                tot_err += response.get("skipped_error", 0)
            time1 = time.time()
            self.logger.info(f"{tot_articles} added to database | {tot_skipped} skipped (duplicates) | {tot_err} unsaved (errors)")
            self.logger.info(f"Elapsed time for {len(url_batch)} RSS feeds in batch ID {batch_id}: {time1-time0}")

    async def create_indexes(self, conn):
        conn = await self._get_conn()
        # Index sur Title (pour vérifier les doublons rapidement)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_title 
            ON articles(Title)
        """)
        # Index sur ID_RSS (pour filtrer par feed rapidement)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_id_rss 
            ON articles(ID_RSS)
        """)
        # Index sur Date (pour trier par date rapidement)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_date 
            ON articles(Date)
        """)
        await conn.commit()
        self.logger.info("Indexes created successfully!")

    async def main(self):
        conn = await self._get_conn()
        await self._init_db(conn)
        cursor = await conn.execute("SELECT DISTINCT URL FROM feeds WHERE LastState = 1") #On récupère tous les URLs qui sont VALIDES (i.e qui ont fonctionné à l'itération n-1)
        URLs = await cursor.fetchall()
        batch_size = self.config.get("url_batch_size", 300)
        batches = [URLs[i:i + batch_size] for i in range(0, len(URLs), batch_size)]
        
        for batch_id, batch in enumerate(batches):
            await self.process_batch(batch, batch_id)
            await conn.execute("VACUUM")
            await conn.commit()
            await asyncio.sleep(0.5)
        await self.cleanup()

    async def query_db(self, conditions, limit, order_by = "ID DESC"):
        conn = await self._get_conn()
        query = "SELECT * FROM articles"
        params = []
        #Ajout des conditions WHERE à la requête
        if conditions:
            allowed_columns = {"ID", "ID_RSS", "Article_URL", "Title", "Description", "Date"}
            where_clauses = []
            for column, value in conditions.items():
                if column not in allowed_columns:
                    raise ValueError(f"Colonne non autorisée: {column}")
                where_clauses.append(f"{column} = ?")
                params.append(value)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
        
        #Ajout du ORDER BY
        allowed_orders = {"ID", "ID DESC", "Title", "Title DESC", "Date", "Date DESC"}
        if order_by not in allowed_orders:
            self.logger.warning(f"ORDER BY clause not authorized in query_db() {order_by}")
            order_by = "ID_DESC"
        query += f" ORDER BY {order_by}"

        #Ajout de LIMIT
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()     
            return [dict(row) for row in rows]       

    async def run_periodic(self, interval_minutes = None):
        if self._running:
            self.logger.info("Le fetcher est déjà en cours d'exécution")
            return
        self._running = True
        interval = interval_minutes or self.config.get("interval_minutes", 20)
        interval_seconds = interval*60
        self.logger.info(f"Démarrage du fetcher périodique (intervalle {interval} minutes)")
        try:
            while self._running:
                await self.main()
                self.logger.info(f"prochaine exécution dans {interval} minutes")
                await asyncio.sleep(interval_seconds)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.warning("Interruption clavier")
            self._running = False #on arrèete l'exécution!
        finally:
            await self.cleanup()

    async def cleanup(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Session HTTP fermée")
        if self.conn:
            await self.conn.close()
            self.logger.info("connexion DB fermée")
    
    def stop(self):
        self._running = False

if __name__ == '__main__':
    fetcher = AsyncFetcher(r'app\config\config.json')
    #profiler = cProfile.Profile()

    #profiler.enable()

    asyncio.run(fetcher.run_periodic())
    #profiler.disable()


    #profiler.dump_stats('single_entry.profile')