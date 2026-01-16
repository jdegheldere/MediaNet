#!/usr/bin/env python3
"""
Main entry point for MediaNet RSS Collector
Run with: python -m app.main
"""

import asyncio
from logging import handlers
import signal
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from collector.rss_collector import RSSCollector, MemoryEfficientCollector
from utils.config import load_config

class Application:
    """Main application controller"""
    
    def __init__(self):
        self.collector = None
        self.is_running = False
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = load_config()
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Configure logging to file and console"""
        log_dir = Path('/app/logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler (rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'rss_collector.log',
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Collector-specific logger
        self.logger = logging.getLogger('RSSCollector')
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
        
        if self.collector:
            # We'll need to handle async shutdown properly
            asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Graceful shutdown procedure"""
        self.logger.info("Performing graceful shutdown...")
        # Add any cleanup tasks here
        await asyncio.sleep(1)  # Give time for pending tasks
        sys.exit(0)
    
    def get_feed_files(self):
        """Discover all feed files in feeds directory"""
        feed_dir = Path('/app/feeds')
        if not feed_dir.exists():
            self.logger.warning(f"Feeds directory not found: {feed_dir}")
            return []
        
        txt_files = list(feed_dir.glob('*.txt'))
        self.logger.info(f"Found {len(txt_files)} feed files")
        return [str(f) for f in txt_files]
    
    async def run_once(self):
        """Run a single collection cycle"""
        feed_files = self.get_feed_files()
        if not feed_files:
            self.logger.error("No feed files found. Please add .txt files to /app/feeds/")
            return
        
        # Create collector instance
        db_path = '/app/data/rss_articles.db'
        collector = MemoryEfficientCollector(db_path, cache_size=1000)
        
        # Load recent URLs for duplicate checking
        collector.load_recent_urls(hours=24)
        
        # Process feeds
        new_articles = await collector.process_feeds(
            collector.load_feed_urls(feed_files),
            max_concurrent=int(self.config.get('max_concurrent', 20))
        )
        
        self.logger.info(f"Cycle complete: {new_articles} new articles")
        return new_articles
    
    async def run_continuous(self):
        """Run continuous collection"""
        self.is_running = True
        
        feed_files = self.get_feed_files()
        if not feed_files:
            self.logger.error("No feed files found. Exiting.")
            return
        
        # Create collector instance
        db_path = '/app/data/rss_articles.db'
        self.collector = MemoryEfficientCollector(db_path, cache_size=1000)
        
        # Load recent URLs
        self.collector.load_recent_urls(hours=24)
        
        self.logger.info("Starting RSS collector in continuous mode")
        self.logger.info(f"Database: {db_path}")
        self.logger.info(f"Feed files: {len(feed_files)}")
        
        try:
            await self.collector.run_continuously(
                feed_files,
                check_interval_minutes=int(self.config.get('check_interval', 5))
            )
        except asyncio.CancelledError:
            self.logger.info("Collection cancelled")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.is_running = False
    
    def run(self, mode='continuous'):
        """Run the application"""
        self.logger.info(f"Starting MediaNet RSS Collector in {mode} mode")
        
        try:
            if mode == 'once':
                asyncio.run(self.run_once())
            else:
                asyncio.run(self.run_continuous())
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self.logger.info("Application stopped")

def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RSS Article Collector')
    parser.add_argument('--mode', choices=['once', 'continuous'], 
                       default='continuous', help='Run mode')
    parser.add_argument('--interval', type=int, default=5,
                       help='Check interval in minutes (continuous mode only)')
    
    args = parser.parse_args()
    
    # Set environment variable for interval
    if args.interval:
        os.environ['CHECK_INTERVAL'] = str(args.interval)
    
    app = Application()
    app.run(mode=args.mode)

if __name__ == "__main__":
    main()