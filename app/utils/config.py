"""
Configuration management
"""
import os
from pathlib import Path
import json

def load_config():
    """Load configuration from environment and config file"""
    config = {
        # Default values
        'max_concurrent': int(os.getenv('MAX_CONCURRENT', '20')),
        'check_interval': int(os.getenv('CHECK_INTERVAL', '5')),
        'cache_size': int(os.getenv('CACHE_SIZE', '1000')),
        'db_path': os.getenv('DB_PATH', '/app/data/rss_articles.db'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    }
    
    # Try to load from config file
    config_file = Path('/app/config/config.json')
    if config_file.exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    return config