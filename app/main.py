from utils.AsyncFetcher import AsyncFetcher
import asyncio



if __name__ == '__main__':
    fetcher = AsyncFetcher('config/config.json')
    asyncio.run(fetcher.run_periodic())