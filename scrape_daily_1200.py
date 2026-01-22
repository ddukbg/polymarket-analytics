from polymarket_api import PolymarketAPI
import logging
from pathlib import Path
import time

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    filename='logs/scrapes.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def scrape_with_retry(max_retries=2, delay=30):
    """Attempt full scrape with retries at the top level"""
    for attempt in range(max_retries):
        try:
            logging.info(f"12:00 Daily scrape starting (attempt {attempt + 1}/{max_retries})")
            
            api = PolymarketAPI()
            # The function now has built-in retries per category (max_retries=3, retry_delay=5)
            snapshot_df = api.scrape_leaderboard_snapshot(
                limit=100, 
                timeframes=['day'],
                max_retries=3,  # Retries per category
                retry_delay=5   # Delay between category retries
            )
            
            # Success
            logging.info(f"12:00 Daily scrape completed: {len(snapshot_df)} records")
            
            # Warn if we got significantly fewer than expected (10 categories * 100 limit = 1000)
            if len(snapshot_df) < 800:
                logging.warning(f"Low record count: {len(snapshot_df)} (expected ~1000)")
            
            return snapshot_df
            
        except Exception as e:
            logging.error(f"12:00 Daily scrape attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
            
            if attempt < max_retries - 1:
                logging.info(f"Waiting {delay} seconds before full retry...")
                time.sleep(delay)
            else:
                logging.error(f"12:00 Daily scrape failed after {max_retries} full attempts")
                raise

# Run the scrape
try:
    result = scrape_with_retry(max_retries=2, delay=30)
    logging.info("12:00 scrape job completed successfully")
except Exception as e:
    logging.error(f"12:00 scrape job completely failed: {e}")