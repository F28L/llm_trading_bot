# config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TradingConfig:
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    
    # Portfolio Settings
    initial_cash: float = 100000.0
    
    # Risk Management
    max_position_size: float = 0.20  # 10% max per position
    max_daily_loss: float = 0.05     # 5% max daily loss
    max_sector_concentration: float = 0.30  # 30% max per sector
    
    # Trading Settings
    cycle_interval_minutes: int = 3
    min_signal_confidence: float = 0.6
    
    # Database
    database_path: str = "trading_agent.db"
    
    # News Sources
    news_sources: list = None
    
    def __post_init__(self):
        if self.news_sources is None:
            self.news_sources = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html"
            ]
