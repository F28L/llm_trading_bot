# config.py
import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TradingConfig:
    """Configuration class for the AI Trading Bot"""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    
    # Portfolio Settings
    initial_cash: float = 100000.0
    
    # Risk Management
    max_position_size: float = 0.10  # 10% max per position
    max_daily_loss: float = 0.05     # 5% max daily loss
    max_sector_concentration: float = 0.30  # 30% max per sector
    
    # Trading Settings
    cycle_interval_minutes: int = 15
    min_signal_confidence: float = 0.6
    
    # Stock Universe Settings  
    max_universe_size: int = 50
    min_market_cap: float = 1e9  # $1B minimum market cap
    max_stock_price: float = 1000  # Max $1000 per share
    min_daily_volume: float = 1e6  # 1M shares daily volume
    excluded_sectors: Optional[List[str]] = None  # Sectors to avoid
    include_etfs: bool = True
    use_trending_discovery: bool = True
    
    # Database
    database_path: str = "trading_agent.db"
    
    # News Sources
    news_sources: Optional[List[str]] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"
    
    # Advanced Trading Settings
    enable_options_trading: bool = False
    enable_crypto_trading: bool = False
    enable_forex_trading: bool = False
    
    # Performance Settings
    cache_duration_seconds: int = 60
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    
    def __post_init__(self):
        """Initialize default values for list fields"""
        if self.news_sources is None:
            self.news_sources = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.marketwatch.com/marketwatch/marketpulse/",
                "https://feeds.benzinga.com/benzinga",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom"
            ]
        
        if self.excluded_sectors is None:
            self.excluded_sectors = []  # Can add sectors like ['Real Estate', 'Utilities']
            
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required API keys
        if not self.openai_api_key:
            errors.append("OpenAI API key is required")
            
        # Validate numerical ranges
        if not 0 < self.max_position_size <= 1:
            errors.append("max_position_size must be between 0 and 1")
            
        if not 0 < self.max_daily_loss <= 1:
            errors.append("max_daily_loss must be between 0 and 1")
            
        if not 0 < self.max_sector_concentration <= 1:
            errors.append("max_sector_concentration must be between 0 and 1")
            
        if self.initial_cash <= 0:
            errors.append("initial_cash must be positive")
            
        if self.cycle_interval_minutes <= 0:
            errors.append("cycle_interval_minutes must be positive")
            
        if not 0 <= self.min_signal_confidence <= 1:
            errors.append("min_signal_confidence must be between 0 and 1")
            
        if self.max_universe_size <= 0:
            errors.append("max_universe_size must be positive")
            
        if self.min_market_cap <= 0:
            errors.append("min_market_cap must be positive")
            
        if self.max_stock_price <= 0:
            errors.append("max_stock_price must be positive")
            
        if self.min_daily_volume <= 0:
            errors.append("min_daily_volume must be positive")
            
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True
    
    def get_risk_settings(self) -> dict:
        """Get risk management settings as a dictionary"""
        return {
            "max_position_size": self.max_position_size,
            "max_daily_loss": self.max_daily_loss,
            "max_sector_concentration": self.max_sector_concentration
        }
    
    def get_universe_criteria(self) -> dict:
        """Get stock universe filtering criteria"""
        return {
            "max_symbols": self.max_universe_size,
            "min_market_cap": self.min_market_cap,
            "max_price": self.max_stock_price,
            "min_volume": self.min_daily_volume,
            "exclude_sectors": self.excluded_sectors,
            "include_etfs": self.include_etfs
        }
    
    def print_summary(self):
        """Print a summary of current configuration"""
        print("🔧 Trading Bot Configuration Summary")
        print("=" * 50)
        print(f"💰 Initial Cash: ${self.initial_cash:,.2f}")
        print(f"🛡️  Max Position Size: {self.max_position_size:.1%}")
        print(f"📉 Max Daily Loss: {self.max_daily_loss:.1%}")
        print(f"🏢 Max Sector Concentration: {self.max_sector_concentration:.1%}")
        print(f"⏱️  Cycle Interval: {self.cycle_interval_minutes} minutes")
        print(f"🎯 Min Signal Confidence: {self.min_signal_confidence:.1f}")
        print(f"🌐 Max Universe Size: {self.max_universe_size} stocks")
        print(f"💼 Min Market Cap: ${self.min_market_cap/1e9:.1f}B")
        print(f"💵 Max Stock Price: ${self.max_stock_price}")
        print(f"📊 Min Daily Volume: {self.min_daily_volume/1e6:.1f}M shares")
        print(f"📰 News Sources: {len(self.news_sources)} feeds")
        if self.excluded_sectors:
            print(f"🚫 Excluded Sectors: {', '.join(self.excluded_sectors)}")
        print("=" * 50)