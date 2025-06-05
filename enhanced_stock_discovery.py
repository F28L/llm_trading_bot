import yfinance as yf
import requests
import pandas as pd
from typing import List, Dict, Set
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class StockMention:
    symbol: str
    company_name: str
    mentions: int
    sentiment_score: float
    sources: List[str]
    market_cap: float = 0.0
    volume: float = 0.0
    price_change_pct: float = 0.0

class StockUniverseManager:
    def __init__(self):
        # Base universe of major stocks to consider
        self.base_universe = {
            # Tech Giants
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            
            # Financial
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp.',
            'WFC': 'Wells Fargo & Company',
            'GS': 'Goldman Sachs Group Inc.',
            
            # Healthcare
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'ABBV': 'AbbVie Inc.',
            
            # Consumer
            'KO': 'Coca-Cola Company',
            'PEP': 'PepsiCo Inc.',
            'WMT': 'Walmart Inc.',
            'HD': 'Home Depot Inc.',
            
            # Energy
            'XOM': 'Exxon Mobil Corporation',
            'CVX': 'Chevron Corporation',
            
            # Industrial
            'BA': 'Boeing Company',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric Company',
        }
        
        # ETFs for sector exposure
        self.etf_universe = {
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'IWM': 'iShares Russell 2000 ETF',
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund',
            'XLV': 'Health Care Select Sector SPDR Fund',
            'XLI': 'Industrial Select Sector SPDR Fund',
        }
        
        self.trending_symbols = set()
        self.stock_mentions = {}
        
    def get_sp500_symbols(self) -> List[str]:
        """Get current S&P 500 symbols from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (remove dots, etc.)
            cleaned_symbols = []
            for symbol in symbols:
                if isinstance(symbol, str):
                    # Handle symbols with dots (like BRK.B)
                    cleaned_symbol = symbol.replace('.', '-')
                    cleaned_symbols.append(cleaned_symbol)
            
            logger.info(f"Retrieved {len(cleaned_symbols)} S&P 500 symbols")
            return cleaned_symbols[:100]  # Limit for performance
            
        except ImportError as e:
            logger.warning(f"Missing dependency for S&P 500 parsing: {e}")
            logger.info("Using base universe instead of S&P 500 symbols")
            return list(self.base_universe.keys())
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
            logger.info("Falling back to base universe")
            return list(self.base_universe.keys())
    
    def get_trending_stocks_from_reddit(self) -> List[str]:
        """Get trending stocks from Reddit mentions (placeholder - would need Reddit API)"""
        # This is a placeholder - you'd implement with Reddit API
        # For now, return some commonly discussed stocks
        reddit_trending = [
            'GME', 'AMC', 'PLTR', 'BB', 'NOK', 'WISH', 'CLOV', 'SPCE',
            'SQ', 'ROKU', 'SHOP', 'ZM', 'SNAP', 'TWTR', 'UBER', 'LYFT'
        ]
        return reddit_trending
    
    def get_trending_stocks_from_yahoo(self) -> List[str]:
        """Get trending stocks from Yahoo Finance"""
        try:
            # Yahoo Finance trending tickers endpoint
            url = "https://query2.finance.yahoo.com/v1/finance/trending/US"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                trending = []
                for item in data.get('finance', {}).get('result', [{}])[0].get('quotes', []):
                    symbol = item.get('symbol', '')
                    if symbol and len(symbol) <= 5:  # Filter out weird symbols
                        trending.append(symbol)
                
                logger.info(f"Retrieved {len(trending)} trending symbols from Yahoo")
                return trending[:20]  # Limit to top 20
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo trending stocks: {e}")
        
        return []
    
    def extract_symbols_from_news(self, news_content: str) -> Set[str]:
        """Extract stock symbols from news content using pattern matching"""
        symbols = set()
        
        # Common patterns for stock symbols in text
        patterns = [
            r'\b([A-Z]{1,5})\s+(?:stock|shares|ticker)',
            r'\$([A-Z]{1,5})\b',  # $AAPL format
            r'\(([A-Z]{1,5})\)',  # (AAPL) format
            r'\b([A-Z]{2,5})\s+(?:rose|fell|gained|lost|dropped)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, news_content.upper())
            for match in matches:
                if len(match) >= 2 and len(match) <= 5:
                    symbols.add(match)
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'NEW', 'CEO', 'CFO', 'USA', 'USD', 'SEC', 'FDA', 'API', 'CEO', 'IPO'}
        symbols = symbols - false_positives
        
        return symbols
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is a real, tradeable stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if it's a valid stock
            if (info.get('quoteType') in ['EQUITY', 'ETF'] and 
                info.get('regularMarketPrice') and 
                info.get('regularMarketPrice') > 0.01 and  # Penny stock filter
                info.get('marketCap', 0) > 100000000):  # Min $100M market cap
                return True
                
        except Exception:
            pass
        
        return False
    
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """Get basic fundamental data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return {}
            
            current_price = float(hist['Close'].iloc[-1])
            prev_price = float(hist['Close'].iloc[0]) if len(hist) > 1 else current_price
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'market_cap': info.get('marketCap', 0),
                'current_price': current_price,
                'price_change_pct': price_change_pct,
                'volume': info.get('regularMarketVolume', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
            }
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return {}
    
    def discover_tradeable_universe(self, max_symbols: int = 50) -> List[str]:
        """Discover and return a diverse universe of tradeable stocks"""
        all_symbols = set()
        
        # 1. Add base universe
        all_symbols.update(self.base_universe.keys())
        
        # 2. Add some S&P 500 stocks
        sp500_symbols = self.get_sp500_symbols()
        all_symbols.update(sp500_symbols[:30])
        
        # 3. Add trending stocks
        trending_yahoo = self.get_trending_stocks_from_yahoo()
        all_symbols.update(trending_yahoo)
        
        # 4. Add Reddit trending (placeholder)
        trending_reddit = self.get_trending_stocks_from_reddit()
        all_symbols.update(trending_reddit[:10])
        
        # 5. Add ETFs for diversification
        all_symbols.update(list(self.etf_universe.keys())[:5])
        
        # Validate symbols and get fundamentals
        valid_symbols = []
        for symbol in list(all_symbols)[:max_symbols * 2]:  # Check more than we need
            if self.validate_symbol(symbol):
                valid_symbols.append(symbol)
                if len(valid_symbols) >= max_symbols:
                    break
        
        logger.info(f"Discovered {len(valid_symbols)} valid tradeable symbols")
        return valid_symbols
    
    def filter_by_criteria(self, symbols: List[str], criteria: Dict) -> List[str]:
        """Filter stocks by various criteria"""
        filtered_symbols = []
        
        for symbol in symbols:
            fundamentals = self.get_stock_fundamentals(symbol)
            if not fundamentals:
                continue
            
            # Apply filters
            if criteria.get('min_market_cap') and fundamentals.get('market_cap', 0) < criteria['min_market_cap']:
                continue
            
            if criteria.get('max_price') and fundamentals.get('current_price', 0) > criteria['max_price']:
                continue
            
            if criteria.get('min_volume') and fundamentals.get('volume', 0) < criteria['min_volume']:
                continue
            
            if criteria.get('exclude_sectors') and fundamentals.get('sector') in criteria['exclude_sectors']:
                continue
            
            filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def get_sector_allocation(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Group symbols by sector for diversification"""
        sectors = {}
        
        for symbol in symbols:
            fundamentals = self.get_stock_fundamentals(symbol)
            sector = fundamentals.get('sector', 'Unknown')
            
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)
        
        return sectors

# Enhanced News Analyzer with dynamic symbol discovery
class EnhancedNewsAnalyzer:
    def __init__(self, openai_api_key: str):
        from main import NewsAnalyzer  # Import the original
        self.base_analyzer = NewsAnalyzer(openai_api_key)
        self.universe_manager = StockUniverseManager()
        
    async def fetch_real_financial_news(self) -> List:
        """Fetch real financial news from multiple sources"""
        news_items = []
        
        try:
            # Try to import feedparser
            import feedparser
            
            feeds = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.marketwatch.com/marketwatch/marketpulse/",
                "https://feeds.benzinga.com/benzinga",
            ]
            
            for feed_url in feeds:
                try:
                    logger.info(f"Fetching news from {feed_url}")
                    feed = feedparser.parse(feed_url)
                    
                    if not feed.entries:
                        logger.warning(f"No entries found in feed: {feed_url}")
                        continue
                    
                    for entry in feed.entries[:3]:  # Limit per source
                        # Extract symbols from content
                        content = entry.get('summary', '') + ' ' + entry.get('title', '')
                        symbols = self.universe_manager.extract_symbols_from_news(content)
                        
                        # Validate symbols
                        valid_symbols = []
                        for symbol in list(symbols)[:5]:  # Limit symbols per article
                            if self.universe_manager.validate_symbol(symbol):
                                valid_symbols.append(symbol)
                        
                        # Only add news if we found valid symbols
                        if valid_symbols:
                            from main import NewsItem
                            news_item = NewsItem(
                                title=entry.get('title', 'No title')[:200],  # Limit title length
                                content=content[:500],  # Limit content length
                                source=feed_url.split('//')[1].split('/')[0],
                                timestamp=datetime.now(),
                                symbols=valid_symbols
                            )
                            news_items.append(news_item)
                            logger.info(f"Added news item: {news_item.title[:50]}... with symbols: {valid_symbols}")
                            
                except Exception as e:
                    logger.error(f"Error parsing feed {feed_url}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("feedparser not installed, using enhanced sample news")
            
        # If no real news found or feedparser not available, use enhanced sample news
        if not news_items:
            logger.info("No real news items found, using sample news with discovered universe")
            from main import NewsItem
            
            # Get current tradeable universe
            tradeable_symbols = self.universe_manager.discover_tradeable_universe(20)
            
            sample_news = [
                NewsItem(
                    title="Market Analysis: Tech Stocks Show Mixed Performance Amid AI Developments",
                    content="Technology stocks showed mixed performance today with major players experiencing varied results as AI developments continue to reshape the sector...",
                    source="MarketWatch",
                    timestamp=datetime.now(),
                    symbols=[s for s in tradeable_symbols if s in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']][:3]
                ),
                NewsItem(
                    title="Energy Sector Rallies on Supply Chain Developments",
                    content="Energy stocks gained momentum as global supply chain developments and policy changes create new opportunities in the sector...",
                    source="Bloomberg",
                    timestamp=datetime.now(),
                    symbols=[s for s in tradeable_symbols if s in ['XOM', 'CVX', 'XLE', 'COP']][:2]
                ),
                NewsItem(
                    title="Financial Services Sector Responds to Federal Reserve Commentary",
                    content="Banking and financial services stocks moved following the latest Federal Reserve commentary on monetary policy and economic outlook...",
                    source="Reuters",
                    timestamp=datetime.now(),
                    symbols=[s for s in tradeable_symbols if s in ['JPM', 'BAC', 'WFC', 'XLF', 'GS']][:3]
                ),
                NewsItem(
                    title="Healthcare Innovations Drive Sector Performance",
                    content="Healthcare stocks advanced on news of breakthrough treatments and regulatory approvals across multiple therapeutic areas...",
                    source="Yahoo Finance",
                    timestamp=datetime.now(),
                    symbols=[s for s in tradeable_symbols if s in ['JNJ', 'PFE', 'UNH', 'ABBV', 'XLV']][:2]
                ),
            ]
            
            # Filter sample news to only include items with valid symbols
            for news in sample_news:
                if news.symbols:  # Only add if we have valid symbols
                    news_items.append(news)
        
        logger.info(f"Fetched {len(news_items)} news items covering {sum(len(item.symbols) for item in news_items)} symbol mentions")
        return news_items
    
    def generate_enhanced_trading_signals(self, news_items: List, current_positions: Dict) -> List:
        """Generate trading signals with expanded universe"""
        # Get current tradeable universe
        tradeable_universe = self.universe_manager.discover_tradeable_universe(30)
        
        # Filter to symbols mentioned in news or trending
        mentioned_symbols = set()
        for item in news_items:
            mentioned_symbols.update(item.symbols)
        
        # Combine mentioned symbols with some trending ones
        target_symbols = list(mentioned_symbols) + [s for s in tradeable_universe if s not in mentioned_symbols][:10]
        
        # Use original signal generation but with expanded context
        try:
            signals = self.base_analyzer.generate_trading_signals(news_items, current_positions)
            
            # If no signals generated, create some based on trending stocks
            if not signals and target_symbols:
                from main import TradingSignal, OrderType
                
                # Generate signals for top trending stocks not in portfolio
                available_symbols = [s for s in target_symbols if s not in current_positions][:3]
                
                enhanced_signals = []
                for symbol in available_symbols:
                    fundamentals = self.universe_manager.get_stock_fundamentals(symbol)
                    if fundamentals and fundamentals.get('market_cap', 0) > 1000000000:  # $1B+ market cap
                        # Determine action based on market trends
                        price_change = fundamentals.get('price_change_pct', 0)
                        
                        if price_change > 2:  # Strong positive momentum
                            action = OrderType.BUY
                            reasoning = f"Strong momentum: +{price_change:.1f}% price change, market cap ${fundamentals.get('market_cap', 0)/1e9:.1f}B"
                        elif price_change < -3:  # Significant decline, might be opportunity
                            action = OrderType.BUY
                            reasoning = f"Potential opportunity: {price_change:.1f}% decline, market cap ${fundamentals.get('market_cap', 0)/1e9:.1f}B"
                        else:  # Stable, consider holding if we have position, otherwise small buy
                            action = OrderType.BUY if symbol not in current_positions else OrderType.HOLD
                            if action == OrderType.HOLD:
                                reasoning = f"Stable position: {price_change:.1f}% change, maintaining current allocation"
                            else:
                                reasoning = f"Stable entry opportunity: {price_change:.1f}% change, market cap ${fundamentals.get('market_cap', 0)/1e9:.1f}B"
                        
                        target_allocation = 0.05 if action == OrderType.BUY else (current_positions[symbol].market_value / 100000.0 if symbol in current_positions else 0.05)  # Assume 100k portfolio for calculation
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            action=action,
                            confidence=0.7,
                            target_allocation=min(target_allocation, 0.10),  # Cap at 10%
                            reasoning=reasoning,
                            news_sources=['Market Trends']
                        )
                        enhanced_signals.append(signal)
                
                signals.extend(enhanced_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating enhanced signals: {e}")
            return []

# Usage example
def get_enhanced_news_analyzer(openai_api_key: str):
    """Factory function to create enhanced news analyzer"""
    return EnhancedNewsAnalyzer(openai_api_key)