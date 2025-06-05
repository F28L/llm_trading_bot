import asyncio
import aiohttp
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import requests
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"

@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

@dataclass
class Transaction:
    id: str
    timestamp: datetime
    symbol: str
    order_type: OrderType
    shares: float
    price: float
    total_value: float
    fees: float
    status: OrderStatus
    reason: str = ""

@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    timestamp: datetime
    symbols: List[str]
    sentiment_score: float = 0.0
    relevance_score: float = 0.0

@dataclass
class TradingSignal:
    symbol: str
    action: OrderType
    confidence: float
    target_allocation: float
    reasoning: str
    news_sources: List[str]

class RiskManager:
    def __init__(self, max_position_size=0.1, max_daily_loss=0.05, max_sector_concentration=0.3):
        self.max_position_size = max_position_size  # 10% max per position
        self.max_daily_loss = max_daily_loss        # 5% max daily loss
        self.max_sector_concentration = max_sector_concentration  # 30% max per sector
        self.daily_start_value = None
        
    def check_position_size(self, symbol: str, allocation: float, portfolio_value: float) -> bool:
        """Check if position size is within risk limits"""
        if allocation > self.max_position_size:
            logger.warning(f"Position size {allocation:.2%} exceeds max {self.max_position_size:.2%} for {symbol}")
            return False
        return True
    
    def check_daily_loss(self, current_portfolio_value: float) -> bool:
        """Check if daily loss limit is exceeded"""
        if self.daily_start_value is None:
            self.daily_start_value = current_portfolio_value
            return True
            
        daily_pnl = (current_portfolio_value - self.daily_start_value) / self.daily_start_value
        if daily_pnl < -self.max_daily_loss:
            logger.warning(f"Daily loss {daily_pnl:.2%} exceeds limit {self.max_daily_loss:.2%}")
            return False
        return True
    
    def reset_daily_tracking(self, portfolio_value: float):
        """Reset daily tracking at market open"""
        self.daily_start_value = portfolio_value

class DatabaseManager:
    def __init__(self, db_path="trading_agent.db"):
        self.db_path = db_path
        self.lock = Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    order_type TEXT,
                    shares REAL,
                    price REAL,
                    total_value REAL,
                    fees REAL,
                    status TEXT,
                    reason TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    shares REAL,
                    avg_cost REAL,
                    realized_pnl REAL,
                    updated_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    timestamp TEXT PRIMARY KEY,
                    total_value REAL,
                    cash REAL,
                    positions_value REAL,
                    daily_pnl REAL,
                    total_pnl REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    confidence REAL,
                    target_allocation REAL,
                    reasoning TEXT,
                    executed BOOLEAN
                )
            ''')
    
    def save_transaction(self, transaction: Transaction):
        """Save transaction to database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction.id,
                    transaction.timestamp.isoformat(),
                    transaction.symbol,
                    transaction.order_type.value,
                    transaction.shares,
                    transaction.price,
                    transaction.total_value,
                    transaction.fees,
                    transaction.status.value,
                    transaction.reason
                ))
    
    def update_position(self, symbol: str, shares: float, avg_cost: float, realized_pnl: float):
        """Update position in database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?)
                ''', (symbol, shares, avg_cost, realized_pnl, datetime.now().isoformat()))
    
    def save_portfolio_snapshot(self, total_value: float, cash: float, positions_value: float, 
                               daily_pnl: float, total_pnl: float):
        """Save portfolio snapshot with logging"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                timestamp = datetime.now().isoformat()
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO portfolio_history VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, total_value, cash, positions_value, daily_pnl, total_pnl))
                    
                    logger.debug(f"Saved portfolio snapshot: Total=${total_value:.2f}, Cash=${cash:.2f}, Positions=${positions_value:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error saving portfolio snapshot: {e}")
                    raise

    def remove_position(self, symbol: str):
        """Remove position from database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))

    def save_trading_signal(self, signal: 'TradingSignal', executed: bool = False):
        """Save trading signal to database"""
        import uuid
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trading_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()),
                    datetime.now().isoformat(),
                    signal.symbol,
                    signal.action.value,
                    signal.confidence,
                    signal.target_allocation,
                    signal.reasoning,
                    executed
                ))

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('''
                SELECT * FROM portfolio_history 
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days), conn)

class MarketDataProvider:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # 1 minute cache
    
    def get_current_price(self, symbol: str) -> float:
        """Get current stock price with caching"""
        now = time.time()
        
        if (symbol in self.cache and 
            symbol in self.cache_expiry and 
            now < self.cache_expiry[symbol]):
            return self.cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self.cache[symbol] = price
                self.cache_expiry[symbol] = now + self.cache_duration
                return price
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            
        return self.cache.get(symbol, 0.0)
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple stock prices efficiently"""
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.get_current_price(symbol)
        return prices

class NewsAnalyzer:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        ]
    
    async def fetch_news(self) -> List[NewsItem]:
        """Fetch news from multiple sources"""
        news_items = []
        
        # For demo purposes, using NewsAPI (you'll need an API key)
        # Replace with actual RSS parsing or API calls
        try:
            # This is a placeholder - implement actual news fetching
            sample_news = [
                NewsItem(
                    title="Apple Reports Strong Q4 Earnings",
                    content="Apple Inc. reported better than expected quarterly earnings...",
                    source="Yahoo Finance",
                    timestamp=datetime.now(),
                    symbols=["AAPL"]
                ),
                NewsItem(
                    title="Tesla Announces New Factory Plans",
                    content="Tesla Inc. announced plans for a new manufacturing facility...",
                    source="Bloomberg",
                    timestamp=datetime.now(),
                    symbols=["TSLA"]
                )
            ]
            news_items.extend(sample_news)
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        
        return news_items
    
    def analyze_news_sentiment(self, news_item: NewsItem) -> Tuple[float, float]:
        """Analyze news sentiment and relevance using LLM"""
        try:
            prompt = f"""
            Analyze this financial news for trading signals:
            
            Title: {news_item.title}
            Content: {news_item.content}
            Symbols: {', '.join(news_item.symbols)}
            
            Provide:
            1. Sentiment score (-1 to 1, where -1 is very negative, 1 is very positive)
            2. Relevance score (0 to 1, where 1 is highly relevant for trading)
            3. Brief reasoning
            
            Respond in JSON format:
            {{"sentiment": 0.5, "relevance": 0.8, "reasoning": "explanation"}}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("sentiment", 0.0), result.get("relevance", 0.0)
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return 0.0, 0.0
    
    def generate_trading_signals(self, news_items: List[NewsItem], 
                               current_positions: Dict[str, Position]) -> List[TradingSignal]:
        """Generate trading signals based on news analysis"""
        try:
            # Prepare context about current positions
            positions_context = "\n".join([
                f"{symbol}: {pos.shares:.2f} shares, {pos.unrealized_pnl:.2f} P&L"
                for symbol, pos in current_positions.items()
            ]) if current_positions else "No current positions"
            
            news_context = "\n".join([
                f"- {item.title} (Sentiment: {item.sentiment_score:.2f}, Relevance: {item.relevance_score:.2f}) [Symbols: {', '.join(item.symbols)}]"
                for item in news_items if item.relevance_score > 0.5
            ]) if news_items else "No relevant news found"
            
            prompt = f"""
            You are an AI trading system. Based on the following news analysis and current portfolio positions, generate trading signals.
            
            Current Positions:
            {positions_context}
            
            Recent News:
            {news_context}
            
            Rules:
            - Maximum 10% allocation per position
            - Generate BUY signals for new positions or increasing existing ones
            - Generate SELL signals to reduce or close existing positions
            - Generate HOLD signals to maintain current positions when news supports keeping them
            - Target allocation should be between 0.02 (2%) and 0.10 (10%)
            - For HOLD signals, target_allocation should match current allocation
            - Confidence should be between 0.6 and 1.0
            
            Signal Types:
            - BUY: Acquire new position or increase existing position
            - SELL: Reduce or close existing position
            - HOLD: Maintain current position size (no trading action)
            
            Respond with ONLY a valid JSON array of trading signals. No other text or explanation.
            
            Format:
            [
                {{
                    "symbol": "AAPL",
                    "action": "BUY",
                    "confidence": 0.8,
                    "target_allocation": 0.05,
                    "reasoning": "Positive earnings news and strong sector sentiment",
                    "news_sources": ["MarketWatch", "Bloomberg"]
                }},
                {{
                    "symbol": "MSFT",
                    "action": "HOLD",
                    "confidence": 0.7,
                    "target_allocation": 0.08,
                    "reasoning": "Stable performance, maintain current position amid market uncertainty",
                    "news_sources": ["Reuters"]
                }}
            ]
            
            If no trading signals are appropriate, return an empty array: []
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content.strip()
            logger.info(f"Raw GPT response: {response_content[:200]}...")  # Log first 200 chars for debugging
            
            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if "```json" in response_content:
                    response_content = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    response_content = response_content.split("```")[1].split("```")[0].strip()
                
                # Parse JSON
                signals_data = json.loads(response_content)
                
                # Validate it's a list
                if not isinstance(signals_data, list):
                    logger.warning("GPT response is not a list, returning empty signals")
                    return []
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {response_content}")
                # Try to extract JSON using regex as fallback
                import re
                json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
                if json_match:
                    try:
                        signals_data = json.loads(json_match.group())
                    except:
                        logger.error("Fallback JSON extraction also failed")
                        return []
                else:
                    logger.error("No JSON array found in response")
                    return []
            
            signals = []
            
            for signal_data in signals_data:
                try:
                    # Validate required fields
                    required_fields = ['symbol', 'action', 'confidence', 'target_allocation', 'reasoning']
                    if not all(field in signal_data for field in required_fields):
                        logger.warning(f"Signal missing required fields: {signal_data}")
                        continue
                    
                    # Validate action
                    if signal_data["action"] not in ["BUY", "SELL", "HOLD"]:
                        logger.warning(f"Invalid action in signal: {signal_data['action']}")
                        continue
                    
                    # Validate ranges
                    if not (0.6 <= signal_data["confidence"] <= 1.0):
                        logger.warning(f"Confidence out of range: {signal_data['confidence']}")
                        continue
                    
                    if not (0.01 <= signal_data["target_allocation"] <= 0.10):
                        logger.warning(f"Target allocation out of range: {signal_data['target_allocation']}")
                        continue
                    
                    signal = TradingSignal(
                        symbol=signal_data["symbol"],
                        action=OrderType(signal_data["action"]),
                        confidence=signal_data["confidence"],
                        target_allocation=signal_data["target_allocation"],
                        reasoning=signal_data["reasoning"],
                        news_sources=signal_data.get("news_sources", [])
                    )
                    signals.append(signal)
                    
                except Exception as e:
                    logger.error(f"Error processing signal data: {signal_data}, error: {e}")
                    continue
            
            logger.info(f"Successfully generated {len(signals)} valid trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return []

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.transaction_history: List[Transaction] = []
        self.db = DatabaseManager()
        self.market_data = MarketDataProvider()
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def update_positions_prices(self):
        """Update current prices for all positions"""
        if not self.positions:
            return
            
        symbols = list(self.positions.keys())
        current_prices = self.market_data.get_multiple_prices(symbols)
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.current_price)
            position.current_price = current_price
            position.market_value = position.shares * current_price
            position.unrealized_pnl = position.market_value - (position.shares * position.avg_cost)
    
    def execute_order(self, symbol: str, order_type: OrderType, shares: float, reason: str = "") -> bool:
        """Execute a trading order"""
        current_price = self.market_data.get_current_price(symbol)
        if current_price <= 0:
            logger.error(f"Cannot get valid price for {symbol}")
            return False
        
        total_value = shares * current_price
        fees = 0.0  # Assuming no fees for paper trading
        
        transaction_id = f"{symbol}_{int(time.time() * 1000)}"
        
        if order_type == OrderType.BUY:
            if total_value > self.cash:
                logger.warning(f"Insufficient cash for {symbol} purchase: need ${total_value:.2f}, have ${self.cash:.2f}")
                return False
            
            # Execute buy order
            self.cash -= total_value
            
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                new_shares = pos.shares + shares
                new_avg_cost = ((pos.shares * pos.avg_cost) + total_value) / new_shares
                pos.shares = new_shares
                pos.avg_cost = new_avg_cost
                pos.market_value = new_shares * current_price
                pos.unrealized_pnl = pos.market_value - (new_shares * new_avg_cost)
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=current_price,
                    current_price=current_price,
                    market_value=total_value,
                    unrealized_pnl=0.0
                )
            
            # Update database
            pos = self.positions[symbol]
            self.db.update_position(symbol, pos.shares, pos.avg_cost, pos.realized_pnl)
        
        elif order_type == OrderType.SELL:
            if symbol not in self.positions or self.positions[symbol].shares < shares:
                logger.warning(f"Insufficient shares to sell {symbol}: need {shares}, have {self.positions.get(symbol, Position('', 0, 0, 0, 0, 0)).shares}")
                return False
            
            # Execute sell order
            pos = self.positions[symbol]
            realized_pnl = (current_price - pos.avg_cost) * shares
            pos.realized_pnl += realized_pnl
            pos.shares -= shares
            
            self.cash += total_value
            
            # Update database
            self.db.update_position(symbol, pos.shares, pos.avg_cost, pos.realized_pnl)
            
            # Remove position if fully sold
            if pos.shares <= 0.001:  # Account for floating point precision
                del self.positions[symbol]
                # Remove from database too
                self.db.remove_position(symbol)
        
        # Record transaction
        transaction = Transaction(
            id=transaction_id,
            timestamp=datetime.now(),
            symbol=symbol,
            order_type=order_type,
            shares=shares,
            price=current_price,
            total_value=total_value,
            fees=fees,
            status=OrderStatus.EXECUTED,
            reason=reason
        )
        
        self.transaction_history.append(transaction)
        self.db.save_transaction(transaction)
        
        logger.info(f"Executed {order_type.value} order: {shares:.3f} shares of {symbol} at ${current_price:.2f}")
        return True
    
    def get_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        # Make sure we have current prices
        self.update_positions_prices()
        
        current_value = self.get_portfolio_value()
        total_pnl = current_value - self.initial_cash
        total_return = total_pnl / self.initial_cash
        
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        positions_value = sum(pos.market_value for pos in self.positions.values())
        
        metrics = {
            "total_value": current_value,
            "cash": self.cash,
            "positions_value": positions_value,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "num_positions": len(self.positions)
        }
        
        # Debug logging
        logger.debug(f"Performance metrics: Total=${current_value:.2f}, Cash=${self.cash:.2f}, Positions=${positions_value:.2f}")
        
        return metrics

class TradingAgent:
    def __init__(self, openai_api_key: str, initial_cash: float = 100000.0):
        self.portfolio = Portfolio(initial_cash)
        self.news_analyzer = NewsAnalyzer(openai_api_key)
        self.risk_manager = RiskManager()
        self.db = DatabaseManager()
        self.running = False
        
        # Enhanced: Add stock universe management
        try:
            from enhanced_stock_discovery import get_enhanced_news_analyzer
            self.enhanced_analyzer = get_enhanced_news_analyzer(openai_api_key)
            self.use_enhanced_discovery = True
            logger.info("Enhanced stock discovery enabled")
        except ImportError:
            self.use_enhanced_discovery = False
            logger.info("Using basic stock discovery")
        
    async def run_trading_cycle(self):
        """Main trading cycle with enhanced stock discovery"""
        logger.info("Starting trading cycle...")
        
        # Update position prices
        self.portfolio.update_positions_prices()
        
        # Check risk management
        portfolio_value = self.portfolio.get_portfolio_value()
        if not self.risk_manager.check_daily_loss(portfolio_value):
            logger.warning("Daily loss limit exceeded, skipping trading cycle")
            return
        
        # Fetch and analyze news (enhanced or basic)
        if self.use_enhanced_discovery:
            news_items = await self.enhanced_analyzer.fetch_real_financial_news()
            signals = self.enhanced_analyzer.generate_enhanced_trading_signals(news_items, self.portfolio.positions)
        else:
            news_items = await self.news_analyzer.fetch_news()
            # Analyze sentiment for each news item
            for news_item in news_items:
                sentiment, relevance = self.news_analyzer.analyze_news_sentiment(news_item)
                news_item.sentiment_score = sentiment
                news_item.relevance_score = relevance
            signals = self.news_analyzer.generate_trading_signals(news_items, self.portfolio.positions)
        
        logger.info(f"Generated {len(signals)} trading signals from {len(news_items)} news items")
        
        # Execute trades based on signals
        executed_signals = []
        hold_signals = []
        
        for signal in signals:
            # Log all signals to database
            is_executed = False
            
            if signal.confidence < 0.6:  # Only trade on high confidence signals
                self.db.save_trading_signal(signal, executed=False)
                continue
                
            if not self.risk_manager.check_position_size(signal.symbol, signal.target_allocation, portfolio_value):
                self.db.save_trading_signal(signal, executed=False)
                continue
            
            current_allocation = 0.0
            if signal.symbol in self.portfolio.positions:
                current_allocation = self.portfolio.positions[signal.symbol].market_value / portfolio_value
            
            target_value = portfolio_value * signal.target_allocation
            
            if signal.action == OrderType.HOLD:
                # Log HOLD decision but don't execute any trades
                if signal.symbol in self.portfolio.positions:
                    hold_signals.append(signal)
                    is_executed = True  # HOLD is considered "executed" since we're maintaining position
                    logger.info(f"HOLD signal for {signal.symbol}: maintaining {current_allocation:.1%} allocation - {signal.reasoning}")
                else:
                    logger.warning(f"HOLD signal for {signal.symbol} ignored - no existing position")
                
                self.db.save_trading_signal(signal, executed=is_executed)
                continue
            
            elif signal.action == OrderType.BUY and current_allocation < signal.target_allocation:
                # Calculate shares to buy
                current_price = self.portfolio.market_data.get_current_price(signal.symbol)
                if current_price > 0:
                    additional_value = target_value - (current_allocation * portfolio_value)
                    shares_to_buy = additional_value / current_price
                    
                    if shares_to_buy > 0 and additional_value <= self.portfolio.cash:
                        success = self.portfolio.execute_order(
                            signal.symbol, 
                            OrderType.BUY, 
                            shares_to_buy,
                            f"AI Signal: {signal.reasoning}"
                        )
                        if success:
                            executed_signals.append(signal)
                            is_executed = True
            
            elif signal.action == OrderType.SELL and signal.symbol in self.portfolio.positions:
                # Calculate shares to sell
                position = self.portfolio.positions[signal.symbol]
                target_shares = target_value / position.current_price if position.current_price > 0 else 0
                shares_to_sell = position.shares - target_shares
                
                if shares_to_sell > 0:
                    success = self.portfolio.execute_order(
                        signal.symbol,
                        OrderType.SELL,
                        min(shares_to_sell, position.shares),
                        f"AI Signal: {signal.reasoning}"
                    )
                    if success:
                        executed_signals.append(signal)
                        is_executed = True
            
            # Save signal to database
            self.db.save_trading_signal(signal, executed=is_executed)
        
        # Save portfolio snapshot
        metrics = self.portfolio.get_performance_metrics()
        self.db.save_portfolio_snapshot(
            metrics["total_value"],
            metrics["cash"],
            metrics["positions_value"],
            0.0,  # Daily P&L calculation would need previous day's value
            metrics["total_pnl"]
        )
        
        # Log current status
        logger.info(f"Portfolio Value: ${metrics['total_value']:.2f}, P&L: ${metrics['total_pnl']:.2f} ({metrics['total_return']:.2%})")
        logger.info(f"Cash: ${metrics['cash']:.2f}, Positions: ${metrics['positions_value']:.2f}")
        logger.info(f"Active positions: {', '.join(self.portfolio.positions.keys()) if self.portfolio.positions else 'None'}")
        
        # Log signal summary
        if executed_signals or hold_signals:
            signal_summary = []
            if executed_signals:
                signal_summary.append(f"{len(executed_signals)} executed")
            if hold_signals:
                signal_summary.append(f"{len(hold_signals)} held")
            logger.info(f"Trading signals: {', '.join(signal_summary)}")
        else:
            logger.info("No trading signals executed this cycle")
        
    async def start_trading(self, cycle_interval_minutes: int = 15):
        """Start the automated trading loop"""
        self.running = True
        logger.info(f"Starting AI trading agent with {cycle_interval_minutes} minute cycles...")
        
        while self.running:
            try:
                await self.run_trading_cycle()
                await asyncio.sleep(cycle_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_trading(self):
        """Stop the trading agent"""
        self.running = False
        logger.info("Trading agent stopped")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        self.portfolio.update_positions_prices()
        metrics = self.portfolio.get_performance_metrics()
        
        positions_summary = []
        for symbol, pos in self.portfolio.positions.items():
            positions_summary.append({
                "symbol": symbol,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl / (pos.shares * pos.avg_cost) if pos.shares > 0 else 0
            })
        
        return {
            "portfolio_metrics": metrics,
            "positions": positions_summary,
            "recent_transactions": [asdict(t) for t in self.portfolio.transaction_history[-10:]]
        }

# Example usage and testing
if __name__ == "__main__":
    # You'll need to provide your OpenAI API key
    OPENAI_API_KEY = "your-openai-api-key-here"
    
    if OPENAI_API_KEY == "your-openai-api-key-here":
        print("Please set your OpenAI API key in the OPENAI_API_KEY variable")
        print("You can get one at: https://platform.openai.com/api-keys")
        exit(1)
    
    # Create and run the trading agent
    agent = TradingAgent(OPENAI_API_KEY, initial_cash=100000.0)
    
    # Example: Run a single trading cycle for testing
    async def test_trading_cycle():
        print("Running test trading cycle...")
        await agent.run_trading_cycle()
        summary = agent.get_portfolio_summary()
        
        print("\n=== PORTFOLIO SUMMARY ===")
        metrics = summary['portfolio_metrics']
        print(f"Total Value: ${metrics['total_value']:,.2f}")
        print(f"Cash: ${metrics['cash']:,.2f}")
        print(f"Positions Value: ${metrics['positions_value']:,.2f}")
        print(f"Total P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_return']:.2%})")
        print(f"Number of Positions: {metrics['num_positions']}")
        
        if summary['positions']:
            print("\n=== CURRENT POSITIONS ===")
            for pos in summary['positions']:
                print(f"{pos['symbol']}: {pos['shares']:.3f} shares @ ${pos['current_price']:.2f}")
                print(f"  Market Value: ${pos['market_value']:,.2f}")
                print(f"  P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_pct']:.2%})")
        
        if summary['recent_transactions']:
            print("\n=== RECENT TRANSACTIONS ===")
            for txn in summary['recent_transactions'][-5:]:  # Show last 5
                print(f"{txn['timestamp'][:19]}: {txn['order_type']} {txn['shares']:.3f} {txn['symbol']} @ ${txn['price']:.2f}")
                if txn['reason']:
                    print(f"  Reason: {txn['reason']}")
    
    print("AI Trading Bot - Paper Trading System")
    print("====================================")
    print("1. Make sure you have set your OpenAI API key")
    print("2. The system will use sample news data for demonstration")
    print("3. All trades are paper trades - no real money involved")
    print("")
    
    # To run continuously (uncomment the line below):
    # asyncio.run(agent.start_trading(cycle_interval_minutes=15))
    
    # For testing, run a single cycle:
    try:
        asyncio.run(test_trading_cycle())
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        print(f"Error running trading bot: {e}")
        print("Make sure you have set a valid OpenAI API key")