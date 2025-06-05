# AI Trading Bot - Paper Trading System

A sophisticated AI-powered trading system that uses GPT-4 for news analysis and automated trading decisions across a dynamic universe of stocks.

## 🚀 Features

- **AI-Powered Trading**: GPT-4 analyzes financial news and generates trading signals
- **Dynamic Stock Discovery**: Automatically discovers trending stocks from multiple sources
- **Risk Management**: Built-in position limits, daily loss limits, and sector diversification
- **Paper Trading**: Safe simulation environment with $100K virtual cash
- **Real-Time Data**: Live market prices via Yahoo Finance
- **Complete Logging**: SQLite database tracks all transactions and performance
- **Enhanced Universe**: Trades across 50+ stocks including S&P 500, trending stocks, and ETFs

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   News Sources  │───▶│   AI Analysis    │───▶│ Trading Signals │
│  (RSS/APIs)     │    │    (GPT-4)       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────▼─────────┐
│  Risk Manager   │◀───│   Portfolio      │◀───│ Signal Execution  │
│                 │    │   Management     │    │                   │
└─────────────────┘    └──────────────────┘    └───────────────────┘
                                  │
                       ┌──────────▼──────────┐
                       │   Database Logger   │
                       │    (SQLite)         │
                       └─────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Poetry
- OpenAI API Key

### Setup Steps

1. **Clone and Navigate to Project**
   ```bash
   cd llm_trading_bot
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   poetry add yfinance pandas numpy openai aiohttp requests feedparser python-dotenv
   ```

3. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Set Your OpenAI API Key**
   ```bash
   # In your .env file:
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the System**
   ```bash
   poetry shell
   python run_bot.py test
   ```

## 🎯 Quick Start

### Single Test Cycle
```bash
python run_bot.py test
```

### Continuous Trading (15-minute cycles)
```bash
python run_bot.py live
```

### Configuration
Edit `config.py` to customize:
- Portfolio size and risk limits
- Trading cycle intervals
- Stock universe criteria
- News sources

## 📈 Sample Output

```
🤖 AI Trading Bot Starting...
💰 Initial Cash: $100,000.00
🛡️ Enhanced stock discovery enabled
📊 Executing trading cycle...

INFO - Retrieved 47 S&P 500 symbols
INFO - Discovered 42 valid tradeable symbols
INFO - Generated 6 trading signals from 12 news items

🟢 2025-06-04 14:45:12 | BUY 26.455 MSFT @ $378.45
   💭 AI Signal: Strong cloud revenue growth in earnings coverage
🟢 2025-06-04 14:45:13 | BUY 15.234 GOOGL @ $131.89
   💭 AI Signal: AI developments driving positive sentiment

📈 PORTFOLIO SUMMARY
💼 Total Value: $100,500.00
💵 Cash: $85,000.00
📊 Positions Value: $15,500.00
📈 Total P&L: $500.00 (0.5%)
🎯 Active Positions: 3
```

## 🏗️ Project Structure

```
llm_trading_bot/
├── main.py                      # Core trading system
├── config.py                    # Configuration settings
├── run_bot.py                   # Bot runner script
├── enhanced_stock_discovery.py  # Enhanced stock discovery
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── trading_agent.db            # SQLite database (generated)
└── pyproject.toml              # Poetry dependencies
```

## 🧠 How It Works

### 1. **News Analysis Pipeline**
- Fetches financial news from multiple RSS feeds
- Uses GPT-4 to analyze sentiment and relevance
- Extracts stock symbols mentioned in news
- Validates symbols and checks fundamental data

### 2. **Stock Universe Discovery**
- S&P 500 constituents (live from Wikipedia)
- Yahoo Finance trending stocks
- Reddit/social media trends (configurable)
- Sector ETFs for diversification
- Validates all symbols for tradeability

### 3. **AI Trading Signals**
- GPT-4 processes news sentiment and portfolio context
- Generates BUY/SELL signals with confidence scores
- Considers risk management and position sizing
- Provides reasoning for each trading decision

### 4. **Risk Management**
- **Position Limits**: Max 10% per stock
- **Daily Loss Limits**: Max 5% daily drawdown
- **Sector Limits**: Max 30% per sector
- **Market Cap Filtering**: Min $1B market cap
- **Liquidity Requirements**: Min daily volume thresholds

### 5. **Execution & Logging**
- Paper trading with real market prices
- Fractional share support
- Complete transaction history
- Portfolio performance tracking
- SQLite database for persistence

## 📊 Database Schema

### Tables
- **`transactions`**: All buy/sell orders with timestamps and reasoning
- **`positions`**: Current stock holdings and P&L
- **`portfolio_history`**: Portfolio value snapshots over time
- **`news_analysis`**: Processed news items and sentiment scores

### Query Examples
```python
import sqlite3
conn = sqlite3.connect('trading_agent.db')

# Get recent transactions
cursor = conn.execute('SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 10')

# Get portfolio performance
cursor = conn.execute('SELECT * FROM portfolio_history ORDER BY timestamp DESC LIMIT 30')
```

## ⚙️ Configuration Options

### Portfolio Settings
```python
initial_cash: float = 100000.0          # Starting cash
max_position_size: float = 0.10         # 10% max per position  
max_daily_loss: float = 0.05            # 5% max daily loss
```

### Stock Universe Settings
```python
max_universe_size: int = 50             # Max stocks to consider
min_market_cap: float = 1e9             # $1B minimum market cap
max_stock_price: float = 1000           # Max $1000 per share
include_etfs: bool = True               # Include sector ETFs
```

### Trading Settings
```python
cycle_interval_minutes: int = 15        # Trading cycle frequency
min_signal_confidence: float = 0.6      # Minimum confidence to trade
```

## 🔧 Advanced Features

### Custom News Sources
Add RSS feeds in `config.py`:
```python
news_sources = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://your-custom-feed.com/rss",
    # Add more sources
]
```

### Sector Filtering
Exclude specific sectors:
```python
excluded_sectors = ['Real Estate', 'Utilities']
```

### API Integration
The system supports additional data sources:
- Alpha Vantage (premium market data)
- Polygon.io (real-time feeds)
- NewsAPI (enhanced news coverage)

## 🚨 Risk Disclaimers

### Important Notices
- **Paper Trading Only**: This system uses virtual money for simulation
- **Educational Purpose**: Designed for learning algorithmic trading concepts
- **No Investment Advice**: Not a substitute for professional financial advice
- **Market Risks**: Real trading involves substantial risk of loss

### Before Live Trading
- Thoroughly test and understand the system
- Implement additional risk controls
- Comply with financial regulations
- Never invest more than you can afford to lose
- Consider professional consultation

## 🔮 Future Enhancements

### Phase 1 (Current)
- ✅ Multi-source stock discovery
- ✅ Real news RSS feeds
- ✅ Sector diversification
- ✅ Enhanced filtering

### Phase 2 (Planned)
- 🔄 Real broker API integration (Alpaca, Interactive Brokers)
- 📱 Social sentiment analysis (Reddit, Twitter)
- 📊 Technical analysis indicators
- 🎯 Options trading strategies

### Phase 3 (Advanced)
- 🤖 Multi-model AI ensemble
- 📈 Backtesting framework
- 🌐 International markets
- ⚡ High-frequency infrastructure

## 🐛 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
```bash
# Check your .env file exists and contains:
OPENAI_API_KEY=your_actual_key_here
```

**"Module not found: feedparser"**
```bash
poetry add feedparser python-dotenv
```

**"Cannot get valid price"**
- Yahoo Finance API might be temporarily down
- Check internet connection
- Verify stock symbol is valid

**"Enhanced stock discovery disabled"**
- Normal fallback behavior
- Install missing dependencies: `poetry add feedparser`
- Check `enhanced_stock_discovery.py` exists

### Debug Mode
```python
# Enable debug logging in main.py
logging.basicConfig(level=logging.DEBUG)
```

### Database Issues
```bash
# Reset database if corrupted
rm trading_agent.db
python run_bot.py test  # Will recreate tables
```

## 📞 Support

### Getting Help
- Check the troubleshooting section above
- Review log outputs for error details
- Ensure all dependencies are installed
- Verify API keys are valid and have credits

### Contributing
- Fork the repository
- Create feature branches
- Add tests for new functionality
- Submit pull requests

## 📄 License

This project is for educational purposes. See LICENSE file for details.

---

**⚠️ Disclaimer**: This software is provided for educational and research purposes only. The authors are not responsible for any financial losses incurred through the use of this system. Always do your own research and consider consulting with financial professionals before making investment decisions.