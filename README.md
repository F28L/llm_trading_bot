# AI Trading Bot Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
poetry add yfinance pandas numpy openai aiohttp requests
```

### 2. Set up Environment Variables
Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
NEWSAPI_KEY=your_newsapi_key_here_optional
```

### 3. Project Structure
```
llm_trading_bot/
├── main.py              # Main trading system code
├── config.py            # Configuration settings
├── run_bot.py           # Bot runner script
├── .env                 # Environment variables
├── trading_agent.db     # SQLite database (created automatically)
└── pyproject.toml       # Poetry dependencies
```

### 4. Get API Keys

**OpenAI API Key (Required):**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add it to your `.env` file

**NewsAPI Key (Optional):**
1. Go to https://newsapi.org/register
2. Get your free API key
3. Add it to your `.env` file

### 5. Run the Bot

**Test Mode (Single Trading Cycle):**
```bash
python run_bot.py test
```

**Live Mode (Continuous Trading):**
```bash
python run_bot.py live
```

## Configuration Options

Edit `config.py` to customize:

- `initial_cash`: Starting portfolio value (default: $100,000)
- `max_position_size`: Maximum % per position (default: 10%)
- `max_daily_loss`: Maximum daily loss % (default: 5%)
- `cycle_interval_minutes`: How often to run trading cycles (default: 15 min)
- `min_signal_confidence`: Minimum confidence to execute trades (default: 0.6)

## Understanding the Output

### Portfolio Summary
- **Total Value**: Current portfolio worth (cash + positions)
- **Cash**: Available cash for new trades
- **Positions Value**: Current value of all stock positions
- **Total P&L**: Profit/Loss since start
- **Active Positions**: Number of stocks currently held

### Position Details
For each stock position:
- **Symbol**: Stock ticker
- **Shares**: Number of shares owned (including fractional)
- **Current Price**: Latest market price
- **Market Value**: Current worth of position
- **P&L**: Unrealized profit/loss and percentage

### Recent Transactions
- **🟢 BUY**: Purchase transactions
- **🔴 SELL**: Sale transactions
- **Reason**: AI-generated reasoning for the trade

## Safety Features

1. **Paper Trading**: No real money at risk
2. **Risk Limits**: Automatic position size and loss limits
3. **Database Logging**: All transactions saved to SQLite
4. **Graceful Shutdown**: Ctrl+C stops safely

## Troubleshooting

**"OpenAI API key not found"**
- Make sure your `.env` file exists and contains `OPENAI_API_KEY=your_key_here`
- Check that the key is valid and has credits

**"Could not get valid price"**
- Yahoo Finance API might be temporarily down
- Stock symbol might be invalid
- Check your internet connection

**"Insufficient cash/shares"**
- This is normal - the bot respects available funds
- Check risk management settings if too restrictive

## Next Steps

1. **Monitor Performance**: Watch the bot run in test mode first
2. **Adjust Risk Settings**: Modify `config.py` based on your risk tolerance
3. **Add Real News Sources**: Integrate actual news APIs
4. **Build Dashboard**: Create a web interface for monitoring

## Development

To modify the trading logic:
1. Edit the `NewsAnalyzer.generate_trading_signals()` method
2. Adjust the LLM prompts for different trading strategies
3. Add new risk management rules in `RiskManager`
4. Implement additional data sources

## Database

The system creates a SQLite database with these tables:
- `transactions`: All buy/sell orders
- `positions`: Current stock holdings
- `portfolio_history`: Portfolio value over time
- `news_analysis`: Processed news and sentiment

Access with any SQLite browser or Python:
```python
import sqlite3
conn = sqlite3.connect('trading_agent.db')
# Query your data
```

## Disclaimer

This is a paper trading system for educational purposes. Always:
- Test thoroughly before any real trading
- Understand the risks of algorithmic trading
- Comply with relevant financial regulations
- Never invest more than you can afford to lose