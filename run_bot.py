# run_bot.py
import asyncio
import sys
import signal
from datetime import datetime
from config import TradingConfig
from main import TradingAgent

class TradingBotRunner:
    def __init__(self):
        self.config = TradingConfig()
        self.agent = None
        self.running = False
    
    def validate_config(self):
        """Validate configuration before starting"""
        if not self.config.openai_api_key:
            print("❌ OpenAI API key not found!")
            print("Please set the OPENAI_API_KEY environment variable or update config.py")
            print("Get your API key at: https://platform.openai.com/api-keys")
            return False
        
        if self.config.initial_cash <= 0:
            print("❌ Initial cash must be greater than 0")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown on Ctrl+C"""
        def signal_handler(sig, frame):
            print("\n🛑 Shutdown signal received...")
            self.stop_bot()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_bot(self, mode="test"):
        """Start the trading bot"""
        if not self.validate_config():
            return
        
        self.setup_signal_handlers()
        
        print("🤖 AI Trading Bot Starting...")
        print("=" * 50)
        print(f"💰 Initial Cash: ${self.config.initial_cash:,.2f}")
        print(f"⚡ Cycle Interval: {self.config.cycle_interval_minutes} minutes")
        print(f"🛡️  Max Position Size: {self.config.max_position_size:.1%}")
        print(f"📉 Max Daily Loss: {self.config.max_daily_loss:.1%}")
        print("=" * 50)
        
        # Create trading agent
        self.agent = TradingAgent(
            openai_api_key=self.config.openai_api_key,
            initial_cash=self.config.initial_cash
        )
        
        # Update risk manager settings
        self.agent.risk_manager.max_position_size = self.config.max_position_size
        self.agent.risk_manager.max_daily_loss = self.config.max_daily_loss
        self.agent.risk_manager.max_sector_concentration = self.config.max_sector_concentration
        
        if mode == "test":
            print("🧪 Running in TEST mode (single cycle)")
            asyncio.run(self.run_test_cycle())
        elif mode == "live":
            print("🚀 Running in LIVE mode (continuous)")
            asyncio.run(self.run_continuous())
    
    async def run_test_cycle(self):
        """Run a single test cycle"""
        try:
            print("\n📊 Executing trading cycle...")
            await self.agent.run_trading_cycle()
            
            # Display results
            summary = self.agent.get_portfolio_summary()
            self.display_portfolio_summary(summary)
            
        except Exception as e:
            print(f"❌ Error in test cycle: {e}")
    
    async def run_continuous(self):
        """Run continuous trading cycles"""
        try:
            self.running = True
            cycle_count = 0
            
            while self.running:
                cycle_count += 1
                print(f"\n🔄 Trading Cycle #{cycle_count}")
                print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                await self.agent.run_trading_cycle()
                
                # Display summary every 5 cycles
                if cycle_count % 5 == 0:
                    summary = self.agent.get_portfolio_summary()
                    self.display_portfolio_summary(summary)
                
                # Wait for next cycle
                print(f"⏳ Waiting {self.config.cycle_interval_minutes} minutes until next cycle...")
                await asyncio.sleep(self.config.cycle_interval_minutes * 60)
                
        except Exception as e:
            print(f"❌ Error in continuous mode: {e}")
    
    def display_portfolio_summary(self, summary):
        """Display formatted portfolio summary"""
        print("\n" + "=" * 60)
        print("📈 PORTFOLIO SUMMARY")
        print("=" * 60)
        
        metrics = summary['portfolio_metrics']
        print(f"💼 Total Value: ${metrics['total_value']:,.2f}")
        print(f"💵 Cash: ${metrics['cash']:,.2f}")
        print(f"📊 Positions Value: ${metrics['positions_value']:,.2f}")
        print(f"📈 Total P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_return']:.2%})")
        print(f"🎯 Active Positions: {metrics['num_positions']}")
        
        if summary['positions']:
            print("\n📋 CURRENT POSITIONS:")
            print("-" * 60)
            for pos in summary['positions']:
                pnl_symbol = "📈" if pos['unrealized_pnl'] >= 0 else "📉"
                print(f"{pos['symbol']:6s} | {pos['shares']:8.3f} shares @ ${pos['current_price']:8.2f}")
                print(f"       | Value: ${pos['market_value']:10,.2f} | P&L: {pnl_symbol} ${pos['unrealized_pnl']:8.2f} ({pos['unrealized_pnl_pct']:6.1%})")
        
        if summary['recent_transactions']:
            print("\n📝 RECENT TRANSACTIONS:")
            print("-" * 60)
            for txn in summary['recent_transactions'][-3:]:  # Show last 3
                if txn['order_type'] == "BUY":
                    action_symbol = "🟢"
                elif txn['order_type'] == "SELL":
                    action_symbol = "🔴"
                else:  # HOLD or other
                    action_symbol = "🟡"
                    
                # Handle both string and datetime timestamp formats
                timestamp_str = txn['timestamp']
                if not isinstance(timestamp_str, str):
                    timestamp_str = timestamp_str.isoformat()
                print(f"{action_symbol} {timestamp_str[:19]} | {txn['order_type']} {txn['shares']:.3f} {txn['symbol']} @ ${txn['price']:.2f}")
                if txn['reason']:
                    print(f"   💭 {txn['reason']}")
        
        print("=" * 60)
    
    def stop_bot(self):
        """Stop the trading bot gracefully"""
        self.running = False
        if self.agent:
            self.agent.stop_trading()
        print("🛑 Trading bot stopped")

def main():
    """Main entry point"""
    runner = TradingBotRunner()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ["test", "live"]:
            runner.start_bot(mode)
        else:
            print("Usage: python run_bot.py [test|live]")
            print("  test: Run a single trading cycle")
            print("  live: Run continuous trading cycles")
    else:
        # Default to test mode
        runner.start_bot("test")

if __name__ == "__main__":
    main()