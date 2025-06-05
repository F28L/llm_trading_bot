# db_visualizer.py - Database Visualization Tools

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import json

class TradingBotDatabaseAnalyzer:
    def __init__(self, db_path="trading_agent.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            # Get latest portfolio snapshot
            portfolio_df = pd.read_sql_query("""
                SELECT * FROM portfolio_history 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, self.conn)
            
            if portfolio_df.empty:
                return {"error": "No portfolio data found"}
            
            latest = portfolio_df.iloc[0]
            
            # Get current positions
            positions_df = pd.read_sql_query("""
                SELECT * FROM positions 
                WHERE shares > 0
                ORDER BY shares * avg_cost DESC
            """, self.conn)
            
            # Get transaction count
            txn_count = pd.read_sql_query("""
                SELECT COUNT(*) as count FROM transactions
            """, self.conn).iloc[0]['count']
            
            return {
                "total_value": latest['total_value'],
                "cash": latest['cash'],
                "positions_value": latest['positions_value'],
                "total_pnl": latest['total_pnl'],
                "num_positions": len(positions_df),
                "total_transactions": txn_count,
                "last_updated": latest['timestamp']
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_chart_data(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio performance data for charting"""
        query = """
            SELECT 
                timestamp,
                total_value,
                cash,
                positions_value,
                total_pnl,
                datetime(timestamp) as date
            FROM portfolio_history 
            WHERE datetime(timestamp) > datetime('now', '-{} days')
            ORDER BY timestamp
        """.format(days)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_transactions_summary(self, days: int = 30) -> pd.DataFrame:
        """Get recent transactions"""
        query = """
            SELECT 
                timestamp,
                symbol,
                order_type,
                shares,
                price,
                total_value,
                reason,
                datetime(timestamp) as date
            FROM transactions 
            WHERE datetime(timestamp) > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_positions_analysis(self) -> pd.DataFrame:
        """Get current positions with analysis"""
        query = """
            SELECT 
                symbol,
                shares,
                avg_cost,
                realized_pnl,
                shares * avg_cost as total_invested,
                updated_at
            FROM positions 
            WHERE shares > 0
            ORDER BY shares * avg_cost DESC
        """
        
        return pd.read_sql_query(query, self.conn)
    
    def get_trading_signals_analysis(self, days: int = 7) -> pd.DataFrame:
        """Get recent trading signals"""
        query = """
            SELECT 
                timestamp,
                symbol,
                action,
                confidence,
                target_allocation,
                reasoning,
                executed,
                datetime(timestamp) as date
            FROM trading_signals 
            WHERE datetime(timestamp) > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)
        
        return pd.read_sql_query(query, self.conn)
    
    def print_portfolio_summary(self):
        """Print a formatted portfolio summary"""
        summary = self.get_portfolio_summary()
        
        if "error" in summary:
            print(f"❌ Error: {summary['error']}")
            return
        
        print("\n" + "="*60)
        print("📊 PORTFOLIO SUMMARY")
        print("="*60)
        print(f"💼 Total Value: ${summary['total_value']:,.2f}")
        print(f"💵 Cash: ${summary['cash']:,.2f}")
        print(f"📈 Positions Value: ${summary['positions_value']:,.2f}")
        print(f"📊 Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"🎯 Active Positions: {summary['num_positions']}")
        print(f"📝 Total Transactions: {summary['total_transactions']}")
        print(f"🕐 Last Updated: {summary['last_updated']}")
        print("="*60)
    
    def print_recent_transactions(self, limit: int = 10):
        """Print recent transactions"""
        df = self.get_transactions_summary(days=30)
        
        if df.empty:
            print("No transactions found")
            return
        
        print(f"\n📝 RECENT TRANSACTIONS (Last {limit})")
        print("-" * 80)
        
        for _, txn in df.head(limit).iterrows():
            action_symbol = "🟢" if txn['order_type'] == "BUY" else "🔴" if txn['order_type'] == "SELL" else "🟡"
            print(f"{action_symbol} {txn['timestamp'][:19]} | {txn['order_type']} {txn['shares']:.3f} {txn['symbol']} @ ${txn['price']:.2f}")
            if pd.notna(txn['reason']):
                print(f"   💭 {txn['reason']}")
        print("-" * 80)
    
    def print_current_positions(self):
        """Print current positions"""
        df = self.get_positions_analysis()
        
        if df.empty:
            print("No current positions")
            return
        
        print(f"\n📋 CURRENT POSITIONS ({len(df)} holdings)")
        print("-" * 70)
        
        for _, pos in df.iterrows():
            print(f"{pos['symbol']:6s} | {pos['shares']:8.3f} shares @ ${pos['avg_cost']:8.2f}")
            print(f"       | Invested: ${pos['total_invested']:10,.2f} | P&L: ${pos['realized_pnl']:8.2f}")
        print("-" * 70)
    
    def print_trading_signals(self, days: int = 7):
        """Print recent trading signals"""
        df = self.get_trading_signals_analysis(days)
        
        if df.empty:
            print(f"No trading signals in last {days} days")
            return
        
        print(f"\n🎯 TRADING SIGNALS (Last {days} days)")
        print("-" * 80)
        
        for _, signal in df.iterrows():
            status = "✅" if signal['executed'] else "❌"
            action_symbol = "🟢" if signal['action'] == "BUY" else "🔴" if signal['action'] == "SELL" else "🟡"
            
            print(f"{status} {action_symbol} {signal['timestamp'][:19]} | {signal['action']} {signal['symbol']}")
            print(f"   📊 Confidence: {signal['confidence']:.1%} | Target: {signal['target_allocation']:.1%}")
            print(f"   💭 {signal['reasoning']}")
        print("-" * 80)

def create_performance_chart(db_path="trading_agent.db", days=30, save_path="portfolio_performance.png"):
    """Create portfolio performance chart"""
    with TradingBotDatabaseAnalyzer(db_path) as analyzer:
        df = analyzer.get_performance_chart_data(days)
        
        if df.empty:
            print("No portfolio data to chart")
            return
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Create the plot
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Portfolio value over time
        ax1.plot(df['datetime'], df['total_value'], 'b-', linewidth=2, label='Total Portfolio Value')
        ax1.fill_between(df['datetime'], df['cash'], alpha=0.3, color='green', label='Cash')
        ax1.fill_between(df['datetime'], df['cash'], df['total_value'], alpha=0.3, color='blue', label='Positions')
        
        ax1.set_title(f'Portfolio Performance - Last {days} Days', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # P&L over time
        ax2.plot(df['datetime'], df['total_pnl'], 'g-' if df['total_pnl'].iloc[-1] >= 0 else 'r-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(df['datetime'], 0, df['total_pnl'], 
                        alpha=0.3, 
                        color='green' if df['total_pnl'].iloc[-1] >= 0 else 'red')
        
        ax2.set_title('Profit & Loss Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('P&L ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to {save_path}")
        plt.show()

def create_trading_analysis_charts(db_path="trading_agent.db"):
    """Create comprehensive trading analysis charts"""
    with TradingBotDatabaseAnalyzer(db_path) as analyzer:
        # Get data
        transactions_df = analyzer.get_transactions_summary(days=30)
        signals_df = analyzer.get_trading_signals_analysis(days=30)
        positions_df = analyzer.get_positions_analysis()
        
        if transactions_df.empty and signals_df.empty:
            print("No trading data to analyze")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Transaction types pie chart
        if not transactions_df.empty:
            txn_counts = transactions_df['order_type'].value_counts()
            colors = ['lightgreen', 'lightcoral', 'gold']
            ax1.pie(txn_counts.values, labels=txn_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(txn_counts)])
            ax1.set_title('Transaction Types Distribution')
        else:
            ax1.text(0.5, 0.5, 'No transactions yet', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Transaction Types Distribution')
        
        # 2. Trading signals analysis
        if not signals_df.empty:
            signal_counts = signals_df['action'].value_counts()
            execution_rate = signals_df['executed'].mean() * 100
            
            bars = ax2.bar(signal_counts.index, signal_counts.values, 
                          color=['lightgreen', 'lightcoral', 'gold'])
            ax2.set_title(f'Trading Signals ({execution_rate:.1f}% Execution Rate)')
            ax2.set_ylabel('Signal Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No signals yet', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Trading Signals')
        
        # 3. Portfolio allocation
        if not positions_df.empty:
            top_positions = positions_df.nlargest(8, 'total_invested')
            ax3.barh(top_positions['symbol'], top_positions['total_invested'])
            ax3.set_title('Current Position Sizes')
            ax3.set_xlabel('Investment Value ($)')
            
            # Format x-axis as currency
            ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        else:
            ax3.text(0.5, 0.5, 'No positions yet', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Current Position Sizes')
        
        # 4. Transaction volume over time
        if not transactions_df.empty:
            transactions_df['date'] = pd.to_datetime(transactions_df['timestamp']).dt.date
            daily_volume = transactions_df.groupby('date')['total_value'].sum()
            
            ax4.bar(daily_volume.index, daily_volume.values, alpha=0.7)
            ax4.set_title('Daily Trading Volume')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Volume ($)')
            ax4.tick_params(axis='x', rotation=45)
            
            # Format y-axis as currency
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        else:
            ax4.text(0.5, 0.5, 'No transactions yet', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Daily Trading Volume')
        
        plt.tight_layout()
        plt.savefig('trading_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Trading analysis charts saved to trading_analysis.png")
        plt.show()

def main():
    """Main function for database visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot Database Analyzer')
    parser.add_argument('--db', default='trading_agent.db', help='Database file path')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--action', choices=['summary', 'chart', 'analysis', 'all'], 
                       default='summary', help='What to display')
    
    args = parser.parse_args()
    
    try:
        with TradingBotDatabaseAnalyzer(args.db) as analyzer:
            if args.action in ['summary', 'all']:
                analyzer.print_portfolio_summary()
                analyzer.print_current_positions()
                analyzer.print_recent_transactions(10)
                analyzer.print_trading_signals(args.days)
            
            if args.action in ['chart', 'all']:
                create_performance_chart(args.db, args.days)
            
            if args.action in ['analysis', 'all']:
                create_trading_analysis_charts(args.db)
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()


# Quick usage functions
def quick_summary(db_path="trading_agent.db"):
    """Quick portfolio summary"""
    with TradingBotDatabaseAnalyzer(db_path) as analyzer:
        analyzer.print_portfolio_summary()
        analyzer.print_current_positions()

def quick_chart(db_path="trading_agent.db", days=30):
    """Quick performance chart"""
    create_performance_chart(db_path, days)

def quick_analysis(db_path="trading_agent.db"):
    """Quick trading analysis"""
    create_trading_analysis_charts(db_path)