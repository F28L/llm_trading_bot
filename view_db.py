#!/usr/bin/env python3
# view_db.py - Simple CLI database viewer for AI Trading Bot

import sqlite3
import sys
from datetime import datetime

def connect_db(db_path="trading_agent.db"):
    """Connect to the database"""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def view_portfolio_summary(conn):
    """View portfolio summary"""
    print("\n" + "="*60)
    print("📊 PORTFOLIO SUMMARY")
    print("="*60)
    
    # Latest portfolio data
    cursor = conn.execute("""
        SELECT * FROM portfolio_history 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    if row:
        print(f"💼 Total Value: ${row[1]:,.2f}")
        print(f"💵 Cash: ${row[2]:,.2f}")
        print(f"📈 Positions Value: ${row[3]:,.2f}")
        print(f"📊 Total P&L: ${row[5]:,.2f}")
        print(f"🕐 Last Updated: {row[0]}")
    else:
        print("No portfolio data found")
    
    # Position count
    cursor = conn.execute("SELECT COUNT(*) FROM positions WHERE shares > 0")
    pos_count = cursor.fetchone()[0]
    print(f"🎯 Active Positions: {pos_count}")
    
    # Transaction count
    cursor = conn.execute("SELECT COUNT(*) FROM transactions")
    txn_count = cursor.fetchone()[0]
    print(f"📝 Total Transactions: {txn_count}")
    
    print("="*60)

def view_positions(conn):
    """View current positions"""
    print("\n📋 CURRENT POSITIONS")
    print("-" * 70)
    
    cursor = conn.execute("""
        SELECT symbol, shares, avg_cost, realized_pnl, shares * avg_cost as invested
        FROM positions 
        WHERE shares > 0 
        ORDER BY shares * avg_cost DESC
    """)
    
    positions = cursor.fetchall()
    if not positions:
        print("No current positions")
        return
    
    print(f"{'Symbol':<8} {'Shares':<10} {'Avg Cost':<10} {'Invested':<12} {'P&L':<10}")
    print("-" * 70)
    
    for pos in positions:
        symbol, shares, avg_cost, realized_pnl, invested = pos
        print(f"{symbol:<8} {shares:<10.3f} ${avg_cost:<9.2f} ${invested:<11,.2f} ${realized_pnl:<9.2f}")
    
    print("-" * 70)

def view_recent_transactions(conn, limit=10):
    """View recent transactions"""
    print(f"\n📝 RECENT TRANSACTIONS (Last {limit})")
    print("-" * 90)
    
    cursor = conn.execute("""
        SELECT timestamp, symbol, order_type, shares, price, total_value, reason
        FROM transactions 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    transactions = cursor.fetchall()
    if not transactions:
        print("No transactions found")
        return
    
    for txn in transactions:
        timestamp, symbol, order_type, shares, price, total_value, reason = txn
        action_symbol = "🟢" if order_type == "BUY" else "🔴" if order_type == "SELL" else "🟡"
        
        print(f"{action_symbol} {timestamp[:19]} | {order_type} {shares:.3f} {symbol} @ ${price:.2f}")
        if reason:
            print(f"   💭 {reason}")
    
    print("-" * 90)

def view_trading_signals(conn, days=7):
    """View recent trading signals"""
    print(f"\n🎯 TRADING SIGNALS (Last {days} days)")
    print("-" * 100)
    
    cursor = conn.execute("""
        SELECT timestamp, symbol, action, confidence, target_allocation, reasoning, executed
        FROM trading_signals 
        WHERE datetime(timestamp) > datetime('now', '-{} days')
        ORDER BY timestamp DESC
    """.format(days))
    
    signals = cursor.fetchall()
    if not signals:
        print(f"No trading signals in last {days} days")
        return
    
    for signal in signals:
        timestamp, symbol, action, confidence, target_allocation, reasoning, executed = signal
        status = "✅" if executed else "❌"
        action_symbol = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
        
        print(f"{status} {action_symbol} {timestamp[:19]} | {action} {symbol}")
        print(f"   📊 Confidence: {confidence:.1%} | Target: {target_allocation:.1%}")
        print(f"   💭 {reasoning}")
    
    print("-" * 100)

def view_performance_stats(conn):
    """View performance statistics"""
    print("\n📈 PERFORMANCE STATISTICS")
    print("-" * 50)
    
    # Portfolio growth
    cursor = conn.execute("""
        SELECT 
            MIN(total_value) as min_value,
            MAX(total_value) as max_value,
            COUNT(*) as data_points
        FROM portfolio_history
    """)
    
    row = cursor.fetchone()
    if row and row[0]:
        min_val, max_val, points = row
        growth = ((max_val - min_val) / min_val) * 100 if min_val > 0 else 0
        print(f"📊 Portfolio Range: ${min_val:,.2f} - ${max_val:,.2f}")
        print(f"📈 Max Growth: {growth:.2f}%")
        print(f"📋 Data Points: {points}")
    
    # Transaction statistics
    cursor = conn.execute("""
        SELECT 
            order_type,
            COUNT(*) as count,
            SUM(total_value) as total_volume
        FROM transactions 
        GROUP BY order_type
    """)
    
    print(f"\n💼 Transaction Summary:")
    for row in cursor.fetchall():
        order_type, count, volume = row
        print(f"   {order_type}: {count} trades, ${volume:,.2f} volume")
    
    # Signal statistics
    cursor = conn.execute("""
        SELECT 
            action,
            COUNT(*) as total,
            SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed
        FROM trading_signals 
        GROUP BY action
    """)
    
    print(f"\n🎯 Signal Summary:")
    for row in cursor.fetchall():
        action, total, executed = row
        execution_rate = (executed / total * 100) if total > 0 else 0
        print(f"   {action}: {total} signals, {executed} executed ({execution_rate:.1f}%)")
    
    print("-" * 50)

def view_database_info(conn):
    """View database schema and table info"""
    print("\n🗄️ DATABASE INFORMATION")
    print("-" * 50)
    
    # Get all tables
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """)
    
    tables = cursor.fetchall()
    print(f"📊 Tables in database: {len(tables)}")
    
    for table_name in tables:
        table = table_name[0]
        
        # Get row count
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        
        print(f"   📋 {table}: {count} records")
    
    print("-" * 50)

def export_data(conn, table_name, output_file):
    """Export table data to CSV"""
    try:
        import pandas as pd
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(df)} records from {table_name} to {output_file}")
        
    except ImportError:
        print("❌ pandas not installed. Cannot export to CSV.")
    except Exception as e:
        print(f"❌ Export error: {e}")

def interactive_menu():
    """Interactive menu for database exploration"""
    db_path = "trading_agent.db"
    
    print("🤖 AI Trading Bot Database Viewer")
    print("=" * 40)
    
    conn = connect_db(db_path)
    if not conn:
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Portfolio Summary")
        print("2. Current Positions")
        print("3. Recent Transactions")
        print("4. Trading Signals")
        print("5. Performance Stats")
        print("6. Database Info")
        print("7. Export Data")
        print("8. All Data")
        print("9. Exit")
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        try:
            if choice == '1':
                view_portfolio_summary(conn)
            elif choice == '2':
                view_positions(conn)
            elif choice == '3':
                limit = input("How many transactions? (default 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                view_recent_transactions(conn, limit)
            elif choice == '4':
                days = input("How many days? (default 7): ").strip()
                days = int(days) if days.isdigit() else 7
                view_trading_signals(conn, days)
            elif choice == '5':
                view_performance_stats(conn)
            elif choice == '6':
                view_database_info(conn)
            elif choice == '7':
                print("\nAvailable tables: transactions, positions, portfolio_history, trading_signals")
                table = input("Enter table name: ").strip()
                output = input("Enter output filename (e.g., data.csv): ").strip()
                if table and output:
                    export_data(conn, table, output)
            elif choice == '8':
                view_portfolio_summary(conn)
                view_positions(conn)
                view_recent_transactions(conn, 5)
                view_trading_signals(conn, 7)
                view_performance_stats(conn)
            elif choice == '9':
                break
            else:
                print("Invalid choice. Please enter 1-9.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    conn.close()
    print("👋 Goodbye!")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        db_path = "trading_agent.db"
        conn = connect_db(db_path)
        
        if not conn:
            return
            
        if sys.argv[1] == '--summary':
            view_portfolio_summary(conn)
        elif sys.argv[1] == '--positions':
            view_positions(conn)
        elif sys.argv[1] == '--transactions':
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            view_recent_transactions(conn, limit)
        elif sys.argv[1] == '--signals':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            view_trading_signals(conn, days)
        elif sys.argv[1] == '--stats':
            view_performance_stats(conn)
        elif sys.argv[1] == '--info':
            view_database_info(conn)
        elif sys.argv[1] == '--all':
            view_portfolio_summary(conn)
            view_positions(conn)
            view_recent_transactions(conn, 10)
            view_trading_signals(conn, 7)
            view_performance_stats(conn)
        else:
            print("Usage: python view_db.py [--summary|--positions|--transactions|--signals|--stats|--info|--all]")
            print("\nOptions:")
            print("  --summary     Portfolio summary")
            print("  --positions   Current positions")
            print("  --transactions [limit]  Recent transactions")
            print("  --signals [days]  Trading signals")
            print("  --stats       Performance statistics")
            print("  --info        Database information")
            print("  --all         All data")
            print("\nRun without arguments for interactive mode")
        
        conn.close()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()