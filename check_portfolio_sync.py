#!/usr/bin/env python3
# check_portfolio_sync.py - Check portfolio sync between memory and database

import sqlite3
from datetime import datetime

def check_portfolio_sync(db_path="trading_agent.db"):
    """Check if portfolio data is in sync"""
    
    print("🔍 Portfolio Sync Checker")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    
    try:
        # 1. Check positions table
        print("\n📋 POSITIONS TABLE:")
        cursor = conn.execute("""
            SELECT symbol, shares, avg_cost, shares * avg_cost as invested, realized_pnl
            FROM positions 
            WHERE shares > 0 
            ORDER BY shares * avg_cost DESC
        """)
        
        positions = cursor.fetchall()
        total_invested = 0
        
        if positions:
            print(f"{'Symbol':<8} {'Shares':<10} {'Avg Cost':<10} {'Invested':<12} {'P&L':<10}")
            print("-" * 60)
            
            for pos in positions:
                symbol, shares, avg_cost, invested, realized_pnl = pos
                print(f"{symbol:<8} {shares:<10.3f} ${avg_cost:<9.2f} ${invested:<11,.2f} ${realized_pnl:<9.2f}")
                total_invested += invested
            
            print("-" * 60)
            print(f"{'TOTAL':<8} {'':<10} {'':<10} ${total_invested:<11,.2f}")
        else:
            print("No positions found")
        
        # 2. Check latest portfolio history
        print("\n📊 LATEST PORTFOLIO HISTORY:")
        cursor = conn.execute("""
            SELECT timestamp, total_value, cash, positions_value, total_pnl
            FROM portfolio_history 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        
        latest = cursor.fetchone()
        if latest:
            timestamp, total_value, cash, positions_value, total_pnl = latest
            print(f"Timestamp: {timestamp}")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Cash: ${cash:,.2f}")
            print(f"Positions Value: ${positions_value:,.2f}")
            print(f"Total P&L: ${total_pnl:,.2f}")
            
            # 3. Check for inconsistencies
            print("\n🔍 CONSISTENCY CHECK:")
            calculated_total = cash + positions_value
            
            print(f"Portfolio History Total: ${total_value:,.2f}")
            print(f"Calculated Total (Cash + Positions): ${calculated_total:,.2f}")
            print(f"Positions Table Total: ${total_invested:,.2f}")
            print(f"Portfolio History Positions: ${positions_value:,.2f}")
            
            # Check if values match
            if abs(total_value - calculated_total) < 0.01:
                print("✅ Portfolio total is consistent")
            else:
                print(f"❌ Portfolio total inconsistent: {abs(total_value - calculated_total):.2f} difference")
            
            if abs(positions_value - total_invested) < 0.01:
                print("✅ Positions value is consistent")
            else:
                print(f"❌ Positions value inconsistent: {abs(positions_value - total_invested):.2f} difference")
            
            # Check cash calculation
            expected_cash = 100000 - total_invested  # Assuming $100k initial
            if abs(cash - expected_cash) < 0.01:
                print("✅ Cash calculation is consistent")
            else:
                print(f"❌ Cash inconsistent: Expected ${expected_cash:.2f}, Got ${cash:.2f}")
                
        else:
            print("No portfolio history found")
        
        # 4. Check recent transactions
        print("\n📝 RECENT TRANSACTIONS:")
        cursor = conn.execute("""
            SELECT timestamp, symbol, order_type, shares, price, total_value
            FROM transactions 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        
        transactions = cursor.fetchall()
        if transactions:
            for txn in transactions:
                timestamp, symbol, order_type, shares, price, total_value = txn
                action = "🟢" if order_type == "BUY" else "🔴"
                print(f"{action} {timestamp[:19]} | {order_type} {shares:.3f} {symbol} @ ${price:.2f} = ${total_value:.2f}")
        else:
            print("No transactions found")
        
        # 5. Check portfolio history count
        print("\n📈 PORTFOLIO HISTORY:")
        cursor = conn.execute("SELECT COUNT(*) FROM portfolio_history")
        history_count = cursor.fetchone()[0]
        print(f"Total snapshots: {history_count}")
        
        if history_count > 0:
            cursor = conn.execute("""
                SELECT timestamp FROM portfolio_history 
                ORDER BY timestamp ASC 
                LIMIT 1
            """)
            first_snapshot = cursor.fetchone()[0]
            print(f"First snapshot: {first_snapshot}")
            print(f"Latest snapshot: {timestamp if latest else 'None'}")
        
    except Exception as e:
        print(f"❌ Error checking sync: {e}")
    
    finally:
        conn.close()

def force_portfolio_sync(db_path="trading_agent.db"):
    """Force a portfolio sync by recalculating from current positions"""
    
    print("\n🔄 Forcing Portfolio Sync...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Calculate current portfolio state from positions
        cursor = conn.execute("""
            SELECT SUM(shares * avg_cost) as total_invested
            FROM positions 
            WHERE shares > 0
        """)
        
        result = cursor.fetchone()
        total_invested = result[0] if result[0] else 0
        
        # Assume initial cash was $100,000
        initial_cash = 100000
        current_cash = initial_cash - total_invested
        total_value = current_cash + total_invested
        total_pnl = total_value - initial_cash
        
        # Insert new portfolio snapshot
        timestamp = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO portfolio_history 
            (timestamp, total_value, cash, positions_value, daily_pnl, total_pnl)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, total_value, current_cash, total_invested, 0.0, total_pnl))
        
        conn.commit()
        
        print(f"✅ Portfolio sync completed:")
        print(f"   Total Value: ${total_value:,.2f}")
        print(f"   Cash: ${current_cash:,.2f}")
        print(f"   Positions: ${total_invested:,.2f}")
        print(f"   P&L: ${total_pnl:,.2f}")
        
    except Exception as e:
        print(f"❌ Error forcing sync: {e}")
        conn.rollback()
    
    finally:
        conn.close()

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--force-sync':
        force_portfolio_sync()
        print("\n" + "=" * 50)
        check_portfolio_sync()
    else:
        check_portfolio_sync()
        
        print("\n" + "=" * 50)
        response = input("Force portfolio sync? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            force_portfolio_sync()

if __name__ == "__main__":
    main()