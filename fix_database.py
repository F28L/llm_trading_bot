#!/usr/bin/env python3
# fix_database.py - Repair database inconsistencies

import sqlite3
from collections import defaultdict
from datetime import datetime

def fix_positions_table(db_path="trading_agent.db"):
    """Rebuild positions table from transaction history"""
    
    print("🔧 Repairing positions table from transaction history...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Get all transactions
        cursor = conn.execute("""
            SELECT symbol, order_type, shares, price, total_value
            FROM transactions 
            ORDER BY timestamp ASC
        """)
        
        transactions = cursor.fetchall()
        
        if not transactions:
            print("❌ No transactions found")
            return
        
        # Calculate positions from scratch
        positions = defaultdict(lambda: {'shares': 0.0, 'total_cost': 0.0, 'realized_pnl': 0.0})
        
        for symbol, order_type, shares, price, total_value in transactions:
            if order_type == "BUY":
                positions[symbol]['shares'] += shares
                positions[symbol]['total_cost'] += total_value
                
            elif order_type == "SELL":
                if positions[symbol]['shares'] >= shares:
                    # Calculate realized P&L for this sale
                    avg_cost = positions[symbol]['total_cost'] / positions[symbol]['shares'] if positions[symbol]['shares'] > 0 else 0
                    realized_pnl = (price - avg_cost) * shares
                    positions[symbol]['realized_pnl'] += realized_pnl
                    
                    # Reduce position
                    cost_reduction = (shares / positions[symbol]['shares']) * positions[symbol]['total_cost']
                    positions[symbol]['shares'] -= shares
                    positions[symbol]['total_cost'] -= cost_reduction
                else:
                    print(f"⚠️  Warning: Oversold {symbol} - {shares} shares when only {positions[symbol]['shares']} available")
        
        # Clear existing positions table
        conn.execute("DELETE FROM positions")
        
        # Insert corrected positions
        positions_added = 0
        for symbol, data in positions.items():
            if data['shares'] > 0.001:  # Only keep positions with meaningful shares
                avg_cost = data['total_cost'] / data['shares'] if data['shares'] > 0 else 0
                
                conn.execute("""
                    INSERT INTO positions (symbol, shares, avg_cost, realized_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol,
                    data['shares'],
                    avg_cost,
                    data['realized_pnl'],
                    datetime.now().isoformat()
                ))
                positions_added += 1
                
                print(f"✅ {symbol}: {data['shares']:.3f} shares @ ${avg_cost:.2f} avg cost")
        
        conn.commit()
        print(f"\n🎉 Successfully rebuilt positions table with {positions_added} active positions")
        
        # Verify the fix
        cursor = conn.execute("SELECT COUNT(*) FROM positions WHERE shares > 0")
        db_positions = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT total_value FROM portfolio_history ORDER BY timestamp DESC LIMIT 1")
        latest_portfolio = cursor.fetchone()
        
        if latest_portfolio:
            print(f"📊 Database now shows {db_positions} positions")
            print(f"💼 Latest portfolio value: ${latest_portfolio[0]:,.2f}")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def verify_database_consistency(db_path="trading_agent.db"):
    """Check database consistency"""
    
    print("\n🔍 Verifying database consistency...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Check positions count
        cursor = conn.execute("SELECT COUNT(*) FROM positions WHERE shares > 0")
        positions_count = cursor.fetchone()[0]
        
        # Check latest portfolio snapshot
        cursor = conn.execute("""
            SELECT total_value, cash, positions_value 
            FROM portfolio_history 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        portfolio_data = cursor.fetchone()
        
        # Check transaction count
        cursor = conn.execute("SELECT COUNT(*) FROM transactions")
        txn_count = cursor.fetchone()[0]
        
        print(f"📊 Positions in database: {positions_count}")
        print(f"📝 Total transactions: {txn_count}")
        
        if portfolio_data:
            total_value, cash, positions_value = portfolio_data
            print(f"💼 Portfolio value: ${total_value:,.2f}")
            print(f"💵 Cash: ${cash:,.2f}")
            print(f"📈 Positions value: ${positions_value:,.2f}")
            
            if positions_count > 0 and positions_value > 0:
                print("✅ Database appears consistent")
            elif positions_count == 0 and positions_value == 0:
                print("✅ Database consistent (no positions)")
            else:
                print("⚠️  Inconsistency detected between positions table and portfolio history")
        else:
            print("❌ No portfolio history found")
    
    except Exception as e:
        print(f"❌ Error verifying database: {e}")
    finally:
        conn.close()

def show_current_positions(db_path="trading_agent.db"):
    """Show current positions from corrected database"""
    
    print("\n📋 CURRENT POSITIONS (After Fix)")
    print("-" * 70)
    
    conn = sqlite3.connect(db_path)
    
    try:
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
        
        print(f"{'Symbol':<8} {'Shares':<10} {'Avg Cost':<10} {'Invested':<12} {'Realized P&L':<12}")
        print("-" * 70)
        
        total_invested = 0
        for pos in positions:
            symbol, shares, avg_cost, realized_pnl, invested = pos
            print(f"{symbol:<8} {shares:<10.3f} ${avg_cost:<9.2f} ${invested:<11,.2f} ${realized_pnl:<11.2f}")
            total_invested += invested
        
        print("-" * 70)
        print(f"{'TOTAL':<8} {'':<10} {'':<10} ${total_invested:<11,.2f}")
        
    except Exception as e:
        print(f"❌ Error showing positions: {e}")
    finally:
        conn.close()

def main():
    """Main repair function"""
    print("🔧 AI Trading Bot Database Repair Tool")
    print("=" * 50)
    
    # Check current state
    verify_database_consistency()
    
    # Ask user if they want to proceed with repair
    print("\n" + "=" * 50)
    response = input("Do you want to rebuild the positions table? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # Backup database first
        import shutil
        backup_name = f"trading_agent_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        try:
            shutil.copy("trading_agent.db", backup_name)
            print(f"📁 Database backed up to: {backup_name}")
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
        
        # Fix the database
        fix_positions_table()
        
        # Verify fix
        verify_database_consistency()
        
        # Show corrected positions
        show_current_positions()
        
    else:
        print("👋 Repair cancelled")

if __name__ == "__main__":
    main()