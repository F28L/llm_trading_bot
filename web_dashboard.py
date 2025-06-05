# web_dashboard.py - Complete web dashboard for trading bot

from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import os
import sqlite3
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime, timedelta

app = Flask(__name__)

class TradingBotDashboardAnalyzer:
    def __init__(self, db_path="trading_agent.db"):
        self.db_path = db_path
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def get_portfolio_summary(self):
        """Get current portfolio summary with real-time calculations"""
        try:
            with self.get_connection() as conn:
                # Get current positions and calculate their current market value
                positions_df = pd.read_sql_query("""
                    SELECT symbol, shares, avg_cost, realized_pnl
                    FROM positions 
                    WHERE shares > 0
                """, conn)
                
                # Calculate positions value (we'll use avg_cost as approximation since we don't store current prices in DB)
                total_invested = 0
                if not positions_df.empty:
                    positions_df['market_value'] = positions_df['shares'] * positions_df['avg_cost']
                    total_invested = positions_df['market_value'].sum()
                
                # Get latest portfolio snapshot for total value
                portfolio_df = pd.read_sql_query("""
                    SELECT total_value, total_pnl FROM portfolio_history 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, conn)
                
                if portfolio_df.empty:
                    return {"error": "No portfolio data found"}
                
                latest = portfolio_df.iloc[0]
                
                # Calculate cash as: total_value - positions_value
                total_value = float(latest['total_value'])
                current_cash = total_value - total_invested
                
                # Get transaction count
                txn_count = pd.read_sql_query("""
                    SELECT COUNT(*) as count FROM transactions
                """, conn).iloc[0]['count']
                
                return {
                    "total_value": total_value,
                    "cash": current_cash,  # Real-time calculated cash
                    "positions_value": total_invested,  # Real-time calculated positions value
                    "total_pnl": float(latest['total_pnl']),
                    "num_positions": len(positions_df),
                    "total_transactions": int(txn_count),
                    "last_updated": "Real-time calculation"
                }
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_chart_data(self, days=30):
        """Get portfolio performance data for charting"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT 
                        timestamp,
                        total_value,
                        cash,
                        positions_value,
                        total_pnl
                    FROM portfolio_history 
                    WHERE datetime(timestamp) > datetime('now', '-{} days')
                    ORDER BY timestamp
                """.format(days)
                
                return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error getting performance data: {e}")
            return pd.DataFrame()
    
    def get_transactions_summary(self, days=30):
        """Get recent transactions"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT 
                        timestamp,
                        symbol,
                        order_type,
                        shares,
                        price,
                        total_value,
                        reason
                    FROM transactions 
                    WHERE datetime(timestamp) > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                """.format(days)
                
                return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error getting transactions: {e}")
            return pd.DataFrame()
    
    def get_positions_analysis(self):
        """Get current positions with real-time market values"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT 
                        symbol,
                        shares,
                        avg_cost,
                        realized_pnl,
                        shares * avg_cost as total_invested
                    FROM positions 
                    WHERE shares > 0
                    ORDER BY shares * avg_cost DESC
                """
                
                df = pd.read_sql_query(query, conn)
                
                # Add current market value calculation
                # For now, we'll use avg_cost as current_price since we don't store live prices
                # In a real system, you'd fetch current prices here
                if not df.empty:
                    df['current_price'] = df['avg_cost']  # Placeholder - could fetch real prices
                    df['market_value'] = df['shares'] * df['current_price']
                    df['unrealized_pnl'] = df['market_value'] - df['total_invested']
                
                return df
        except Exception as e:
            print(f"Error getting positions: {e}")
            return pd.DataFrame()
    
    def get_trading_signals_analysis(self, days=7):
        """Get recent trading signals"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT 
                        timestamp,
                        symbol,
                        action,
                        confidence,
                        target_allocation,
                        reasoning,
                        executed
                    FROM trading_signals 
                    WHERE datetime(timestamp) > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                """.format(days)
                
                return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error getting signals: {e}")
            return pd.DataFrame()

# Initialize analyzer
analyzer = TradingBotDashboardAnalyzer()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    try:
        summary = analyzer.get_portfolio_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance chart data"""
    days = request.args.get('days', 30, type=int)
    
    try:
        df = analyzer.get_performance_chart_data(days)
        
        if df.empty:
            return jsonify({"error": "No performance data"})
        
        # Create Plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86AB', width=3),
            hovertemplate='<b>Portfolio Value</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_pnl'],
            mode='lines',
            name='P&L',
            line=dict(color='#A23B72', width=2),
            yaxis='y2',
            hovertemplate='<b>P&L</b><br>Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Portfolio Performance - Last {days} Days',
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            xaxis=dict(title='Date', showgrid=True, gridcolor='#ECF0F1'),
            yaxis=dict(
                title='Portfolio Value ($)', 
                side='left',
                showgrid=True,
                gridcolor='#ECF0F1',
                tickformat='$,.0f'
            ),
            yaxis2=dict(
                title='P&L ($)', 
                side='right', 
                overlaying='y',
                tickformat='$,.0f'
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2C3E50")
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": graphJSON})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/transactions')
def api_transactions():
    """API endpoint for recent transactions"""
    limit = request.args.get('limit', 20, type=int)
    
    try:
        df = analyzer.get_transactions_summary(30)
        
        if df.empty:
            return jsonify({"transactions": []})
        
        transactions = []
        for _, txn in df.head(limit).iterrows():
            transactions.append({
                "timestamp": txn['timestamp'],
                "symbol": txn['symbol'],
                "action": txn['order_type'],
                "shares": round(float(txn['shares']), 3),
                "price": round(float(txn['price']), 2),
                "total": round(float(txn['total_value']), 2),
                "reason": txn['reason'] if pd.notna(txn['reason']) else ""
            })
        
        return jsonify({"transactions": transactions})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/positions')
def api_positions():
    """API endpoint for current positions"""
    try:
        df = analyzer.get_positions_analysis()
        
        if df.empty:
            return jsonify({"positions": []})
        
        positions = []
        for _, pos in df.iterrows():
            position_data = {
                "symbol": pos['symbol'],
                "shares": round(float(pos['shares']), 3),
                "avg_cost": round(float(pos['avg_cost']), 2),
                "total_invested": round(float(pos['total_invested']), 2),
                "realized_pnl": round(float(pos['realized_pnl']), 2)
            }
            
            # Add market value and unrealized P&L if available
            if 'market_value' in pos:
                position_data['market_value'] = round(float(pos['market_value']), 2)
                position_data['current_price'] = round(float(pos['current_price']), 2)
                position_data['unrealized_pnl'] = round(float(pos['unrealized_pnl']), 2)
            
            positions.append(position_data)
        
        return jsonify({"positions": positions})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/debug')
def api_debug():
    """Debug endpoint to see raw database values"""
    try:
        with analyzer.get_connection() as conn:
            # Get latest portfolio history
            portfolio_raw = pd.read_sql_query("""
                SELECT * FROM portfolio_history 
                ORDER BY timestamp DESC LIMIT 1
            """, conn)
            
            # Get all positions
            positions_raw = pd.read_sql_query("""
                SELECT * FROM positions WHERE shares > 0
            """, conn)
            
            # Get recent transactions
            transactions_raw = pd.read_sql_query("""
                SELECT * FROM transactions 
                ORDER BY timestamp DESC LIMIT 5
            """, conn)
            
            return jsonify({
                "portfolio_history": portfolio_raw.to_dict('records') if not portfolio_raw.empty else [],
                "positions": positions_raw.to_dict('records') if not positions_raw.empty else [],
                "recent_transactions": transactions_raw.to_dict('records') if not transactions_raw.empty else []
            })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/signals')
def api_signals():
    """API endpoint for trading signals"""
    days = request.args.get('days', 7, type=int)
    
    try:
        df = analyzer.get_trading_signals_analysis(days)
        
        if df.empty:
            return jsonify({"signals": []})
        
        signals = []
        for _, signal in df.iterrows():
            signals.append({
                "timestamp": signal['timestamp'],
                "symbol": signal['symbol'],
                "action": signal['action'],
                "confidence": round(float(signal['confidence']), 2),
                "target_allocation": round(float(signal['target_allocation']), 3),
                "reasoning": signal['reasoning'],
                "executed": bool(signal['executed'])
            })
        
        return jsonify({"signals": signals})
        
    except Exception as e:
        return jsonify({"error": str(e)})

# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.95);
            color: #2C3E50;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        
        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 8px;
        }
        
        .stat-label {
            color: #7F8C8D;
            font-size: 1.1em;
            font-weight: 500;
        }
        
        .chart-container, .table-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .chart-container h2, .table-container h2 {
            color: #2C3E50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }
        
        th, td {
            padding: 15px 12px;
            text-align: left;
            border-bottom: 1px solid #ECF0F1;
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85em;
        }
        
        tr:hover {
            background-color: #F8F9FA;
        }
        
        .positive { color: #27AE60; font-weight: 600; }
        .negative { color: #E74C3C; font-weight: 600; }
        .neutral { color: #7F8C8D; font-weight: 600; }
        
        .buy { background-color: rgba(39, 174, 96, 0.1); }
        .sell { background-color: rgba(231, 76, 60, 0.1); }
        .hold { background-color: rgba(241, 196, 15, 0.1); }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #7F8C8D;
            font-size: 1.1em;
        }
        
        .refresh-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .status-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            font-size: 0.9em;
            z-index: 1000;
        }
        
        .status-online {
            background: #27AE60;
        }
        
        .status-offline {
            background: #E74C3C;
        }
        
        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            th, td {
                padding: 10px 8px;
                font-size: 0.9em;
            }
        }
    </style>
</head>
<body>
    <div class="status-indicator status-online" id="statusIndicator">🟢 LIVE</div>
    
    <div class="container">
        <div class="header">
            <h1>🤖 AI Trading Bot Dashboard</h1>
            <p>Real-time portfolio monitoring and intelligent trading analysis</p>
            <button class="refresh-btn" onclick="loadAllData()">🔄 Refresh All Data</button>
        </div>

        <div class="stats-grid" id="statsGrid">
            <div class="loading">Loading portfolio data...</div>
        </div>

        <div class="chart-container">
            <h2>📈 Portfolio Performance</h2>
            <div id="performanceChart" style="height: 450px;"></div>
        </div>

        <div class="grid-2">
            <div class="table-container">
                <h2>💼 Current Positions</h2>
                <div id="positionsTable">
                    <div class="loading">Loading positions...</div>
                </div>
            </div>

            <div class="table-container">
                <h2>📝 Recent Transactions</h2>
                <div id="transactionsTable">
                    <div class="loading">Loading transactions...</div>
                </div>
            </div>
        </div>

        <div class="table-container">
            <h2>🎯 Trading Signals</h2>
            <div id="signalsTable">
                <div class="loading">Loading signals...</div>
            </div>
        </div>
    </div>

    <script>
        let refreshInterval;
        
        // Fetch and display portfolio summary
        async function loadPortfolioSummary() {
            try {
                const response = await fetch('/api/portfolio');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('statsGrid').innerHTML = `<div class="stat-card"><div class="stat-value">Error</div><div class="stat-label">${data.error}</div></div>`;
                    return;
                }

                const totalReturn = ((data.total_value - 100000) / 100000) * 100; // Assuming 100k initial
                
                const statsHTML = `
                    <div class="stat-card">
                        <div class="stat-value">$${data.total_value.toLocaleString()}</div>
                        <div class="stat-label">Total Portfolio Value</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value ${data.total_pnl >= 0 ? 'positive' : 'negative'}">$${data.total_pnl.toLocaleString()}</div>
                        <div class="stat-label">Total P&L</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value ${totalReturn >= 0 ? 'positive' : 'negative'}">${totalReturn.toFixed(2)}%</div>
                        <div class="stat-label">Total Return</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.num_positions}</div>
                        <div class="stat-label">Active Positions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.total_transactions}</div>
                        <div class="stat-label">Total Trades</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">$${data.cash.toLocaleString()}</div>
                        <div class="stat-label">Available Cash</div>
                    </div>
                `;
                
                document.getElementById('statsGrid').innerHTML = statsHTML;
                updateStatus(true);
            } catch (error) {
                console.error('Error loading portfolio summary:', error);
                updateStatus(false);
            }
        }

        // Load performance chart
        async function loadPerformanceChart() {
            try {
                const response = await fetch('/api/performance?days=30');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('performanceChart').innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                    return;
                }

                const chartData = JSON.parse(data.chart);
                Plotly.newPlot('performanceChart', chartData.data, chartData.layout, {responsive: true});
            } catch (error) {
                console.error('Error loading performance chart:', error);
                document.getElementById('performanceChart').innerHTML = '<div class="loading">Error loading chart</div>';
            }
        }

        // Load positions table
        async function loadPositions() {
            try {
                const response = await fetch('/api/positions');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('positionsTable').innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                    return;
                }

                if (data.positions.length === 0) {
                    document.getElementById('positionsTable').innerHTML = '<div class="loading">No positions yet</div>';
                    return;
                }

                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Shares</th>
                                <th>Avg Cost</th>
                                <th>Invested</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                data.positions.forEach(pos => {
                    tableHTML += `
                        <tr>
                            <td><strong>${pos.symbol}</strong></td>
                            <td>${pos.shares}</td>
                            <td>$${pos.avg_cost}</td>
                            <td>$${pos.total_invested.toLocaleString()}</td>
                            <td class="${pos.realized_pnl >= 0 ? 'positive' : 'negative'}">$${pos.realized_pnl}</td>
                        </tr>
                    `;
                });

                tableHTML += '</tbody></table>';
                document.getElementById('positionsTable').innerHTML = tableHTML;
            } catch (error) {
                console.error('Error loading positions:', error);
            }
        }

        // Load transactions table
        async function loadTransactions() {
            try {
                const response = await fetch('/api/transactions?limit=10');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('transactionsTable').innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                    return;
                }

                if (data.transactions.length === 0) {
                    document.getElementById('transactionsTable').innerHTML = '<div class="loading">No transactions yet</div>';
                    return;
                }

                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Action</th>
                                <th>Symbol</th>
                                <th>Shares</th>
                                <th>Price</th>
                                <th>Total</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                data.transactions.forEach(txn => {
                    const actionClass = txn.action.toLowerCase();
                    const timestamp = new Date(txn.timestamp).toLocaleString();
                    
                    tableHTML += `
                        <tr class="${actionClass}">
                            <td>${timestamp}</td>
                            <td><strong>${txn.action}</strong></td>
                            <td>${txn.symbol}</td>
                            <td>${txn.shares}</td>
                            <td>$${txn.price}</td>
                            <td>$${txn.total.toLocaleString()}</td>
                        </tr>
                    `;
                });

                tableHTML += '</tbody></table>';
                document.getElementById('transactionsTable').innerHTML = tableHTML;
            } catch (error) {
                console.error('Error loading transactions:', error);
            }
        }

        // Load trading signals
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals?days=7');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('signalsTable').innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                    return;
                }

                if (data.signals.length === 0) {
                    document.getElementById('signalsTable').innerHTML = '<div class="loading">No recent signals</div>';
                    return;
                }

                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Action</th>
                                <th>Symbol</th>
                                <th>Confidence</th>
                                <th>Target %</th>
                                <th>Status</th>
                                <th>Reasoning</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                data.signals.forEach(signal => {
                    const actionClass = signal.action.toLowerCase();
                    const timestamp = new Date(signal.timestamp).toLocaleString();
                    const executed = signal.executed ? '✅ Executed' : '❌ Not Executed';
                    
                    tableHTML += `
                        <tr class="${actionClass}">
                            <td>${timestamp}</td>
                            <td><strong>${signal.action}</strong></td>
                            <td>${signal.symbol}</td>
                            <td>${(signal.confidence * 100).toFixed(0)}%</td>
                            <td>${(signal.target_allocation * 100).toFixed(1)}%</td>
                            <td>${executed}</td>
                            <td style="max-width: 300px; word-wrap: break-word;">${signal.reasoning}</td>
                        </tr>
                    `;
                });

                tableHTML += '</tbody></table>';
                document.getElementById('signalsTable').innerHTML = tableHTML;
            } catch (error) {
                console.error('Error loading signals:', error);
            }
        }
        
        function updateStatus(isOnline) {
            const indicator = document.getElementById('statusIndicator');
            if (isOnline) {
                indicator.className = 'status-indicator status-online';
                indicator.textContent = '🟢 LIVE';
            } else {
                indicator.className = 'status-indicator status-offline';
                indicator.textContent = '🔴 OFFLINE';
            }
        }

        // Load all data
        function loadAllData() {
            loadPortfolioSummary();
            loadPerformanceChart();
            loadPositions();
            loadTransactions();
            loadSignals();
        }

        // Initialize dashboard
        window.addEventListener('load', function() {
            loadAllData();
            
            // Auto-refresh every 30 seconds
            refreshInterval = setInterval(() => {
                loadPortfolioSummary();
                loadPositions();
                loadTransactions();
                loadSignals();
            }, 30000);
            
            // Refresh chart every 5 minutes
            setInterval(loadPerformanceChart, 300000);
        });
        
        // Clean up on page unload
        window.addEventListener('beforeunload', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>
"""

def create_dashboard_template():
    """Create the dashboard HTML template file"""
    import os
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Write the dashboard HTML
    with open('templates/dashboard.html', 'w') as f:
        f.write(DASHBOARD_HTML)
    
    print("✅ Dashboard template created at templates/dashboard.html")

def run_dashboard(port=8080, debug=False):
    """Run the web dashboard"""
    create_dashboard_template()
    print(f"🚀 Starting AI Trading Bot Web Dashboard")
    print("=" * 50)
    print(f"🌐 URL: http://localhost:{port}")
    print(f"📱 Mobile friendly: Yes")
    print(f"🔄 Auto-refresh: Every 30 seconds")
    print("=" * 50)
    print("📊 Dashboard features:")
    print("   • Real-time portfolio summary")
    print("   • Interactive performance charts") 
    print("   • Current positions table")
    print("   • Recent transactions log")
    print("   • Trading signals analysis")
    print("   • Live status indicator")
    print("   • Responsive design")
    print("=" * 50)
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")

if __name__ == "__main__":
    run_dashboard()