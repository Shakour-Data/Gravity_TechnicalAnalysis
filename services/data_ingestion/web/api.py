import logging
import os
import re
import sys

# Add src to path for runtime imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, parent_dir)

from flask import Flask, jsonify, request  # noqa: E402
from flask_cors import CORS  # noqa: E402
import sqlite3  # noqa: E402
import pandas as pd  # noqa: E402

from config import DB_FILE  # noqa: E402  # type: ignore
from encoding_utils import ensure_utf8_console  # noqa: E402  # type: ignore

ensure_utf8_console()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gravity_api")


def _safe_limit(raw_limit, default=100, max_limit=5000):
    """Validate and clamp limit."""
    try:
        value = int(raw_limit)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return min(value, max_limit)


def _safe_sector_id(raw_sector_id):
    """Convert sector id to int or None."""
    if raw_sector_id is None:
        return None
    try:
        return int(raw_sector_id)
    except (TypeError, ValueError):
        return None


def _safe_ticker(raw_ticker):
    """Restrict ticker to a safe subset."""
    if raw_ticker is None:
        return None
    ticker = str(raw_ticker).strip().upper()
    if not ticker or not re.fullmatch(r"[A-Z0-9._-]{1,15}", ticker):
        return None
    return ticker

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_FILE)

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get market summary statistics"""
    try:
        conn = get_db_connection()

        # Get total companies
        companies_count = pd.read_sql("SELECT COUNT(*) as count FROM companies", conn).iloc[0]['count']

        # Get total price records
        price_records = pd.read_sql("SELECT COUNT(*) as count FROM price_data", conn).iloc[0]['count']

        # Get latest update date
        latest_update = pd.read_sql("SELECT MAX(date) as latest FROM price_data", conn).iloc[0]['latest']

        # Get sectors count
        sectors_count = pd.read_sql("SELECT COUNT(*) as count FROM sectors", conn).iloc[0]['count']

        conn.close()

        return jsonify({
            'status': 'success',
            'data': {
                'companies': companies_count,
                'price_records': price_records,
                'latest_update': latest_update,
                'sectors': sectors_count
            }
        })
    except Exception:
        logger.exception("failed to fetch sectors")
        return jsonify({
            'status': 'error',
            'message': 'internal error'
        }), 500

@app.route('/api/companies', methods=['GET'])
def get_companies():
    """Get companies list with optional filtering"""
    try:
        conn = get_db_connection()

        sector_id = _safe_sector_id(request.args.get('sector_id'))
        limit = _safe_limit(request.args.get('limit', 100))

        query = """
            SELECT c.ticker, c.name, c.sector_id, s.sector_name
            FROM companies c
            LEFT JOIN sectors s ON c.sector_id = s.sector_id
        """

        params = []
        if sector_id is not None:
            query += " WHERE c.sector_id = ?"
            params.append(sector_id)

        query += " ORDER BY c.name LIMIT ?"
        params.append(limit)

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        return jsonify({
            'status': 'success',
            'data': df.to_dict('records')
        })
    except Exception:
        logger.exception("failed to fetch companies")
        return jsonify({
            'status': 'error',
            'message': 'internal error'
        }), 500

@app.route('/api/price-data/<ticker>', methods=['GET'])
def get_price_data(ticker):
    """Get price data for a specific ticker"""
    try:
        conn = get_db_connection()

        limit = _safe_limit(request.args.get('limit', 100))
        safe_ticker = _safe_ticker(ticker)
        if safe_ticker is None:
            return jsonify({'status': 'error', 'message': 'invalid ticker'}), 400

        query = """
            SELECT date, adj_open, adj_high, adj_low, adj_close, adj_final, adj_volume
            FROM price_data
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(safe_ticker, limit))
        conn.close()

        return jsonify({
            'status': 'success',
            'ticker': safe_ticker,
            'data': df.to_dict('records')
        })
    except Exception:
        logger.exception("failed to fetch price data for %s", ticker)
        return jsonify({
            'status': 'error',
            'message': 'internal error'
        }), 500

@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    """Get all sectors"""
    try:
        conn = get_db_connection()

        query = """
            SELECT s.sector_id, s.sector_name, COUNT(c.company_id) as company_count
            FROM sectors s
            LEFT JOIN companies c ON s.sector_id = c.sector_id
            GROUP BY s.sector_id, s.sector_name
            ORDER BY s.sector_name
        """

        df = pd.read_sql(query, conn)
        conn.close()

        return jsonify({
            'status': 'success',
            'data': df.to_dict('records')
        })
    except Exception:
        logger.exception("failed to fetch sectors")
        return jsonify({
            'status': 'error',
            'message': 'internal error'
        }), 500

@app.route('/api/market-indices', methods=['GET'])
def get_market_indices():
    """Get market indices data (joined with index metadata)."""
    try:
        conn = get_db_connection()

        limit = _safe_limit(request.args.get('limit', 100))

        query = """
            SELECT
                mi.date,
                mi.j_date,
                mi.index_code,
                ii.index_name_fa AS index_name,
                mi.open,
                mi.high,
                mi.low,
                mi.close
            FROM market_indices mi
            LEFT JOIN indices_info ii ON mi.index_code = ii.index_code
            ORDER BY mi.date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(limit,))
        conn.close()

        return jsonify({
            'status': 'success',
            'data': df.to_dict('records')
        })
    except Exception:
        logger.exception("failed to fetch market indices")
        return jsonify({
            'status': 'error',
            'message': 'internal error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'version': '2.0.0'
        })
    except Exception:
        logger.exception("health check failed")
        return jsonify({
            'status': 'unhealthy',
            'error': 'internal error'
        }), 500

if __name__ == '__main__':
    print("🔌 Starting GravityTseHisPrice API Server...")
    print("📡 API will be available at: http://127.0.0.1:5000/")
    print("📖 API Documentation: http://127.0.0.1:5000/health")
    print("❌ Press Ctrl+C to stop the server")
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
