"""
DB explorer endpoints for UI: list tables, query rows, backup, schema/info.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from gravity_tech.database.database_manager import DatabaseManager, DatabaseType
from pydantic import BaseModel

logger = structlog.get_logger()

router = APIRouter(tags=["DB Explorer"], prefix="/db")

FORBIDDEN_TABLES = {"unauthorized"}


class TableListResponse(BaseModel):
    tables: list[str]


class QueryResponse(BaseModel):
    table: str
    rows: list[dict[str, Any]]
    total: int


class TableSchemaResponse(BaseModel):
    table: str
    columns: list[dict[str, Any]]


class DatabaseInfoResponse(BaseModel):
    info: dict[str, Any]
    stats: dict[str, int]


def _list_tables(dbm: DatabaseManager) -> list[str]:
    if dbm.db_type == DatabaseType.JSON_FILE:
        return sorted(dbm.json_data.keys())

    conn = dbm.get_connection()
    cursor = conn.cursor()
    tables: list[str] = []
    try:
        if dbm.db_type == DatabaseType.SQLITE:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
        else:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
            )
            tables = [row[0] for row in cursor.fetchall()]
    except Exception as exc:
        logger.error("db_list_tables_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to list tables") from exc
    return tables


def _get_table_columns(dbm: DatabaseManager, table: str) -> list[str]:
    if dbm.db_type == DatabaseType.JSON_FILE:
        records = dbm.json_data.get(table, [])
        if records and isinstance(records, list) and isinstance(records[0], dict):
            return list(records[0].keys())
        return []

    conn = dbm.get_connection()
    cursor = conn.cursor()
    try:
        if dbm.db_type == DatabaseType.SQLITE:
            cursor.execute(f"PRAGMA table_info({table})")
            return [row[1] for row in cursor.fetchall()]
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception as exc:
        logger.error("db_get_columns_failed", table=table, error=str(exc))
        return []
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            if dbm.db_type == DatabaseType.POSTGRESQL:
                dbm.release_connection(conn)
        except Exception:
            pass


@router.get("/tables", response_model=TableListResponse)
async def list_tables() -> TableListResponse:
    """Return tables available for UI browsing (all tables)."""
    dbm = DatabaseManager(auto_setup=True)
    return TableListResponse(tables=_list_tables(dbm))


@router.get("/info", response_model=DatabaseInfoResponse)
async def db_info() -> DatabaseInfoResponse:
    """Return database info + per-table row counts."""
    dbm = DatabaseManager(auto_setup=True)
    return DatabaseInfoResponse(info=dbm.get_database_info(), stats=dbm.get_statistics())


@router.get("/schema", response_model=TableSchemaResponse)
async def table_schema(table: str = Query(..., description="Table name")) -> TableSchemaResponse:
    dbm = DatabaseManager(auto_setup=True)
    available = _list_tables(dbm)
    if table not in available:
        raise HTTPException(status_code=404, detail="Table not found")
    return TableSchemaResponse(table=table, columns=dbm.get_table_schema(table))


@router.get("/backup")
async def download_backup(tables: list[str] | None = Query(None, description="Optional list of tables")):
    """Download a backup (JSON payload)."""
    dbm = DatabaseManager(auto_setup=True)
    backup_data = dbm.create_backup(tables)
    return JSONResponse(
        content=backup_data,
        headers={"Content-Disposition": "attachment; filename=gravity_db_backup.json"},
    )


@router.get("/query", response_model=QueryResponse)
async def query_table(
    table: str = Query(..., description="Table name"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    symbol: str | None = Query(None, description="Optional symbol filter"),
) -> QueryResponse:
    """Query a table with optional symbol filter and pagination."""
    dbm = DatabaseManager(auto_setup=True)
    available = _list_tables(dbm)
    if table in FORBIDDEN_TABLES:
        raise HTTPException(status_code=403, detail="Table access forbidden")
    if table not in available:
        raise HTTPException(status_code=404, detail="Table not found")

    # JSON fallback
    if dbm.db_type == DatabaseType.JSON_FILE:
        records = dbm.json_data.get(table, [])
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol]
        total = len(records)
        rows = records[offset : offset + limit]
        return QueryResponse(table=table, rows=rows, total=total)

    conn = dbm.get_connection()
    cursor = conn.cursor()
    params: list[Any] = []

    placeholder = "?" if dbm.db_type == DatabaseType.SQLITE else "%s"
    where_clause = ""
    if symbol:
        where_clause = f"WHERE symbol = {placeholder}"
        params.append(symbol)

    # Order by a known column if present
    order_clause = ""
    columns = _get_table_columns(dbm, table)
    for col in ["created_at", "timestamp", "id", "rowid"]:
        if col in columns:
            order_clause = f"ORDER BY {col} DESC"
            break

    query = f"SELECT * FROM {table} {where_clause} {order_clause} LIMIT {placeholder} OFFSET {placeholder}"
    params.extend([limit, offset])

    try:
        cursor.execute(query, tuple(params))
        cols = [desc[0] for desc in cursor.description]
        fetched = cursor.fetchall()
        rows = [dict(zip(cols, row, strict=True)) for row in fetched]

        # total count (best-effort)
        count_query = f"SELECT COUNT(*) FROM {table} {where_clause}"
        cursor.execute(count_query, tuple(params[:-2]) if symbol else ())
        total = cursor.fetchone()[0]
    except Exception as exc:
        logger.error("db_query_failed", table=table, error=str(exc))
        raise HTTPException(status_code=500, detail="Query failed") from exc

    return QueryResponse(table=table, rows=rows, total=total)


@router.get("/ui", response_class=HTMLResponse)
async def db_ui():
    """
    Minimal DB explorer UI (vanilla JS) for browsing tables, schema, stats, backup.
    """
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DB Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0b1220;
      --card: #111a2e;
      --accent: #3dd598;
      --accent-2: #21d4fd;
      --text: #e8edf5;
      --muted: #93a4c2;
      --border: #1f2c45;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    header h1 { margin: 0; font-size: 18px; letter-spacing: 0.5px; }
    .nav { margin-left: auto; display: inline-flex; gap: 10px; flex-wrap: wrap; }
    .nav a { color: var(--muted); font-size: 13px; text-decoration: none; padding: 6px 10px; border-radius: 8px; border: 1px solid var(--border); background: #0f1629; }
    .nav a:hover { color: var(--text); border-color: var(--accent); }
    main { padding: 20px 24px; display: grid; grid-template-columns: 360px 1fr; gap: 16px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.35); }
    label { display: block; margin-bottom: 6px; color: var(--muted); font-size: 13px; }
    select, input, button { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: #0f1629; color: var(--text); }
    button { margin-top: 10px; background: linear-gradient(135deg, #38ef7d, #11998e); color: #0b1220; font-weight: 700; cursor: pointer; border: none; transition: transform 0.1s ease; }
    button.secondary { background: linear-gradient(135deg, #21d4fd, #3dd598); color: #0b1220; }
    button:hover { filter: brightness(1.05); transform: translateY(-1px); }
    .table-wrap { max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
    th { position: sticky; top: 0; background: #0f182d; color: var(--muted); }
    .muted { color: var(--muted); font-size: 12px; }
    #chart-card { height: 360px; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; background: #0f1629; color: var(--muted); font-size: 12px; }
    .schema-box { background: #0f1629; border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-top: 10px; max-height: 200px; overflow: auto; font-size: 12px; color: var(--muted); }
    .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    @media(max-width: 980px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>DB Explorer</h1>
    <span class="pill">تمام عملیات دیتابیس با یک کلیک</span>
    <div class="nav">
      <a href="/api/v1/db/home" target="_blank">خانه</a>
      <a href="/api/docs" target="_blank">Swagger</a>
      <a href="/api/v1/db/ui">DB UI</a>
      <a href="/api/v1/backtest" target="_blank">Backtest API</a>
    </div>
  </header>
  <main>
    <section class="card">
      <div class="grid-2">
        <div>
          <label>Table</label>
          <select id="tableSelect"></select>
        </div>
        <div>
          <label>Limit</label>
          <input id="limitInput" type="number" value="50" min="1" max="500" />
        </div>
      </div>
      <div style="margin-top:10px;">
        <label>Symbol (اختیاری)</label>
        <input id="symbolInput" placeholder="مثال: FOLD" />
      </div>
      <div class="grid-2" style="margin-top:10px; gap:8px;">
        <button id="fetchBtn">🔍 نمایش داده‌ها</button>
        <button id="exportBtn" class="secondary">⬇️ خروجی CSV</button>
      </div>
      <div class="grid-2" style="margin-top:10px; gap:8px;">
        <button id="schemaBtn" class="secondary">📑 نمایش شِما</button>
        <button id="backupBtn">💾 بکاپ همه جداول</button>
      </div>
      <div class="schema-box" id="schemaBox" style="display:none;"></div>
      <div style="margin-top:14px;">
        <label>نمایش نمودار (ستون عددی/زمانی)</label>
        <select id="columnSelect"></select>
        <button id="chartBtn" style="margin-top:8px;">📈 ترسیم</button>
      </div>
      <div style="margin-top:14px;">
        <label>وضعیت دیتابیس</label>
        <div id="dbInfo" class="schema-box">Loading...</div>
      </div>
    </section>
    <section class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <div class="pill" id="tableMeta">Rows: 0</div>
        <div class="pill" id="tableNamePill">—</div>
      </div>
      <div class="table-wrap">
        <table id="dataTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
      <div id="chart-card" style="margin-top:12px;">
        <canvas id="chartCanvas"></canvas>
      </div>
    </section>
  </main>
  <script>
    const tablesEl = document.getElementById('tableSelect');
    const symbolEl = document.getElementById('symbolInput');
    const limitEl = document.getElementById('limitInput');
    const fetchBtn = document.getElementById('fetchBtn');
    const exportBtn = document.getElementById('exportBtn');
    const columnSelect = document.getElementById('columnSelect');
    const chartBtn = document.getElementById('chartBtn');
    const schemaBtn = document.getElementById('schemaBtn');
    const backupBtn = document.getElementById('backupBtn');
    const schemaBox = document.getElementById('schemaBox');
    const dbInfoBox = document.getElementById('dbInfo');
    const thead = document.querySelector('#dataTable thead');
    const tbody = document.querySelector('#dataTable tbody');
    const tableMeta = document.getElementById('tableMeta');
    const tableNamePill = document.getElementById('tableNamePill');
    let currentRows = [];
    let chart;

    async function loadTables() {
      const res = await fetch('/api/v1/db/tables');
      const data = await res.json();
      data.tables.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t; opt.textContent = t;
        tablesEl.appendChild(opt);
      });
      if (data.tables.length) tablesEl.value = data.tables[0];
    }

    async function loadInfo() {
      const res = await fetch('/api/v1/db/info');
      const data = await res.json();
      const info = data.info || {};
      const stats = data.stats || {};
      dbInfoBox.style.display = 'block';
      dbInfoBox.innerHTML = `
        نوع: <b>${info.type || '-'}</b><br>
        مسیر/کانکشن: <span style="color:#e8edf5">${info.path || info.connection || '-'}</span><br>
        تعداد جداول: ${info.table_count ?? '-'}<br>
        اندازه (SQLite): ${info.size || '-'}<br>
        شمارنده ردیف‌ها: ${Object.keys(stats).length ? Object.entries(stats).map(([k,v]) => `${k}: ${v}`).join(' | ') : '-'}
      `;
    }

    async function fetchTable() {
      const table = tablesEl.value;
      const symbol = symbolEl.value.trim();
      const limit = parseInt(limitEl.value || '50', 10);
      const params = new URLSearchParams({ table, limit });
      if (symbol) params.append('symbol', symbol);
      const res = await fetch('/api/v1/db/query?' + params.toString());
      if (!res.ok) { alert('Query failed'); return; }
      const data = await res.json();
      currentRows = data.rows || [];
      renderTable(table, currentRows, data.total || currentRows.length);
    }

    async function showSchema() {
      const table = tablesEl.value;
      const res = await fetch('/api/v1/db/schema?table=' + encodeURIComponent(table));
      if (!res.ok) { alert('Schema fetch failed'); return; }
      const data = await res.json();
      schemaBox.style.display = 'block';
      schemaBox.innerHTML = data.columns.map(c => `${c.name} (${c.type})${c.nullable ? '' : ' NOT NULL'}`).join('<br>');
    }

    async function doBackup() {
      const res = await fetch('/api/v1/db/backup');
      if (!res.ok) { alert('Backup failed'); return; }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'gravity_db_backup.json';
      a.click();
      URL.revokeObjectURL(url);
    }

    function exportCSV() {
      if (!currentRows.length) return;
      const cols = Object.keys(currentRows[0]);
      const csv = [cols.join(',')].concat(currentRows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(','))).join('\\n');
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (tablesEl.value || 'table') + '.csv';
      a.click();
      URL.revokeObjectURL(url);
    }

    function renderTable(table, rows, total) {
      tableMeta.textContent = `Rows: ${total}`;
      tableNamePill.textContent = table || '—';
      if (!rows.length) {
        thead.innerHTML = ''; tbody.innerHTML = '';
        columnSelect.innerHTML = '';
        return;
      }
      const cols = Object.keys(rows[0]);
      thead.innerHTML = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
      tbody.innerHTML = rows.map(r => '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>').join('');
      columnSelect.innerHTML = '';
      cols.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        columnSelect.appendChild(opt);
      });
    }

    function visualize() {
      if (!currentRows.length) return;
      const col = columnSelect.value;
      if (!col) return;
      const labels = currentRows.map((_, idx) => idx + 1);
      const data = currentRows.map(r => Number(r[col]) || 0);
      const ctx = document.getElementById('chartCanvas').getContext('2d');
      if (chart) chart.destroy();
      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{ label: col, data, borderColor: '#3dd598', fill: false, tension: 0.2 }]
        },
        options: { plugins: { legend: { labels: { color: '#e8edf5' } } }, scales: { x: { ticks: { color: '#93a4c2' }}, y:{ ticks:{color:'#93a4c2'}}}}
      });
    }

    fetchBtn.onclick = fetchTable;
    chartBtn.onclick = visualize;
    schemaBtn.onclick = showSchema;
    backupBtn.onclick = doBackup;
    exportBtn.onclick = exportCSV;
    loadTables().then(fetchTable);
    loadInfo();
  </script>
</body>
</html>
        """
    )


@router.get("/home", response_class=HTMLResponse)
async def db_home():
    """Index with linked pages so users don't need to memorize URLs."""
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="fa">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gravity Technical Analysis - Home</title>
  <style>
    :root { --bg:#0b1220; --card:#111a2e; --accent:#3dd598; --text:#e8edf5; --muted:#93a4c2; --border:#1f2c45; }
    body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 32px; }
    h1 { margin-top: 0; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    code { background: #0f1629; padding: 2px 6px; border-radius: 6px; }
    ul { padding-left: 18px; }
    .nav { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
    .nav a { padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); background: #0f1629; color: var(--muted); }
    .nav a:hover { color: var(--text); border-color: var(--accent); }
  </style>
</head>
<body>
  <div class="nav">
    <a href="/api/v1/db/ui" target="_blank">DB UI</a>
    <a href="/api/docs" target="_blank">Swagger</a>
    <a href="/metrics" target="_blank">Metrics</a>
    <a href="/api/v1/backtest" target="_blank">Backtest API</a>
  </div>
  <h1>Gravity Technical Analysis - Quick Links</h1>
  <div class="card">
    <h3>رابط کاربری دیتابیس</h3>
    <ul>
      <li><a href="/api/v1/db/ui" target="_blank">DB Explorer UI (همه عملیات دیتابیس)</a></li>
      <li><a href="/api/v1/db/tables" target="_blank">لیست جداول (JSON)</a></li>
      <li><a href="/api/v1/db/backup" target="_blank">دانلود بکاپ کامل</a></li>
    </ul>
  </div>
  <div class="card">
    <h3>APIs و مانیتورینگ</h3>
    <ul>
      <li><a href="/api/docs" target="_blank">Swagger /api/docs</a></li>
      <li><a href="/metrics" target="_blank">Prometheus metrics</a></li>
      <li><a href="/api/v1/backtest" target="_blank">Backtest endpoint</a></li>
    </ul>
  </div>
  <div class="card">
    <h3>دستورات CLI (نمونه قابل کپی)</h3>
    <ul>
      <li><code>python -m gravity_tech.cli.run_backtesting --symbol FOLD --source db --limit 1200 --persist --auto-tune</code></li>
      <li><code>python -m gravity_tech.cli.run_complete_pipeline --symbol BTCUSDT --interval 1h --limit 500 --trend-weights ml_models/multi_horizon/indicator_weights_btcusdt.json --momentum-weights ml_models/multi_horizon/dimension_weights_btcusdt.json --volatility-weights ml_models/multi_horizon/dimension_weights_btcusdt.json</code></li>
    </ul>
  </div>
  <div class="card">
    <h3>دسترسی سریع به داده</h3>
    <p>لیست جداول: <code>/api/v1/db/tables</code> — کوئری نمونه: <code>/api/v1/db/query?table=YOUR_TABLE&amp;limit=50</code></p>
  </div>
</body>
</html>
        """
    )
