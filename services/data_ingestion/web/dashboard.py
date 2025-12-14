import os
import sys
import io
import contextlib
import sqlite3
import subprocess
from pathlib import Path

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, parent_dir)

from config import DB_FILE  # noqa: E402
from encoding_utils import ensure_utf8_console  # noqa: E402
from fetcher import DataFetcher  # noqa: E402

ensure_utf8_console()

external_stylesheets = [
    dbc.themes.ZEPHYR,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "GravityTseHisPrice | داشبورد بازار"

palette = {
    "ink": "#0f172a",
    "panel": "#111827",
    "card": "#0b1324",
    "accent": "#2dd4bf",
    "accent_alt": "#f59e0b",
    "muted": "#94a3b8",
    "text": "#e2e8f0",
    "line": "#1f2937",
}

CLI_COMMANDS = [
    {"value": "create-db", "label": "ایجاد پایگاه داده و جداول"},
    {"value": "create-indices-tables", "label": "ایجاد جداول شاخص‌ها"},
    {"value": "load-market-indices", "label": "بارگذاری شاخص‌های بازار"},
    {"value": "load-sector-indices", "label": "بارگذاری شاخص‌های صنایع"},
    {"value": "load-initial", "label": "بارگذاری داده‌های اولیه"},
    {"value": "init-all", "label": "ایجاد DB و بارگذاری کامل"},
    {"value": "load-all-prices", "label": "بارگذاری همه قیمت‌ها"},
    {"value": "reload-table", "label": "بارگذاری مجدد جدول از JSON"},
    {"value": "drop-table", "label": "حذف جدول"},
    {"value": "update-db", "label": "به‌روزرسانی همه جداول"},
    {"value": "update-table", "label": "به‌روزرسانی جدول از JSON"},
    {"value": "list-sectors", "label": "لیست صنایع"},
    {"value": "list-companies", "label": "لیست شرکت‌ها بر اساس صنعت"},
    {"value": "get-price-data", "label": "نمایش قیمت یک نماد"},
]


def to_persian_numbers(text):
    """Convert English digits inside text to Persian digits."""
    if text is None:
        return "نامشخص"

    if not isinstance(text, str):
        text = str(text)

    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    translation_table = str.maketrans(english_digits, persian_digits)
    return text.translate(translation_table)


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_FILE)


def get_market_summary():
    """Get market summary statistics."""
    conn = get_db_connection()
    try:
        companies_count = pd.read_sql(
            "SELECT COUNT(*) as count FROM companies", conn
        ).iloc[0]["count"]
        price_records = pd.read_sql(
            "SELECT COUNT(*) as count FROM price_data", conn
        ).iloc[0]["count"]
        latest_update = pd.read_sql(
            "SELECT MAX(date) as latest FROM price_data", conn
        ).iloc[0]["latest"]
        sectors_count = pd.read_sql(
            "SELECT COUNT(*) as count FROM sectors", conn
        ).iloc[0]["count"]

        return {
            "companies": companies_count,
            "price_records": price_records,
            "latest_update": latest_update,
            "sectors": sectors_count,
        }
    finally:
        conn.close()


def get_recent_price_data(ticker=None, limit=100):
    """Get recent price data for visualization."""
    conn = get_db_connection()
    try:
        if ticker:
            query = """
                SELECT date, adj_close, adj_volume, ticker
                FROM price_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
            """
            params = (ticker, limit)
        else:
            query = """
                SELECT date, adj_close, adj_volume, ticker
                FROM price_data
                ORDER BY date DESC
                LIMIT ?
            """
            params = (limit,)

        df = pd.read_sql(query, conn, params=params)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"], format="ISO8601", errors="coerce")
        df = df.sort_values("date")
        return df
    finally:
        conn.close()


def get_sectors_data():
    """Get sectors data for visualization."""
    conn = get_db_connection()
    try:
        query = """
            SELECT s.sector_name, COUNT(c.company_id) as company_count
            FROM sectors s
            LEFT JOIN companies c ON s.sector_id = c.sector_id
            GROUP BY s.sector_id, s.sector_name
            ORDER BY company_count DESC
        """
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def get_top_companies_by_volume(limit=10):
    """Get top companies by trading volume."""
    conn = get_db_connection()
    try:
        query = f"""
            SELECT c.ticker, c.name, AVG(p.adj_volume) as avg_volume,
                   AVG(p.adj_close) as avg_price, COUNT(p.id) as records_count
            FROM companies c
            JOIN price_data p ON c.ticker = p.ticker
            GROUP BY c.ticker, c.name
            HAVING records_count > 10
            ORDER BY avg_volume DESC
            LIMIT {limit}
        """
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def get_database_tables_info():
    """Get information about all database tables and their row counts."""
    conn = get_db_connection()
    try:
        tables_query = (
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = pd.read_sql(tables_query, conn)

        table_info = []
        for table_name in tables["name"]:
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = pd.read_sql(count_query, conn)
            row_count = count_result.iloc[0]["count"]

            pragma_query = f"PRAGMA table_info({table_name})"
            columns = pd.read_sql(pragma_query, conn)

            table_info.append(
                {
                    "table_name": table_name,
                    "row_count": row_count,
                    "column_count": len(columns),
                    "columns": columns["name"].tolist(),
                }
            )

        return table_info
    finally:
        conn.close()


def get_table_preview(table_name, limit=5):
    """Get a preview of table data."""
    conn = get_db_connection()
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def get_table_data(table_name, limit=200):
    """Fetch table data (latest rows first)."""
    conn = get_db_connection()
    try:
        # Use ROWID to return latest inserted rows first when no explicit ordering exists.
        query = f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT {limit}"
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def run_cli_command(command, table=None, file_path=None, sector_id=None, ticker=None, limit=None):
    """Run the existing CLI with the provided arguments and capture logs."""
    main_path = os.path.join(parent_dir, "main.py")
    cmd = [sys.executable, main_path, command]

    if command in {"reload-table", "update-table"}:
        if not table or not file_path:
            return False, "برای این دستور جدول و فایل JSON الزامی است."
        cmd += [table, file_path]
    elif command == "drop-table":
        if not table:
            return False, "نام جدول لازم است."
        cmd += [table]
    elif command == "list-companies":
        if sector_id is None:
            return False, "کد صنعت لازم است."
        cmd += [str(sector_id)]
    elif command == "get-price-data":
        if not ticker:
            return False, "نماد لازم است."
        cmd += [ticker]
        if limit:
            cmd += ["--limit", str(limit)]

    # Validate file path if provided
    if file_path and command in {"reload-table", "update-table"}:
        if not Path(file_path).exists():
            return False, f"فایل یافت نشد: {file_path}"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=parent_dir, check=False
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        success = result.returncode == 0
        return success, output.strip() or "بدون خروجی."
    except Exception as exc:  # pragma: no cover
        return False, f"خطای اجرای دستور: {exc}"


def build_stat_card(title, value_id, icon, tone="blue"):
    """Reusable stat card component."""
    return dbc.Card(
        dbc.CardBody(
            html.Div(
                [
                    html.Div(html.I(className=f"fa-solid fa-{icon}"), className="stat-icon"),
                    html.Div(
                        [html.P(title, className="stat-label"), html.H2(id=value_id, className="stat-value")]
                    ),
                ],
                className=f"stat-card-content tone-{tone}",
            )
        ),
        className="stat-card h-100 shadow-soft",
    )


app.layout = html.Div(
    className="app-shell",
    dir="rtl",
    children=[
        dcc.Interval(id="init-trigger", interval=800, max_intervals=1),
        html.Section(
            className="hero-section",
            children=dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Span(
                                        "تصویر روشن از بازار سرمایه",
                                        className="eyebrow",
                                    ),
                                    html.H1(
                                        "داشبورد GravityTseHisPrice",
                                        className="hero-title",
                                    ),
                                    html.P(
                                        "پوشش کامل شرکت‌ها، صنایع و شاخص‌ها با داده‌های تعدیل‌شده و به‌روزرسانی سریع. "
                                        "همه چیز آماده یک گزارش شفاف و حرفه‌ای است.",
                                        className="hero-lead",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                [html.I(className="fa-solid fa-rotate icon-left"), "به‌روزرسانی دیتابیس"],
                                                id="hero-update-btn",
                                                color="success",
                                                className="hero-cta",
                                            ),
                                            dbc.Button(
                                                [html.I(className="fa-solid fa-chart-column icon-left"), "نمایش خلاصه بازار"],
                                                id="hero-status-btn",
                                                color="secondary",
                                                outline=True,
                                                className="hero-cta",
                                            ),
                                        ],
                                        className="hero-cta-wrap",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Badge(
                                                "به‌روزرسانی با یک کلیک",
                                                className="hero-badge",
                                            ),
                                            dbc.Badge(
                                                "تحلیل آماده ارائه",
                                                className="hero-badge alt",
                                            ),
                                        ],
                                        className="hero-badges",
                                    ),
                                ],
                                lg=7,
                                md=12,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.P("چک‌لیست آماده‌باش", className="mini-title"),
                                            html.Ul(
                                                [
                                                    html.Li("دیتابیس و API هم‌مسیر و هماهنگ"),
                                                    html.Li("گزارش لحظه‌ای تعداد رکوردها و آخرین تاریخ"),
                                                    html.Li("پیش‌نمایش جداول و داده‌ها بدون خروج از داشبورد"),
                                                ],
                                                className="hero-list",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("مسیر پایگاه داده:", className="muted-label"),
                                                    html.Code(DB_FILE, className="db-path"),
                                                ],
                                                className="mini-row",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("آدرس داشبورد:", className="muted-label"),
                                                    html.Code("http://127.0.0.1:8050/", className="db-path"),
                                                ],
                                                className="mini-row",
                                            ),
                                        ]
                                    ),
                                    className="glass-card shadow-soft",
                                ),
                                lg=5,
                                md=12,
                            ),
                        ],
                        className="align-items-center g-4",
                    )
                ],
                fluid=True,
            ),
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            build_stat_card("تعداد شرکت‌ها", "companies-count", "building", tone="teal"),
                            lg=3,
                            md=6,
                        ),
                        dbc.Col(
                            build_stat_card("تعداد رکوردهای قیمت", "price-records-count", "chart-line", tone="amber"),
                            lg=3,
                            md=6,
                        ),
                        dbc.Col(
                            build_stat_card("تعداد صنایع", "sectors-count", "grid-2", tone="blue"),
                            lg=3,
                            md=6,
                        ),
                        dbc.Col(
                            build_stat_card("آخرین تاریخ به‌روزرسانی", "latest-update", "calendar-days", tone="slate"),
                            lg=3,
                            md=6,
                        ),
                    ],
                    className="g-3 mt-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.I(className="fa-solid fa-chart-area header-icon"),
                                                            html.Span("روند قیمت تعدیل‌شده", className="header-title"),
                                                        ],
                                                        className="header-wrap",
                                                    ),
                                                    md=7,
                                                    xs=12,
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id="ticker-dropdown",
                                                        placeholder="نماد را جست‌وجو کنید...",
                                                        className="ticker-dropdown",
                                                        clearable=True,
                                                    ),
                                                    md=5,
                                                    xs=12,
                                                ),
                                            ],
                                            className="g-2 align-items-center",
                                        )
                                    ),
                                    dbc.CardBody(dcc.Graph(id="price-chart", className="chart-card")),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=8,
                            md=12,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-circle-notch header-icon"),
                                                    html.Span("ترکیب صنایع", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(dcc.Graph(id="sectors-chart", className="chart-card")),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=4,
                            md=12,
                        ),
                    ],
                    className="g-3 mt-1",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-table header-icon"),
                                                    html.Span("مرورگر جداول با فیلتر", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            id="table-select",
                                                            placeholder="یک جدول را انتخاب کنید",
                                                            className="ticker-dropdown mb-2",
                                                        ),
                                                        md=4,
                                                        xs=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="table-filter",
                                                            placeholder="فیلتر متنی (جست‌وجو در همه ستون‌ها)",
                                                            className="form-control mb-2",
                                                        ),
                                                        md=5,
                                                        xs=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="table-limit",
                                                            type="number",
                                                            value=100,
                                                            min=10,
                                                            max=500,
                                                            step=10,
                                                            placeholder="تعداد ردیف",
                                                            className="form-control mb-2",
                                                        ),
                                                        md=3,
                                                        xs=12,
                                                    ),
                                                ],
                                                className="g-2 align-items-center",
                                            ),
                                            dcc.Loading(
                                                type="default",
                                                children=html.Div(id="table-viewer", className="action-log"),
                                            ),
                                        ]
                                    ),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=12,
                            md=12,
                        )
                    ],
                    className="g-3 mt-1 mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-ranking-star header-icon"),
                                                    html.Span("پرتفوی پُرحجم", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(html.Div(id="top-companies-table")),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=12,
                            md=12,
                        ),
                    ],
                    className="g-3 mt-1",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-bolt header-icon"),
                                                    html.Span("دستورات سریع", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Button(
                                                [html.I(className="fa-solid fa-rotate icon-left"), "به‌روزرسانی داده‌ها"],
                                                id="update-btn",
                                                color="primary",
                                                className="w-100 mb-2 action-btn",
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(className="fa-solid fa-chart-column icon-left"),
                                                    "نمایش خلاصه بازار",
                                                ],
                                                id="status-btn",
                                                color="info",
                                                className="w-100 mb-2 action-btn",
                                            ),
                                            dbc.Button(
                                                [html.I(className="fa-solid fa-download icon-left"), "تهیه نسخه پشتیبان"],
                                                id="backup-btn",
                                                color="warning",
                                                className="w-100 action-btn",
                                            ),
                                        ]
                                    ),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=4,
                            md=12,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-terminal header-icon"),
                                                    html.Span("گزارش عملیات و خطاها", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            type="default",
                                            children=html.Div(id="action-output", className="action-log"),
                                        )
                                    ),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=8,
                            md=12,
                        ),
                    ],
                    className="g-3 mt-1 mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-terminal header-icon"),
                                                    html.Span("دستورات CLI در داشبورد", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            ),
                                            html.Span("همه دستورات main.py در دسترس است.", className="muted-label"),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Dropdown(
                                                id="cli-command",
                                                options=CLI_COMMANDS,
                                                placeholder="یک دستور را انتخاب کنید",
                                                className="mb-2 ticker-dropdown",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            id="cli-table",
                                                            options=[
                                                                {"label": "companies", "value": "companies"},
                                                                {"label": "sectors", "value": "sectors"},
                                                                {"label": "markets", "value": "markets"},
                                                                {"label": "panels", "value": "panels"},
                                                                {"label": "price_data", "value": "price_data"},
                                                                {"label": "last_updates", "value": "last_updates"},
                                                            ],
                                                            placeholder="جدول (در صورت نیاز)",
                                                            className="mb-2 ticker-dropdown",
                                                        ),
                                                        md=6,
                                                        xs=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="cli-file",
                                                            placeholder="مسیر فایل JSON (در صورت نیاز)",
                                                            className="form-control mb-2",
                                                        ),
                                                        md=6,
                                                        xs=12,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="cli-sector",
                                                            type="number",
                                                            placeholder="کد صنعت برای list-companies",
                                                            className="form-control mb-2",
                                                        ),
                                                        md=6,
                                                        xs=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="cli-ticker",
                                                            placeholder="نماد برای get-price-data",
                                                            className="form-control mb-2",
                                                        ),
                                                        md=6,
                                                        xs=12,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Input(
                                                            id="cli-limit",
                                                            type="number",
                                                            placeholder="حد (اختیاری)",
                                                            className="form-control mb-3",
                                                        ),
                                                        md=6,
                                                        xs=12,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            [html.I(className="fa-solid fa-play icon-left"), "اجرای دستور"],
                                                            id="cli-run-btn",
                                                            color="success",
                                                            className="w-100 action-btn mb-3",
                                                        ),
                                                        md=6,
                                                        xs=12,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "راهنما: برای reload/update-table مسیر JSON و نام جدول لازم است. برای list-companies کد صنعت و برای get-price-data نماد (و حد اختیاری) را وارد کنید.",
                                                        className="text-muted small mb-0",
                                                    )
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=6,
                            md=12,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-scroll header-icon"),
                                                    html.Span("خروجی و لاگ دستورات CLI", className="header-title"),
                                                ],
                                                className="header-wrap",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            id="cli-loading",
                                            type="default",
                                            children=html.Div(id="cli-log", className="action-log"),
                                        )
                                    ),
                                ],
                                className="panel-card shadow-soft h-100",
                            ),
                            lg=6,
                            md=12,
                        ),
                    ],
                    className="g-3 mt-1 mb-4",
                ),
                html.Footer(
                    [
                        html.P(
                            "GravityTseHisPrice - پایش روزانه بازار سرمایه | نسخه 2.0.0",
                            className="footer-title",
                        ),
                        html.P(
                            "بازطراحی برای گزارش‌های حرفه‌ای و تجربه کاربری پاک و یکدست.",
                            className="footer-subtitle",
                        ),
                    ],
                    className="app-footer text-center",
                ),
            ],
            fluid=True,
            className="content-shell pb-5",
        ),
    ],
)


@app.callback(
    [
        Output("companies-count", "children"),
        Output("price-records-count", "children"),
        Output("sectors-count", "children"),
        Output("latest-update", "children"),
    ],
    [Input("init-trigger", "n_intervals"), Input("update-btn", "n_clicks"), Input("hero-update-btn", "n_clicks")],
)
def update_summary(init_tick, n_clicks, hero_clicks):
    """Update summary statistics."""
    try:
        summary = get_market_summary()
        return [
            to_persian_numbers(f"{summary['companies']:,}"),
            to_persian_numbers(f"{summary['price_records']:,}"),
            to_persian_numbers(f"{summary['sectors']}"),
            summary["latest_update"] or "نامشخص",
        ]
    except Exception:
        return ["—", "—", "—", "—"]


@app.callback(
    Output("ticker-dropdown", "options"),
    [Input("init-trigger", "n_intervals"), Input("update-btn", "n_clicks"), Input("hero-update-btn", "n_clicks")],
)
def update_ticker_options(init_tick, n_clicks, hero_clicks):
    """Update ticker dropdown options."""
    try:
        conn = get_db_connection()
        tickers = pd.read_sql("SELECT DISTINCT ticker, name FROM companies ORDER BY ticker", conn)
        conn.close()

        options = [
            {"label": f"{row['ticker']} - {row['name']}", "value": row["ticker"]} for _, row in tickers.iterrows()
        ]
        return options
    except Exception:
        return []


@app.callback(Output("price-chart", "figure"), Input("ticker-dropdown", "value"))
def update_price_chart(selected_ticker):
    """Update price chart based on selected ticker."""
    if selected_ticker:
        df = get_recent_price_data(selected_ticker, limit=500)
    else:
        df = get_recent_price_data(limit=500)

    if df.empty:
        return go.Figure()

    fig = go.Figure()

    if selected_ticker:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["adj_close"],
                mode="lines",
                name=f"قیمت {selected_ticker}",
                line=dict(color=palette["accent"], width=3),
                fill="tozeroy",
                fillcolor="rgba(45, 212, 191, 0.08)",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["adj_volume"],
                name="حجم",
                yaxis="y2",
                marker_color=palette["accent_alt"],
                opacity=0.35,
            )
        )
        fig.update_layout(
            title=f"روند قیمت و حجم | {selected_ticker}",
            yaxis2=dict(title="حجم", overlaying="y", side="right"),
        )
    else:
        df_avg = df.groupby("date")["adj_close"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=df_avg["date"],
                y=df_avg["adj_close"],
                mode="lines",
                name="میانگین بازار",
                line=dict(color=palette["accent"], width=3),
                fill="tozeroy",
                fillcolor="rgba(45, 212, 191, 0.08)",
            )
        )
        fig.update_layout(title="میانگین قیمت تعدیل‌شده کل بازار")

    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"], size=12),
        xaxis_title="تاریخ",
        yaxis_title="قیمت تعدیل‌شده",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

    return fig


@app.callback(
    Output("sectors-chart", "figure"),
    [Input("init-trigger", "n_intervals"), Input("update-btn", "n_clicks"), Input("hero-update-btn", "n_clicks")],
)
def update_sectors_chart(init_tick, n_clicks, hero_clicks):
    """Update sectors distribution chart."""
    try:
        df = get_sectors_data()
        if df.empty:
            return go.Figure()

        fig = px.pie(
            df,
            values="company_count",
            names="sector_name",
            title="سهم صنایع از شرکت‌ها",
            hole=0.45,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            pull=[0.04] * len(df),
            marker=dict(line=dict(color=palette["ink"], width=1)),
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette["text"]),
        )
        return fig
    except Exception:
        return go.Figure()


@app.callback(
    Output("top-companies-table", "children"),
    [Input("init-trigger", "n_intervals"), Input("update-btn", "n_clicks"), Input("hero-update-btn", "n_clicks")],
)
def update_top_companies_table(init_tick, n_clicks, hero_clicks):
    """Update top companies table."""
    try:
        df = get_top_companies_by_volume(10)
        if df.empty:
            return html.P("داده‌ای برای نمایش وجود ندارد.", className="text-muted")

        rows = []
        for _, row in df.iterrows():
            rows.append(
                html.Tr(
                    [
                        html.Td(row["ticker"], className="fw-bold"),
                        html.Td(row["name"]),
                        html.Td(to_persian_numbers(f"{row['avg_volume']:.0f}")),
                        html.Td(to_persian_numbers(f"{row['avg_price']:.0f}")),
                    ]
                )
            )

        table = dbc.Table(
            [
                html.Thead(
                    html.Tr([html.Th("نماد"), html.Th("نام شرکت"), html.Th("میانگین حجم"), html.Th("میانگین قیمت")])
                ),
                html.Tbody(rows),
            ],
            bordered=False,
            hover=True,
            responsive=True,
            striped=True,
            className="data-table",
        )
        return table
    except Exception as e:
        return html.P(f"خطا در بارگذاری جدول: {str(e)}", className="text-danger")


@app.callback(
    Output("table-select", "options"),
    [Input("init-trigger", "n_intervals"), Input("update-btn", "n_clicks"), Input("hero-update-btn", "n_clicks")],
)
def update_table_options(init_tick, n_clicks, hero_clicks):
    """Populate table dropdown options."""
    try:
        tables = get_database_tables_info()
        return [{"label": t["table_name"], "value": t["table_name"]} for t in tables]
    except Exception:
        return []


@app.callback(
    Output("table-viewer", "children"),
    [
        Input("table-select", "value"),
        Input("table-filter", "value"),
        Input("table-limit", "value"),
        Input("init-trigger", "n_intervals"),
        Input("update-btn", "n_clicks"),
        Input("hero-update-btn", "n_clicks"),
    ],
)
def render_table_viewer(table_name, filter_text, limit, init_tick, n_clicks, hero_clicks):
    """Show selected table with optional text filter and row limit."""
    if not table_name:
        return html.P("یک جدول را انتخاب کنید.", className="text-muted")

    try:
        display_limit = int(limit) if limit else 100
        display_limit = max(10, min(display_limit, 500))
    except Exception:
        display_limit = 100

    try:
        # Pull a generous slice of latest rows to allow filtering across data, then trim to display_limit
        fetch_limit = max(display_limit * 5, 500)
        df = get_table_data(table_name, limit=fetch_limit)
        if df.empty:
            return html.P("داده‌ای یافت نشد.", className="text-muted")

        if filter_text:
            mask = df.apply(lambda col: col.astype(str).str.contains(filter_text, case=False, na=False))
            df = df[mask.any(axis=1)]
            if df.empty:
                return html.P("هیچ ردیفی با این فیلتر یافت نشد.", className="text-muted")

        df = df.head(display_limit)

        rows = []
        for _, row in df.iterrows():
            rows.append(html.Tr([html.Td(to_persian_numbers(val) if isinstance(val, (int, float)) else val) for val in row]))

        table = dbc.Table(
            [
                html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                html.Tbody(rows),
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="data-table",
        )
        return table
    except Exception as e:
        return dbc.Alert(f"خطا در نمایش جدول: {e}", color="danger")


@app.callback(
    Output("action-output", "children"),
    [
        Input("update-btn", "n_clicks"),
        Input("hero-update-btn", "n_clicks"),
        Input("status-btn", "n_clicks"),
        Input("hero-status-btn", "n_clicks"),
        Input("backup-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def handle_actions(update_clicks, hero_update_clicks, status_clicks, hero_status_clicks, backup_clicks):
    """Handle quick action buttons."""
    ctx = callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id in {"update-btn", "hero-update-btn"}:
        if not (update_clicks or hero_update_clicks):
            return ""

        try:
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                DataFetcher.run()

            logs = output_buffer.getvalue().strip()
            trigger_label = "دکمه به‌روزرسانی" if button_id == "update-btn" else "دکمه هدر"
            if logs:
                log_lines = logs.split("\n")
                formatted_logs = [html.P(line, className="log-line") for line in log_lines]
                return dbc.Alert(
                    [
                        html.H6(f"نتیجه اجرای {trigger_label}"),
                        html.Div(formatted_logs, className="log-scroll"),
                    ],
                    color="success",
                    className="mb-0",
                )
            return dbc.Alert(f"{trigger_label} با موفقیت اجرا شد (خروجی متنی ندارد).", color="success", className="mb-0")

        except Exception as e:
            error_msg = str(e)

            if "timeout" in error_msg.lower():
                return dbc.Alert(
                    [
                        html.H6("اتصال به منبع داده تایم‌اوت شد."),
                        html.P("وضعیت اینترنت/VPN/پراکسی را بررسی کنید و دوباره تلاش کنید."),
                    ],
                    color="warning",
                    className="mb-0",
                )

            if "MaxRetryError" in error_msg:
                return dbc.Alert(
                    [
                        html.H6("پاسخی از سرور منبع دریافت نشد."),
                        html.P("ممکن است محدودیت یا قطعی موقت باشد؛ کمی بعد دوباره تلاش کنید."),
                    ],
                    color="warning",
                    className="mb-0",
                )

            return dbc.Alert(f"خطا در اجرای به‌روزرسانی: {error_msg}", color="danger", className="mb-0")

    if button_id in {"status-btn", "hero-status-btn"}:
        try:
            summary = get_market_summary()
            companies_count = to_persian_numbers(f"{summary['companies']:,}")
            price_count = to_persian_numbers(f"{summary['price_records']:,}")
            sectors_count = to_persian_numbers(summary["sectors"])
            latest = summary["latest_update"] or "نامشخص"

            return dbc.Alert(
                [
                    html.H6("خلاصه وضعیت دیتابیس"),
                    html.P(f"تعداد شرکت‌ها: {companies_count}"),
                    html.P(f"رکوردهای قیمت: {price_count}"),
                    html.P(f"تعداد صنایع: {sectors_count}"),
                    html.P(f"آخرین تاریخ: {latest}"),
                ],
                color="info",
                className="mb-0",
            )
        except Exception as e:
            return dbc.Alert(f"خطا در دریافت وضعیت: {e}", color="danger", className="mb-0")

    if button_id == "backup-btn":
        return dbc.Alert("تهیه نسخه پشتیبان به‌زودی اضافه می‌شود.", color="secondary", className="mb-0")

    return ""


@app.callback(
    Output("cli-log", "children"),
    Input("cli-run-btn", "n_clicks"),
    [
        State("cli-command", "value"),
        State("cli-table", "value"),
        State("cli-file", "value"),
        State("cli-sector", "value"),
        State("cli-ticker", "value"),
        State("cli-limit", "value"),
    ],
    prevent_initial_call=True,
)
def handle_cli_commands(n_clicks, command, table, file_path, sector_id, ticker, limit):
    """Expose all CLI commands inside the dashboard."""
    if not n_clicks:
        return ""

    if not command:
        return dbc.Alert("ابتدا یک دستور را انتخاب کنید.", color="warning", className="mb-0")

    success, logs = run_cli_command(
        command=command,
        table=table,
        file_path=file_path,
        sector_id=sector_id,
        ticker=ticker,
        limit=limit,
    )

    log_lines = [html.P(line, className="log-line") for line in logs.split("\n") if line.strip()]
    if not log_lines:
        log_lines = [html.P("بدون خروجی.", className="log-line")]

    return dbc.Alert(
        [
            html.H6(f"نتیجه اجرای دستور: {command}"),
            html.P("وضعیت: " + ("موفق" if success else "ناموفق"), className="mb-2"),
            html.Div(log_lines, className="log-scroll"),
        ],
        color="success" if success else "danger",
        className="mb-0",
    )


if __name__ == "__main__":
    print("🚀 Starting GravityTseHisPrice Dashboard...")
    print("🌐 Open: http://127.0.0.1:8050/")
    print("⌨️  Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
