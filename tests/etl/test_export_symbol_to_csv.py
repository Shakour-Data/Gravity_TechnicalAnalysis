
import sys
import tempfile
from pathlib import Path

# Ensure the parent of 'scripts' is in sys.path
from pathlib import Path as _Path
from unittest import mock

import pandas as pd
import pytest

_root = _Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import scripts.etl.export_symbol_to_csv as export_mod


def test_list_unique_symbols_handles_empty(monkeypatch):
    # Patch psycopg2.connect to simulate empty DB
    class DummyCursor:
        def execute(self, *a, **k): pass
        def fetchall(self): return []
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    result = export_mod.list_unique_symbols()
    assert result == []

def test_export_symbol_data_creates_dir_and_summary(monkeypatch):
    # Patch DB and pandas
    dummy_tables = ["table1"]
    dummy_columns = [("symbol", "character varying")]
    # The DataFrame must not be empty and must match the query logic
    dummy_df = pd.DataFrame({"symbol": ["SYM"], "date": ["2021-01-01"]})
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                self._fetch = dummy_columns
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
        def escape_string(self, s): return s.encode() if isinstance(s, str) else s
        def escape_identifier(self, s): return s
    class DummyEngine:
        pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    monkeypatch.setattr(export_mod, "create_engine", lambda *a, **k: DummyEngine())
def test_export_symbol_data_creates_dir_and_summary(monkeypatch):
    # Use mock for everything
    dummy_df = pd.DataFrame({"symbol": ["SYM"], "date": ["2021-01-01"]})
    mock_conn = mock.MagicMock()
    mock_cursor = mock.MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.side_effect = [
        [("table1",)],  # tables
        [("symbol", "character varying")]  # columns
    ]
    class MockIdentifier:
        def __init__(self, name):
            self.name = name
        def as_string(self, conn):
            return self.name

    class MockSQL:
        def __init__(self, sql_str):
            self.sql_str = sql_str
        def format(self, *args):
            formatted = self.sql_str
            for arg in args:
                if hasattr(arg, 'as_string'):
                    formatted = formatted.replace('{}', arg.as_string(None), 1)
                else:
                    formatted = formatted.replace('{}', str(arg), 1)
            return MockSQL(formatted)
        def as_string(self, conn):
            return self.sql_str

    monkeypatch.setattr(export_mod, 'sql', mock.MagicMock())
    export_mod.sql.SQL = MockSQL
    export_mod.sql.Identifier = MockIdentifier
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: mock_conn)
    monkeypatch.setattr(export_mod, "create_engine", lambda *a, **k: mock.MagicMock())
    with mock.patch('pandas.read_sql_query', return_value=dummy_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            export_mod.export_symbol_data("SYM", export_base=outdir, fuzzy=False)
            files = list((outdir / "SYM").glob("*.csv"))
            assert any("_summary.csv" in str(f) for f in files)
        print("files:", files)
        assert any("_summary.csv" in str(f) for f in files)

def test_export_symbol_data_handles_no_data(monkeypatch):
    # Patch DB and pandas to return no data
    dummy_tables = ["table1"]
    dummy_columns = [("symbol", "character varying")]
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                self._fetch = dummy_columns
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    class DummyEngine:
        pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    monkeypatch.setattr(export_mod, "create_engine", lambda *a, **k: DummyEngine())
    monkeypatch.setattr(export_mod.pd, "read_sql_query", lambda *a, **k: pd.DataFrame())
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        export_mod.export_symbol_data("NOTFOUND", export_base=outdir, fuzzy=False)
        # Should not raise, and no summary file should be created
        symdir = outdir / "NOTFOUND"
        assert not any(symdir.glob("*.csv"))

def test_list_unique_symbols_handles_values(monkeypatch):
    dummy_tables = ["table1"]
    dummy_columns = [("symbol",)]
    dummy_values = [("SYM1",), ("SYM2",)]
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                self._fetch = dummy_columns
            elif "DISTINCT" in str(q):
                self._fetch = dummy_values
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    result = export_mod.list_unique_symbols()
    assert result == ["SYM1", "SYM2"]


def test_export_symbol_data_handles_db_error(monkeypatch):
    class DummyConn:
        def cursor(self):
            raise Exception("db fail")
        def close(self): pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    # Should not raise, just log critical
    export_mod.export_symbol_data("SYM", export_base=Path(tempfile.gettempdir()), fuzzy=False)


def test_list_unique_symbols_handles_db_error(monkeypatch):
    class DummyConn:
        def cursor(self):
            raise Exception("db fail")
        def close(self): pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    result = export_mod.list_unique_symbols()
    assert result == []


def test_export_symbol_data_fuzzy(monkeypatch):
    # Use mock for everything
    dummy_df = pd.DataFrame({"name": ["SOMENAME"], "date": ["2022-01-01"]})
    mock_conn = mock.MagicMock()
    mock_cursor = mock.MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.side_effect = [
        [("table1",)],  # tables
        [("name", "text")]  # columns
    ]
    mock_conn.escape_string = mock.MagicMock(return_value=b"SOMENAME")
    mock_conn.escape_identifier = mock.MagicMock(return_value=b'"name"')
    mock_conn.encoding = 'utf-8'
    class MockIdentifier:
        def __init__(self, name):
            self.name = name
        def as_string(self, conn):
            return self.name

    class MockSQL:
        def __init__(self, sql_str):
            self.sql_str = sql_str
        def format(self, *args):
            formatted = self.sql_str
            for arg in args:
                if hasattr(arg, 'as_string'):
                    formatted = formatted.replace('{}', arg.as_string(None), 1)
                else:
                    formatted = formatted.replace('{}', str(arg), 1)
            return MockSQL(formatted)
        def as_string(self, conn):
            return self.sql_str

    monkeypatch.setattr(export_mod, 'sql', mock.MagicMock())
    export_mod.sql.SQL = MockSQL
    export_mod.sql.Identifier = MockIdentifier
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: mock_conn)
    monkeypatch.setattr(export_mod, "create_engine", lambda *a, **k: mock.MagicMock())
    with mock.patch('pandas.read_sql_query', return_value=dummy_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            export_mod.export_symbol_data("SOMENAME", export_base=outdir, fuzzy=True)
            files = list((outdir / "SOMENAME").glob("*.csv"))
            assert any("_summary.csv" in str(f) for f in files)


def test_export_symbol_data_no_text_cols(monkeypatch):
    dummy_tables = ["table1"]
    dummy_columns = [("id", "integer")]  # No text columns
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                self._fetch = dummy_columns
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    class DummyEngine:
        pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    monkeypatch.setattr(export_mod, "create_engine", lambda *a, **k: DummyEngine())
    monkeypatch.setattr(export_mod.pd, "read_sql_query", lambda *a, **k: pd.DataFrame())
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        export_mod.export_symbol_data("SYM", export_base=outdir, fuzzy=False)
        # Should not create summary since no text cols
        assert not (outdir / "SYM" / "_summary.csv").exists()


def test_export_symbol_data_table_error(monkeypatch):
    dummy_tables = ["table1"]
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                raise Exception("table error")
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    class DummyEngine:
        pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    monkeypatch.setattr(export_mod, "create_engine", lambda *a, **k: DummyEngine())
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        export_mod.export_symbol_data("SYM", export_base=outdir, fuzzy=False)
        # Should handle error gracefully


def test_list_unique_symbols_no_matching_cols(monkeypatch):
    dummy_tables = ["table1"]
    dummy_columns = [("id",)]  # No symbol, index, etc.
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                self._fetch = dummy_columns
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    result = export_mod.list_unique_symbols()
    assert result == []


def test_list_unique_symbols_read_error(monkeypatch):
    dummy_tables = ["table1"]
    dummy_columns = [("symbol",)]
    class DummyCursor:
        def execute(self, q, params=None):
            if "information_schema.tables" in str(q):
                self._fetch = [(t,) for t in dummy_tables]
            elif "information_schema.columns" in str(q):
                self._fetch = dummy_columns
            elif "DISTINCT" in str(q):
                raise Exception("read error")
            else:
                self._fetch = []
        def fetchall(self): return self._fetch
        def close(self): pass
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    monkeypatch.setattr(export_mod.psycopg2, "connect", lambda *a, **k: DummyConn())
    result = export_mod.list_unique_symbols()
    assert result == []






@pytest.mark.skip(reason="Hard to mock local import")
def test_main_with_symbol(monkeypatch):
    import sys
    called = {}
    monkeypatch.setattr(sys, "argv", ["prog", "SYM"])
    monkeypatch.setattr(export_mod, "export_symbol_data", lambda *a, **k: called.setdefault("export", True))
    # Mock argparse in the module
    mock_argparse = mock.MagicMock()
    mock_parser = mock.MagicMock()
    mock_argparse.ArgumentParser.return_value = mock_parser
    mock_args = mock.MagicMock()
    mock_args.symbol = "SYM"
    mock_args.list = False
    mock_args.fuzzy = False
    mock_args.auto = False
    mock_args.out = str(export_mod.EXPORT_BASE)
    mock_parser.parse_args.return_value = mock_args
    monkeypatch.setattr(export_mod, "argparse", mock_argparse)
    # Run the main block
    exec(open(export_mod.__file__).read().split('if __name__ == "__main__":')[1], export_mod.__dict__)
    assert "export" in called
