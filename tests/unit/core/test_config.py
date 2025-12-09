"""
Unit tests for src/config.py

Tests configuration loading and path resolution.
"""

import importlib
import os
from unittest.mock import patch

from src.config import APP_DB_FILE, BASE_DIR


class TestConfig:
    """Test configuration module functionality."""

    def test_base_dir_exists(self):
        """Test that BASE_DIR points to a valid directory."""
        assert BASE_DIR.exists()
        assert BASE_DIR.is_dir()

    def test_app_db_file_path(self):
        """Test that APP_DB_FILE is constructed correctly."""
        expected_path = os.path.join(BASE_DIR, 'data', 'tool_performance.db')
        assert APP_DB_FILE == expected_path

    @patch('pathlib.Path.exists')
    def test_tse_db_file_found(self, mock_exists):
        """Test TSE_DB_FILE selection when database file exists."""
        # Mock that the first candidate exists
        mock_exists.side_effect = [True, False, False]

        # Re-import to trigger the logic
        import src.config
        importlib.reload(src.config)

        # Should select the first candidate
        expected = r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db'
        assert src.config.TSE_DB_FILE == expected

    @patch('pathlib.Path.exists')
    def test_tse_db_file_fallback(self, mock_exists):
        """Test TSE_DB_FILE fallback when no database file exists."""
        # Mock that no candidates exist
        mock_exists.return_value = False

        # Re-import to trigger the logic
        import src.config
        importlib.reload(src.config)

        # Should fallback to first candidate
        expected = r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db'
        assert src.config.TSE_DB_FILE == expected

    def test_tse_db_candidates_structure(self):
        """Test that TSE_DB_CANDIDATES contains expected paths."""
        from pathlib import Path

        from src.config import _TSE_DB_CANDIDATES

        assert len(_TSE_DB_CANDIDATES) == 3

        # First candidate should be the Windows path
        assert str(_TSE_DB_CANDIDATES[0]) == r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db'

        # Second should be relative to BASE_DIR
        expected_relative = BASE_DIR.parent / 'GravityTseHisPrice' / 'data' / 'tse_data.db'
        assert _TSE_DB_CANDIDATES[1] == expected_relative

        # Third should be in home directory
        expected_home = Path.home() / 'GravityTseHisPrice' / 'data' / 'tse_data.db'
        assert _TSE_DB_CANDIDATES[2] == expected_home
