# Changelog

## [2.0.0] - 2025-12-08
### Added
- Major refactoring: Renamed `MarketMakerDataFetcher` to `DataFetcher` throughout the codebase
- Extended test suite with 56 comprehensive tests achieving 84% code coverage
- New test files: `test_cli_ext.py`, `test_data_fetcher_ext.py`, `test_database_ext.py`, `test_expanded_coverage.py`
- Enhanced database setup and table creation for all test scenarios
- Improved error handling and code quality with Ruff linting compliance

### Changed
- Complete codebase modernization with better naming conventions
- Test infrastructure overhaul with proper database mocking and isolation
- CLI commands now fully tested and documented
- Database operations optimized for better testability

### Fixed
- Resolved all test failures and database setup issues
- Fixed import order and linting errors across all modules
- Improved test reliability with proper cleanup and isolation

### Removed
- Legacy `test_market_maker_data_fetcher.py` file (replaced with improved versions)

## [1.1.1] - 2025-12-05
### Fixed
- Minor bug fixes and small improvements for release stability.
- No breaking changes; all files preserved.

## [1.1.0] - 2025-12-05
### Added
- CLI test coverage increased to >80%.
- `sector_id` column and `update_price_data_sectors()` method added to database.
- Fetcher refactored for testability and dynamic import of gravity_tse.
- README and database documentation improved.

### Changed
- All main CLI commands now covered by tests.
- Database operations compatible with future migration to Postgres/Supabase.

### Fixed
- Various test and coverage issues.
- Minor bugs in fetcher and database logic.

---
Older changes are available in previous commits.