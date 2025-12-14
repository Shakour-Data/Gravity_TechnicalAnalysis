# Services Directory

This folder groups the long-running components that now live inside the
unified Gravity Technical Analysis repository.

- `data_ingestion/`: Raw-market ETL (ex GravityTseHisPrice). Responsible for
  downloading TSE reference/price data into a local SQLite cache that is
  later replicated into PostgreSQL (`tse_input.*` schema).
- `analysis-api/` (root project): Still located at the repository root
  because it already shipped with its own packaging structure. The
  FastAPI/ML service continues to live in `src/`.

Use `docker-compose.stack.yml` for a one-shot local deployment that
starts PostgreSQL, the ingestion worker, and the FastAPI analysis API.
