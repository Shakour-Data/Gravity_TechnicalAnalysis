"""Initial migration for historical_scores table

Revision ID: ecb5692dd22d
Revises:
Create Date: 2025-11-20 13:06:53.194008
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ecb5692dd22d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Ensure target schema exists for analytics tables
    op.execute(sa.text("CREATE SCHEMA IF NOT EXISTS tech_analysis"))

    # Create historical_scores table
    op.create_table(
        "historical_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("trend_score", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("trend_confidence", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("momentum_score", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("momentum_confidence", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("combined_score", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("combined_confidence", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("trend_weight", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column("momentum_weight", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column("trend_signal", sa.String(length=20), nullable=False),
        sa.Column("momentum_signal", sa.String(length=20), nullable=False),
        sa.Column("combined_signal", sa.String(length=20), nullable=False),
        sa.Column(
            "volume_score",
            sa.Numeric(precision=10, scale=4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "volatility_score",
            sa.Numeric(precision=10, scale=4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "cycle_score",
            sa.Numeric(precision=10, scale=4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "support_resistance_score",
            sa.Numeric(precision=10, scale=4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("recommendation", sa.String(length=20), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("price_at_analysis", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("raw_data", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.UniqueConstraint("symbol", "timestamp", "timeframe", name="unique_score_entry"),
    )

    # Daily dimension scores (per-dimension aggregates)
    op.create_table(
        "daily_dimension_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dimension", sa.String(length=50), nullable=False),
        sa.Column("score", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("weight", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("signal", sa.String(length=20), nullable=True),
        sa.Column("features", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.UniqueConstraint("symbol", "timeframe", "ts", "dimension", name="uq_daily_dim"),
        schema="tech_analysis",
    )
    op.create_index(
        "idx_daily_dim_symbol_ts",
        "daily_dimension_scores",
        ["symbol", "ts"],
        unique=False,
        schema="tech_analysis",
    )
    op.create_index(
        "idx_daily_dim_dimension",
        "daily_dimension_scores",
        ["dimension"],
        unique=False,
        schema="tech_analysis",
    )

    # Daily indicator values (per-indicator measurements)
    op.create_table(
        "daily_indicator_values",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dimension", sa.String(length=50), nullable=False),
        sa.Column("indicator_name", sa.String(length=100), nullable=False),
        sa.Column("indicator_params", sa.Text(), nullable=True),
        sa.Column("value", sa.Numeric(precision=20, scale=10), nullable=True),
        sa.Column("score", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("signal", sa.String(length=20), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("weight", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("source_window", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.UniqueConstraint(
            "symbol", "timeframe", "ts", "dimension", "indicator_name", name="uq_daily_indicator"
        ),
        schema="tech_analysis",
    )
    op.create_index(
        "idx_daily_ind_symbol_ts",
        "daily_indicator_values",
        ["symbol", "ts"],
        unique=False,
        schema="tech_analysis",
    )
    op.create_index(
        "idx_daily_ind_dimension",
        "daily_indicator_values",
        ["dimension"],
        unique=False,
        schema="tech_analysis",
    )
    op.create_index(
        "idx_daily_ind_indicator",
        "daily_indicator_values",
        ["indicator_name"],
        unique=False,
        schema="tech_analysis",
    )

    # Create indexes
    op.create_index(
        "idx_historical_scores_symbol_time",
        "historical_scores",
        ["symbol", "timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_historical_scores_timeframe", "historical_scores", ["timeframe"], unique=False
    )
    # op.create_index('idx_historical_scores_date', 'historical_scores', [sa.text('DATE(timestamp)')], unique=False)


def downgrade() -> None:
    op.drop_index(
        "idx_daily_ind_indicator", table_name="daily_indicator_values", schema="tech_analysis"
    )
    op.drop_index(
        "idx_daily_ind_dimension", table_name="daily_indicator_values", schema="tech_analysis"
    )
    op.drop_index(
        "idx_daily_ind_symbol_ts", table_name="daily_indicator_values", schema="tech_analysis"
    )
    op.drop_table("daily_indicator_values", schema="tech_analysis")

    op.drop_index(
        "idx_daily_dim_dimension", table_name="daily_dimension_scores", schema="tech_analysis"
    )
    op.drop_index(
        "idx_daily_dim_symbol_ts", table_name="daily_dimension_scores", schema="tech_analysis"
    )
    op.drop_table("daily_dimension_scores", schema="tech_analysis")

    op.drop_index("idx_historical_scores_timeframe", table_name="historical_scores")
    op.drop_index("idx_historical_scores_symbol_time", table_name="historical_scores")
    op.drop_table("historical_scores")
