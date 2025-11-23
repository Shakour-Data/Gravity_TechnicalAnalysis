"""Initial migration for historical_scores table

Revision ID: ecb5692dd22d
Revises: 
Create Date: 2025-11-20 13:06:53.194008

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ecb5692dd22d'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create historical_scores table
    op.create_table('historical_scores',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('trend_score', sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column('trend_confidence', sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column('momentum_score', sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column('momentum_confidence', sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column('combined_score', sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column('combined_confidence', sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column('trend_weight', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('momentum_weight', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('trend_signal', sa.String(length=20), nullable=False),
        sa.Column('momentum_signal', sa.String(length=20), nullable=False),
        sa.Column('combined_signal', sa.String(length=20), nullable=False),
        sa.Column('recommendation', sa.String(length=20), nullable=False),
        sa.Column('action', sa.String(length=20), nullable=False),
        sa.Column('price_at_analysis', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timestamp', 'timeframe', name='unique_score_entry')
    )
    
    # Create indexes
    op.create_index('idx_historical_scores_symbol_time', 'historical_scores', ['symbol', 'timestamp'], unique=False)
    op.create_index('idx_historical_scores_timeframe', 'historical_scores', ['timeframe'], unique=False)
    op.create_index('idx_historical_scores_date', 'historical_scores', [sa.text('DATE(timestamp)')], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_historical_scores_date', table_name='historical_scores')
    op.drop_index('idx_historical_scores_timeframe', table_name='historical_scores')
    op.drop_index('idx_historical_scores_symbol_time', table_name='historical_scores')
    
    # Drop table
    op.drop_table('historical_scores')
