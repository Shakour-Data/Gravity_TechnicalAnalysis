"""
Database models for input tables.
Used for Alembic autogeneration.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Sector(Base):
    __tablename__ = 'sectors'
    sector_id = Column(Integer, primary_key=True)
    sector_name = Column(Text, unique=True)
    sector_name_en = Column(Text)
    us_sector = Column(Text)

class Market(Base):
    __tablename__ = 'markets'
    market_id = Column(Integer, primary_key=True)
    market_name = Column(Text, unique=True)

class Panel(Base):
    __tablename__ = 'panels'
    panel_id = Column(Integer, primary_key=True)
    panel_name = Column(Text, unique=True)

class Company(Base):
    __tablename__ = 'companies'
    company_id = Column(Text, primary_key=True)
    ticker = Column(Text, unique=True)
    name = Column(Text)
    sector_id = Column(Integer, ForeignKey('sectors.sector_id'))
    panel_id = Column(Integer, ForeignKey('panels.panel_id'))
    market_id = Column(Integer, ForeignKey('markets.market_id'))

class PriceData(Base):
    __tablename__ = 'price_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adj_close = Column(Float)