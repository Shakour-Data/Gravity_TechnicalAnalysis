"""
API v1 Routes

Main router configuration for API version 1 endpoints.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from fastapi import APIRouter
from gravity_tech.api.v1 import ml as ml_router
from gravity_tech.api.v1 import tools as tools_router
from gravity_tech.api.v1 import backtest as backtest_router
from gravity_tech.api.v1 import db_explorer as db_router
from gravity_tech.api.v1 import analysis as analysis_router

router = APIRouter()
router.include_router(ml_router.router)
router.include_router(tools_router.router)
router.include_router(backtest_router.router)
router.include_router(db_router.router)
router.include_router(analysis_router.router)
