"""
Database Schema Validator

Validates that database schema is consistent and all required tables/columns exist.
Runs before data loading to catch schema issues early.
"""

from typing import List, Dict, Optional
import structlog

logger = structlog.get_logger()


class SchemaValidator:
    """Validate database schema consistency"""
    
    def __init__(self, db_engine):
        """Initialize validator with database engine"""
        self.engine = db_engine
    
    def validate_tables_exist(self, expected_tables: List[str]) -> bool:
        """
        Check if all required tables exist.
        
        Args:
            expected_tables: List of table names to check
        
        Returns:
            True if all tables exist, False otherwise
        """
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        
        missing = set(expected_tables) - set(existing_tables)
        
        if missing:
            logger.error(
                "missing_tables",
                tables=list(missing),
                existing=existing_tables
            )
            return False
        
        logger.info("all_tables_exist", tables=expected_tables)
        return True
    
    def validate_column_types(
        self,
        table: str,
        expected_cols: Dict[str, str]
    ) -> bool:
        """
        Validate column types match expected schema.
        
        Args:
            table: Table name
            expected_cols: Dict of {column_name: expected_type}
        
        Returns:
            True if all columns have correct types
        """
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        
        try:
            existing_cols = inspector.get_columns(table)
            col_map = {col['name']: str(col['type']) for col in existing_cols}
            
            mismatches = []
            for col_name, expected_type in expected_cols.items():
                if col_name not in col_map:
                    mismatches.append(
                        f"{col_name}: missing (expected {expected_type})"
                    )
                elif col_map[col_name] != expected_type:
                    mismatches.append(
                        f"{col_name}: {col_map[col_name]} "
                        f"(expected {expected_type})"
                    )
            
            if mismatches:
                logger.error(
                    "column_type_mismatches",
                    table=table,
                    mismatches=mismatches
                )
                return False
            
            logger.info(
                "column_types_valid",
                table=table,
                columns=len(expected_cols)
            )
            return True
            
        except Exception as e:
            logger.error(
                "schema_validation_error",
                table=table,
                error=str(e)
            )
            return False
    
    def get_table_info(self, table: str) -> Optional[Dict]:
        """Get table structure information"""
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        
        try:
            columns = inspector.get_columns(table)
            pk = inspector.get_pk_constraint(table)
            indexes = inspector.get_indexes(table)
            
            return {
                "table": table,
                "columns": [
                    {
                        "name": col['name'],
                        "type": str(col['type']),
                        "nullable": col['nullable']
                    }
                    for col in columns
                ],
                "primary_key": pk,
                "indexes": indexes,
                "column_count": len(columns)
            }
            
        except Exception as e:
            logger.error(
                "get_table_info_failed",
                table=table,
                error=str(e)
            )
            return None
