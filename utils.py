"""
Utility functions for the agent.

Provides common utilities like Spark session management.
"""

from typing import Optional

from environment import is_databricks_mode


def get_spark_session(force_standalone: bool = False):
    """
    Get Spark session (DatabricksSession or SparkSession).
    
    Args:
        force_standalone: If True, force use of SparkSession even in Databricks
    
    Returns:
        Spark session instance
    """
    # Force standalone mode if requested
    if force_standalone:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    
    # Try Databricks session first if in Databricks mode
    if is_databricks_mode():
        try:
            from databricks.connect import DatabricksSession
            return DatabricksSession.builder.getOrCreate()
        except (ImportError, AttributeError, Exception):
            # Fall back to SparkSession if DatabricksSession fails
            pass
    
    # Use standard SparkSession
    from pyspark.sql import SparkSession
    return SparkSession.builder.getOrCreate()

