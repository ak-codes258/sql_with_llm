import logging
import json
from typing import Optional, Dict, Any, List
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError, BadRequest
from google.oauth2 import service_account
import os
import configparser
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_config() -> Dict[str, Any]:
    """Load configuration from config.ini file."""
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs', 'config.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config.read(config_path)
    
    # Load BigQuery configuration
    bq_config = {
        'project': config.get('Database', 'bigquery_project'),
        'dataset': config.get('Database', 'bigquery_dataset'),
        'service_account_file': config.get('Database', 'service_account_file')
    }
    
    return bq_config


def create_bigquery_client(config: Dict[str, str]) -> bigquery.Client:
    """Create and return a BigQuery client using service account credentials."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            config['service_account_file'],
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        client = bigquery.Client(
            project=config['project'],
            credentials=credentials
        )
        
        logger.info(f"Successfully created BigQuery client for project '{config['project']}'")
        return client
    except Exception as e:
        logger.error(f"Failed to create BigQuery client: {e}")
        raise


def execute_query(query: str, client: bigquery.Client = None) -> bigquery.QueryJob:
    """
    Execute a BigQuery query and return the query job.
    
    Args:
        query: The SQL query string to execute
        client: Optional BigQuery client. If None, will create a new client
        
    Returns:
        bigquery.QueryJob: The query job object
    
    Raises:
        ValueError: If query is empty
        GoogleAPICallError: If query execution fails
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if client is None:
        # If no client provided, create one using service account credentials
        config = get_config()
        client = create_bigquery_client(config)
    
    try:
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        return query_job
    except GoogleAPICallError as e:
        logger.error(f"Query execution failed: {e}")
        raise


def get_query_plan_dry_run(query: str, client: bigquery.Client = None) -> Dict[str, Any]:
    """
    Get comprehensive query plan using BigQuery dry run for LLM optimization.
    This method provides detailed analysis WITHOUT executing the query or incurring costs.
    
    Args:
        query: The SQL query string to analyze
        client: Optional BigQuery client. If None, will create a new client
        
    Returns:
        Dict[str, Any]: Comprehensive query plan and execution details
        
    Raises:
        ValueError: If query is empty
        BadRequest: If query has syntax errors
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if client is None:
        # If no client provided, create one using service account credentials
        config = get_config()
        client = create_bigquery_client(config)
    
    try:
        # Configure dry run - validates without executing
        job_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            use_legacy_sql=False  # Ensure GoogleSQL dialect
        )
        
        # Execute dry run - NO actual data processing
        query_job = client.query(query, job_config=job_config)
        
        # Extract comprehensive plan information
        plan_info = {
            "validation_status": "valid",
            "query_metadata": {
                "total_bytes_processed": query_job.total_bytes_processed,
                "total_bytes_billed": query_job.total_bytes_billed,
                "creation_time": str(query_job.created),
                "cache_hit": query_job.cache_hit,
                "referenced_tables": [str(ref) for ref in query_job.referenced_tables] if query_job.referenced_tables else [],
                "statement_type": getattr(query_job, 'statement_type', 'SELECT'),
                "estimated_cost_usd": _calculate_estimated_cost(query_job.total_bytes_processed),
                "data_scan_gb": round(query_job.total_bytes_processed / (1024**3), 2) if query_job.total_bytes_processed else 0
            },
            "performance_estimates": _estimate_performance_metrics(query_job.total_bytes_processed),
            "optimization_context": {
                "large_scan_warning": query_job.total_bytes_processed > 1_000_000_000 if query_job.total_bytes_processed else False,  # > 1GB
                "full_table_scan_likely": _detect_full_table_scan(query),
                "join_complexity": _analyze_join_complexity(query),
                "aggregation_present": _detect_aggregations(query)
            }
        }
        
        # Add detailed query plan if available
        if hasattr(query_job, 'query_plan') and query_job.query_plan:
            plan_info["execution_stages"] = _extract_execution_stages(query_job.query_plan)
        
        # Add schema validation information
        if query_job.referenced_tables:
            plan_info["schema_analysis"] = _analyze_table_references(query_job.referenced_tables, client)
        
        return plan_info
        
    except BadRequest as e:
        return {
            "validation_status": "invalid",
            "error": str(e),
            "error_type": "syntax_error",
            "query_metadata": {
                "total_bytes_processed": 0,
                "estimated_cost_usd": 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting query plan: {e}")
        return {
            "validation_status": "error",
            "error": str(e),
            "error_type": "execution_error",
            "query_metadata": {
                "total_bytes_processed": 0,
                "estimated_cost_usd": 0
            }
        }


def _calculate_estimated_cost(bytes_processed: int) -> float:
    """Calculate estimated query cost based on bytes processed"""
    if bytes_processed:
        tb_processed = bytes_processed / (1024**4)  # Convert to TB
        return round(tb_processed * 5, 4)  # BigQuery on-demand pricing: $5 per TB
    return 0.0


def _estimate_performance_metrics(bytes_processed: int) -> Dict[str, Any]:
    """Estimate performance metrics based on bytes processed"""
    if not bytes_processed:
        return {
            "estimated_slots": 0,
            "execution_time_estimate_seconds": 0,
            "performance_tier": "unknown"
        }
    
    # Rough estimation based on BigQuery performance characteristics
    gb_processed = bytes_processed / (1024**3)
    
    # Estimate slots (BigQuery auto-scaling)
    estimated_slots = min(2000, max(1, int(gb_processed / 0.1)))  # ~100MB per slot
    
    # Estimate execution time (very rough approximation)
    # Assumes ~25 MB/second per slot processing rate
    mb_processed = bytes_processed / (1024**2)
    estimated_seconds = max(1, int(mb_processed / (25 * estimated_slots)))
    
    # Performance tier classification
    if gb_processed < 1:
        performance_tier = "fast"
    elif gb_processed < 10:
        performance_tier = "moderate"
    elif gb_processed < 100:
        performance_tier = "slow"
    else:
        performance_tier = "very_slow"
    
    return {
        "estimated_slots": estimated_slots,
        "execution_time_estimate_seconds": estimated_seconds,
        "performance_tier": performance_tier,
        "data_processed_gb": round(gb_processed, 2)
    }


def _extract_execution_stages(query_plan) -> List[Dict[str, Any]]:
    """Extract detailed execution stages from query plan"""
    stages = []
    
    for stage in query_plan:
        stage_info = {
            "stage_id": getattr(stage, 'id', 'unknown'),
            "name": getattr(stage, 'name', 'unknown'),
            "status": getattr(stage, 'status', 'unknown'),
            "records_read": getattr(stage, 'records_read', 0),
            "records_written": getattr(stage, 'records_written', 0),
            "parallel_inputs": getattr(stage, 'parallel_inputs', 0),
            "compute_ratio_avg": getattr(stage, 'compute_ratio_avg', 0),
            "compute_ratio_max": getattr(stage, 'compute_ratio_max', 0),
            "read_ratio_avg": getattr(stage, 'read_ratio_avg', 0),
            "read_ratio_max": getattr(stage, 'read_ratio_max', 0),
            "write_ratio_avg": getattr(stage, 'write_ratio_avg', 0),
            "write_ratio_max": getattr(stage, 'write_ratio_max', 0),
            "steps": []
        }
        
        # Extract step details
        if hasattr(stage, 'steps') and stage.steps:
            for step in stage.steps:
                step_info = {
                    "kind": getattr(step, 'kind', 'unknown'),
                    "substeps": list(getattr(step, 'substeps', []))
                }
                stage_info["steps"].append(step_info)
        
        stages.append(stage_info)
    
    return stages


def _analyze_table_references(referenced_tables, client: bigquery.Client) -> Dict[str, Any]:
    """Analyze referenced tables for optimization insights"""
    table_analysis = {
        "table_count": len(referenced_tables),
        "tables": []
    }
    
    for table_ref in referenced_tables:
        try:
            table = client.get_table(table_ref)
            table_info = {
                "table_id": str(table_ref),
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "partitioned": table.time_partitioning is not None or table.range_partitioning is not None,
                "clustered": bool(table.clustering_fields),
                "partition_field": getattr(table.time_partitioning, 'field', None) if table.time_partitioning else None,
                "clustering_fields": table.clustering_fields or []
            }
            table_analysis["tables"].append(table_info)
        except Exception as e:
            logger.warning(f"Could not analyze table {table_ref}: {e}")
            table_analysis["tables"].append({
                "table_id": str(table_ref),
                "error": f"Analysis failed: {str(e)}"
            })
    
    return table_analysis


def _detect_full_table_scan(query: str) -> bool:
    """Detect if query likely performs full table scan"""
    query_lower = query.lower()
    
    # Simple heuristics for full table scan detection
    has_where = 'where' in query_lower
    has_limit = 'limit' in query_lower
    has_select_star = 'select *' in query_lower.replace(' ', '')
    
    # Likely full scan if no WHERE clause and SELECT *
    return not has_where and has_select_star and not has_limit


def _analyze_join_complexity(query: str) -> Dict[str, Any]:
    """Analyze JOIN complexity in the query"""
    query_lower = query.lower()
    
    join_types = ['inner join', 'left join', 'right join', 'full join', 'cross join']
    join_count = sum(query_lower.count(join_type) for join_type in join_types)
    
    return {
        "join_count": join_count,
        "complexity": "low" if join_count <= 1 else "medium" if join_count <= 3 else "high",
        "has_cross_join": 'cross join' in query_lower
    }


def _detect_aggregations(query: str) -> bool:
    """Detect if query contains aggregation functions"""
    query_lower = query.lower()
    agg_functions = ['count(', 'sum(', 'avg(', 'min(', 'max(', 'group by']
    return any(func in query_lower for func in agg_functions)


def format_plan_for_llm(plan_data: Dict[str, Any]) -> str:
    """
    Format the comprehensive query plan data for LLM consumption.
    
    Args:
        plan_data: Dictionary containing query plan and metadata
        
    Returns:
        str: Formatted query plan optimized for LLM processing
    """
    if plan_data["validation_status"] != "valid":
        return f"""
--- Query Validation Failed ---
Status: {plan_data['validation_status']}
Error: {plan_data.get('error', 'Unknown error')}
Error Type: {plan_data.get('error_type', 'unknown')}

This query cannot be optimized due to syntax or execution errors.
Please fix the query syntax before requesting optimization.
"""
    
    metadata = plan_data["query_metadata"]
    performance = plan_data["performance_estimates"]
    context = plan_data["optimization_context"]
    
    output = []
    output.append("=== BigQuery Query Analysis for LLM Optimization ===\n")
    
    # Query Metadata Section
    output.append("--- Query Execution Metadata ---")
    output.append(f"Validation Status     : {plan_data['validation_status']}")
    output.append(f"Total Bytes Processed : {metadata['total_bytes_processed']:,} bytes ({metadata['data_scan_gb']} GB)")
    output.append(f"Total Bytes Billed    : {metadata['total_bytes_billed']:,} bytes")
    output.append(f"Estimated Cost        : ${metadata['estimated_cost_usd']}")
    output.append(f"Statement Type        : {metadata['statement_type']}")
    output.append(f"Cache Hit Potential   : {metadata['cache_hit']}")
    output.append(f"Referenced Tables     : {len(metadata['referenced_tables'])}")
    
    for table in metadata['referenced_tables']:
        output.append(f"  - {table}")
    
    # Performance Estimates Section
    output.append("\n--- Performance Estimates ---")
    output.append(f"Estimated Slots Required : {performance['estimated_slots']}")
    output.append(f"Estimated Execution Time : {performance['execution_time_estimate_seconds']} seconds")
    output.append(f"Performance Tier         : {performance['performance_tier']}")
    output.append(f"Data Processed (GB)      : {performance['data_processed_gb']}")
    
    # Optimization Context Section
    output.append("\n--- Optimization Context ---")
    output.append(f"Large Data Scan Warning  : {context['large_scan_warning']}")
    output.append(f"Full Table Scan Likely   : {context['full_table_scan_likely']}")
    output.append(f"Join Complexity          : {context['join_complexity']['complexity']} ({context['join_complexity']['join_count']} joins)")
    output.append(f"Contains Aggregations    : {context['aggregation_present']}")
    output.append(f"Has Cross Join           : {context['join_complexity']['has_cross_join']}")
    
    # Schema Analysis Section
    if "schema_analysis" in plan_data:
        schema = plan_data["schema_analysis"]
        output.append("\n--- Referenced Tables Analysis ---")
        output.append(f"Total Tables: {schema['table_count']}")
        
        for table_info in schema["tables"]:
            if "error" not in table_info:
                output.append(f"\nTable: {table_info['table_id']}")
                output.append(f"  Rows: {table_info['num_rows']:,}")
                output.append(f"  Size: {table_info['num_bytes']:,} bytes")
                output.append(f"  Partitioned: {table_info['partitioned']}")
                output.append(f"  Clustered: {table_info['clustered']}")
                if table_info['partition_field']:
                    output.append(f"  Partition Field: {table_info['partition_field']}")
                if table_info['clustering_fields']:
                    output.append(f"  Clustering Fields: {', '.join(table_info['clustering_fields'])}")
    
    # Execution Stages Section
    if "execution_stages" in plan_data:
        output.append("\n--- Query Execution Stages ---")
        for i, stage in enumerate(plan_data["execution_stages"], 1):
            output.append(f"\nStage {i}: {stage['name']}")
            output.append(f"  Status           : {stage['status']}")
            output.append(f"  Records Read     : {stage['records_read']:,}")
            output.append(f"  Records Written  : {stage['records_written']:,}")
            output.append(f"  Parallel Inputs  : {stage['parallel_inputs']}")
            output.append(f"  Compute Ratio    : {stage['compute_ratio_avg']}")
            
            if stage['steps']:
                output.append("  Steps:")
                for step in stage['steps']:
                    output.append(f"    - {step['kind']}")
                    if step['substeps']:
                        for substep in step['substeps']:
                            output.append(f"        {substep}")
    
    output.append("\n=== End of Query Analysis ===")
    return "\n".join(output)


# Example usage
if __name__ == "__main__":
    try:
        # Example query for testing
        query = """
        SELECT * 
        FROM `linen-striker-454116-c9.techsteer.test_data_event_store`
        WHERE payment_status = 'COMPLETED' 
        AND sender_transaction_amount > 500 
        AND TIMESTAMP_MILLIS(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        """
        
        # Get comprehensive query plan using dry run
        plan_data = get_query_plan_dry_run(query)
        
        # Format for LLM
        formatted_plan = format_plan_for_llm(plan_data)
        print(formatted_plan)
        
        # Also save as JSON for programmatic access
        with open('query_plan_analysis.json', 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        print("\nDetailed plan data saved to 'query_plan_analysis.json'")
        
    except Exception as e:
        logger.error(f"Error: {e}")
