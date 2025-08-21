import os
import logging
import json
from typing import Dict, Any, List
import configparser
from google.cloud import bigquery
from google.oauth2 import service_account
from query_executor import get_query_plan_dry_run
from sql_extractor import extract_table_columns_from_query
from prompt import generate_llm_prompt
from llm_response import call_llm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.ini file.
    
    Returns:
        Dict containing configuration settings
    """
    config = configparser.ConfigParser()
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'config.ini')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    config.read(config_file)
    
    # Load BigQuery config
    bq_config = {
        'project': config.get('Database', 'bigquery_project'),
        'dataset': config.get('Database', 'bigquery_dataset'),
        'service_account_file': config.get('Database', 'service_account_file')
    }
    
    # Load LLM config if present
    llm_config = {}
    if 'LLM' in config:
        llm_config['model'] = config.get('LLM', 'model')
        llm_config['api_key'] = config.get('LLM', 'api_key')
        llm_config['embedding_model'] = config.get('LLM', 'embedding_model')
    
    return {**bq_config, **llm_config}

def load_query() -> Dict[str, Any]:
    """Load a single query from query.json file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(script_dir, 'query.json')
    
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query file not found at {query_path}")
    
    try:
        with open(query_path, 'r') as f:
            query_data = json.load(f)
            query = query_data.get('query', {})
            if not query or not all(key in query for key in ['sql']):
                raise ValueError("Query object must contain 'sql' field")
            return query
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse query file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading query: {e}")
        raise

def get_query_plan_analysis(query: str, client: bigquery.Client = None) -> Dict[str, Any]:
    """
    Get comprehensive query plan analysis including dry run and execution details.
    
    Args:
        query: The SQL query string to analyze
        client: Optional BigQuery client
        
    Returns:
        Dict containing query plan analysis
    """
    try:
        # Get dry run analysis first
        plan_data = get_query_plan_dry_run(query, client)
        
        # Save analysis to JSON
        analysis_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_plan_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
            logger.info(f"Saved query plan analysis to {analysis_file}")
            
        return plan_data
    except Exception as e:
        logger.error(f"Failed to get query plan analysis: {e}")
        raise

def analyze_query_plan(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the query plan and extract key metrics.
    
    Args:
        plan_data: Dictionary containing query plan information
        
    Returns:
        Dict containing analyzed metrics
    """
    try:
        analysis = {
            'validation_status': plan_data.get('validation_status', 'unknown'),
            'error': plan_data.get('error', None),
            'error_type': plan_data.get('error_type', None),
            'query_metadata': plan_data.get('query_metadata', {}),
            'estimated_cost': plan_data['query_metadata'].get('estimated_cost_usd', 0),
            'bytes_processed': plan_data['query_metadata'].get('total_bytes_processed', 0)
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Failed to analyze query plan: {e}")
        return {'error': str(e)}

def extract_query_details(sql_query: str, config: Dict[str, Any]) -> List[Dict[str, List[str]]]:
    """
    Extract table and column details from the SQL query.
    
    Args:
        sql_query: The SQL query string to analyze
        config: Configuration containing project and dataset info
        
    Returns:
        List of dictionaries containing extracted tables and columns
    """
    try:
        # Get project and dataset from config
        project = config['project']
        dataset = config['dataset']
        
        # Extract tables and columns
        extracted_data = extract_table_columns_from_query(
            sql_query,
            default_project=project,
            default_dataset=dataset
        )
        
        # Save extracted data to JSON
        details_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_details.json')
        with open(details_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
            logger.info(f"Saved query details to {details_file}")
        
        return extracted_data
    except Exception as e:
        logger.error(f"Failed to extract query details: {e}")
        raise

def create_table_context(query_details: List[Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    """
    Create table context by matching query details with metadata.
    
    Args:
        query_details: List of dictionaries containing tables and columns from query
        
    Returns:
        List of dictionaries containing table context with schema details
    """
    try:
        # Get latest metadata file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_dir = os.path.join(script_dir, 'metadata')
        
        # Find latest table schemas file
        table_schemas_file = None
        latest_time = 0
        
        for file in os.listdir(metadata_dir):
            if file.startswith('table_schemas_') and file.endswith('.json'):
                timestamp = int(file.split('_')[-1].split('.')[0])
                if timestamp > latest_time:
                    latest_time = timestamp
                    table_schemas_file = os.path.join(metadata_dir, file)
        
        if not table_schemas_file:
            raise FileNotFoundError("No table schemas metadata file found")
            
        # Load table schemas metadata
        with open(table_schemas_file, 'r') as f:
            table_schemas = json.load(f)
            
        # Create table context
        table_context = []
        
        for table_dict in query_details:
            for table_name, columns in table_dict.items():
                # Find matching table in metadata
                matching_table = None
                for schema in table_schemas:
                    # Compare table names (case-insensitive)
                    if schema['table_name'].lower() == table_name.lower():
                        matching_table = schema
                        break
                
                if matching_table:
                    # Get relevant columns from schema
                    relevant_columns = []
                    for column in matching_table['schema']:
                        if column['column_name'].lower() in [col.lower() for col in columns]:
                            relevant_columns.append({
                                'name': column['column_name'],
                                'data_type': column['data_type'],
                                'mode': column['mode'],
                                'description': column['description']
                            })
                    
                    context = {
                        'table_name': table_name,
                        'description': matching_table.get('table_description', ''),
                        'columns': relevant_columns,
                        'primary_keys': matching_table.get('primary_keys', []),
                        'foreign_keys': matching_table.get('foreign_keys', []),
                        'num_rows': matching_table.get('num_rows'),
                        'size_bytes': matching_table.get('size_bytes'),
                        'last_modified': matching_table.get('last_modified'),
                        'created': matching_table.get('created'),
                        'partitioning': {
                            'type': matching_table.get('partitioning_type'),
                            'field': matching_table.get('partitioning', {}).get('field'),
                            'require_partition_filter': matching_table.get('partitioning', {}).get('require_partition_filter')
                        },
                        'clustering_fields': matching_table.get('clustering_fields', [])
                    }
                    table_context.append(context)
        
        # Save table context to JSON
        context_file = os.path.join(script_dir, 'table_context.json')
        with open(context_file, 'w') as f:
            json.dump(table_context, f, indent=2)
            logger.info(f"Saved table context to {context_file}")
        
        return table_context
        
    except Exception as e:
        logger.error(f"Failed to create table context: {e}")
        raise

def main():
    try:
        # Load configurations
        config = load_config()
        logger.info("Configuration loaded successfully!")
        
        # Load query
        query_data = load_query()
        logger.info("Query loaded successfully!")
        
        # Log the query details
        logger.info(f"Query loaded:")
        logger.info(f"SQL: {query_data['sql']}")
        
        # Get query plan analysis
        logger.info("Generating query plan analysis...")
        plan_data = get_query_plan_analysis(query_data['sql'])
        
        # Analyze the plan
        analysis = analyze_query_plan(plan_data)
        logger.info("Query plan analysis complete!")
        
        # Log key metrics
        logger.info("\nQuery Plan Analysis:")
        logger.info(f"Status: {analysis['validation_status']}")
        logger.info(f"Estimated Cost: ${analysis['estimated_cost']:.2f}")
        logger.info(f"Bytes Processed: {analysis['bytes_processed']}")
        
        if analysis['error']:
            logger.error(f"Error in query plan: {analysis['error']}")
        
        # Extract table and column details
        logger.info("\nExtracting table and column details...")
        extracted_details = extract_query_details(query_data['sql'], config)
        
        # Log extracted details
        logger.info("\nExtracted Tables and Columns:")
        for i, table_dict in enumerate(extracted_details, 1):
            for table_name, columns in table_dict.items():
                logger.info(f"\n{i}. Table: {table_name}")
                logger.info(f"   Columns: {columns}")
        
        logger.info(f"\nTotal tables found: {len(extracted_details)}")
        
        # Create table context
        logger.info("\nCreating table context from metadata...")
        table_context = create_table_context(extracted_details)
        
        # Generate LLM prompt
        logger.info("\nGenerating optimization prompt...")
        prompt = generate_llm_prompt(
            sql_query=query_data['sql'],
            table_context=table_context,
            query_plan_info=analysis,
            model="gpt-4-0125-preview"
        )
        
        # Save prompt to file
        prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_prompt.json')
        with open(prompt_file, 'w') as f:
            json.dump(prompt, f, indent=2)
            logger.info(f"Saved optimization prompt to {prompt_file}")
        
        # Call LLM for optimization
        logger.info("\nCalling LLM for optimization...")
        
        # Load LLM config from config
        llm_config = {
            "model": "gpt-4-0125-preview",
            "system_message": "You are a skilled BigQuery SQL optimizer. Please respond with a JSON object exactly matching this structure:\n{\n  \"business_goal\": \"string\",\n  \"current_bottlenecks\": [\"string\"],\n  \"sql_optimizations\": {\n    \"optimized_sql\": \"string\",\n    \"changes_made\": [\"string\"],\n    \"performance_impact\": \"string\"\n  },\n  \"schema_recommendations\": {\n    \"partitioning\": [\"string\"],\n    \"clustering\": [\"string\"],\n    \"other\": [\"string\"]\n  },\n  \"cost_savings\": \"string\"\n}\nDo not include any additional text or explanations outside of this JSON structure."
        }
        
        # Add API key from config
        if 'api_key' in config:
            llm_config['api_key'] = config['api_key']
        
        # Call LLM
        llm_response = call_llm(
            prompt=prompt,
            llm_config=llm_config,
            max_tokens=2048,
            temperature=0.1
        )
        
        # Save LLM response
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_output.json')
        with open(output_file, 'w') as f:
            json.dump(llm_response, f, indent=2)
            logger.info(f"Saved LLM response to {output_file}")
        
        # Print optimization results
        logger.info("\nOptimization Results:")
        if 'error' in llm_response:
            logger.error(f"LLM Error: {llm_response['error']}")
        else:
            raw_response_str = llm_response.get('raw_response', '{}')
            if raw_response_str.startswith("```json"):
                raw_response_str = raw_response_str[7:]
            if raw_response_str.endswith("```"):
                raw_response_str = raw_response_str[:-3]
            
            try:
                optimization_results = json.loads(raw_response_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response JSON: {e}")
                optimization_results = {}

            logger.info("\nBusiness Goal:")
            logger.info(optimization_results.get('business_goal', "No business goal provided"))
            
            logger.info("\nCurrent Bottlenecks:")
            for bottleneck in optimization_results.get('current_bottlenecks', []):
                logger.info(f"- {bottleneck}")
            
            logger.info("\nSQL Optimizations:")
            logger.info("Optimized SQL:")
            logger.info(optimization_results.get('sql_optimizations', {}).get('optimized_sql', "No optimized SQL provided"))
            
            logger.info("\nChanges Made:")
            for change in optimization_results.get('sql_optimizations', {}).get('changes_made', []):
                logger.info(f"- {change}")
            
            logger.info("\nPerformance Impact:")
            logger.info(optimization_results.get('sql_optimizations', {}).get('performance_impact', "No performance impact provided"))
            
            logger.info("\nSchema Recommendations:")
            logger.info("Partitioning:")
            for field in optimization_results.get('schema_recommendations', {}).get('partitioning', []):
                logger.info(f"- {field}")
            
            logger.info("Clustering:")
            for field in optimization_results.get('schema_recommendations', {}).get('clustering', []):
                logger.info(f"- {field}")
            
            logger.info("Other Suggestions:")
            for suggestion in optimization_results.get('schema_recommendations', {}).get('other', []):
                logger.info(f"- {suggestion}")
            
            logger.info("\nCost Savings:")
            logger.info(optimization_results.get('cost_savings', "No cost savings provided"))
        
        # Log table context
        logger.info("\nTable Context:")
        for i, context in enumerate(table_context, 1):
            logger.info(f"\n{i}. Table: {context['table_name']}")
            logger.info(f"   Description: {context['description']}")
            logger.info("   Columns:")
            for col in context['columns']:
                logger.info(f"     - {col['name']}: {col['data_type']} ({col['mode']})")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()