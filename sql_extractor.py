import sqlglot
from sqlglot import exp
from typing import Dict, List, Set
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryTableColumnExtractor:
    def __init__(self, default_project: str = None, default_dataset: str = None):
        """
        Initialize the extractor with optional default project and dataset
        for handling unqualified table references.
        """
        self.default_project = default_project
        self.default_dataset = default_dataset
        
    def extract_tables_and_columns(self, sql_query: str) -> List[Dict[str, List[str]]]:
        """
        Extract tables and their associated columns from a BigQuery SQL query.
        
        Args:
            sql_query (str): The SQL query to parse
            
        Returns:
            List[Dict[str, List[str]]]: List of dictionaries where each dict has
                                      one fully qualified table name as key and
                                      list of its columns as value
        """
        try:
            # Parse the SQL query using sqlglot with BigQuery dialect
            parsed = sqlglot.parse(sql_query, read='bigquery')
            
            if not parsed:
                logger.warning("Could not parse SQL query, falling back to regex extraction")
                return self._fallback_extraction(sql_query)
            
            # Dictionary to store table -> columns mapping
            table_columns = {}
            
            # Process each parsed statement
            for statement in parsed:
                self._extract_from_statement(statement, table_columns)
            
            # Convert to list of dictionaries format
            result = []
            for table_name, columns in table_columns.items():
                result.append({table_name: list(columns)})
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {e}")
            return self._fallback_extraction(sql_query)
    
    def _extract_from_statement(self, statement: exp.Expression, table_columns: Dict[str, Set[str]]):
        """Extract tables and columns from a parsed SQL statement"""
        
        # Dictionary to map aliases to full table names
        alias_to_table = {}
        
        # First pass: collect all table references and their aliases
        for table_node in statement.find_all(exp.Table):
            full_table_name = self._get_full_table_name(table_node)
            
            # Get alias if present
            alias = None
            if hasattr(table_node, 'alias') and table_node.alias:
                alias = table_node.alias
            elif hasattr(table_node, 'alias_or_name'):
                alias = table_node.alias_or_name
            
            # Map alias to full table name
            if alias and alias != full_table_name:
                alias_to_table[alias] = full_table_name
            
            # Initialize column set for this table
            if full_table_name not in table_columns:
                table_columns[full_table_name] = set()
        
        # Second pass: collect all column references
        for column_node in statement.find_all(exp.Column):
            column_name = column_node.name
            table_ref = None
            
            # Try to get the table reference for this column
            if hasattr(column_node, 'table') and column_node.table:
                table_ref = column_node.table
            elif hasattr(column_node, 'db') and column_node.db:
                table_ref = column_node.db
            
            # Resolve table reference
            if table_ref:
                # Check if it's an alias
                if table_ref in alias_to_table:
                    full_table_name = alias_to_table[table_ref]
                else:
                    # Try to find matching table by name
                    full_table_name = self._resolve_table_reference(table_ref, table_columns.keys())
                
                if full_table_name and full_table_name in table_columns:
                    table_columns[full_table_name].add(column_name)
            else:
                # Column without explicit table reference - add to all tables
                # This handles cases like SELECT column_name FROM table WHERE condition
                for table_name in table_columns.keys():
                    table_columns[table_name].add(column_name)
    
    def _get_full_table_name(self, table_node: exp.Table) -> str:
        """Extract fully qualified table name from table node"""
        
        # Get the table identifier
        if hasattr(table_node, 'this') and table_node.this:
            table_id = str(table_node.this)
        else:
            table_id = str(table_node)
        
        # Remove backticks if present
        table_id = table_id.strip('`')
        
        # If already fully qualified, return as-is
        if table_id.count('.') >= 2:
            return table_id
        
        # If partially qualified (dataset.table), add default project
        if table_id.count('.') == 1 and self.default_project:
            return f"{self.default_project}.{table_id}"
        
        # If unqualified (just table name), add default project and dataset
        if table_id.count('.') == 0 and self.default_project and self.default_dataset:
            return f"{self.default_project}.{self.default_dataset}.{table_id}"
        
        return table_id
    
    def _resolve_table_reference(self, table_ref: str, known_tables: List[str]) -> str:
        """Resolve a table reference to a full table name"""
        
        # Direct match
        if table_ref in known_tables:
            return table_ref
        
        # Try to find by last part of qualified name
        for table_name in known_tables:
            if table_name.split('.')[-1] == table_ref:
                return table_name
        
        return table_ref
    
    def _fallback_extraction(self, sql_query: str) -> List[Dict[str, List[str]]]:
        """Fallback method using regex when sqlglot parsing fails"""
        logger.info("Using fallback regex extraction method")
        
        result = []
        
        # Extract table references using regex
        # Pattern for fully qualified table names with backticks
        table_pattern = r'`([^`]+\.[^`]+\.[^`]+)`'
        # Pattern for table aliases
        alias_pattern = r'`[^`]+\.[^`]+\.[^`]+`\s+(?:AS\s+)?(\w+)'
        
        # Find all table references
        table_matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
        alias_matches = re.findall(alias_pattern, sql_query, re.IGNORECASE)
        
        # Create alias mapping
        alias_to_table = {}
        for i, table in enumerate(table_matches):
            if i < len(alias_matches):
                alias_to_table[alias_matches[i]] = table
        
        # Extract column references
        # Simple pattern for column references (this is very basic)
        column_pattern = r'\b(\w+)\.(\w+)\b'
        column_matches = re.findall(column_pattern, sql_query)
        
        # Group columns by table
        table_columns = {}
        for table in table_matches:
            table_columns[table] = set()
        
        # Map columns to tables
        for table_alias, column in column_matches:
            if table_alias in alias_to_table:
                full_table = alias_to_table[table_alias]
                if full_table in table_columns:
                    table_columns[full_table].add(column)
        
        # Convert to required format
        for table_name, columns in table_columns.items():
            if columns:  # Only include tables with identified columns
                result.append({table_name: list(columns)})
        
        return result

def extract_table_columns_from_query(sql_query: str, 
                                   default_project: str = None, 
                                   default_dataset: str = None) -> List[Dict[str, List[str]]]:
    """
    Main function to extract tables and their columns from a BigQuery SQL query.
    
    Args:
        sql_query (str): The SQL query to parse
        default_project (str): Default project for unqualified table references
        default_dataset (str): Default dataset for unqualified table references
        
    Returns:
        List[Dict[str, List[str]]]: List of dictionaries with table names as keys
                                   and their column lists as values
    """
    extractor = BigQueryTableColumnExtractor(default_project, default_dataset)
    return extractor.extract_tables_and_columns(sql_query)

# Example usage and testing
if __name__ == "__main__":
    # Test with a complex BigQuery SQL query
    test_query = """
    SELECT 
        u.user_id, 
        u.email, 
        u.created_date,
        o.order_id, 
        o.total_amount, 
        o.order_date,
        p.product_name,
        p.category,
        SUM(oi.quantity) as total_quantity
    FROM `my-project.analytics.users` u
    JOIN `my-project.analytics.orders` o ON u.user_id = o.user_id
    JOIN `my-project.analytics.products` p ON o.product_id = p.product_id
    LEFT JOIN `my-project.analytics.order_items` oi ON o.order_id = oi.order_id
    WHERE u.created_date >= '2024-01-01'
        AND o.total_amount > 100
        AND p.category IN ('Electronics', 'Books')
    GROUP BY u.user_id, u.email, u.created_date, o.order_id, o.total_amount, o.order_date, p.product_name, p.category
    ORDER BY total_quantity DESC
    LIMIT 100
    """
    
    print("Testing BigQuery Table-Column Extractor")
    print("=" * 50)
    
    result = extract_table_columns_from_query(
        test_query, 
        default_project="my-project",
        default_dataset="analytics"
    )
    
    print("Extracted Tables and Columns:")
    for i, table_dict in enumerate(result, 1):
        for table_name, columns in table_dict.items():
            print(f"\n{i}. Table: {table_name}")
            print(f"   Columns: {columns}")
    
    print(f"\nTotal tables found: {len(result)}")
    
    # Test JSON serialization
    import json
    print("\nJSON Output:")
    print(json.dumps(result, indent=2))
