import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPICallError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BigQueryMetadataExtractor:
    """
    Extracts and stores detailed metadata from tables in a BigQuery dataset.
    Metadata is saved to timestamped JSON files in a metadata directory.
    """
    
    def __init__(self, project_id: str, dataset_id: str, service_account_file: str):
        """
        Initialize the metadata extractor.
        
        Args:
            project_id: BigQuery project ID
            dataset_id: BigQuery dataset ID
            service_account_file: Path to service account credentials file
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_ref = f"{project_id}.{dataset_id}"
        self.metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata')
        
        try:
            # Create metadata directory if it doesn't exist
            os.makedirs(self.metadata_dir, exist_ok=True)
            
            # Initialize BigQuery client
            self.credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.client = bigquery.Client(project=self.project_id, credentials=self.credentials)
            logger.info(f"Successfully initialized BigQuery client for project '{self.project_id}'.")
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def _get_table_list(self) -> List[bigquery.table.TableListItem]:
        """Get list of all tables in the dataset."""
        try:
            return list(self.client.list_tables(self.dataset_ref))
        except GoogleAPICallError as e:
            logger.error(f"Failed to list tables: {e}")
            raise
    
    def _get_table_schema(self, table: bigquery.table.TableListItem) -> Dict[str, Any]:
        """Get schema information for a table."""
        full_table_id = f"{self.project_id}.{self.dataset_id}.{table.table_id}"
        try:
            table_obj = self.client.get_table(full_table_id)
            
            # Get table description
            table_description = table_obj.description if table_obj.description else "No description available"
            
            # Get column information
            schema = []
            primary_keys = []
            foreign_keys = []
            
            # Get column names for statistics
            column_names = []
            for field in table_obj.schema:
                column_info = {
                    'column_name': field.name,
                    'data_type': field.field_type,
                    'mode': field.mode,
                    'description': field.description if field.description else "",
                    'is_nullable': field.mode == 'NULLABLE',
                    'is_required': field.mode == 'REQUIRED',
                    'is_repeated': field.mode == 'REPEATED'
                }
                
                # Check for primary key annotation (if using BigQuery's _PARTITIONTIME or similar)
                if field.name.lower() in ['_partitiontime', '_partitiondate', '_partitiontimestamp']:
                    column_info['is_partition_key'] = True
                
                schema.append(column_info)
                column_names.append(f"{field.name}")
            
            # Try to get primary keys (if they exist)
            try:
                # Query for primary key information (if available)
                primary_key_query = f"""
                    SELECT column_name
                    FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE`
                    WHERE table_name = '{table.table_id}'
                    AND constraint_name LIKE '%PRIMARY%'
                """
                primary_key_job = self.client.query(primary_key_query)
                for row in primary_key_job:
                    primary_keys.append(row.column_name)
            except Exception as e:
                logger.warning(f"Could not retrieve primary key information: {e}")
            
            return {
                'table_name': full_table_id,
                'table_description': table_description,
                'schema': schema,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys,
                'num_rows': table_obj.num_rows,
                'size_bytes': table_obj.num_bytes,
                'last_modified': str(table_obj.modified),
                'created': str(table_obj.created),
                'partitioning_type': table_obj.partitioning_type,
                'clustering_fields': table_obj.clustering_fields or [],
                'partitioning': {
                    'type': table_obj.partitioning_type,
                    'field': table_obj.time_partitioning.field if table_obj.time_partitioning else None,
                    'require_partition_filter': table_obj.require_partition_filter
                }
            }
            
        except GoogleAPICallError as e:
            logger.error(f"Failed to get schema for {table.table_id}: {e}")
            raise
    
    def _get_dataset_metadata(self) -> Dict[str, Any]:
        """Get metadata about the entire dataset."""
        try:
            dataset = self.client.get_dataset(self.dataset_ref)
            return {
                'dataset_id': self.dataset_id,
                'project_id': self.project_id,
                'location': dataset.location,
                'description': dataset.description,
                'num_tables': len(list(self._get_table_list())),
                'created': str(dataset.created),
                'last_modified': str(dataset.modified)
            }
        except GoogleAPICallError as e:
            logger.error(f"Failed to get dataset metadata: {e}")
            raise
    
    def _clear_metadata_directory(self):
        """Clear all existing metadata files in the directory."""
        try:
            # Get list of all files in the metadata directory
            files = os.listdir(self.metadata_dir)
            
            # Filter for JSON files
            json_files = [f for f in files if f.endswith('.json')]
            
            # Delete each JSON file
            for file in json_files:
                file_path = os.path.join(self.metadata_dir, file)
                os.remove(file_path)
                logger.info(f"Deleted existing metadata file: {file}")
                
        except Exception as e:
            logger.warning(f"Could not clear metadata directory: {e}")
    
    def save_metadata(self):
        """
        Extract and save all metadata to timestamped files.
        Creates two files:
        1. Dataset metadata
        2. Table schemas
        """
        try:
            # Clear existing metadata files first
            self._clear_metadata_directory()
            
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get dataset metadata
            dataset_metadata = self._get_dataset_metadata()
            
            # Get table schemas
            tables = self._get_table_list()
            table_schemas = []
            for table in tables:
                schema = self._get_table_schema(table)
                table_schemas.append(schema)
            
            # Save dataset metadata
            dataset_file = os.path.join(self.metadata_dir, f"dataset_metadata_{timestamp}.json")
            with open(dataset_file, 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            logger.info(f"Saved dataset metadata to {dataset_file}")
            
            # Save table schemas
            tables_file = os.path.join(self.metadata_dir, f"table_schemas_{timestamp}.json")
            with open(tables_file, 'w') as f:
                json.dump(table_schemas, f, indent=2)
            logger.info(f"Saved table schemas to {tables_file}")
            
            return dataset_file, tables_file
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Example usage (replace with your actual config)
        project_id = "your-project-id"
        dataset_id = "your-dataset-id"
        service_account_file = "path/to/your/service-account.json"
        
        extractor = BigQueryMetadataExtractor(project_id, dataset_id, service_account_file)
        dataset_file, tables_file = extractor.save_metadata()
        print(f"Metadata saved successfully to {dataset_file} and {tables_file}")
    except Exception as e:
        logger.error(f"Error: {e}")
