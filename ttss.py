import configparser
import json
from src.llm.prompt_templates import get_prompt_for_generating_sql_query, get_prompt_for_sql_optimization
import os
from src.llm.llm_connector import LLMConnector
from src.services.ragQueryPipline import RAGPipeline
from src.utils.config_reader import load_config
from src.utils.logger import logger

class ConvertTextToSqlRequest():
    def __init__(self, query: str, llm_type: str, market: str):
        self.query = query
        self.llm_type = llm_type
        self.market = market
        self.config = load_config(market)
        logger.info("TEXT_TO_SQL_INIT - Query: %s, LLM Type: %s", query[:100] + "..." if len(query) > 100 else query, llm_type)

    def convert_text_to_sql_using_llm(self):
        logger.info("Generating SQL for: '%s'", self.query)
        if self.llm_type =='openai':
            # Get Table and columns contexts using RAGPipeline with the provided market
            rag_pipeline = RAGPipeline(self.market)
            context_data = rag_pipeline.query(self.query)

            # Process the structured context to extract tables and columns
            matched_tables = self._extract_tables_from_rag(context_data)
            matched_cols = self._extract_columns_from_rag(context_data)
            
            # Close the RAG pipeline connection
            rag_pipeline.close()

            # Create prompt for text to sql
            # Define system instruction
            system_message = {
                "role": "system",
                "content": (
                    "You are a skilled Data Analyst with deep expertise in BigQuery and GoogleSQL dialect. "
                    "Your task is to generate accurate and optimized SQL queries only. "
                    "Respond strictly with the SQL query as per instructions—no explanations, comments, or additional text."
                )
            }
            # Load middle conversation from JSON
            MESSAGES_PATH = self.config.get('Database', 'messages_path')
            with open(MESSAGES_PATH, 'r') as file:
                middle_conversation = json.load(file)
            
            TEMPLATE = get_prompt_for_generating_sql_query(matched_tables, matched_cols, self.query)
            
            # Define the last user prompt dynamically
            last_prompt = {
                "role": "user",
                "content": f"{TEMPLATE}"
            }
            
            prompt_messages = [system_message] + middle_conversation + [last_prompt]

            llm = LLMConnector(prompt_messages, self.config)
            res = llm.get_llm_response()
            logger.debug("LLM response: %s", res)
            
            # Try different possible markers in order of preference
            markers = ["Optimised Query:-", "Optimised Query:", "Optimized Query:-", "Optimized Query:"]
            
            for marker in markers:
                if marker in res:
                    parts = res.split(marker, 1)  # Split only once
                    if len(parts) > 1:
                        return parts[1].strip()
            
            # If no marker found, return the full response
            logger.warning("No query marker found in LLM response")
            return res
        else:
            logger.warning("TEXT_TO_SQL_UNSUPPORTED_LLM - Type: %s", self.llm_type)
            return f"Unsupported LLM: {self.llm_type}"

    def _extract_tables_from_rag(self, context_data):
        """Extract and format table information from RAG pipeline results. """
        tables_info = []
        
        def format_columns(columns_str):
            if not columns_str:
                return ""
            return ", ".join([col.strip() for col in columns_str.replace('•', ',').replace('–', ',').split(',') if col.strip()])
        
        # Process structured context first
        for item in context_data.get('structured_context', []):
            context = item.get('key_value_context', {})
            if context.get('type') != 'table':
                continue
                
            # Extract table name and handle display name if present
            table_name = context.get('name', '').replace(' (Event Store)', '').strip()
            if not table_name:
                continue
                
            # Build the table header
            table_info = [f"# Table: {table_name} (Event Store)", ""]
            
            # Add description (remove duplicate 'Description:' if present)
            description = context.get('Description', '').replace('Description:', '').strip()
            if description:
                table_info.extend([f"Description: {description}", ""])
            
            # Start columns section
            columns_section = ["Columns:"]
            
            # Add key columns
            key_columns = format_columns(context.get('• Key columns', ''))
            if key_columns:
                columns_section.append(f"  • Key columns: {key_columns}")
            
            # Add filterable columns
            filterable = format_columns(context.get('• Filterable columns', ''))
            if filterable:
                columns_section.append(f"  • Filterable columns: {filterable}")
            
            # Add aggregatable columns
            aggregatable = format_columns(context.get('• Aggregatable columns', ''))
            if aggregatable:
                columns_section.append(f"  • Aggregatable columns: {aggregatable}")
            
            # Add sortable columns
            sortable = format_columns(context.get('• Sortable columns', ''))
            if sortable:
                columns_section.append(f"  • Sortable columns: {sortable}")
            
            # Add joins
            joins = context.get('Joins', '')
            if joins:
                join_lines = [f"- {join.strip()}" for join in joins.split('–') if join.strip()]
                if join_lines:
                    columns_section.extend(["", "Joins:"] + join_lines)
            
            # Add business terms
            business_terms = context.get('Business terms', '')
            if business_terms:
                terms = ", ".join([term.strip() for term in business_terms.split(',')])
                columns_section.extend(["", f"Business terms: {terms}"])
            
            # Add sample queries
            sample_queries = context.get('Sample queries', '')
            if sample_queries:
                queries = [q.strip() for q in sample_queries.split('`') if q.strip() and not q.strip().startswith('SELECT')]
                if len(queries) >= 2:  # At least one complete query (description + SQL)
                    columns_section.extend(["", "Sample queries:"])
                    for i in range(0, len(queries)-1, 2):
                        columns_section.append(f"  {i//2 + 1}. {queries[i].strip()}")
                        columns_section.append(f"     `{queries[i+1].strip()}`")
            
            # Add tags
            tags = context.get('Tags', '')
            if tags:
                tag_list = ", ".join([tag.strip() for tag in tags.split(',')])
                columns_section.extend(["", f"Tags: {tag_list}"])
            
            # Combine all sections
            table_info.extend(columns_section)
            tables_info.append('\n'.join(line for line in table_info if line))
        
        # If no tables found in structured context, try text context
        if not tables_info:
            for item in context_data.get('text_context', []):
                if item.get('context_type') == 'table':
                    tables_info.append(item['context'])
        
        return '\n\n'.join(tables_info) if tables_info else 'No relevant tables found in the context.'
    
    def _extract_columns_from_rag(self, context_data):
        """Extract and format column information from RAG pipeline results.
        
        Args:
            context_data: Dictionary containing structured context from RAG pipeline
            
        Returns:
            str: Formatted string containing column information
        """
        columns_info = []
        
        # Process both structured and text contexts
        for context_source in ['structured_context', 'text_context']:
            for item in context_data.get(context_source, []):
                # Handle both structured and text context formats
                if context_source == 'structured_context':
                    context = item.get('key_value_context', {})
                    if context.get('type') != 'column':
                        continue
                        
                    column_name = context.get('name', '').strip()
                    table = context.get('Table', '').strip()
                    data_type = context.get('Data Type', '').strip()
                    description = context.get('Description', '').strip()
                    
                    # Extract additional column properties
                    is_primary = bool(context.get('Primary Key', '').strip())
                    is_nullable = context.get('Nullable', '').strip()
                    sample_values = context.get('Sample Values', '').strip()
                    
                    # Build column info
                    column_info = [
                        f"Column: {column_name}",
                        f"Table: {table}" if table else "",
                        f"Type: {data_type}" if data_type else "",
                        f"Description: {description}" if description else "",
                        "Primary Key: Yes" if is_primary else "",
                        f"Nullable: {is_nullable}" if is_nullable else "",
                        f"Sample Values: {sample_values}" if sample_values else ""
                    ]
                    
                    # Join non-empty parts with newlines
                    column_info = '\n'.join(filter(None, column_info))
                    columns_info.append(column_info)
                    
                else:  # text_context
                    context = item.get('context', '')
                    if 'Column:' in context and ('Table:' in context or 'Type:' in context):
                        columns_info.append(context)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for column in columns_info:
            if column not in seen:
                seen.add(column)
                unique_columns.append(column)
        
        # Join all column information with double newlines for better readability
        if not unique_columns:
            return 'No relevant columns found in the context.'
            
        return '\n\n'.join(unique_columns)
    
    def optimize_sql_query_using_llm(self):
        if self.llm_type =='openai':
            rag_pipeline = RAGPipeline(self.market)
            context_data = rag_pipeline.query(self.query)
            matched_tables = self._extract_tables_from_rag(context_data)
            matched_cols = self._extract_columns_from_rag(context_data)
            rag_pipeline.close()
            system_message = {
                "role": "system",
                "content": (
                    "You are a skilled Data Analyst with deep expertise in BigQuery and GoogleSQL dialect. "
                    "Your task is to generate accurate and optimized SQL queries only. "
                    "Respond strictly with the SQL query as per instructions—no explanations, comments, or additional text."
                )
            }

            MESSAGES_PATH = self.config.get('Database', 'messages_path')
            with open(MESSAGES_PATH, 'r') as file:
                middle_conversation = json.load(file)
            
            TEMPLATE = get_prompt_for_sql_optimization(matched_tables, matched_cols, self.query)
            
            # Define the last user prompt dynamically
            last_prompt = {
                "role": "user",
                "content": f"{TEMPLATE}"
            }
            
            prompt_messages = [system_message] + middle_conversation + [last_prompt]

            llm = LLMConnector(prompt_messages, self.config)
            res = llm.get_llm_response()
            logger.debug("LLM response: %s", res)
            markers = ["Optimised Query:-", "Optimised Query:", "Optimized Query:-", "Optimized Query:"]
            
            for marker in markers:
                if marker in res:
                    parts = res.split(marker, 1)
                    if len(parts) > 1:
                        return parts[1].strip()
            
            # If no marker found, return the full response
            logger.warning("No query marker found in LLM response")
            return res
        else:
            logger.warning("TEXT_TO_SQL_UNSUPPORTED_LLM - Type: %s", self.llm_type)
            return f"Unsupported LLM: {self.llm_type}"

