import warnings
import json
import os
import re
import nltk
import httpx
from fastapi import APIRouter, FastAPI, Request, HTTPException,Depends
from concurrent.futures import TimeoutError
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from langchain_community.document_loaders import JSONLoader
from src.database.postgres_loader import PostgresLoader

from src.utils.config_reader import load_config
from src.utils.nlp_utils import *
from src.services.guardrails_service import *
from src.services.text_to_sql_service import ConvertTextToSqlRequest
from src.database.query_executor import execute
from src.database.postgres_loader import PostgresLoader
from src.services.cost_estimator import Estimate

from src.utils.mock_ldap import verify_token
from src.database.postgres_vector_loader import PostgresVectorLoader
from src.services.ragQueryPipline import RAGPipeline
from src.utils.logger import logger

nltk.download('punkt')
nltk.download('words')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

router = APIRouter(tags=["query"])

warnings.filterwarnings("ignore")

# Custom HTTP client
custom_http_client = httpx.Client(verify=False)

def remove_numbers_and_dots(s: str) -> str:
    return re.sub(r'[\d\.]+', '', s)

@router.get("/health")
async def health():
    logger.info("Health check requested")
    return "Welcome to TruLens"

@router.get("/")
async def index():
    logger.info("Root endpoint accessed")
    return HTMLResponse("<h1>Welcome to TruLens</h1>")

class QueryRequest(BaseModel):
    query: str
    llm_type: str
    market: str
    
#Route for generating SQL query with logic and gaurdrails
@router.post("/generate_query")
async def generate_query(req: QueryRequest, user: dict = Depends(verify_token)):
    username = user.get('username', 'unknown')
    logger.info("GENERATE_QUERY started - User: %s, Market: %s, Query: '%s'",
                username, req.market, req.query[:100] + "..." if len(req.query) > 100 else req.query)

    if os.getenv("TEST_MODE") == "true":
        logger.info("TEST_MODE enabled - returning mock response")
        return JSONResponse(content={"message": "Mocked response in test mode", "user": user}, status_code=200)

    try:
        query = req.query
        llm_type = req.llm_type
        market = req.market

        logger.debug("Loading config for market: %s", market)
        config = load_config(market)

        logger.debug("Starting query validation pipeline")

        # Validation 1: Invalid utterances
        if invalid_utterance_in_prompt(query):
            logger.warning("VALIDATION_FAILED - Invalid BI query detected for user: %s", username)
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["The input doesn't appear to be a business-related query. Please provide a data question or request."], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )

        # Validation 2: English validation
        if validate_english(query):
            logger.warning("VALIDATION_FAILED - English validation failed for user: %s, query: %s", username, query[:50])
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["The query doesn't seem to start with a valid English phrase or has malicious intent. Please revise and try again."], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )

        # Validation 3: Invalid Domain Query
        logger.debug("Validating query intent for analytical purpose")
        validation_res = validate_query_for_invalid_domain_query(query, config)
        if validation_res == 'False':
            logger.warning("Sorry Invalid Domain Query: %s", username)
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["Sorry Invalid Domain Query"], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )


        request = ConvertTextToSqlRequest(query, llm_type, market)
        sql_query = request.convert_text_to_sql_using_llm()
        logger.info("SQL generated successfully for user: %s, SQL: %s", username, sql_query[:200] + "..." if len(sql_query) > 200 else sql_query)

        # SQL Security Validations
        logger.debug("Starting SQL security validations")

        # Check for modification queries
        if check_for_modification_in_query(sql_query=sql_query):
            logger.error("SECURITY_VIOLATION - Modification query detected for user: %s, SQL: %s", username, sql_query)
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["Query blocked: Modifications to the database (e.g., UPDATE, DELETE, DROP, etc.) are not permitted. Please try again with a valid query."], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )

        # Check for SQL injection
        if validate_query_for_sql_injection(sql_query, config) == 'True':
            logger.error("SECURITY_VIOLATION - SQL Injection detected for user: %s, SQL: %s", username, sql_query)
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["SQL Injection detected in the Query. Query execution blocked by guardrails !!"], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )

        ans = {'sql_query_generated': sql_query}
        logger.info("GENERATE_QUERY completed successfully for user: %s", username)
        return JSONResponse(content=ans)

    except Exception as e:
        logger.exception("GENERATE_QUERY_ERROR - Unexpected error for user: %s, Query: %s, Error: %s",
                        username, query[:100], str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@router.post("/optimise_query")
async def optimized_query(req: QueryRequest, user: dict = Depends(verify_token)):
    username = user.get('username', 'unknown')
    logger.info("GENERATE_QUERY started - User: %s, Market: %s, Query: '%s'",
                username, req.market, req.query[:100] + "..." if len(req.query) > 100 else req.query)

    if os.getenv("TEST_MODE") == "true":
        logger.info("TEST_MODE enabled - returning mock response")
        return JSONResponse(content={"message": "Mocked response in test mode", "user": user}, status_code=200)

    try:
        query = req.query
        llm_type = req.llm_type
        market = req.market
        logger.debug("Loading config for market: %s", market)
        config = load_config(market)
        logger.debug("Starting query validation pipeline")
        # English validation
        if validate_english(query):
            logger.warning("VALIDATION_FAILED - English validation failed for user: %s, query: %s", username, query[:50])
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["The query doesn't seem to start with a valid English phrase or has malicious intent. Please revise and try again."], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )
        request = ConvertTextToSqlRequest(query, llm_type, market)
        sql_query = request.optimize_sql_query_using_llm()
        logger.info("SQL generated successfully for user: %s, SQL: %s", username, sql_query[:200] + "..." if len(sql_query) > 200 else sql_query)
        ans = {'sql_query_generated': sql_query}
        logger.info("GENERATE_QUERY completed successfully for user: %s", username)
        return JSONResponse(content=ans)

    except Exception as e:
        logger.exception("GENERATE_QUERY_ERROR - Unexpected error for user: %s, Query: %s, Error: %s",
                        username, query[:100], str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


class QueryPayload(BaseModel):
    query: str
    market: str

@router.post("/execute_query")
def execute_query(payload: QueryPayload, user: dict = Depends(verify_token)):
    username = user.get('username', 'unknown')
    logger.info("EXECUTE_QUERY started - User: %s, Market: %s, Query: %s", 
                username, payload.market, payload.query[:100] + "..." if len(payload.query) > 100 else payload.query)
    
    if os.getenv("TEST_MODE") == "true":
        logger.info("TEST_MODE enabled - returning mock response")
        return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200) 
    
    try:
        logger.debug("Estimating query cost and execution time for user: %s", username)
        estimate = Estimate(payload.query, payload.market)
        cost_result = estimate.estimate_query_cost()

        if cost_result.get("status") == "error":
            logger.error("EXECUTE_QUERY_ESTIMATION_ERROR - User: %s, Error: %s", username, cost_result.get("error_message"))
            raise Exception(cost_result["error_message"])
            
        cost_usd = cost_result.get('estimated_cost_usd', 0)
        bytes_processed = cost_result.get('bytes_processed', 0)
        
        # Heuristic timeout calculation (1GB/s processing speed)
        estimated_seconds = bytes_processed / (1024 ** 3) if bytes_processed > 0 else 0
        
        logger.info("EXECUTE_QUERY_ESTIMATES - User: %s, Cost: $%.6f, Estimated time: %.1fs", 
                   username, cost_usd, estimated_seconds)
        
        # Check cost limit ($10.00)
        if cost_usd > 10.0:
            logger.warning("EXECUTE_QUERY_BLOCKED - Cost limit exceeded - User: %s, Cost: $%.6f > $10.00", 
                          username, cost_usd)
            return JSONResponse(
                status_code=400,
                content={
                    'result': [], 
                    'metadata': "", 
                    'sql_query': "", 
                    'textual_summary': [f"Query execution blocked: Cost ${cost_usd:.2f} exceeds limit of $10.00"], 
                    'followup_prompts': [], 
                    "x-axis": "", 
                    "typeOFgraph": ""
                }
            )
        
        # Check time limit (30 seconds)
        if estimated_seconds > 30:
            logger.warning("EXECUTE_QUERY_BLOCKED - Time limit exceeded - User: %s, Estimated time: %.1fs > 30s", 
                          username, estimated_seconds)
            return JSONResponse(
                status_code=400,
                content={
                    'result': [], 
                    'metadata': "", 
                    'sql_query': "", 
                    'textual_summary': [f"Query execution blocked: Estimated execution time {estimated_seconds:.1f}s exceeds limit of 30s"], 
                    'followup_prompts': [], 
                    "x-axis": "", 
                    "typeOFgraph": ""
                }
            )
        
        # Execute the query with a 30-second timeout
        timeout = 30
        
        logger.debug("Executing query with timeout: %ss for user: %s", timeout, username)
        
        if is_dml_query(payload.query):
            logger.error("SECURITY_VIOLATION - DML query detected for user: %s, SQL: %s", username, payload.query)
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["DML query detected in the Query. Query execution blocked by guardrails !!"], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )
        
        if payload.query == '':
            logger.error("SECURITY_VIOLATION - Query is None for user: %s", username)
            return JSONResponse(
                status_code=400,
                content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': ["No query was detected. Kindly input a valid search query to proceed."], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
            )
        
        result = execute(payload.query, payload.market, timeout=timeout)
        
        result_count = len(result) if isinstance(result, list) else "N/A"
        logger.info("EXECUTE_QUERY completed - User: %s, Rows returned: %s, Result: %s", username, result_count, result)
        
        return JSONResponse(
            status_code=200,
            content={'result': result, 'metadata': "", 'sql_query': "", 'textual_summary': [], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
        )
    except Exception as e:
        logger.exception("EXECUTE_QUERY_ERROR - User: %s, Market: %s, Query: %s, Error: %s", 
                        username, payload.market, payload.query[:50], str(e))
        return JSONResponse(
            status_code=500,
            content={'result': [], 'metadata': "", 'sql_query': "", 'textual_summary': [f"BigQuery Error: {str(e)}"], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
        )

class MetadataRequest(BaseModel):
    metadata_type: str  # "table" or "column"
    market: str

@router.post("/postgres_loader/")
async def postgres_loader(request: MetadataRequest):
    logger.info("POSTGRES_LOADER started - Type: %s, Market: %s", request.metadata_type, request.market)

    if os.getenv("TEST_MODE") == "true":
        logger.info("TEST_MODE enabled - returning mock response")
        return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200)

    try:
        logger.debug("Loading config for market: %s", request.market)
        config = load_config(request.market)
        tables_path = config.get('Database', 'tables_path')
        columns_path = config.get('Database', 'columns_path')

        project_id = config.get('Database', 'bigquery_project')
        dataset_id = config.get('Database', 'bigquery_dataset')
        logger.debug("Config loaded - Project: %s, Dataset: %s", project_id, dataset_id)

        postgres_loader = PostgresLoader(request.market)

        if request.metadata_type == "table":
            logger.info("Processing table metadata from: %s", tables_path)
            postgres_loader.create_table_config()

            data = JSONLoader(
                file_path=tables_path,
                jq_schema='.',
                text_content=False,
                json_lines=False
            ).load()

            record_count = 0
            for doc in data:
                record = json.loads(doc.page_content)
                record['data_source_id'] = project_id
                record['data_namespace'] = dataset_id
                postgres_loader.insert_table_metadata(record, "table_config")
                record_count += 1

            logger.info("POSTGRES_LOADER completed - %d table records inserted for market: %s",
                       record_count, request.market)
            return {"status": "success", "message": "table_config metadata inserted successfully"}

        elif request.metadata_type == "column":
            logger.info("Processing column metadata from: %s", columns_path)
            postgres_loader.create_column_config()

            data = JSONLoader(
                file_path=columns_path,
                jq_schema='.[]',
                text_content=False,
                json_lines=False
            ).load()

            record_count = 0
            for doc in data:
                record = json.loads(doc.page_content)
                record['data_source_id'] = project_id
                record['data_namespace'] = dataset_id
                postgres_loader.insert_column_metadata(record, "column_config")
                record_count += 1

            logger.info("POSTGRES_LOADER completed - %d column records inserted for market: %s",
                       record_count, request.market)
            return {"status": "success", "message": "column_config metadata inserted successfully"}

        else:
            logger.error("POSTGRES_LOADER_ERROR - Invalid metadata_type: %s", request.metadata_type)
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata_type. Must be 'table' or 'column'."
            )

    except FileNotFoundError as e:
        logger.error("POSTGRES_LOADER_ERROR - File not found: %s", str(e))
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error("POSTGRES_LOADER_ERROR - JSON decode error: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.exception("POSTGRES_LOADER_ERROR - Market: %s, Type: %s, Error: %s",
                        request.market, request.metadata_type, str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/estimate")
def estimate_cost(payload: QueryPayload, user: dict = Depends(verify_token)):
    username = user.get('username', 'unknown')
    logger.info("ESTIMATE_COST started - User: %s, Market: %s, Query: %s",
                username, payload.market, payload.query[:100] + "..." if len(payload.query) > 100 else payload.query)

    if os.getenv("TEST_MODE") == "true":
        logger.info("TEST_MODE enabled - returning mock response")
        return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200)

    try:
        logger.debug("Creating cost estimator for user: %s", username)
        estimate = Estimate(payload.query, payload.market)
        result = estimate.estimate_query_cost()

        if result.get("status") == "error":
            logger.error("ESTIMATE_COST_ERROR - User: %s, Error: %s", username, result.get("error_message"))
            raise HTTPException(status_code=400, detail=result["error_message"])

        # Log cost details for monitoring
        cost_usd = result.get('estimated_cost_usd', 0)
        bytes_processed = result.get('bytes_processed', 0)
        logger.info("ESTIMATE_COST completed - User: %s, Cost: $%.6f, Bytes: %d",
                   username, cost_usd, bytes_processed)

        return JSONResponse(
            status_code=200,
            content={'result': result, 'metadata': "", 'sql_query': "", 'textual_summary': [], 'followup_prompts': [], "x-axis": "", "typeOFgraph": ""}
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error("Error estimating cost")
        logger.exception("ESTIMATE_COST_ERROR - User: %s, Market: %s, Query: %s, Error: %s", 
                        username, payload.market, payload.query[:50], str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")

class RAGQueryRequest(BaseModel):
    market: str
    question: str

# This endpoint queries the RAG pipeline for a given question and market and returns top relevant contexts.
@router.post("/rag_query")
def query_rag(request: RAGQueryRequest):
    logger.info("RAG_QUERY started - Question: %s, Market: %s", request.question, request.market)

    if os.getenv("TEST_MODE") == "true":
        logger.info("TEST MODE Enabled - returning mock response")
        return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200)
    try:
        logger.debug("Creating RAG_Pipeline instance")
        query = RAGPipeline(request.market)
        result = query.query(request.question)
        logger.debug("Returning result: %s", result)
        return JSONResponse(content={"result": result}, status_code=200)

    except Exception as e:
        logger.error("RAG_QUERY_ERROR found: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pg_vector_loader")
def pg_vector_loader(request: MetadataRequest):
    if os.getenv("TEST_MODE") == "true":
        return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200)
    
    config = load_config(request.market)
    tables_path = config.get('Database', 'tables_path')
    columns_path = config.get('Database', 'columns_path')
    project_id = config.get('Database', 'bigquery_project')
    dataset_id = config.get('Database', 'bigquery_dataset')

    postgres_vector_loader = PostgresVectorLoader(request.market)

    if request.metadata_type == "table":
        # postgres_vector_loader.create_table_context()
        data = JSONLoader(
            file_path=tables_path,
            jq_schema='.',
            text_content=False,
            json_lines=False
        ).load()

        for doc in data:
            record = json.loads(doc.page_content)
            record['data_source_id'] = project_id
            record['data_namespace'] = dataset_id
            postgres_vector_loader.insert_table_context(record, "table_context")

        return JSONResponse(
            content={"status": "success", "message": "table_context metadata inserted successfully"},
            status_code=200
        )

    elif request.metadata_type == "column":
        # postgres_vector_loader.create_column_context()
        try:
            data = JSONLoader(
                file_path=columns_path,
                jq_schema='.[]',
                text_content=False,
                json_lines=False
            ).load()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading column metadata: {str(e)}"
            )
    
        for doc in data:
            record = json.loads(doc.page_content)
            record['data_source_id'] = project_id
            record['data_namespace'] = dataset_id
            postgres_vector_loader.insert_column_context(record, "column_context")
            
        return JSONResponse(
            content={"status": "success", "message": "column_context metadata inserted successfully"},
            status_code=200
        )
        
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid metadata_type. Must be 'table' or 'column'."
        )

    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"Unexpected error: {str(e)}"
    #     )

@router.post("/pg_vector_loader")
def pg_vector_loader(request: MetadataRequest):
    if os.getenv("TEST_MODE") == "true":
        return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200)
    
    config = load_config(request.market)
    db_tables_path = config.get('Database', 'tables_path')
    db_column_path = config.get('Database', 'columns_path')
    project_id = config.get('Database', 'bigquery_project')
    dataset_id = config.get('Database', 'bigquery_dataset')

    postgres_vector_loader = PostgresVectorLoader(request.market)

    if request.metadata_type == "table":
        data = JSONLoader(
            file_path=db_tables_path,
            jq_schema='.',
            text_content=False,
            json_lines=False
        ).load()

        for doc in data:
            records = json.loads(doc.page_content)
            # Check if records is a list or a single record
            if isinstance(records, list):
                for record in records:
                    record['data_source_id'] = project_id
                    record['data_namespace'] = dataset_id
                    postgres_vector_loader.insert_table_context(record, "table_context")
            else:
                # Handle case where it's a single record
                records['data_source_id'] = project_id
                records['data_namespace'] = dataset_id
                postgres_vector_loader.insert_table_context(records, "table_context")

        return JSONResponse(
            content={"status": "success", "message": "table_context metadata inserted successfully"},
            status_code=200
        )

    elif request.metadata_type == "column":
        try:
            data = JSONLoader(
                file_path=db_column_path,
                jq_schema='.[]',
                text_content=False,
                json_lines=False
            ).load()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading column metadata: {str(e)}"
            )
    
        for doc in data:
            record = json.loads(doc.page_content)
            record['data_source_id'] = project_id
            record['data_namespace'] = dataset_id
            postgres_vector_loader.insert_column_context(record, "column_context")
            
        return JSONResponse(
            content={"status": "success", "message": "column_context metadata inserted successfully"},
            status_code=200
        )
        
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid metadata_type. Must be 'table' or 'column'."
        )

# @router.post("/postgres_loader/")
# async def postgres_loader(request: MetadataRequest):
#     logger.info("LOAD_METADATA: %s", request.metadata_type)
#     if os.getenv("TEST_MODE") == "true":
#         return JSONResponse(content={"message": "Mocked response in test mode"}, status_code=200)
    
class PromptQueryLogRequest(BaseModel):
    prompt: str
    query: str
    edited_query: str | None = None
    market: str
@router.post("/log-prompt-query")
async def log_prompt_query(request: PromptQueryLogRequest, user: dict = Depends(verify_token)):

    logger.info("LOG_PROMPT_QUERY started - User: %s", user.get('username'))
    try:
        # Initialize PostgresLoader
        loader = PostgresLoader(market = request.market)

        loader.create_prompt_query_log_table()

        # Call your existing function
        loader.add_prompt_query_log(
            user_id=user.get('username'),
            prompt=request.prompt,
            query=request.query,
            edited_query=request.edited_query
        )

        logger.info("LOG_PROMPT_QUERY completed successfully for user: %s", user.get('username'))
        return JSONResponse(
            content={"status": "success", "message": "Prompt and query logged successfully"},
            status_code=200
        )
    except Exception as e:
        logger.exception("LOG_PROMPT_QUERY_ERROR - Error logging prompt/query for user: %s, Error: %s",
                         user.get('username'), str(e))
        raise HTTPException(status_code=500, detail=f"Failed to log prompt/query: {str(e)}")


#commentouts converted to fucntion
def process_metadata(request):
    """
    Processes metadata of type 'table' or 'column' based on the request.
    This function is defined but not called in the current implementation.
    """
    config = load_config(request.market)
    market_config_dir = os.path.join("config", request.market)
    DB_TABLES_DIR = os.path.join(market_config_dir, "tables")
    DB_COLUMNS_DIR = os.path.join(market_config_dir, "columns")

    project_id = config.get('Database', 'bigquery_project')
    dataset_id = config.get('Database', 'bigquery_dataset')

    postgres_loader = PostgresLoader(request.market)

    if request.metadata_type == "table":
        postgres_loader.create_table_config()
        for filename in os.listdir(DB_TABLES_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(DB_TABLES_DIR, filename)
                data = JSONLoader(
                    file_path=filepath,
                    jq_schema='.',
                    text_content=False,
                    json_lines=False
                ).load()

                for doc in data:
                    record = json.loads(doc.page_content)
                    record['data_source_id'] = project_id
                    record['data_namespace'] = dataset_id
                    postgres_loader.insert_table_metadata(record, "table_config")

        logger.info("Loaded %s metadata", request.metadata_type)
        return {"status": "success", "message": "table_config metadata inserted successfully"}

    elif request.metadata_type == "column":
        postgres_loader.create_column_config()
        for filename in os.listdir(DB_COLUMNS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(DB_COLUMNS_DIR, filename)
                data = JSONLoader(
                    file_path=filepath,
                    jq_schema='.[]',
                    text_content=False,
                    json_lines=False
                ).load()

                for doc in data:
                    record = json.loads(doc.page_content)
                    record['data_source_id'] = project_id
                    record['data_namespace'] = dataset_id
                    postgres_loader.insert_column_metadata(record, "column_config")

        logger.info("Loaded %s metadata", request.metadata_type)
        return {"status": "success", "message": "column_config metadata inserted successfully"}

    else:
        logger.error("Metadata load failed: Invalid metadata_type")
        raise HTTPException(
            status_code=400,
            detail="Invalid metadata_type. Must be 'table' or 'column'."
        )

class LatestPrompts(BaseModel):
    market: str

@router.post("/latest_prompts")
async def get_latest_prompts(request: LatestPrompts, user: dict = Depends(verify_token)):
    logger.info("LATEST_PROMPTS started - User: %s, Market: %s", user.get('username', 'unknown'), request.market)

    try:
        loader = PostgresLoader(request.market)
        prompts = loader.fetch_latest_prompts()
        print(prompts)
        return JSONResponse(content={"prompts": prompts}, status_code=200)
    except Exception as e:
        logger.error("LATEST_PROMPTS_ERROR - %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest prompts: {str(e)}")
