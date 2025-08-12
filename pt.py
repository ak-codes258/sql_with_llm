def get_prompt_for_analytical_intent(query: str):
    prompt = f'''Following query is the analytical query or not: 
                {query}
                Just print True or False and nothing else'''
    return prompt
 
def get_prompt_for_generating_sql_query(tables: str, schema: str, content: str):
    prompt = f'''
    ### Table Details
    The following are the table descriptions and join keys, separated by '|':
    
    TABLES: {tables}

    ### Column Schema
    Below are the matched table-column mappings, also separated by '|':

    MATCHED_SCHEMA: {schema}

    ### User Request
    Write a BigQuery SQL query that fulfills this requirement:

    {content}

    ### Rules and Guidelines (Follow all strictly)
    1. Use **only** column names listed in the MATCHED_SCHEMA section.
    2. Associate each column strictly with its corresponding table from MATCHED_SCHEMA.
    3. If the user prompt matches any known prompt examples, return the corresponding sample SQL query.
    4. Use `SAFE_DIVIDE`, `SAFE_CAST` where applicable.
    5. Use descriptive aliases for all column names in the SELECT clause, especially when:
       - The column name is long or complex
       - The column is a calculated field
       - Multiple columns from different tables have similar names
       Example: `SELECT u.user_id AS user_identifier, o.order_date AS purchase_date`
    6. Always use the full column reference (table.column or table_alias.column) in GROUP BY and ORDER BY clauses
    7. Use `UNNEST` when dealing with STRUCT or ARRAY<STRUCT> fields.
    8. Ensure clean formatting and consistent indentation.
    8. Use `GROUP BY` if the query includes aggregation functions (e.g., `COUNT`).
    9. Do **not** include GCP project IDs or dataset references in the output.
    10. Optimize the query for performance:
        - Use efficient joins and indexed columns.
        - Avoid redundant calculations or unnecessary casting.
    11. Remove `CAST()` where not needed.
    12. Use BigQuery functions based on data type:
        - For timestamps, use `TIMESTAMP_SUB` or `DATE_ADD` with `INTERVAL ... DAY`.
        - Convert months to days using a 30-day month approximation.
    13. Give the priority to return the values of the column provided in context in 'sample_values' section or else return the most similar with sample values given.
    14. If "columns_metadata" has "mode": "REPEATED", like "ids", use the provided sql_patterns. For repeated fields, never use direct dot notation. Always use UNNEST for repeated field access
    
    ### Output Format Rules (STRICT)
    - The output must be exactly in this format:
      Optimised Query:-SELECT * FROM table
    - Do NOT include any markdown formatting (```sql or ```)
    - Do NOT include any additional text, explanations, or formatting
    - The output must:
      * Start with "Optimised Query:-"
      * Follow immediately with the SQL query
      * Contain no other text before or after
    - Example of correct output:
      Optimised Query:-SELECT COUNT(*) FROM users WHERE status = 'active'
    - Example of INCORRECT output:
      ```sql
      SELECT COUNT(*) FROM users WHERE status = 'active'
      ```
      Or any output that includes markdown formatting or additional text.

    Respond with ONLY the SQL query in the exact format specified above.
    '''
    return prompt

def get_prompt_for_sql_optimization(tables: str, schema: str, content: str):
    prompt = f'''
        Table Details
        The following are the table descriptions and join keys, separated by '|':

        TABLES: {tables}

        Column Schema
        Below are the matched table-column mappings, also separated by '|':

        MATCHED_SCHEMA: {schema}

        User Request
        Optimization Request for existed Query:
        {content}

    #### Rules and Guidelines (Follow all strictly)

       ** CRITICAL MISSION: ACHIEVE MINIMUM 30% PERFORMANCE OPTIMIZATION **
        Transform the query to achieve AT LEAST 30% better performance while maintaining IDENTICAL output results.

        STRICT METADATA COMPLIANCE RULES (MANDATORY)
        - ** Don't Change the Table Names from the Original Query, If required optimize the Query by using Joins or any window functions **
        - ** ONLY use table names provided in TABLES section above **
        - ** ONLY use column names provided in MATCHED_SCHEMA section above **
        - ** NEVER assume or invent column names not explicitly listed **
        - ** NEVER assume table structures beyond what's provided **
        - ** If a column is not in MATCHED_SCHEMA, DO NOT use it in the query **
        - ** If a table is not in TABLES, DO NOT reference it **

        METADATA-DRIVEN OPTIMIZATION STRATEGIES

       ####1. COLUMN-SPECIFIC OPTIMIZATIONS
              ** BEFORE applying any optimization, VERIFY VERY CAREFULLY: **
              - Must column exists in MATCHED_SCHEMA With data type
              - Table exists in TABLES
              - Data type is appropriate for the operation

             ** FOR EACH COLUMN IN MATCHED_SCHEMA: **
              - If temporal column (created_at, updated_at, date, timestamp): ADD mandatory date filter
              - If it possible make partitioning on temporal column
              - Use With clause for temporal column Self-Join Optimization and String operations
              - If text/string column: ADD IS NOT NULL and TRIM() != '' filters
              - If ID column: ADD IS NOT NULL filter

         ####2. TABLE-AWARE JOIN OPTIMIZATION
               ** ONLY JOIN TABLES that exist in TABLES metadata: **
                - Use exact table names from TABLES section
                - Apply join conditions ONLY on columns confirmed in MATCHED_SCHEMA
                - Never assume foreign key relationships not specified in TABLES


        ####3. METADATA VALIDATION REQUIREMENTS TIGHTLY MATCHED WITH SCHEMA TABLES AND COLUMNS 100%
                  ** BEFORE writing query, validate: **
                  1. Every table name must exist in TABLES section check twice if it is matched with schema
                  2. Every column must exist in MATCHED_SCHEMA section
                  3. No assumptions about table structures or relationships
                  4. All filters and joins must strictly use provided metadata
                  6. QUERY STRUCTURE AND FORMAT

  '''
    return prompt

def get_prompt_verify_sql_injection(sql_query: str):
    prompt = f"""
        You are a cybersecurity expert specializing in SQL injection detection.  
        Analyze the given SQL query and determine if it **has SQL injection intent**.  

        **Rules for detection:**  
        - Query contains patterns like `OR 1=1`, `UNION SELECT`, `--`, `DROP TABLE`, etc.  
        - Use of **string concatenation** that modifies query logic dynamically.  
        - **Stacked queries** (`;` multiple statements in one query).  
        - **Unusual use of quotes or comments** that might escape intended query structure.  

        **Return only `True` or `False` (without any explanation):**  
        - **`True`** → If SQL injection risk is detected.  
        - **`False`** → If no SQL injection risk is detected.  

        **Analyze this SQL query and return only True or False:**
        {sql_query}
    """
    return prompt



def get_prompt_verify_invalid_domain_query(content: str):
    prompt = f"""
Role:
You are a domain-aware reasoning engine analyzing whether a natural language query belongs to a financial transaction and case resolution system. Your job is to **semantically** evaluate relevance, even when domain terms are not used explicitly.

Instructions:
Use the following four analysis categories to determine if the question is related to the domain. You may infer relationships based on synonyms, user intent, or implied logic.

1. **TABLE CONTEXT (Implied or Direct)**:
   - Does the query refer to or imply operations on transactional or case-related tables?
     Examples: transactions, events, queues, updates, resolution
   - Even if table names like `event_store` aren't used, check for logical references to system processes.

2. **DOMAIN COLUMNS & FLOW**:
   - Does it refer to data typically captured in finance or case systems? e.g.:
     - Users, payments, fraud, queue, resolution, timestamps
   - Does it imply relationships like transaction → case → state change?

3. **INTENT DETECTION**:
   - Does the question imply investigation, monitoring, or resolution? For example:
     - Fraud detection, payment flow, case updates, trade history
   - Use synonyms: "trade" may imply "transaction", "Gaurav" may be a userId

4. **DOMAIN ALIGNMENT**:
   - Is the question aligned with core purposes of the system?
     - Tracking financial events
     - Workflow processing
     - Risk/fraud management
     - User or case lifecycle

Scoring:
Return **True** if **at least 2 of the 4 categories** are met and the intent aligns with the domain  
Return **False** if there is no alignment in **any** of the above

Input Question:
"{content}"

--- Reasoning Process ---
1. Table Context Check:
[Your reasoning here]

2. Column/Data Flow Analysis:
[Your reasoning here]

3. Intent Detection:
[Your reasoning here]

4. Domain Alignment:
[Your reasoning here]


Recheck the query and return the result.

Final Output:
Only return **True** or **False**
"""
    return prompt
