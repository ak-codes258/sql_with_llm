# prompt.py

import os
import json
from typing import Any, Dict, Optional

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def generate_llm_prompt(sql_query: str, table_context: Any, query_plan_info: Any, model: str = "gpt-4.0") -> str:
    table_context_str = json.dumps(table_context, indent=2)
    query_plan_str = json.dumps(query_plan_info, indent=2)

    prompt = f"""
You are an expert Google BigQuery SQL optimizer working with model {model}.  
Given a complex SQL query along with its schema context and detailed execution plan analysis, your tasks are:
- Identify and explain all key bottlenecks in the query (data scan size, partitioning, clustering, full table scans, column usage, execution plan issues, etc.)
- Rewrite the SQL for optimal efficiency and cost, using BigQuery best practices (partition filtering, clustering, column pruning, avoiding SELECT *, etc.)
- Suggest specific schema changes (partitioning/clustering fields, indices, table alterations)
- Estimate cost and performance improvement, with justification for each recommendation.

Provide your output as a JSON block inside Markdown fences with the following fields:

{{
"bottlenecks": ["..."],
"optimized_sql": "Improved SQL goes here",
"schema_recommendations": ["..."],
"estimated_cost_savings": "...",
"justification": "..."
}}


## Table and Column Context
{table_context_str}

## Query Execution Plan Analysis
{query_plan_str}

Only suggest improvements backed by the provided schema and plan context.
"""
    return prompt.strip()

def build_prompt_and_save(
    query_path: str,
    table_context_path: str,
    query_plan_path: str,
    out_txt_path: Optional[str] = "prompt.txt",
    model: str = "gpt-4-0125-preview"
) -> str:
    # Load inputs
    query_data = load_json(query_path)
    sql_query = query_data['query']['sql']
    table_context = load_json(table_context_path)
    query_plan_info = load_json(query_plan_path)

    prompt = generate_llm_prompt(
        sql_query=sql_query,
        table_context=table_context,
        query_plan_info=query_plan_info,
        model=model
    )

    # Save to prompt.txt
    if out_txt_path:
        with open(out_txt_path, "w") as f:
            f.write(prompt)
    return prompt

# If run as script, build prompt with defaults from files in the current directory
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt = build_prompt_and_save(
        os.path.join(script_dir, 'query.json'),
        os.path.join(script_dir, 'table_context.json'),
        os.path.join(script_dir, 'query_plan_analysis.json'),
        os.path.join(script_dir, 'prompt.txt')
    )
    print("Prompt written to prompt.txt")
