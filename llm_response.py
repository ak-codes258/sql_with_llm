import openai
import os
import json
from typing import Dict, Any, Optional

def load_llm_config(config_path: str) -> Dict[str, Any]:
    """Loads LLM configuration JSON."""
    with open(config_path, "r") as f:
        return json.load(f)

def call_llm(
    prompt: str,
    llm_config: Dict[str, Any],
    max_tokens: int = 2048,
    temperature: float = 0.1,
    stop: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Calls LLM (OpenAI's GPT-4.0) with the given prompt.
    Ensures JSON output inside Markdown block.
"""
    api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key in config or OPENAI_API_KEY env variable.")

    model = llm_config.get("model", "gpt-4")
    openai.api_key = api_key

    messages = [
        {
            "role": "system",
            "content": llm_config.get(
                "system_message",
                "You are a skilled BigQuery SQL optimizer. Always answer ONLY using a JSON block inside Markdown fences with all required fields as described."
            ),
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        # Extract text
        text = response.choices[0].message.content

        # Optional: find JSON block between triple backticks
        import re
        match = re.search(r"``````", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                return parsed
            except Exception:
                pass  # fallback to returning raw text

        return {"raw_response": text}

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example usage:
    # prompt.txt has the prompt
    # llm_config.json has llm config as described in docstring

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # You can customize these file names/paths
    PROMPT_PATH = os.path.join(script_dir, 'prompt.txt')
    LLM_CONFIG_PATH = os.path.join(script_dir, 'llm_config.json')

    with open(PROMPT_PATH, "r") as f:
        prompt = f.read().strip()

    llm_config = load_llm_config(LLM_CONFIG_PATH)

    response = call_llm(prompt, llm_config)
    print("\n=== LLM Response ===")
    print(json.dumps(response, indent=2))

    # Optional: Save LLM result to file
    out_path = os.path.join(script_dir, "llm_sql_optimization_response.json")
    with open(out_path, "w") as f:
        json.dump(response, f, indent=2)
    print(f"\nResponse saved to {out_path}")
