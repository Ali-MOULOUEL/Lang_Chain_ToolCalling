from langchain_core.tools import tool

from utils.llm_utils import get_llm_model, generate_llm_output, extract_response
from prompts import code_generation_prompt


# Définition des outils avec le décorateur @tool
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b

@tool
def code_generation(query: str) -> tuple[str, str]:
    """Generate Python code according to the query."""
    llm_model_name = "gpt-3.5-turbo-1106"
    model_temperature = 0.01
    llm = get_llm_model(llm_model_name, model_temperature)
    prompt = code_generation_prompt(query)
    response = generate_llm_output(llm, prompt)
    if response is None:
        return "Error: No response from the model.", ""
    code, markdown = extract_response(response)
    return code, markdown