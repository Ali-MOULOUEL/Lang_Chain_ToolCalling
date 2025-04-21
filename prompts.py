from typing import List, Optional

def code_generation_prompt(query: str) -> str:
    """
    Generates a prompt for a language model to generate Python code that creates Highcharts charts based on user queries.

    Parameters
    ----------
    query : str
        The specific problem or scenario described by the user.
    cells : List[str]
        A list of code from Jupyter Notebook cells, serving as a reference to avoid redundant operations.
    cell_outputs : List[str]
        A list of outputs from Jupyter Notebook cells, serving as a reference to avoid redundant operations.
    data_definitions : Optional[str]
        A dictionary containing additional context, such as explanations of .csv file column names. This is optional.

    Returns
    -------
    str
        A JSON-like string representing the prompt for the language model.
    """
    llm_base_prompt = {
        "role": "World-class Python programming expert.",
        "description": (
            "You are a Python expert specializing in generating executable Python code. ",
            "Your goal is to create Python solutions that align with the user's query and integrate seamlessly with existing code."
        ),
        "task": [
            "Follow these structured steps to generate code:",
            "1. Analyze the inputs:",
            "    - `query`: The user's problem or request.",
            "2. Determine the scope of the user's query:",
            "    - If the query requires functionality not yet covered by `cell_contents`, generate new Python code.",
            "    - If the query relates to data visualization (e.g., Highcharts), generate the JSON structure for the chart.",
            "    - If the query is a general question, generate Python code to produce the requested result.",
            "3. Ensure efficient code generation:",
            "    - Do not duplicate efforts; leverage existing code or outputs whenever possible.",
            "    - For .csv/.xlsx files or SQL queries, reuse previously executed results or modify existing pandas operations/queries as needed.",
            "    - DO NOT RELOAD the data from the source; use the data already defined in the code cells.",
            "4. Return only the solution as a self-contained, fully executable Python cell.",
            "5. Ensure compatibility with Jupyter Notebook cells:",
            "    - Avoid JSON serialization operations within the code.",
            "6. For chart generation queries, produce a JSON object structured for Highchart.",
            "    - Otherwise, ensure the code prints the output appropriately.",
        ],
        "inputs": {
            "query": query,
        },
        "expected_output_format": [
            "Output should be a JSON object with the following format:",
            "{",
            "    \"code\": \"<generated_python_code>\",",
            "    \"markdown\": \"<explanation_in_markdown>\"",
            "}",
            "where:",
            "    - <generated_python_code> is Python code executable as-is.",
            "    - <explanation_in_markdown> contains a Markdown explanation of the code's purpose and functionality."
        ]
    }
    return str(llm_base_prompt)

