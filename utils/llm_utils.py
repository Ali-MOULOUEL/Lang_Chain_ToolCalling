import os
import json
import logging
from langchain_openai import ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

from typing import Optional
from config import OPENAI_MODELS, ANTHROPIC_MODELS, AZURE_MODELS, EMBEDDINGS_AZURE_MODELS

def get_llm_model(llm_model_name: str, model_temperature: float) -> AzureChatOpenAI:
    """
    Get the language model instance based on the model name and temperature.

    Parameters
    ----------
    llm_model_name : str
        The name of the language model.
    model_temperature : float
        The temperature setting for the language model.

    Returns
    -------
    AzureChatOpenAI
        An instance of the AzureChatOpenAI language model.
    """
    if llm_model_name in OPENAI_MODELS:
        return ChatOpenAI(temperature=model_temperature, model=llm_model_name)
    elif llm_model_name in ANTHROPIC_MODELS:
        return ChatAnthropic(temperature=model_temperature, model_name=llm_model_name)
    elif llm_model_name in AZURE_MODELS:
        AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
        azure_deployment = "gpt4o-default" if llm_model_name == "gpt-4o-azure" else "gpt-4"
        return AzureChatOpenAI(
            temperature=model_temperature,
            azure_deployment=azure_deployment,
            api_version=AZURE_API_VERSION,
            timeout=None,
            cache=True,
            model_kwargs={"seed": 590}
        )
    elif llm_model_name in EMBEDDINGS_AZURE_MODELS:
        AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
        azure_deployment = llm_model_name
        return AzureOpenAIEmbeddings(    
            azure_deployment=azure_deployment,
            openai_api_version=AZURE_API_VERSION,
        )
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model_name}")
    
def generate_llm_output(llm_model: AzureChatOpenAI, llm_prompt: str) -> Optional[str]:
    """
    Generate output from the language model based on the provided prompt.

    Parameters
    ----------
    llm_model : AzureChatOpenAI
        The language model instance.
    llm_prompt : str
        The prompt to generate the output.

    Returns
    -------
    str
        The generated output from the language model.
    """
    json_llm = llm_model.bind(response_format={"type": "json_object"})
    messages = [("system", llm_prompt)]
    try:
        ai_msg = json_llm.invoke(messages)
        logging.info(f"LLM output: {ai_msg.content}")
        return None if ai_msg.content is None else ai_msg.content
    except Exception as e:
        logging.error(f"Error generating LLM output: {e}")
        return None

def extract_response(response: str) -> (Optional[str], Optional[str]):
    """
    Extract the code and markdown from the language model response.

    Parameters
    ----------
    response : str
        The response from the language model.

    Returns
    -------
    tuple of str
        A tuple containing the extracted code and markdown.
    """
    try:
        if response is None:
            return None, None
        parsed_json = json.loads(response)
        values = list(parsed_json.values())
        return values[0], values[1]
    except Exception as e:
        logging.error(f"Error extracting response: {e}")
        return None, None
