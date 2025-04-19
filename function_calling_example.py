# function_calling_example.py

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
import os

# Assurez-vous que votre clé API OpenAI est défini dans les variables d'environnement
# ou remplacez 'your-api-key' par votre clé réelle
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

# Liste des outils disponibles
tools = [add, multiply, subtract]

# Initialisation du modèle de chat
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Liaison des outils au modèle
llm_with_tools = llm.bind_tools(tools)

# Message de l'utilisateur
query = "What is 3 * 12? Also, what is 11 + 49, what is 10 - 5?"
messages = [HumanMessage(content=query)]

# Appel initial au modèle
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
# Exécution des outils appelés par le modèle
for tool_call in ai_msg.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]

    # Sélection de l'outil approprié
    selected_tool = {"add": add, "multiply": multiply, "subtract": subtract}[tool_name.lower()]
    tool_output = selected_tool.invoke(tool_args)

    # Ajout de la réponse de l'outil aux messages
    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

# Appel final au modèle avec les résultats des outils
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
