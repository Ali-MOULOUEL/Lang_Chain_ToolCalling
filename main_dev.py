from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from tools import add, multiply, subtract, code_generation
from utils.llm_utils import get_llm_model

# Load environment variables from .env file
load_dotenv()

# Initialization of the chat model
llm = get_llm_model("gpt-3.5-turbo-1106", 0.01)

# List of available tools
# tools = [add, multiply, subtract]
tools = [add, multiply, subtract, code_generation]
tool_node = ToolNode(tools)

# Binding tools to the model
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Example with a single tool call
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    for chunk in app.stream(
        {"messages": [("human", query)]}, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()