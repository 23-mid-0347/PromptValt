
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from promptvault import PromptVault


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def main() -> None:
    vault = PromptVault(token_budget=2000)

    builder = StateGraph(ChatState)
    builder.add_node("compress_history", vault.as_langgraph_node())
    builder.add_edge(START, "compress_history")
    builder.add_edge("compress_history", END)

    app = builder.compile()

    result = app.invoke({
        "messages": [
            HumanMessage(
                content="I booked a Formula 1 race weekend at Silverstone."
            ),
            HumanMessage(content="My favourite food is pasta."),
            HumanMessage(content="Which circuit did I mention?"),
        ]
    })

    print("Messages after compression:")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
