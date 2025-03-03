from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


@tool
def get_weather(location: str) -> str:
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny!"
    elif "boston" in location.lower():
        return "It's rainy!"
    else:
        return f"I am not sure what the weather is in {location}"


tools = [get_weather]


tools_by_name = {tool.name: tool for tool in tools}


@task
def call_model(messages: list[BaseMessage]) -> BaseMessage:
    """Call model with a sequence of messages."""
    response = model.bind_tools(tools).invoke(messages)
    return response


@task
def call_tool(tool_call: dict) -> ToolMessage:
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])


checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def agent(
    messages: list[BaseMessage],
    previous: list[BaseMessage] | None,
) -> BaseMessage:
    if previous is not None:
        messages = add_messages(previous, messages)

    llm_response = call_model(messages).result()
    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]

        # Append to message list
        messages = add_messages(messages, [llm_response, *tool_results])

        # Call model again
        llm_response = call_model(messages).result()

    messages = add_messages(messages, llm_response)
    return entrypoint.final(value=llm_response, save=messages)


def main() -> None:
    config = {"configurable": {"thread_id": "1"}}

    user_message = {"role": "user", "content": "What's the weather in san francisco?"}
    print(user_message)

    for step in agent.stream([user_message], config=config):
        for task_name, message in step.items():
            if task_name == "agent":
                continue  # Just print task updates
            print(f"\n{task_name}:")
            message.pretty_print()

    user_message = {"role": "user", "content": "先ほど何の話をしましたっけ？"}
    print(user_message)

    for step in agent.stream([user_message], config=config):
        for task_name, message in step.items():
            if task_name == "agent":
                continue  # Just print task updates
            print(f"\n{task_name}:")
            message.pretty_print()


if __name__ == "__main__":
    main()
