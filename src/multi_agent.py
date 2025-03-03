"""
TODO: travel_advisorにHIL入れられるのか
"""

import random
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

load_dotenv()


@tool
def get_travel_recommendations():
    """Get recommendation for travel destinations"""
    return random.choice(["aruba", "turks and caicos"])


@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """Get hotel recommendations for a given destination."""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)"
            "Bucuti & Tara Beach Resort (Eagle Beach)",
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    }[location]


@tool(return_direct=True)
def transfer_to_hotel_advisor():
    """Ask hotel advisor agent for help."""
    return "Successfully transferred to hotel advisor"


@tool(return_direct=True)
def transfer_to_travel_advisor():
    """Ask travel advisor agent for help."""
    return "Successfully transferred to travel advisor"


from langchain_core.messages import AIMessage

# from langchain_anthropic import ChatAnthropic
# model = ChatAnthropic(model="claude-3-5-sonnet-latest")
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")

# Define travel advisor ReAct agent
travel_advisor_tools = [
    get_travel_recommendations,
    transfer_to_hotel_advisor,
]
travel_advisor = create_react_agent(
    model,
    travel_advisor_tools,
    state_modifier=(
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)


@task
def call_travel_advisor(messages):
    # You can also add additional logic like changing the input to the agent / output from the agent, etc.
    # NOTE: we're invoking the ReAct agent with the full history of messages in the state
    response = travel_advisor.invoke({"messages": messages})
    return response["messages"]


# Define hotel advisor ReAct agent
hotel_advisor_tools = [get_hotel_recommendations, transfer_to_travel_advisor]
hotel_advisor = create_react_agent(
    model,
    hotel_advisor_tools,
    state_modifier=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
        "You MUST include human-readable response before transferring to another agent."
    ),
)


@task
def call_hotel_advisor(messages):
    response = hotel_advisor.invoke({"messages": messages})
    return response["messages"]


@entrypoint()
def workflow(messages: list[BaseMessage]) -> list[BaseMessage]:
    messages = add_messages([], messages)

    call_active_agent = call_travel_advisor
    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)
        ai_msg = next(m for m in reversed(agent_messages) if isinstance(m, AIMessage))
        if not ai_msg.tool_calls:
            break

        tool_call = ai_msg.tool_calls[-1]
        if tool_call["name"] == "transfer_to_travel_advisor":
            call_active_agent = call_travel_advisor
        elif tool_call["name"] == "transfer_to_hotel_advisor":
            call_active_agent = call_hotel_advisor
        else:
            raise ValueError(f"Expected transfer tool, got '{tool_call['name']}'")

    return messages


from langchain_core.messages import convert_to_messages


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")


for chunk in workflow.stream(
    [
        {
            "role": "user",
            "content": "i wanna go somewhere warm in the caribbean. pick one destination and give me hotel recommendations",
        },
    ],
    subgraphs=True,
):
    pretty_print_messages(chunk)
