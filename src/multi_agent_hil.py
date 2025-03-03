from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

import random
from typing import Literal


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


import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

model = ChatAnthropic(model="claude-3-5-sonnet-latest")

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


checkpointer = MemorySaver()


def string_to_uuid(input_string):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, input_string))


@entrypoint(checkpointer=checkpointer)
def multi_turn_graph(messages, previous):
    previous = previous or []
    messages = add_messages(previous, messages)
    call_active_agent = call_travel_advisor
    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)
        # Find the last AI message
        # If one of the handoff tools is called, the last message returned
        # by the agent will be a ToolMessage because we set them to have
        # "return_direct=True". This means that the last AIMessage will
        # have tool calls.
        # Otherwise, the last returned message will be an AIMessage with
        # no tool calls, which means we are ready for new input.
        ai_msg = next(m for m in reversed(agent_messages) if isinstance(m, AIMessage))
        if not ai_msg.tool_calls:
            user_input = interrupt(value="Ready for user input.")
            # Add user input as a human message
            # NOTE: we generate unique ID for the human message based on its content
            # it's important, since on subsequent invocations previous user input (interrupt) values
            # will be looked up again and we will attempt to add them again here
            # `add_messages` deduplicates messages based on the ID, ensuring correct message history
            human_message = {
                "role": "user",
                "content": user_input,
                "id": string_to_uuid(user_input),
            }
            messages = add_messages(messages, [human_message])
            continue

        tool_call = ai_msg.tool_calls[-1]
        if tool_call["name"] == "transfer_to_hotel_advisor":
            call_active_agent = call_hotel_advisor
        elif tool_call["name"] == "transfer_to_travel_advisor":
            call_active_agent = call_travel_advisor
        else:
            raise ValueError(f"Expected transfer tool, got '{tool_call['name']}'")

    return entrypoint.final(value=agent_messages[-1], save=messages)


thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

inputs = [
    # 1st round of conversation,
    {
        "role": "user",
        "content": "i wanna go somewhere warm in the caribbean",
        "id": str(uuid.uuid4()),
    },
    # Since we're using `interrupt`, we'll need to resume using the Command primitive.
    # 2nd round of conversation,
    Command(
        resume="could you recommend a nice hotel in one of the areas and tell me which area it is.",
    ),
    # 3rd round of conversation,
    Command(
        resume="i like the first one. could you recommend something to do near the hotel?",
    ),
]

for idx, user_input in enumerate(inputs):
    print()
    print(f"--- Conversation Turn {idx + 1} ---")
    print()
    print(f"User: {user_input}")
    print()
    for update in multi_turn_graph.stream(
        user_input,
        config=thread_config,
        stream_mode="updates",
    ):
        for node_id, value in update.items():
            if isinstance(value, list) and value:
                last_message = value[-1]
                if isinstance(last_message, dict) or last_message.type != "ai":
                    continue
                print(f"{node_id}: {last_message.content}")
