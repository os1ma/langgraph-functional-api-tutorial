import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

load_dotenv()


@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    print("Writing essay...")
    time.sleep(1)  # This is a placeholder for a long-running task.
    return f"An essay about topic: {topic}"


@entrypoint()
def sub_workflow(topic: str) -> dict:
    """A simple workflow that writes an essay and asks for a review."""
    essay = write_essay("cat").result()

    is_approved = interrupt(
        {
            # Any json-serializable payload provided to interrupt as argument.
            # It will be surfaced on the client side as an Interrupt when streaming data
            # from the workflow.
            "essay": essay,  # The essay we want reviewed.
            # We can add any additional information that we need.
            # For example, introduce a key called "action" with some instructions.
            "action": "Please approve/reject the essay",
        },
    )

    return {
        "essay": essay,  # The essay that was generated
        "is_approved": is_approved,  # Response from HIL
    }


@entrypoint(checkpointer=MemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that writes an essay and asks for a review."""
    sub_workflow_result = sub_workflow.invoke({"topic": topic})
    return sub_workflow_result


def main() -> None:
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    for item in workflow.stream("cat", config):
        print(item)

    human_review = input("Human review: ")
    for item in workflow.stream(Command(resume=human_review), config):
        print(item)


if __name__ == "__main__":
    main()
