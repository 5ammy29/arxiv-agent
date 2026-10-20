from uuid import UUID

from langchain_core.messages import HumanMessage

from project.graph.workflow import build_workflow
from project.infrastructure.database.checkpointer import get_checkpointer
from project.infrastructure.database.sessions import (
    create_session,
    get_session,
)


def run_query(workflow, query, paper_or_topic, config):
    try:
        input_state = {
            "query": query,
            "messages": [HumanMessage(content=query)],
        }

        if is_arxiv_identifier(paper_or_topic):
            input_state["paper"] = paper_or_topic

        result = workflow.invoke(input_state, config=config)

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    answer = result.get("answer")

    if answer:
        print(f"\n{answer}")
    else:
        print("\nNo answer was generated.")


def interactive_mode(workflow, config):
    print("\narXiv Research Agent")

    while True:
        try:
            paper_or_topic = input(
                "\nEnter an arXiv ID, URL, or topic: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not paper_or_topic:
            print("Please enter an arXiv ID, URL, or topic.")
            continue

        if paper_or_topic.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        run_query(workflow, query, paper_or_topic, config)


def is_arxiv_identifier(value):
    value = value.strip()

    if "/abs/" in value or "/pdf/" in value or value.endswith(".pdf"):
        return True

    parts = value.split("/")

    if len(parts) == 2:
        return True

    return value.replace(".", "").isdigit()


def main():
    user_id = UUID("e18d69e9-a0f5-4817-8a5b-408d0ff7c62b")

    print("arXiv Research Agent")
    print("1. Create a new session")
    print("2. Resume an existing session")

    choice = input("Choose an option (1/2): ").strip()

    if choice == "1":
        session = create_session(
            user_id=user_id,
            title="Research Session",
        )
    elif choice == "2":
        session_id_input = input("Enter session ID: ").strip()

        try:
            session_id = UUID(session_id_input)
        except ValueError:
            print("Invalid session ID.")
            return

        session = get_session(session_id, user_id)

        if session is None:
            print("Session not found.")
            return

        if session["status"] != "active":
            print("This session is not active.")
            return
    else:
        print("Invalid option.")
        return

    config = {
        "configurable": {
            "thread_id": str(session["id"])
        }
    }

    print("Session ID:", session["id"])
    print("Session thread ID:", config["configurable"]["thread_id"])

    with get_checkpointer() as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        interactive_mode(workflow, config)


if __name__ == "__main__":
    main()
