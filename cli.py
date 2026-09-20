from project.graph.workflow import build_workflow

def run_query(workflow, query, paper_or_topic):
    try:
        if is_arxiv_identifier(paper_or_topic):
            result = workflow.invoke({
                "query": query,
                "paper": paper_or_topic,
            })
        else:
            result = workflow.invoke({
                "query": query,
            })
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    answer = result.get("answer")

    if answer:
        print()
        print(answer)
    else:
        print("\nNo answer was generated.")

def interactive_mode(workflow):
    print()
    print("arXiv Research Agent")

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

        run_query(workflow, query, paper_or_topic)

def is_arxiv_identifier(value):
    value = value.strip()

    if "/abs/" in value or "/pdf/" in value:
        return True

    if value.endswith(".pdf"):
        return True

    parts = value.split("/")

    if len(parts) == 2:
        return True

    return value.replace(".", "").isdigit()

def main():
    workflow = build_workflow()
    interactive_mode(workflow)

if __name__ == "__main__":
    main()