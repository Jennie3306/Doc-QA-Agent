"""Terminal interface to the agent. Uses the same graph as the API.

Run with:  python cli.py
"""

import warnings

from agent_graph import build_agent, initial_state
from config import settings

warnings.filterwarnings("ignore")


def run_agent() -> None:
    agent = build_agent()
    chat_history: list[dict] = []

    print("\n" + "=" * 55)
    print("  NVIDIA Document Agent - LangGraph Edition")
    print("  Powered by Nemotron + ChromaDB + LangGraph")
    print("=" * 55)
    print("  Commands: 'quit' | 'clear' | 'history'")
    print("=" * 55 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye! Great work today.")
            break

        if user_input.lower() == "clear":
            chat_history = []
            print("Memory cleared!\n")
            continue

        if user_input.lower() == "history":
            if not chat_history:
                print("No history yet.\n")
            else:
                print(f"\n{len(chat_history) // 2} exchanges in memory:")
                for msg in chat_history:
                    role = "You" if msg["role"] == "user" else "Agent"
                    print(f"  {role}: {msg['content'][:80]}...")
                print()
            continue

        if not user_input:
            continue

        # State is built by initial_state() rather than by hand, so adding a
        # field to AgentState in Phase 1 does not silently break the CLI while
        # the API keeps working.
        print()
        final_state = agent.invoke(initial_state(user_input, chat_history))
        answer = final_state["answer"]
        decision = final_state["decision"]

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        chat_history = chat_history[-settings.max_history_messages :]

        print(f"Agent [{decision.upper()}]: {answer}\n")


if __name__ == "__main__":
    run_agent()
