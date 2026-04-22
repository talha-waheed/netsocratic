class TerminalInteractor:
    """
    Shared terminal I/O used by both the Clarification Agent and the Selection Agent.
    Replace with a different subclass (e.g. WebInteractor) to change the interaction channel.
    """

    # ── output ────────────────────────────────────────────────────────────────

    def display(self, message: str) -> None:
        print(message)

    def display_section(self, title: str, body: str) -> None:
        width = 60
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
        print(body)

    def display_banner(self, text: str) -> None:
        width = 60
        print(f"\n{'─' * width}")
        print(f"  {text}")
        print(f"{'─' * width}")

    # ── input ─────────────────────────────────────────────────────────────────

    def ask(self, prompt: str) -> str:
        """Prompt the user for a single free-text answer."""
        return input(f"\n{prompt}\n> ").strip()

    def ask_questions(self, questions: list[str]) -> list[str]:
        """
        Display a numbered list of questions and collect one answer per question.
        Returns answers in the same order as the questions.
        """
        answers: list[str] = []
        for i, question in enumerate(questions, start=1):
            print(f"\n[Q{i}] {question}")
            answer = input("    Your answer: ").strip()
            answers.append(answer)
        return answers

    def confirm(self, message: str) -> bool:
        """Ask a yes/no question. Returns True for 'y' / 'yes'."""
        resp = input(f"\n{message} [y/n]: ").strip().lower()
        return resp in ("y", "yes")
