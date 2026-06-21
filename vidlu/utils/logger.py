from datetime import datetime


def _format(record: tuple[str, str]) -> str:
    timestamp, text = record
    return f"[{timestamp}] {text}"


class Logger:
    """Append-only, serializable transcript of an experiment's console output.

    A single instance is shared by reference across all training callbacks. It is
    the single source of truth for the human-readable run transcript: every
    message is recorded so it can be (a) printed live and (b) replayed to stdout
    when a run is resumed from a checkpoint.

    Display is delegated to ``emit`` (default ``print``). Callers that show
    ``tqdm`` progress bars should pass ``emit=tqdm.write`` so messages don't
    corrupt active bars. The serialized state is just the records — plain data,
    no functions — so checkpoints stay portable.
    """

    def __init__(self, emit=print):
        self.records: list[tuple[str, str]] = []  # (timestamp, text)
        self.emit = emit

    def log(self, text: str) -> None:
        record = (datetime.now().strftime('%H:%M:%S'), text)
        self.records.append(record)
        self.emit(_format(record))

    def print_all(self) -> None:
        for record in self.records:
            self.emit(_format(record))

    def as_text(self) -> str:
        """Human-readable transcript (used for the per-checkpoint ``log.txt``)."""
        return "\n".join(_format(r) for r in self.records)

    def state_dict(self) -> dict:
        return {"records": list(self.records)}

    def load_state_dict(self, state) -> None:
        self.records = list(state["records"])
