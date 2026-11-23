"""Compare the agent's answers with values calculated directly from SQLite."""

import csv
import re
import sys
import unicodedata
from contextlib import closing
from decimal import Decimal
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
# Make the project's agent importable when running python eval/run_eval.py.
sys.path.insert(0, str(EVAL_DIR.parent))

from agent import _get_agent, _read_only_connection, ask


def plain_text(text):
    # Accept both Jokić and Jokic when comparing player names.
    return unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()


def matches_answer(expected, answer):
    # shortcut: a matching value or phrase can miss context, inspect the notes for close calls.
    if expected is None:
        # NULL means the requested season has no rows or the statistic has no column.
        unavailable = (
            "outside", "only covers", "earliest", "not available", "not recorded",
            "not included", "not covered", "not track", "no data", "cannot", "can't",
            "does not contain", "doesn't contain", "does not include", "doesn't include",
            "doesn't have", "don't have", "isn't available",
        )
        reply = answer.casefold().replace("’", "'")
        return any(phrase in reply for phrase in unavailable)
    if isinstance(expected, (int, float)):
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        return any(Decimal(number.replace(",", "")) == Decimal(str(expected)) for number in numbers)
    return plain_text(expected) in plain_text(answer)


def run_eval():
    _get_agent()  # Check the database and API key before grading any questions.
    results = []
    with (EVAL_DIR / "questions.csv").open(encoding="utf-8", newline="") as file:
        with closing(_read_only_connection()) as db:
            for case in csv.DictReader(file):
                question = case["question"]
                try:
                    row = db.execute(case["reference_sql"]).fetchone()
                    expected = row[0] if row else None
                    answer, _ = ask(question, [])
                    correct = matches_answer(expected, answer)
                    expected_label = expected if expected is not None else "an unavailable-data response"
                    notes = f"Expected {expected_label}. Agent answered {answer}"
                except Exception as error:
                    correct, notes = False, f"Error running this case: {error}"
                grade = "yes" if correct else "no"
                results.append((question, grade, notes))
                print(f"[{grade}] {question}", flush=True)

    if not results:
        raise ValueError("There are no questions in eval/questions.csv.")
    with (EVAL_DIR / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "correct", "notes"])
        writer.writerows(results)
    passed = sum(grade == "yes" for _, grade, _ in results)
    print(f"Accuracy {passed}/{len(results)} ({passed / len(results):.1%}). Saved eval/results.csv.")


if __name__ == "__main__":
    try:
        run_eval()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from None
