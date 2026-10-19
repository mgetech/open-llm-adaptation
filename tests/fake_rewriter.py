"""A rewriter that writes a valid new text for a task, from the facts only."""

from decimal import Decimal


def money(amount: float) -> str:
    return f"{Decimal(str(amount)):,.2f}"


def rewrite(task: dict, variant: int = 0) -> str:
    """A text that keeps every fact of `task` and is not close to its draft."""
    lines = [task["supplier"].upper(), f"Style {variant}: {task['style'][:30]}"]
    day = task["date"].split("-")
    lines.append(f"Date {day[2]}.{day[1]}.{day[0]}")
    for item in task["items"]:
        lines.append(f"* {item['description']} -> {money(item['amount'])}")
    for row in task["vat"]:
        lines.append(f"VAT {row['rate']}% included: {money(row['vat'])}")
    lines += task["keep_verbatim"]
    suffix = "" if task["currency"] == "EUR" else f" {task['currency']}"
    lines.append(f"Total: {money(task['total'])}{suffix}")
    payment = task.get("payment")
    if payment:
        lines += [f"Cash {money(payment['cash'])}", f"Change {money(payment['change'])}"]
    return "\n".join(lines)
