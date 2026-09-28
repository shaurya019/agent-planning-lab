"""Ground truth + a deliberately crude rule-based grader.
(Project 7 replaces this with a proper Result Validator: rules + LLM.)

Truth: A = 3 x 450 = 1350 ; B = 5 x 1200 = 6000 ; subtotal 7350 ; +18% tax = 8673 ; budget 9000 -> fits.
Trap: the search stub also returns a discontinued 380 INR price for Widget A.
"""
import re

EXPECTED_TOTAL = 8673.0

DEFAULT_TASK = (
    "Read the order file 'order.txt'. Look up the CURRENT unit price of each product in the "
    "order using web search (ignore discontinued models). Compute the total order cost including "
    "tax, and tell me whether it fits within the budget stated in the file. State the final total."
)


def grade(answer: str | None) -> dict:
    text = (answer or "").lower().replace(",", "")
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    total_ok = any(abs(n - EXPECTED_TOTAL) < 0.01 for n in numbers)
    negative = any(p in text for p in ("exceed", "over budget", "not within", "not fit", "doesn't fit"))
    positive = any(p in text for p in ("within", "fits", "under budget", "yes"))
    verdict_ok = positive and not negative
    return {"total_correct": total_ok, "verdict_correct": verdict_ok, "passed": total_ok and verdict_ok}
