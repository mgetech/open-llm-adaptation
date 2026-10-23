# 0008 Exact-match metrics, no LLM judge

Status: accepted

Context: The answers are structured, and I want the same score every time I run the evaluation.

Decision: A document is correct when its items match the label (account, VAT rate and the amount within one cent), the reverse-charge flag matches, and the review decision is right. Errors on automated documents are weighted: a wrong VAT rate, a wrong reverse-charge flag or an account of another class (asset, private, expense) is critical, another group of the same class is major, another account in the same group is minor. Intervals come from a bootstrap and the comparison with few-shot from an exact McNemar test.

Alternatives: A language model as judge. Fuzzy text matching.

Consequences: Scores are deterministic and need no other model. A different but valid account counts as an error, because the rules have one answer. The critical-error limit of 0.5% needs about 600 automated test documents to show, and my test sets have a few hundred.
