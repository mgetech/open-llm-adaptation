# 0001 Task and output design

Status: accepted

Context: A wrong sum or a wrong VAT amount in a booking is silent and costly, and a language model is unreliable at arithmetic.

Decision: The model answers with one JSON object that a JSON schema constrains. The accounts and VAT rates (19, 7, 0) are enumerated in the schema, and amounts are copied as printed. Code computes net, VAT, the reverse-charge liability and the booking lines, and checks the sum, the asset limit and the chart.

Alternatives: The model writes booking lines with net and VAT itself. Free text that I parse afterwards.

Consequences: The model cannot invent an account, and the arithmetic is code I can test. It can still choose a wrong account or rate, and the evaluation measures exactly that.
