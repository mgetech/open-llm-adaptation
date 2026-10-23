# 0006 Fail closed

Status: accepted

Context: A wrong automated booking costs more than a document that a person has to look at.

Decision: One function, `Pipeline.process`, serves the eval runner and the gateway. A failed document check, a model error, invalid JSON, a failed booking check or rule, a low or unreadable confidence, and any exception all end in `needs_review` with a reason. A document with an order in its text goes to review before the model is called. If the gateway cannot write the audit line, the request fails.

Alternatives: Return the best answer on a failure. Retry until the model answers.

Consequences: The cost is the automation rate: 0.76 on `dev` and 0.65 on `dev_real` for the shipped configuration. A model outage shows up as reviews, never as bookings. The review itself is a person's job and outside this repository.
