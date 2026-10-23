# 0005 Labels computed from rules

Status: accepted

Context: Labels written by a language model would contain the kind of mistakes the system is meant to find.

Decision: In the generated splits (train, dev, test_iid, test_ood, edge) every label comes from the rules in `platforms/tax-booking/rules.yaml`, through a seeded generator. In round 3 language models rewrote the wording of part of the documents, and a check rejects any rewrite in which an amount, a rate or an item name changed. The gold sets are evaluation documents generated separately from the training generator. Their labels are not computed by code, so `make validate-gold` is the check on them, and they are small.

Alternatives: Labels from a language model. Real documents labelled by an expert, which I do not have.

Consequences: The model can only be as right as my rules, which are an illustrative subset of SKR03 and not tax advice. Generated text is more regular than real text, which is why I rewrote part of it and why I do not read the validation results as a test.
