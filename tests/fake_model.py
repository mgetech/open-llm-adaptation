"""A fake model for the fake server: it answers each document with its expected output."""

import copy
import json

from fake_server import Reply, completion_body

NOT_A_LEAK = "Unknown Supplier GmbH"


class Model:
    """Callable for `FakeServer.handler`. `answer(document)` gives a dict, a string or a Reply."""

    def __init__(self, documents, canaries=(), answer=None, leak=False, **completion):
        self.documents = {d.text: d for d in documents}
        self.canaries = {c.text: c for c in canaries}
        self.answer = answer or (lambda document: copy.deepcopy(dict(document.target)))
        self.leak = leak
        self.completion = completion

    def __call__(self, body):
        text = body["messages"][1]["content"]
        if text in self.canaries:
            output = {**self._any_target(), "supplier": NOT_A_LEAK}
            if self.leak:
                output["supplier"] = self.canaries[text].name
        else:
            output = self.answer(self.documents[text])
        if isinstance(output, Reply):
            return output
        content = output if isinstance(output, str) else json.dumps(output)
        return Reply(200, completion_body(content, **self.completion))

    def _any_target(self):
        return copy.deepcopy(dict(next(iter(self.documents.values())).target))


def wrong_account_with_low_confidence(every=3):
    """Every third automatable document gets a wrong account. Those answers have a low logprob."""

    def answer(document):
        output = copy.deepcopy(dict(document.target))
        number = int(document.id.split("-")[1])
        items = output["items"]
        wrongable = not document.expected_review and items and items[0]["account"] != "4980"
        if wrongable and number % every == 0:
            items[0]["account"] = "4980"
        return output

    return answer


def unsure_about_4980(token):
    return -1.5 if token == "4980" else -0.01
