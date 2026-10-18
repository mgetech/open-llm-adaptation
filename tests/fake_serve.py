"""Run the fake server on its own, for a dry run without a GPU.

It answers every document of every split with its expected output, so the results of a run
against it say nothing about a model. Stop it with Ctrl+C.
"""

import argparse
import time

from adapt.eval.runner import default_splits, load_canaries, load_documents
from fake_model import Model
from fake_server import FakeServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fake OpenAI-compatible server.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    args = parser.parse_args()

    documents = [
        d
        for split in default_splits(args.platform)
        for d in load_documents(args.platform, split, data_round=args.round)
    ]
    server = FakeServer(args.port)
    server.version = "fake"
    server.handler = Model(documents, load_canaries(args.platform, data_round=args.round))
    with server:
        print(f"fake server at {server.url}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
