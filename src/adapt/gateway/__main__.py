"""Run the gateway with `python -m adapt.gateway` (`make serve-local` starts it with vLLM).

It reads `configs/serve.yaml` and the key file. vLLM is reached at `--vllm-url`, which defaults to
the port of the serve file on this machine, or to the `VLLM_URL` environment variable.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from adapt.gateway.app import create_app
from adapt.gateway.keys import KEYS_FILE, load_keys
from adapt.serve import SERVE_FILE, load_serve_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gateway.")
    parser.add_argument("--serve-file", type=Path, default=SERVE_FILE)
    parser.add_argument("--keys-file", type=Path, default=KEYS_FILE)
    parser.add_argument("--vllm-url", help="default: $VLLM_URL, else vLLM on this machine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    try:
        serve = load_serve_config(args.serve_file)
        keys = load_keys(args.keys_file)
        vllm_url = (
            args.vllm_url or os.environ.get("VLLM_URL") or f"http://127.0.0.1:{serve.port}/v1"
        )
        app = create_app(serve, keys, vllm_url)
    except (ValueError, OSError) as error:
        sys.exit(f"error: {error}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
