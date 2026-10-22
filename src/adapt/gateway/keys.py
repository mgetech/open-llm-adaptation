"""API keys of the gateway.

A key file holds the SHA-256 of each key and the platforms the key may use, never the key. The real
file is `configs/keys.yaml`, which git ignores; `configs/keys.example.yaml` shows the format.
`python -m adapt.gateway.keys` makes a new key and prints the line to add to the file.
"""

import argparse
import hashlib
import hmac
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from adapt.models import CONFIGS_DIR

KEYS_FILE = CONFIGS_DIR / "keys.yaml"
EXAMPLE_FILE = CONFIGS_DIR / "keys.example.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiKey(_Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    platforms: list[str] = Field(min_length=1)


class KeyFile(_Model):
    keys: list[ApiKey]

    @model_validator(mode="after")
    def _unique(self) -> "KeyFile":
        for field in ("id", "sha256"):
            values = [getattr(key, field) for key in self.keys]
            if len(set(values)) != len(values):
                raise ValueError(f"a key {field} is used twice")
        return self


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_keys(path: Path = KEYS_FILE) -> KeyFile:
    if not path.is_file():
        raise ValueError(f"no key file at {path}; copy {EXAMPLE_FILE.name} and add your keys")
    try:
        return KeyFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as error:
        raise ValueError(f"{path.name} is not valid: {error}") from error


def find_key(keys: KeyFile, presented: str | None) -> ApiKey | None:
    """The key entry whose hash matches the key a request presented, or None."""
    if not presented:
        return None
    digest = hash_key(presented)
    found = None
    for key in keys.keys:
        # Every hash is compared, so the time does not depend on which key matched.
        if hmac.compare_digest(key.sha256, digest):
            found = key
    return found


def new_key() -> str:
    return secrets.token_urlsafe(32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make an API key and print its line for the file.")
    parser.add_argument(
        "--id", required=True, help="a short name for the key, used in the audit log"
    )
    parser.add_argument("--platforms", nargs="+", default=["tax-booking"])
    args = parser.parse_args(argv)
    key = new_key()
    entry = {"id": args.id, "sha256": hash_key(key), "platforms": args.platforms}
    try:
        ApiKey.model_validate(entry)
    except ValidationError as error:
        sys.exit(f"error: {error}")
    print(f"key (shown once): {key}")
    print("add to configs/keys.yaml under keys:")
    print(yaml.safe_dump([entry], sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
