# Compliance notes

An engineering design for a tax firm that would run this, not legal advice. The project itself uses synthetic data only. For each point I say what the control is and where it lives.

- **Professional secrecy.** The model is open-weight and self-hosted. vLLM runs on a Docker network with no route out, and the only published port is the gateway's. The client calls only the vLLM endpoint it is given, so no document goes to a hosted model API, and serving is offline.
  Where: `deploy/compose.yaml` (`internal: true`), `configs/serve.yaml` (`HF_HUB_OFFLINE`), `src/adapt/client.py`.

- **Personal data in the text.** IBANs, email addresses, phone numbers and tax IDs are replaced by placeholders (`<IBAN>`, `<EMAIL>`, `<PHONE>`, `<TAX_ID>`) in the training data, in the pipeline before the model is called, and before the audit hash. Names and addresses are not replaced yet, so a document with a person's name still reaches the model. Replacing them is a next step.
  Where: `src/adapt/data/pseudonymise.py`, `src/adapt/data/split.py`, `src/adapt/pipeline.py`, `src/adapt/gateway/app.py`.

- **Audit log and erasure.** The log has one JSON line per request with the SHA-256 of the pseudonymised input, the adapter and base revision, the outcome and the latency, never the text. A test checks that no part of the document is in the log, and a request fails if its line cannot be written. If a person's data had got into the training data, erasure means dropping those rows, training the adapter again and running the gate again. Rehearsing that erasure end to end is a next step.
  Where: `src/adapt/gateway/audit.py`, `tests/test_gateway_audit.py`, `make train`, `make gate`.

- **Separation between platforms.** An API key lists the platforms it may use, and the key file holds only the hash of the key. A key without the platform gets 403, the same answer whether the platform exists or not, and each platform has its own adapter.
  Where: `src/adapt/gateway/keys.py`, `src/adapt/gateway/app.py`.

- **EU AI Act.** As I understand the guidance on general-purpose models, a company that modifies one becomes its provider only if the compute it uses for the change is more than a third of the compute used to train the original. The adapter trained for 3.6 GPU hours on two T4 GPUs, which the run manifest records. Working out the share needs a source for the training compute of the base model, and assessing whether a firm's use would count as high-risk is another open point. Both are next steps. A flagged document is meant for a person to check, and the system never books one by itself. The review workflow is outside this repository.
  Where: `results/round3/runs/tax-booking-qwen3-4b-r3-0.json`, `src/adapt/pipeline.py`.

- **Supply chain.** A model needs an allowed licence (`apache-2.0` or `mit`) and a pinned revision. `make models-check` checks the licence against that list and, unless it runs offline, compares the pinned revision, licence and parameter count with the Hub. The adapter is checked by hash when the launcher starts and when the gateway reports ready. Telemetry is off in serving. The adapter weights are licensed for research and demonstration only (CC BY-NC 4.0), and checking whether the terms of the providers of the models that rewrote part of the training text limit that use is a next step. The model card says so.
  Where: `configs/models.yaml`, `configs/serve.yaml` (`VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`), `src/adapt/serve.py`, [`docs/model-card.md`](model-card.md).

- **Demo and production.** The Kaggle runs and the Azure run used synthetic data only. A deployment for a firm would run the same containers on a GPU VM in an EU region of its own subscription ([`deploy/README.md`](../deploy/README.md)).
