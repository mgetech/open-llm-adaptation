# Running on an Azure VM

How I run the stack of `deploy/compose.yaml` on one NVIDIA T4 in Azure, check that it answers, and check that it gives the same results as on Kaggle. The VM is a `Standard_NC4as_T4_v3` (4 vCPUs and one T4 with 16 GB, the same GPU as Kaggle) in Germany West Central, with Ubuntu 22.04 and a 128 GB disk. Check the size on Azure's size page and the quota of your subscription before you start.

Status: I ran these steps once on a VM, and the files in `results/azure/` record that run. I created the VM in the portal with the same size, image and disk instead of with the commands of step 1. The one step that failed was the comparison of step 5, because the evaluation hashed the download notes in the adapter folder; that is fixed in the code.

The VM is billed while it is allocated. Stopping the machine from inside does not stop the billing: deallocate it, and delete the resource group when you are done (step 6).

## What you need

- The Azure CLI, logged in (`az login`), and a quota of at least 4 vCPUs of the NCASv3_T4 family in the region.
- A Hugging Face token that can read the adapter repository. The repository is gated, so the account must have accepted its terms.
- An SSH key. `az vm create --generate-ssh-keys` makes one if you have none.

## 1. Create the VM

```
az group create --name booking-rg --location germanywestcentral
az vm create --resource-group booking-rg --name booking-t4 --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 --os-disk-size-gb 128 --admin-username azureuser --generate-ssh-keys
```

Limit port 22 in the network security group to your own address. Do not open port 8080: I test the gateway from the VM itself. Then `ssh azureuser@<public ip>`.

## 2. Driver and containers

```
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

After the reboot, `nvidia-smi` must show a Tesla T4. Then Docker and the NVIDIA container toolkit (NVIDIA's installation guide has the current repository lines; these are the ones I used):

```
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit make git
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out and in again so the group applies. `docker run --rm --gpus all ubuntu nvidia-smi` must show the T4.

## 3. The project and the models

```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/mgetech/open-llm-adaptation.git && cd open-llm-adaptation
git checkout <tag or commit>
make setup
export HF_TOKEN=<your token>
make fetch-weights WEIGHTS_DIR=weights
make fetch-adapters
```

The host needs the project for the downloads, the key tool and the checks. It does not need vLLM: the container has it. `make fetch-adapters` checks the hash of the adapter against `configs/serve.yaml`.

## 4. Start the stack and check it

```
make serve-check PREPARE=1
chmod a+rwx logs
make compose-env
make compose-up
docker compose --env-file deploy/vllm.env -f deploy/compose.yaml logs -f vllm
```

`make serve-check PREPARE=1` writes `configs/keys.yaml` with the two keys of the test, so it comes first: if the file is missing, Docker creates a folder in its place. It also creates `logs/`. The gateway runs as user 10001 and writes its audit log there, so run `chmod a+rwx logs` before `make compose-up`, or every request fails with a 500. The first start of vLLM takes several minutes. When the log says the application started:

```
make serve-check NO_VLLM=1 OUT=results/azure/smoke-test.json
```

vLLM is on a network without a route out, so the check cannot reach it directly, and `NO_VLLM=1` skips the two direct checks. `/readyz` still checks that vLLM serves the base model and the adapter, and `docker compose ... exec vllm vllm --version` shows the version.

To send a document yourself, open a tunnel to port 8080 (`ssh -N -L 8080:localhost:8080 azureuser@<public ip>`), set the variable `key` of `deploy/gateway.postman_collection.json` to the plain text key in `logs/serve-check-keys.json`, and send one of its requests.

## 5. Evaluate dev and dev_real, and compare with Kaggle

The evaluation runs inside the compose network, because vLLM is not reachable from the host. The container gets the data and the evaluation sets read-only, and writes to `results/azure/`:

```
mkdir -p results/azure
docker compose --env-file deploy/vllm.env -f deploy/compose.yaml run --rm --no-deps \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/data:/app/data:ro" -v "$PWD/platforms:/app/platforms:ro" \
  -v "$PWD/results:/app/results:ro" -v "$PWD/results/azure:/app/results/azure" \
  gateway python -m adapt.eval.runner --system qwen3-4b-sft-r3-0-azure --model tax-booking \
  --split dev --endpoints http://vllm:8000/v1 --round 3 --threshold 0.9817172787547637 \
  --adapter-path adapters/tax-booking/qwen3-4b-r3-0 \
  --model-revision 1cfa9a7208912126459214e8b04321603b3df60c --out-dir results/azure/eval
```

Run it again with `--split dev_real`. Then compare each split with the Kaggle result of the same adapter:

```
make compare FIRST=results/round3/rescored/tax-booking/qwen3-4b-sft-r3-0/dev.json SECOND=results/azure/eval/tax-booking/qwen3-4b-sft-r3-0-azure/dev.json
make compare FIRST=results/round3/rescored/tax-booking/qwen3-4b-sft-r3-0/dev_real.json SECOND=results/azure/eval/tax-booking/qwen3-4b-sft-r3-0-azure/dev_real.json
```

The comparison passes when both runs used the same base model revision, adapter, prompt and threshold, they hold the same documents, and every metric of the VM is inside the confidence interval of the Kaggle run. It also counts the documents with an identical answer. A document that a pipeline rule sends to review before the model is called (an order in the text, for example) has an empty answer on the VM, and the Kaggle file holds the answer the model gave then, so a few answers differ for that reason.

I use dev and dev_real because the test_ood split and the gold files back the claim in `results/final.lock.json`: running them again needs the override that changes the lock.

## 6. Stop

```
docker compose --env-file deploy/vllm.env -f deploy/compose.yaml down
az vm deallocate --resource-group booking-rg --name booking-t4
az group delete --name booking-rg --yes
```

Copy `results/azure/` to your machine before you delete the group. The first command stops the containers, the second stops the billing for the VM and keeps the disk, and the last one deletes everything.
