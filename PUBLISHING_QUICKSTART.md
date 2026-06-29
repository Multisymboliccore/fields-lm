# Public release quickstart

This repository is assembled on the validated H100 machine so the exact canonical Field source can be inserted and verified by SHA-256 before publication.

## 1. Assemble on the H100

```bash
cd /home/ubuntu/fields-lm-public-release-v2
PY=/home/ubuntu/field_infer_env/bin/python
$PY -m pip install -e ".[dev,hf]"
PYTHON="$PY" bash scripts/assemble_release_on_h100.sh
```

## 2. Validate and seal

```bash
cd /home/ubuntu/fields-lm-github-release
PY=/home/ubuntu/field_infer_env/bin/python
$PY -m pip install -e ".[dev,hf]"
PYTHON="$PY" bash scripts/release_preflight.sh
$PY scripts/smoke_test_h100.py 2>&1 | tee H100_SMOKE_TEST.txt
PYTHON="$PY" bash scripts/seal_release.sh
```

The final archive is written to:

```text
/home/ubuntu/fields-lm-github-release.tar.gz
```

## 3. Publish to GitHub

Clone `Multisymboliccore/fields-lm`, extract the sealed archive into the clone, inspect `git status`, commit, and push to `main`.

## 4. Publish the checkpoint to Hugging Face

After the GitHub runtime is public and the checkpoint round-trip report passes, generate the HF folder with `scripts/prepare_hf_release.sh` and upload it to `Multisymboliccore/fields-300m-pg19`.
