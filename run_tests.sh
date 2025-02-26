#!/usr/bin/env bash

set -e -x

# Install the package (necessary for CLI tests).
# Requirements should already be cached in the docker image.
pip install -e .

# Log installed versions
echo "PIP FREEZE:"
pip freeze

exit_if_error() {
  local exit_code=$1
  shift
  printf 'ERROR: %s\n' "$@" >&2
  exit "$exit_code"
}

download_assets() {
  set -e -x
  mkdir -p axlearn/data/tokenizers/sentencepiece
  mkdir -p axlearn/data/tokenizers/bpe
  curl https://huggingface.co/t5-base/resolve/main/spiece.model -o axlearn/data/tokenizers/sentencepiece/t5-base
  curl https://huggingface.co/FacebookAI/roberta-base/raw/main/merges.txt -o axlearn/data/tokenizers/bpe/roberta-base-merges.txt
  curl https://huggingface.co/FacebookAI/roberta-base/raw/main/vocab.json -o axlearn/data/tokenizers/bpe/roberta-base-vocab.json
}

precommit_checks() {
  set -e -x
  pre-commit install
  pre-commit run --all-files || exit_if_error $? "pre-commit failed."
  # Run pytype separately to utilize all cpus and for better output.
  pytype -j auto . || exit_if_error $? "pytype failed."
}

# Collect all background PIDs explicitly.
TEST_PIDS=()

download_assets

MARKER_FILTER="not (gs_login or tpu or high_cpu or fp64 or for_8_devices)"

while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    --skip-pre-commit)
      echo "Skipping precommit"
      SKIP_PRECOMMIT=true
      shift
      ;;
    --run-tpu-tests)
      echo "Running TPU tests"
      MARKER_FILTER="not (gs_login or high_cpu or fp64)"
      shift
      ;;
    -*)
      echo "Invalid option: $arg" >&2
      exit 1
      ;;
    *)
      args+=("$arg")
      shift
      ;;
  esac
done

echo "Remaining args: $1"
echo "Remaining args: $args"

#if [[ "${1:-x}" = "--skip-pre-commit" ]] ; then
#  SKIP_PRECOMMIT=true
#  shift
#fi
#
#MARKER_FILTER="not (gs_login or tpu or high_cpu or fp64)"
#if [[ "${1:-x}" = "--run-tpu-tests" ]] ; then
#  MARKER_FILTER="not (gs_login or high_cpu or fp64)"
#  shift
#fi

# Skip pre-commit on parallel CI because it is run as a separate job.
if [[ "${SKIP_PRECOMMIT:-false}" = "false" ]] ; then
  precommit_checks &
  TEST_PIDS[$!]=1
fi

UNQUOTED_PYTEST_FILES=$(echo $args |  tr -d "'")
pytest --durations=100 -v -n auto \
  -m "${MARKER_FILTER}" ${UNQUOTED_PYTEST_FILES} \
  --dist worksteal &
TEST_PIDS[$!]=1

JAX_ENABLE_X64=1 pytest --durations=100 -v -n auto -v -m "fp64" --dist worksteal &
TEST_PIDS[$!]=1

XLA_FLAGS="--xla_force_host_platform_device_count=8" pytest --durations=100 -v \
  -n auto -v -m "for_8_devices" --dist worksteal &
TEST_PIDS[$!]=1

# Use Bash 5.1's new wait -p feature to quit immediately if any subprocess fails to make error
# finding a bit easier.
while [ ${#TEST_PIDS[@]} -ne 0 ]; do
  wait -n -p PID ${!TEST_PIDS[@]} || exit_if_error $? "Test failed."
  unset TEST_PIDS[$PID]
done
