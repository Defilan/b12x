#!/bin/bash
# Serving A/B for local-inference-lab/b12x#457 on the three-Spark ring. For each arm (roce, nccl):
# deploy the InferenceService (manifests differ only in VLLM_ENABLE_ROCE_ALLREDUCE), wait for Ready,
# record the all-reduce selection from the rank-0 startup log, send a temperature-0 sanity request,
# run decode_ab.py (c=1 and c=2, 5 rounds after a warmup), then delete the service.
# The smoke watcher (memory floor) runs alongside; the gb10-memory-guard drops page cache at boot.
set -u
S=${S:-.}            # working directory (watcher script, client, outputs); paths generalized for publication
D=${D:-.}            # directory holding inferenceservice-{roce,nccl}.yaml
E=$S/ab-out
PY=${PY:-python3}
mkdir -p "$E"
exec > >(tee -a "$E/run-ab.log") 2>&1
for arm in roce nccl; do
  echo "=== arm $arm start $(date -u +%FT%TZ)"
  pkill -f smoke-watch3.sh
  kubectl --context shadowstack -n default delete isvc dsv41-b12x-smoke --ignore-not-found --wait=false >/dev/null
  for i in $(seq 1 60); do [ "$(kubectl --context shadowstack -n default get pods --no-headers 2>/dev/null | grep -c dsv41-b12x-smoke-mn)" = 0 ] && break; sleep 5; done
  mkdir -p "$S/smoke-r3"; nohup "$S/smoke-watch3.sh" >/dev/null 2>&1 &
  kubectl --context shadowstack apply -f "$D/inferenceservice-$arm.yaml" >/dev/null
  for i in $(seq 1 120); do
    ph=$(kubectl --context shadowstack -n default get isvc dsv41-b12x-smoke -o jsonpath='{.status.phase}' 2>/dev/null)
    [ "$ph" = Ready ] && break; sleep 15
  done
  echo "phase=$ph after ~$((i*15)) s"
  [ "$ph" = Ready ] || { echo "arm $arm never Ready"; continue; }
  kubectl --context shadowstack -n default logs dsv41-b12x-smoke-mn-0 -c vllm > "$E/$arm-rank0-startup.log" 2>&1
  grep -E "all-reduce backends|RoCEnante|Using RoCEnante" "$E/$arm-rank0-startup.log" | grep -v "NCCL INFO" | head -4 > "$E/$arm-startup.txt"
  cat "$E/$arm-startup.txt"
  kubectl --context shadowstack -n default port-forward svc/dsv41-b12x-smoke 18120:8080 >/dev/null 2>&1 & PF=$!; sleep 4
  curl -s -m 120 localhost:18120/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Write a Python one-liner that reverses a string."}],"max_tokens":300,"temperature":0}' \
    > "$E/$arm-sanity.json"
  echo "sanity: $(jq -r '.choices[0].message.content' "$E/$arm-sanity.json" | head -c 200 | tr '\n' ' ')"
  "$PY" "$S/decode_ab.py" http://localhost:18120 "$arm" "$E/$arm-decode.json"
  kill $PF
  echo "=== arm $arm done $(date -u +%FT%TZ)"
done
echo "=== A/B done; restoring the RoCE smoke service"
pkill -f smoke-watch3.sh
kubectl --context shadowstack -n default delete isvc dsv41-b12x-smoke --ignore-not-found --wait=false >/dev/null
for i in $(seq 1 60); do [ "$(kubectl --context shadowstack -n default get pods --no-headers 2>/dev/null | grep -c dsv41-b12x-smoke-mn)" = 0 ] && break; sleep 5; done
kubectl --context shadowstack apply -f "$D/inferenceservice-roce.yaml" >/dev/null
nohup "$S/smoke-watch3.sh" >/dev/null 2>&1 &
echo "=== restored $(date -u +%FT%TZ)"
