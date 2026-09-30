# Phase 4 cost/latency report

n claims: 20
mean wall-clock latency/claim (end to end): 21.08s
p95 wall-clock latency/claim (end to end): 45.81s
mean API latency/claim: 17.36s
p95 API latency/claim: 36.09s
mean cost/claim (incl. extraction fallback): $0.0307
total cost: $0.6144
extraction fallback calls: 0 ($0.0000)
prompt cache tokens: 0 written, 0 read (no cache writes: cached prefix is under the model's minimum cacheable length or unmarked)
one-time startup (LoRA load + index build, excluded above): 10.47s
