# Phase 4 cost/latency report

n claims: 10
mean wall-clock latency/claim (end to end): 25.46s
p95 wall-clock latency/claim (end to end): 47.11s
mean API latency/claim: 21.38s
p95 API latency/claim: 41.54s
mean cost/claim (incl. extraction fallback): $0.0405
total cost: $0.4051
extraction fallback calls: 0 ($0.0000)
prompt cache tokens: 0 written, 0 read (no cache writes: cached prefix is under the model's minimum cacheable length or unmarked)
one-time startup (LoRA load + index build, excluded above): 10.76s
