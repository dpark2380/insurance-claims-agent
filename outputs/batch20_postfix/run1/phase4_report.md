# Phase 4 cost/latency report

n claims: 20
mean wall-clock latency/claim (end to end): 26.97s
p95 wall-clock latency/claim (end to end): 47.90s
mean API latency/claim: 21.73s
p95 API latency/claim: 42.92s
mean cost/claim (incl. extraction fallback): $0.0387
total cost: $0.7734
extraction fallback calls: 1 ($0.0027)
prompt cache tokens: 0 written, 0 read (no cache writes: cached prefix is under the model's minimum cacheable length or unmarked)
one-time startup (LoRA load + index build, excluded above): 11.88s
