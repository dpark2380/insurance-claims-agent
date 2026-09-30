# Phase 4 cost/latency report

n claims: 20
mean wall-clock latency/claim (end to end): 56.42s
p95 wall-clock latency/claim (end to end): 598.51s
mean API latency/claim: 50.65s
p95 API latency/claim: 589.03s
mean cost/claim (incl. extraction fallback): $0.0383
total cost: $0.7669
extraction fallback calls: 1 ($0.0027)
prompt cache tokens: 0 written, 0 read (no cache writes: cached prefix is under the model's minimum cacheable length or unmarked)
one-time startup (LoRA load + index build, excluded above): 11.30s
