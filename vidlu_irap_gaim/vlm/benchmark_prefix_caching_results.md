# Prefix Caching Benchmark Results

## Run: Long output (~1500 chars target)

Command:
```bash
python -m vidlu_irap_gaim.vlm.benchmark_prefix_caching --constant-len 11000 --target-output-chars 1500 --num-requests 5
```
Result:
```
  Total time: 7.23 s
  Per request: 1.45 s
  Output lengths: [1192, 537, 537, 537, 809]

--- Summary ---
  Without cache: 16.64 s
  With cache:    7.23 s
  Speedup:       2.30x
```


## Run: Short output (~325 chars)

Command:
```bash
python -m vidlu_irap_gaim.vlm.benchmark_prefix_caching --constant-len 11000 --target-output-chars 250 --num-requests 5
```
Result:
```
  Total time: 2.40 s
  Per request: 0.48 s
  Output lengths: [310, 300, 300, 300, 304]

--- Summary ---
  Without cache: 13.66 s
  With cache:    2.40 s
  Speedup:       5.69x
```

