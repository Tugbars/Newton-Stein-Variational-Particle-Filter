# SVPF Performance Optimization

## Summary

Two targeted optimizations reduced SVPF latency by **15% (mean)** and **55% (P99)**, improving compute efficiency from 55% to 83%.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Mean latency | 175.4 μs | 148.3 μs | **-15%** |
| P99 latency | 427.8 μs | 192.2 μs | **-55%** |
| Throughput (mean) | 5,700/s | 6,700/s | +18% |
| Throughput (P99) | 2,340/s | 5,200/s | **+122%** |
| Compute efficiency | 55% | 83% | +28pp |

Test configuration: N=512 particles, Stein=5, RTX 5080

---

## The Problem

Profiling revealed that **45% of step time was host-device communication overhead**, not actual computation:

```
Before optimization:
  Sync overhead:    16.4 μs (9%)
  Memcpy overhead:  62.4 μs (36%)
  Actual kernels:   96.6 μs (55%)  ← Only 55% useful work!
```

The main culprits:

1. **Four separate synchronous D2H memcpy calls** - each one stalls the pipeline
2. **Two cudaStreamSynchronize calls** - one before graph launch, one after

---

## Fix 1: Batch D2H Transfers with Pinned Memory

### Before (4 sync points)

```c
// Each cudaMemcpy is SYNCHRONOUS - blocks until complete
if (h_loglik_out) cudaMemcpy(h_loglik_out, opt->d_loglik_single, ...);
cudaMemcpy(&vol_local, opt->d_vol_single, ...);
cudaMemcpy(&h_mean_local, opt->d_h_mean_prev, ...);
cudaMemcpy(&bandwidth_local, opt->d_bandwidth, ...);
```

Each `cudaMemcpy` to pageable host memory:
1. Allocates staging buffer
2. Copies GPU → staging
3. Copies staging → host
4. **Synchronizes stream**

Four transfers = four sync points = ~33 μs wasted.

### After (1 sync point)

```c
// Pinned memory allocated once at init
cudaMallocHost(&opt->h_results_pinned, 4 * sizeof(float));

// Async copies - no intermediate sync!
float* results = opt->h_results_pinned;
cudaMemcpyAsync(&results[0], opt->d_loglik_single, ..., stream);
cudaMemcpyAsync(&results[1], opt->d_vol_single, ..., stream);
cudaMemcpyAsync(&results[2], opt->d_h_mean_prev, ..., stream);
cudaMemcpyAsync(&results[3], opt->d_bandwidth, ..., stream);
// Single sync at end
cudaStreamSynchronize(stream);
```

Pinned memory enables true async DMA:
- Direct GPU → host transfer (no staging)
- All four copies queue without blocking
- Single sync waits for all

**Savings: ~20 μs**

---

## Fix 2: Remove Redundant Synchronization

### Before (2 syncs)

```c
cudaStreamSynchronize(stream);           // SYNC #1 - wait for H2D uploads
cudaGraphLaunch(graph_exec, stream);
cudaStreamSynchronize(stream);           // SYNC #2 - wait for graph

cudaMemcpyAsync(...);  // D2H copies
cudaStreamSynchronize(stream);           // SYNC #3 - wait for D2H
```

### After (1 sync)

```c
// No sync needed - stream ordering guarantees H2D completes before graph
cudaGraphLaunch(graph_exec, stream);

// No sync needed - stream ordering guarantees graph completes before D2H
cudaMemcpyAsync(...);
cudaMemcpyAsync(...);
cudaMemcpyAsync(...);
cudaMemcpyAsync(...);

cudaStreamSynchronize(stream);           // SINGLE sync for everything
```

CUDA stream semantics guarantee ordering. Operations on the same stream execute in order. Explicit syncs between them are unnecessary.

**Savings: ~16 μs**

---

## Results

### Overhead Breakdown

| Component | Before | After |
|-----------|--------|-------|
| Sync overhead | 16.4 μs (9%) | 5.7 μs (4%) |
| Memcpy overhead | 62.4 μs (36%) | 18.9 μs (13%) |
| **Actual computation** | 96.6 μs (55%) | 123.7 μs (83%) |

### Latency Distribution

The P99 improvement is particularly significant for HFT:

```
Before: P50=142μs, P99=428μs (3.0x ratio)
After:  P50=149μs, P99=192μs (1.3x ratio)
```

Tail latency is now much more predictable.

---

## Code Changes

### svpf_optimized_graph.cu

```c
// 1. Allocate pinned memory at init
cudaMallocHost(&opt->h_results_pinned, 4 * sizeof(float));

// 2. Single-sync pipeline
cudaGraphLaunch(opt->graph_exec, opt->graph_stream);

float* results = opt->h_results_pinned;
cudaMemcpyAsync(&results[0], opt->d_loglik_single, sizeof(float), 
                cudaMemcpyDeviceToHost, opt->graph_stream);
cudaMemcpyAsync(&results[1], opt->d_vol_single, sizeof(float), 
                cudaMemcpyDeviceToHost, opt->graph_stream);
cudaMemcpyAsync(&results[2], opt->d_h_mean_prev, sizeof(float), 
                cudaMemcpyDeviceToHost, opt->graph_stream);
cudaMemcpyAsync(&results[3], opt->d_bandwidth, sizeof(float), 
                cudaMemcpyDeviceToHost, opt->graph_stream);

cudaStreamSynchronize(opt->graph_stream);  // Single sync
```

### svpf.cuh

```c
typedef struct {
    // ... existing fields ...
    
    // Pinned host memory for fast D2H transfers
    float* h_results_pinned;  // Layout: [loglik, vol, h_mean, bandwidth]
} SVPFOptimizedState;
```

---

## Profiling Tools

The optimizations were guided by a custom profiler (`test/profile_svpf.cu`) that measures:

- Step latency distribution (P50, P90, P99, max)
- Particle count scaling
- Stein iteration scaling
- Individual CUDA operation overhead

Usage:
```bash
./profile_svpf [n_particles] [n_steps] [warmup]
./profile_svpf 512 5000 100
```

For deeper kernel analysis:
```bash
ncu --set full ./profile_svpf 512 5000 100
nsys profile ./profile_svpf 512 5000 100
```

---

## Key Lessons

1. **Profile before optimizing** - 45% overhead was hidden until measured
2. **Sync points are expensive** - Each cudaStreamSynchronize costs ~5-10μs
3. **Pageable → Pinned memory** - Enables true async and reduces variance
4. **Stream ordering is free** - Don't sync when CUDA guarantees order
5. **P99 matters for HFT** - Optimizing tail latency often helps more than mean

---

## Future Opportunities

Current breakdown at N=512, Stein=5:
- Kernels: ~124 μs (83%)
- Overhead: ~25 μs (17%)

Further optimization would require:
- Kernel fusion (predict + gradient)
- Algorithmic changes (fewer Stein steps)
- Shared memory optimization for larger N

At 83% compute efficiency, diminishing returns are likely.
