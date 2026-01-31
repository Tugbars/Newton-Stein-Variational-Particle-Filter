# SVPF Parallel Stein Operator - Summary

## Goal

Parallelize the O(N²) Stein operator computation to improve performance at high particle counts (N ≥ 1024).

## Approach

**Sequential (existing):** One thread computes all N neighbor interactions per particle. Works but becomes slow at large N.

**Parallel (attempted):** One block (256 threads) per particle, each thread handles a strided subset of neighbors, then warp shuffle reduction aggregates results.

## Outcome

- **Standard Stein (no Newton):** Parallel path works correctly, matches sequential accuracy.
- **Full Newton Stein:** Parallel path has persistent accuracy degradation that we could not resolve.

The Full Newton variant involves additional Hessian-weighted preconditioning. Despite numerous attempts to align the parallel implementation with the sequential reference (fused accumulation, shared memory initialization, reduction order), the accuracy gap remained.

## Current State

- Parallel kernels are implemented and functional
- Standard mode: use parallel path for N ≥ 1024
- Full Newton mode: recommend staying on sequential path until root cause is identified

## Files

- `svpf_opt_kernels.cu`: Contains both sequential and parallel kernel implementations
- `svpf_optimized_graph.cu`: Integration logic that selects path based on N and mode
