/**
 * @file svpf_config.h
 * @brief Build configuration for SVPF
 * 
 * Runtime-configurable options for testing different algorithm variants.
 */

#ifndef SVPF_CONFIG_H
#define SVPF_CONFIG_H

// =============================================================================
// STEIN OPERATOR SIGN CONFIGURATION
// =============================================================================
//
// The Stein operator has two terms:
//   φ(x_i) = 1/n Σⱼ [k(xⱼ, xᵢ)·∇log q(xⱼ) + ∇_{xⱼ} k(xⱼ, xᵢ)]
//                    └─── attraction ────┘   └─── repulsion ───┘
//
// For IMQ kernel with diff = x_i - x_j:
//   ∇_{xⱼ} k(xⱼ, xᵢ) = +2·diff/h²·k²
//
// FAN ET AL. (2021): Repulsive term should ADD to spread particles
// CURRENT IMPL: Repulsive term SUBTRACTS (empirically tuned with other aids)
//
// STEIN_SIGN_MODE:
//   0 = LEGACY (subtract) - current behavior, works with MIM/SVLD/guide
//   1 = PAPER  (add)      - mathematically correct per Fan et al.
// =============================================================================

// Default to legacy behavior for backwards compatibility
#ifndef SVPF_STEIN_SIGN_MODE_DEFAULT
#define SVPF_STEIN_SIGN_MODE_DEFAULT 1
#endif

// =============================================================================
// Add this field to SVPFState struct in svpf.cuh:
// =============================================================================
//
//     // Stein operator sign configuration
//     // 0 = legacy (subtract, attraction), 1 = paper (add, repulsion)
//     int stein_repulsive_sign;
//
// =============================================================================

// =============================================================================
// Add this to svpf_create() in svpf_optimized_graph.cu:
// =============================================================================
//
//     state->stein_repulsive_sign = SVPF_STEIN_SIGN_MODE_DEFAULT;
//
// =============================================================================

// =============================================================================
// Add this API function declaration to svpf.cuh:
// =============================================================================
//
//     /**
//      * @brief Set Stein operator repulsive sign mode
//      * 
//      * @param state   Filter state
//      * @param mode    0 = legacy (subtract/attract), 1 = paper (add/repel)
//      * 
//      * Mode 0 (legacy): Particles slightly attract, compensated by SVLD/MIM/guide
//      * Mode 1 (paper):  Particles repel per Fan et al. 2021, may need retuning
//      */
//     void svpf_set_stein_sign_mode(SVPFState* state, int mode);
//
// =============================================================================

#endif // SVPF_CONFIG_H
