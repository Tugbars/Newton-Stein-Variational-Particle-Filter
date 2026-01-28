/**
 * @file svpf_python.cpp
 * @brief PyBind11 wrapper for SVPF CUDA filter
 * 
 * Build with:
 *   pip install pybind11
 *   nvcc -shared -o svpf.so svpf_python.cpp svpf_kernels.cu svpf_optimized_graph.cu \
 *        svpf_opt_kernels.cu $(python3 -m pybind11 --includes) \
 *        -Xcompiler -fPIC --use_fast_math -O3
 * 
 * Usage:
 *   import svpf
 *   filter = svpf.SVPF(n_particles=400, n_stein=8, nu=30.0)
 *   filter.initialize(rho=0.97, sigma_z=0.15, mu=-3.5)
 *   for y in returns:
 *       result = filter.step(y)
 *       print(result.vol_mean)
 */

// Windows doesn't define M_PI by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "svpf.cuh"

namespace py = pybind11;

// =============================================================================
// Python-friendly Result Struct
// =============================================================================

struct PySVPFResult {
    float log_lik;
    float vol_mean;
    float vol_std;
    float h_mean;
    float mu_estimate;
};

// =============================================================================
// Main SVPF Python Class
// =============================================================================

class PySVPF {
public:
    /**
     * @brief Create SVPF filter
     * 
     * @param n_particles Number of particles (recommend 256-1024)
     * @param n_stein Stein iterations per step (recommend 5-10)
     * @param nu Student-t degrees of freedom (30 for quasi-Gaussian)
     */
    PySVPF(int n_particles = 400, int n_stein = 8, float nu = 30.0f) {
        cudaStreamCreate(&stream_);
        state_ = svpf_create(n_particles, n_stein, nu, stream_);
        if (!state_) {
            throw std::runtime_error("Failed to create SVPF filter");
        }
        
        // Set production defaults
        set_defaults();
        
        y_prev_ = 0.0f;
        initialized_ = false;
    }
    
    ~PySVPF() {
        if (state_) {
            svpf_destroy(state_);
        }
        cudaStreamDestroy(stream_);
    }
    
    // Disable copy
    PySVPF(const PySVPF&) = delete;
    PySVPF& operator=(const PySVPF&) = delete;
    
    // ==========================================================================
    // Configuration
    // ==========================================================================
    
    void set_defaults() {
        // Observation model
        state_->nu = 30.0f;
        state_->lik_offset = 0.345f;
        
        // Asymmetric persistence
        state_->use_asymmetric_rho = 1;
        state_->rho_up = 0.98f;
        state_->rho_down = 0.93f;
        
        // SVLD
        state_->use_svld = 1;
        state_->temperature = 0.45f;
        state_->rmsprop_rho = 0.9f;
        state_->rmsprop_eps = 1e-6f;
        
        // Annealing
        state_->use_annealing = 1;
        state_->n_anneal_steps = 5;
        
        // MIM
        state_->use_mim = 1;
        state_->mim_jump_prob = 0.25f;
        state_->mim_jump_scale = 9.0f;
        
        // Adaptive beta (KSD-based)
        state_->use_adaptive_beta = 1;
        
        // Rejuvenation
        state_->use_rejuvenation = 1;
        state_->rejuv_ksd_threshold = 0.05f;
        state_->rejuv_prob = 0.30f;
        state_->rejuv_blend = 0.30f;
        
        // Newton-Stein
        state_->use_newton = 1;
        state_->use_full_newton = 1;
        
        // Exact gradient
        state_->use_exact_gradient = 1;
        
        // Stein steps
        state_->stein_min_steps = 8;
        state_->stein_max_steps = 16;
        state_->ksd_improvement_threshold = 0.05f;
        
        // Guided prediction
        state_->use_guided = 1;
        state_->guided_alpha_base = 0.0f;
        state_->guided_alpha_shock = 0.40f;
        state_->guided_innovation_threshold = 1.5f;
        
        // EKF Guide
        state_->use_guide = 1;
        state_->use_guide_preserving = 1;
        state_->guide_strength = 0.05f;
        state_->guide_mean = 0.0f;
        state_->guide_var = 0.0f;
        state_->guide_K = 0.0f;
        state_->guide_initialized = 0;
        
        // Adaptive guide
        state_->use_adaptive_guide = 1;
        state_->guide_strength_base = 0.05f;
        state_->guide_strength_max = 0.30f;
        state_->guide_innovation_threshold = 1.0f;
        
        // Adaptive mu
        state_->use_adaptive_mu = 1;
        state_->mu_state = -3.5f;
        state_->mu_var = 1.0f;
        state_->mu_process_var = 0.001f;
        state_->mu_obs_var_scale = 11.0f;
        state_->mu_min = -4.0f;
        state_->mu_max = -1.0f;
        
        // Adaptive sigma_z
        state_->use_adaptive_sigma = 1;
        state_->sigma_boost_threshold = 0.95f;
        state_->sigma_boost_max = 3.2f;
        state_->sigma_z_effective = 0.10f;
        
        // Local params
        state_->use_local_params = 0;
        state_->delta_rho = 0.0f;
        state_->delta_sigma = 0.0f;
    }
    
    // --------------------------------------------------------------------------
    // Individual Config Setters
    // --------------------------------------------------------------------------
    
    void set_adaptive_mu(bool enabled, float process_var = 0.001f, 
                         float obs_var_scale = 11.0f,
                         float mu_min = -4.0f, float mu_max = -1.0f) {
        state_->use_adaptive_mu = enabled ? 1 : 0;
        state_->mu_process_var = process_var;
        state_->mu_obs_var_scale = obs_var_scale;
        state_->mu_min = mu_min;
        state_->mu_max = mu_max;
        invalidate_graph();
    }
    
    void set_adaptive_sigma(bool enabled, float threshold = 0.95f, float max_boost = 3.2f) {
        state_->use_adaptive_sigma = enabled ? 1 : 0;
        state_->sigma_boost_threshold = threshold;
        state_->sigma_boost_max = max_boost;
        invalidate_graph();
    }
    
    void set_adaptive_guide(bool enabled, float base = 0.05f, 
                            float max = 0.30f, float threshold = 1.0f) {
        state_->use_adaptive_guide = enabled ? 1 : 0;
        state_->guide_strength_base = base;
        state_->guide_strength_max = max;
        state_->guide_innovation_threshold = threshold;
        invalidate_graph();
    }
    
    void set_asymmetric_rho(bool enabled, float rho_up = 0.98f, float rho_down = 0.93f) {
        state_->use_asymmetric_rho = enabled ? 1 : 0;
        state_->rho_up = rho_up;
        state_->rho_down = rho_down;
        invalidate_graph();
    }
    
    void set_mim(bool enabled, float jump_prob = 0.25f, float jump_scale = 9.0f) {
        state_->use_mim = enabled ? 1 : 0;
        state_->mim_jump_prob = jump_prob;
        state_->mim_jump_scale = jump_scale;
        invalidate_graph();
    }
    
    void set_svld(bool enabled, float temperature = 0.45f) {
        state_->use_svld = enabled ? 1 : 0;
        state_->temperature = temperature;
        invalidate_graph();
    }
    
    void set_annealing(bool enabled, int n_steps = 5) {
        state_->use_annealing = enabled ? 1 : 0;
        state_->n_anneal_steps = n_steps;
        invalidate_graph();
    }
    
    void set_newton(bool enabled) {
        state_->use_newton = enabled ? 1 : 0;
        invalidate_graph();
    }
    
    void set_guide(bool enabled, float strength = 0.05f) {
        state_->use_guide = enabled ? 1 : 0;
        state_->guide_strength = strength;
        invalidate_graph();
    }
    
    // --------------------------------------------------------------------------
    // NEW: Additional Config Setters for Full Production Config
    // --------------------------------------------------------------------------
    
    void set_adaptive_beta(bool enabled) {
        state_->use_adaptive_beta = enabled ? 1 : 0;
        invalidate_graph();
    }
    
    void set_full_newton(bool enabled) {
        state_->use_full_newton = enabled ? 1 : 0;
        invalidate_graph();
    }
    
    void set_exact_gradient(bool enabled) {
        state_->use_exact_gradient = enabled ? 1 : 0;
        invalidate_graph();
    }
    
    void set_stein_steps(int min_steps = 8, int max_steps = 16, float ksd_threshold = 0.05f) {
        state_->stein_min_steps = min_steps;
        state_->stein_max_steps = max_steps;
        state_->ksd_improvement_threshold = ksd_threshold;
        invalidate_graph();
    }
    
    void set_guided_innovation(float alpha_base = 0.0f, float alpha_shock = 0.40f, 
                               float threshold = 1.5f) {
        state_->guided_alpha_base = alpha_base;
        state_->guided_alpha_shock = alpha_shock;
        state_->guided_innovation_threshold = threshold;
        invalidate_graph();
    }
    
    void set_rejuvenation(bool enabled, float ksd_threshold = 0.05f, 
                          float prob = 0.30f, float blend = 0.30f) {
        state_->use_rejuvenation = enabled ? 1 : 0;
        state_->rejuv_ksd_threshold = ksd_threshold;
        state_->rejuv_prob = prob;
        state_->rejuv_blend = blend;
        invalidate_graph();
    }
    
    // --------------------------------------------------------------------------
    // Property setters
    // --------------------------------------------------------------------------
    
    void set_nu(float nu) { 
        state_->nu = nu; 
        // Recompute Student-t constant
        state_->student_t_const = lgammaf((nu + 1.0f) / 2.0f) 
                                 - lgammaf(nu / 2.0f) 
                                 - 0.5f * logf(M_PI * nu);
        invalidate_graph();
    }
    
    void set_lik_offset(float offset) { 
        state_->lik_offset = offset; 
        invalidate_graph();
    }
    
    // Property getters
    float get_nu() const { return state_->nu; }
    float get_lik_offset() const { return state_->lik_offset; }
    int get_n_particles() const { return state_->n_particles; }
    int get_n_stein_steps() const { return state_->n_stein_steps; }
    
    // ==========================================================================
    // Core API
    // ==========================================================================
    
    /**
     * @brief Initialize filter with model parameters
     * 
     * @param rho Persistence (0.9-0.99)
     * @param sigma_z Vol-of-vol (0.1-0.3)
     * @param mu Long-run mean log-vol (-5 to -3)
     * @param gamma Leverage effect (-0.5 to 0)
     * @param seed Random seed
     */
    void initialize(float rho = 0.97f, float sigma_z = 0.15f, 
                   float mu = -3.5f, float gamma = -0.5f,
                   unsigned long long seed = 42) {
        params_.rho = rho;
        params_.sigma_z = sigma_z;
        params_.mu = mu;
        params_.gamma = gamma;
        
        svpf_initialize(state_, &params_, seed);
        
        // Initialize adaptive mu state
        if (state_->use_adaptive_mu) {
            state_->mu_state = mu;
            state_->mu_var = 0.01f;
        }
        
        y_prev_ = 0.0f;
        initialized_ = true;
    }
    
    /**
     * @brief Process single observation
     * 
     * @param y_t Current return observation
     * @return Result with vol_mean, h_mean, log_lik
     */
    PySVPFResult step(float y_t) {
        if (!initialized_) {
            throw std::runtime_error("Filter not initialized. Call initialize() first.");
        }
        
        float loglik, vol, h_mean;
        
        svpf_step_graph(state_, y_t, y_prev_, &params_, &loglik, &vol, &h_mean);
        
        y_prev_ = y_t;
        
        PySVPFResult result;
        result.log_lik = loglik;
        result.vol_mean = vol;
        result.vol_std = 0.0f;  // TODO: compute if needed
        result.h_mean = h_mean;
        result.mu_estimate = state_->use_adaptive_mu ? state_->mu_state : params_.mu;
        
        return result;
    }
    
    /**
     * @brief Process array of observations
     * 
     * @param observations NumPy array of returns [T]
     * @return Tuple of (vol_array, h_mean_array, loglik_array)
     */
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> 
    run(py::array_t<float> observations) {
        if (!initialized_) {
            throw std::runtime_error("Filter not initialized. Call initialize() first.");
        }
        
        py::buffer_info buf = observations.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Observations must be 1-dimensional");
        }
        
        int T = buf.shape[0];
        float* obs_ptr = static_cast<float*>(buf.ptr);
        
        // Allocate output arrays
        auto vol_out = py::array_t<float>(T);
        auto h_mean_out = py::array_t<float>(T);
        auto loglik_out = py::array_t<float>(T);
        
        float* vol_ptr = static_cast<float*>(vol_out.request().ptr);
        float* h_ptr = static_cast<float*>(h_mean_out.request().ptr);
        float* ll_ptr = static_cast<float*>(loglik_out.request().ptr);
        
        // Run filter
        for (int t = 0; t < T; t++) {
            float loglik, vol, h_mean;
            svpf_step_graph(state_, obs_ptr[t], y_prev_, &params_, &loglik, &vol, &h_mean);
            
            vol_ptr[t] = vol;
            h_ptr[t] = h_mean;
            ll_ptr[t] = loglik;
            
            y_prev_ = obs_ptr[t];
        }
        
        return std::make_tuple(vol_out, h_mean_out, loglik_out);
    }
    
    /**
     * @brief Reset filter state without changing configuration
     */
    void reset(unsigned long long seed = 42) {
        svpf_initialize(state_, &params_, seed);
        if (state_->use_adaptive_mu) {
            state_->mu_state = params_.mu;
            state_->mu_var = 0.01f;
        }
        y_prev_ = 0.0f;
    }
    
    /**
     * @brief Get current particle positions
     */
    py::array_t<float> get_particles() const {
        auto result = py::array_t<float>(state_->n_particles);
        float* ptr = static_cast<float*>(result.request().ptr);
        svpf_get_particles(state_, ptr);
        return result;
    }
    
    /**
     * @brief Get effective sample size
     */
    float get_ess() const {
        return svpf_get_ess(state_);
    }
    
    /**
     * @brief Force CUDA graph recapture
     */
    void invalidate_graph() {
        svpf_graph_invalidate(state_);
    }
    
    /**
     * @brief Check if graph is captured
     */
    bool is_graph_captured() const {
        return svpf_graph_is_captured(state_);
    }
    
    /**
     * @brief Update model parameters (triggers graph recapture)
     */
    void set_params(float rho, float sigma_z, float mu, float gamma) {
        params_.rho = rho;
        params_.sigma_z = sigma_z;
        params_.mu = mu;
        params_.gamma = gamma;
        invalidate_graph();
    }
    
private:
    SVPFState* state_;
    SVPFParams params_;
    cudaStream_t stream_;
    float y_prev_;
    bool initialized_;
};

// =============================================================================
// PyBind11 Module Definition
// =============================================================================

PYBIND11_MODULE(pysvpf, m) {
    m.doc() = "SVPF: Stein Variational Particle Filter for Stochastic Volatility";
    
    // Result struct
    py::class_<PySVPFResult>(m, "SVPFResult")
        .def_readonly("log_lik", &PySVPFResult::log_lik, "Log-likelihood increment")
        .def_readonly("vol_mean", &PySVPFResult::vol_mean, "Mean volatility estimate")
        .def_readonly("vol_std", &PySVPFResult::vol_std, "Volatility std (if computed)")
        .def_readonly("h_mean", &PySVPFResult::h_mean, "Mean log-volatility")
        .def_readonly("mu_estimate", &PySVPFResult::mu_estimate, "Current adaptive mu")
        .def("__repr__", [](const PySVPFResult& r) {
            return "<SVPFResult vol=" + std::to_string(r.vol_mean) + 
                   " h=" + std::to_string(r.h_mean) + 
                   " ll=" + std::to_string(r.log_lik) + ">";
        });
    
    // Main SVPF class
    py::class_<PySVPF>(m, "SVPF")
        .def(py::init<int, int, float>(),
             py::arg("n_particles") = 400,
             py::arg("n_stein") = 8,
             py::arg("nu") = 30.0f,
             "Create SVPF filter\n\n"
             "Args:\n"
             "    n_particles: Number of particles (default: 400)\n"
             "    n_stein: Stein iterations per step (default: 8)\n"
             "    nu: Student-t degrees of freedom (default: 30)")
        
        // Core API
        .def("initialize", &PySVPF::initialize,
             py::arg("rho") = 0.97f,
             py::arg("sigma_z") = 0.15f,
             py::arg("mu") = -3.5f,
             py::arg("gamma") = -0.5f,
             py::arg("seed") = 42,
             "Initialize filter with model parameters")
        
        .def("step", &PySVPF::step,
             py::arg("y_t"),
             "Process single observation, return SVPFResult")
        
        .def("run", &PySVPF::run,
             py::arg("observations"),
             "Process array of observations\n\n"
             "Returns: (vol_array, h_mean_array, loglik_array)")
        
        .def("reset", &PySVPF::reset,
             py::arg("seed") = 42,
             "Reset filter state")
        
        // Configuration
        .def("set_adaptive_mu", &PySVPF::set_adaptive_mu,
             py::arg("enabled"),
             py::arg("process_var") = 0.001f,
             py::arg("obs_var_scale") = 11.0f,
             py::arg("mu_min") = -4.0f,
             py::arg("mu_max") = -1.0f,
             "Configure adaptive mu (Kalman filter on mean level)")
        
        .def("set_adaptive_sigma", &PySVPF::set_adaptive_sigma,
             py::arg("enabled"),
             py::arg("threshold") = 0.95f,
             py::arg("max_boost") = 3.2f,
             "Configure adaptive sigma_z (breathing filter)")
        
        .def("set_adaptive_guide", &PySVPF::set_adaptive_guide,
             py::arg("enabled"),
             py::arg("base") = 0.05f,
             py::arg("max") = 0.30f,
             py::arg("threshold") = 1.0f,
             "Configure adaptive guide strength")
        
        .def("set_asymmetric_rho", &PySVPF::set_asymmetric_rho,
             py::arg("enabled"),
             py::arg("rho_up") = 0.98f,
             py::arg("rho_down") = 0.93f,
             "Configure asymmetric persistence")
        
        .def("set_mim", &PySVPF::set_mim,
             py::arg("enabled"),
             py::arg("jump_prob") = 0.25f,
             py::arg("jump_scale") = 9.0f,
             "Configure Mixture Innovation Model (scout particles)")
        
        .def("set_svld", &PySVPF::set_svld,
             py::arg("enabled"),
             py::arg("temperature") = 0.45f,
             "Configure SVLD (Langevin noise)")
        
        .def("set_annealing", &PySVPF::set_annealing,
             py::arg("enabled"),
             py::arg("n_steps") = 5,
             "Configure annealed Stein updates")
        
        .def("set_newton", &PySVPF::set_newton,
             py::arg("enabled"),
             "Enable/disable Newton-Stein (Hessian preconditioning)")
        
        .def("set_guide", &PySVPF::set_guide,
             py::arg("enabled"),
             py::arg("strength") = 0.05f,
             "Configure EKF guide density")
        
        .def("set_params", &PySVPF::set_params,
             py::arg("rho"),
             py::arg("sigma_z"),
             py::arg("mu"),
             py::arg("gamma"),
             "Update model parameters")
        
        // NEW: Additional config setters for full production config
        .def("set_adaptive_beta", &PySVPF::set_adaptive_beta,
             py::arg("enabled"),
             "Enable/disable KSD-adaptive beta tempering")
        
        .def("set_full_newton", &PySVPF::set_full_newton,
             py::arg("enabled"),
             "Enable/disable full Newton-Stein (Hessian preconditioning)")
        
        .def("set_exact_gradient", &PySVPF::set_exact_gradient,
             py::arg("enabled"),
             "Enable/disable exact Student-t gradient")
        
        .def("set_stein_steps", &PySVPF::set_stein_steps,
             py::arg("min_steps") = 8,
             py::arg("max_steps") = 16,
             py::arg("ksd_threshold") = 0.05f,
             "Configure Stein iteration limits and KSD convergence threshold")
        
        .def("set_guided_innovation", &PySVPF::set_guided_innovation,
             py::arg("alpha_base") = 0.0f,
             py::arg("alpha_shock") = 0.40f,
             py::arg("threshold") = 1.5f,
             "Configure innovation-gated guided prediction")
        
        .def("set_rejuvenation", &PySVPF::set_rejuvenation,
             py::arg("enabled"),
             py::arg("ksd_threshold") = 0.05f,
             py::arg("prob") = 0.30f,
             py::arg("blend") = 0.30f,
             "Configure partial rejuvenation (nudge stuck particles toward guide)")
        
        // Properties
        .def_property("nu", &PySVPF::get_nu, &PySVPF::set_nu,
                      "Student-t degrees of freedom")
        .def_property("lik_offset", &PySVPF::get_lik_offset, &PySVPF::set_lik_offset,
                      "Likelihood offset (bias correction)")
        .def_property_readonly("n_particles", &PySVPF::get_n_particles,
                               "Number of particles")
        .def_property_readonly("n_stein_steps", &PySVPF::get_n_stein_steps,
                               "Stein iterations per step")
        
        // Diagnostics
        .def("get_particles", &PySVPF::get_particles,
             "Get current particle positions as NumPy array")
        .def("get_ess", &PySVPF::get_ess,
             "Get effective sample size")
        .def("invalidate_graph", &PySVPF::invalidate_graph,
             "Force CUDA graph recapture")
        .def("is_graph_captured", &PySVPF::is_graph_captured,
             "Check if CUDA graph is captured")
        
        // Special methods
        .def("__repr__", [](const PySVPF& f) {
            return "<SVPF n_particles=" + std::to_string(f.get_n_particles()) +
                   " n_stein=" + std::to_string(f.get_n_stein_steps()) +
                   " nu=" + std::to_string(f.get_nu()) + ">";
        });
    
    // Module-level info
    m.attr("__version__") = "1.0.0";
}