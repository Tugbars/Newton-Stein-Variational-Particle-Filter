/**
 * @file market_data_loader.h
 * @brief Simple loader for market data CSV/binary files
 * 
 * Usage:
 *   float* returns;
 *   int n;
 *   
 *   // Option 1: Load binary (fast)
 *   load_returns_binary("market_data/spy_full.bin", &returns, &n);
 *   
 *   // Option 2: Load CSV (flexible)
 *   load_returns_csv("market_data/spy_full_returns.csv", &returns, &n);
 *   
 *   // Use data...
 *   for (int t = 0; t < n; t++) { ... }
 *   
 *   // Cleanup
 *   free(returns);
 */

#ifndef MARKET_DATA_LOADER_H
#define MARKET_DATA_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load returns from binary file (float32)
 * 
 * @param filepath  Path to .bin file
 * @param returns   Output: allocated array of returns
 * @param n         Output: number of returns
 * @return 0 on success, -1 on error
 */
static inline int load_returns_binary(const char* filepath, float** returns, int* n) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filepath);
        return -1;
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    *n = (int)(size / sizeof(float));
    *returns = (float*)malloc(size);
    
    if (!*returns) {
        fclose(f);
        return -1;
    }
    
    size_t read = fread(*returns, sizeof(float), *n, f);
    fclose(f);
    
    if ((int)read != *n) {
        free(*returns);
        *returns = NULL;
        return -1;
    }
    
    return 0;
}

/**
 * @brief Load returns from CSV file (one return per line)
 * 
 * @param filepath  Path to *_returns.csv file
 * @param returns   Output: allocated array of returns
 * @param n         Output: number of returns
 * @return 0 on success, -1 on error
 */
static inline int load_returns_csv(const char* filepath, float** returns, int* n) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filepath);
        return -1;
    }
    
    // Count lines
    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        count++;
    }
    
    // Allocate
    *n = count;
    *returns = (float*)malloc(count * sizeof(float));
    if (!*returns) {
        fclose(f);
        return -1;
    }
    
    // Read values
    rewind(f);
    for (int i = 0; i < count; i++) {
        if (fgets(line, sizeof(line), f)) {
            (*returns)[i] = (float)atof(line);
        }
    }
    
    fclose(f);
    return 0;
}

/**
 * @brief Load full CSV with all columns
 * 
 * @param filepath  Path to .csv file
 * @param returns   Output: log returns
 * @param prices    Output: prices (optional, can be NULL)
 * @param n         Output: number of observations
 * @return 0 on success, -1 on error
 */
static inline int load_market_data_csv(
    const char* filepath, 
    float** returns, 
    float** prices,
    int* n
) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filepath);
        return -1;
    }
    
    // Skip header
    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }
    
    // Count lines
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        count++;
    }
    
    // Allocate
    *n = count;
    *returns = (float*)malloc(count * sizeof(float));
    if (prices) {
        *prices = (float*)malloc(count * sizeof(float));
    }
    
    // Parse CSV: date,price,return,log_return
    rewind(f);
    fgets(line, sizeof(line), f);  // Skip header again
    
    for (int i = 0; i < count; i++) {
        if (fgets(line, sizeof(line), f)) {
            char* tok = strtok(line, ",");  // date
            tok = strtok(NULL, ",");         // price
            float price = (float)atof(tok);
            tok = strtok(NULL, ",");         // return (skip)
            tok = strtok(NULL, ",");         // log_return
            float log_ret = (float)atof(tok);
            
            (*returns)[i] = log_ret;
            if (prices) {
                (*prices)[i] = price;
            }
        }
    }
    
    fclose(f);
    return 0;
}

/**
 * @brief Compute basic statistics for returns
 */
static inline void compute_return_stats(
    const float* returns, 
    int n,
    float* mean,
    float* std,
    float* min_ret,
    float* max_ret,
    int* n_3sigma,
    int* n_4sigma
) {
    // Mean
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += returns[i];
    }
    *mean = (float)(sum / n);
    
    // Std
    double sq_sum = 0;
    for (int i = 0; i < n; i++) {
        double d = returns[i] - *mean;
        sq_sum += d * d;
    }
    *std = (float)sqrt(sq_sum / n);
    
    // Min/Max and tail counts
    *min_ret = returns[0];
    *max_ret = returns[0];
    *n_3sigma = 0;
    *n_4sigma = 0;
    
    for (int i = 0; i < n; i++) {
        if (returns[i] < *min_ret) *min_ret = returns[i];
        if (returns[i] > *max_ret) *max_ret = returns[i];
        
        float z = (returns[i] - *mean) / (*std);
        if (fabsf(z) > 3.0f) (*n_3sigma)++;
        if (fabsf(z) > 4.0f) (*n_4sigma)++;
    }
}

/**
 * @brief Print return statistics
 */
static inline void print_return_stats(const char* name, const float* returns, int n) {
    float mean, std, min_ret, max_ret;
    int n_3sigma, n_4sigma;
    
    compute_return_stats(returns, n, &mean, &std, &min_ret, &max_ret, &n_3sigma, &n_4sigma);
    
    printf("  %s: n=%d, μ=%.4f, σ=%.4f, range=[%.2f%%, %.2f%%]\n",
           name, n, mean * 252, std * sqrt(252.0f), min_ret * 100, max_ret * 100);
    printf("    Tail events: %d (>3σ), %d (>4σ)\n", n_3sigma, n_4sigma);
}

#ifdef __cplusplus
}
#endif

#endif // MARKET_DATA_LOADER_H
