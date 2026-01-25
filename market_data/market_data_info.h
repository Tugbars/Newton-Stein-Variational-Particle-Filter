// Auto-generated market data info
// Generated: 2026-01-25T21:53:01.149931

#ifndef MARKET_DATA_INFO_H
#define MARKET_DATA_INFO_H

#define SPY_FULL_N 4528
#define SPY_2008_N 375
#define SPY_2020_N 123
#define SPY_2022_N 250
#define QQQ_FULL_N 4528

// Usage:
//   float* returns = (float*)malloc(SPY_FULL_N * sizeof(float));
//   FILE* f = fopen("market_data/spy_full.bin", "rb");
//   fread(returns, sizeof(float), SPY_FULL_N, f);
//   fclose(f);

#endif // MARKET_DATA_INFO_H
