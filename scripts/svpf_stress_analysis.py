# SVPF Extreme Stress Test Visualization
# Run test_svpf_extreme.exe first to generate CSV files

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Set paths - CSVs are in same folder as this script (root/scripts)
DATA_DIR = "."
OUTPUT_DIR = "."  # Save plots here too

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# =====================================================================
# Load Summary Data
# =====================================================================

summary_path = os.path.join(DATA_DIR, 'svpf_stress_summary.csv')
summary = pd.read_csv(summary_path)
print(f"Loaded summary from: {summary_path}")
print(summary.head(10))
print(f"\nTotal scenarios: {len(summary)}")
print(f"Failed (NaN/Inf): {summary['had_nan'].sum() + summary['had_inf'].sum()}")
print(f"Failed to recover: {summary['failed_recovery'].sum()}")

# =====================================================================
# 1. Summary Heatmap: ESS vs Sigma by Scenario
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ESS heatmap
pivot_ess = summary.pivot_table(values='min_ess', index='scenario', columns='sigma', aggfunc='mean')
im1 = axes[0].imshow(pivot_ess.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=512)
axes[0].set_xticks(range(len(pivot_ess.columns)))
axes[0].set_xticklabels([f'{s:.0f}σ' for s in pivot_ess.columns])
axes[0].set_yticks(range(len(pivot_ess.index)))
axes[0].set_yticklabels(pivot_ess.index)
axes[0].set_title('Minimum ESS (higher = better)')
plt.colorbar(im1, ax=axes[0])

# Max h heatmap
pivot_h = summary.pivot_table(values='max_h', index='scenario', columns='sigma', aggfunc='mean')
im2 = axes[1].imshow(pivot_h.values, cmap='RdYlGn_r', aspect='auto', vmin=-5, vmax=5)
axes[1].set_xticks(range(len(pivot_h.columns)))
axes[1].set_xticklabels([f'{s:.0f}σ' for s in pivot_h.columns])
axes[1].set_yticks(range(len(pivot_h.index)))
axes[1].set_yticklabels(pivot_h.index)
axes[1].set_title('Max log-volatility (lower = better)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stress_heatmap.png'), dpi=150)
plt.show()

# =====================================================================
# 2. Load and Plot a Single Timeseries
# =====================================================================

def plot_scenario(filename, title=None):
    """Plot a single stress test scenario timeseries with TRUE vs ESTIMATED comparison"""
    filepath = os.path.join(DATA_DIR, filename) if not os.path.isabs(filename) else filename
    df = pd.read_csv(filepath)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    if title is None:
        title = f"{df['scenario'].iloc[0]} - {df['sigma'].iloc[0]:.0f}σ"
    
    # Compute tracking metrics
    error = df['h_mean'] - df['true_h']
    rmse = np.sqrt((error**2).mean())
    bias = error.mean()
    mae = error.abs().mean()
    max_err = error.abs().max()
    correlation = df['h_mean'].corr(df['true_h'])
    
    # Add metrics to title
    title += f"\n[RMSE={rmse:.3f}, Bias={bias:+.3f}, MAE={mae:.3f}, MaxErr={max_err:.3f}, Corr={correlation:.3f}]"
    
    # Panel 1: Returns
    axes[0].plot(df['t'], df['y_t'], 'b-', alpha=0.7, linewidth=0.8)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Return (y_t)')
    axes[0].set_title(title)
    
    # Mark extreme returns
    extreme_mask = np.abs(df['y_t']) > 0.05  # 5% move
    if extreme_mask.any():
        axes[0].scatter(df.loc[extreme_mask, 't'], df.loc[extreme_mask, 'y_t'], 
                       color='red', s=50, zorder=5, label='Extreme (>5%)')
        axes[0].legend()
    
    # Panel 2: TRUE vs ESTIMATED log-volatility (h)
    axes[1].plot(df['t'], df['true_h'], 'k-', linewidth=2, label='True h', alpha=0.8)
    axes[1].plot(df['t'], df['h_mean'], 'r--', linewidth=1.5, label='Estimated h', alpha=0.8)
    axes[1].axhline(-4.5, color='gray', linestyle=':', alpha=0.5, label='μ = -4.5')
    axes[1].set_ylabel('Log-volatility (h)')
    axes[1].legend(loc='upper right')
    axes[1].fill_between(df['t'], df['true_h'], df['h_mean'], alpha=0.2, color='red')
    
    # Panel 3: TRUE vs ESTIMATED volatility (%)
    axes[2].plot(df['t'], df['true_vol'] * 100, 'k-', linewidth=2, label='True Vol (%)', alpha=0.8)
    axes[2].plot(df['t'], df['vol'] * 100, 'g--', linewidth=1.5, label='Estimated Vol (%)', alpha=0.8)
    axes[2].set_ylabel('Volatility (%)')
    axes[2].legend(loc='upper right')
    axes[2].set_ylim(bottom=0)
    
    # Panel 4: ESS
    axes[3].fill_between(df['t'], df['ess'], alpha=0.3, color='purple')
    axes[3].plot(df['t'], df['ess'], 'purple', linewidth=1)
    axes[3].axhline(10, color='red', linestyle='--', alpha=0.7, label='Collapse threshold (ESS=10)')
    axes[3].axhline(512, color='green', linestyle='--', alpha=0.7, label='Max (N=512)')
    axes[3].set_ylabel('ESS')
    axes[3].set_xlabel('Timestep')
    axes[3].legend(loc='upper right')
    axes[3].set_ylim(0, 550)
    
    plt.tight_layout()
    
    # Print metrics to console
    print(f"\n{'='*60}")
    print(f"Tracking Metrics: {df['scenario'].iloc[0]} @ {df['sigma'].iloc[0]:.0f}σ")
    print(f"{'='*60}")
    print(f"  RMSE(h):      {rmse:.4f}")
    print(f"  Bias(h):      {bias:+.4f} ({'overestimates' if bias > 0 else 'underestimates'} vol)")
    print(f"  MAE(h):       {mae:.4f}")
    print(f"  Max Error:    {max_err:.4f}")
    print(f"  Correlation:  {correlation:.4f}")
    print(f"{'='*60}")
    
    return fig

# Example: Plot 20-sigma single spike
try:
    fig = plot_scenario('svpf_stress_Single_Spike_20sigma.csv')
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_single_spike_20sigma.png'), dpi=150)
    plt.show()
except FileNotFoundError:
    print("CSV file not found - run test_svpf_extreme.exe first")

# =====================================================================
# 3. Compare Multiple Sigma Levels
# =====================================================================

def plot_sigma_comparison(scenario_prefix, sigmas=[10, 20, 30, 40, 50]):
    """Compare filter tracking accuracy across different sigma levels"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigmas)))
    
    for i, sigma in enumerate(sigmas):
        filename = f'svpf_stress_{scenario_prefix}_{sigma:.0f}sigma.csv'
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            # True vol (solid) vs Estimated vol (dashed)
            axes[0].plot(df['t'], df['true_vol'] * 100, color=colors[i], 
                        linewidth=1.5, label=f'{sigma:.0f}σ true', alpha=0.9)
            axes[0].plot(df['t'], df['vol'] * 100, color=colors[i], 
                        linewidth=1.5, linestyle='--', alpha=0.6)
            
            # Tracking error
            tracking_error = (df['h_mean'] - df['true_h']).abs()
            axes[1].plot(df['t'], tracking_error, color=colors[i], 
                        linewidth=1.5, label=f'{sigma:.0f}σ', alpha=0.8)
            
            # ESS
            axes[2].plot(df['t'], df['ess'], color=colors[i], 
                        linewidth=1.5, label=f'{sigma:.0f}σ', alpha=0.8)
        except FileNotFoundError:
            print(f"Not found: {filepath}")
    
    axes[0].set_ylabel('Volatility (%)')
    axes[0].set_title(f'{scenario_prefix.replace("_", " ")}: True (solid) vs Estimated (dashed)')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].set_ylim(bottom=0)
    
    axes[1].set_ylabel('|h_est - h_true|')
    axes[1].set_title('Tracking Error')
    axes[1].legend(loc='upper right')
    
    axes[2].set_ylabel('ESS')
    axes[2].set_xlabel('Timestep')
    axes[2].axhline(10, color='red', linestyle='--', alpha=0.5)
    axes[2].legend(loc='lower right')
    axes[2].set_ylim(0, 550)
    
    plt.tight_layout()
    return fig

# Compare single spike at different magnitudes
try:
    fig = plot_sigma_comparison('Single_Spike', [10, 20, 30, 40, 50])
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_comparison_single_spike.png'), dpi=150)
    plt.show()
except:
    print("Files not found - run test_svpf_extreme.exe first")

# =====================================================================
# 3b. Error Analysis Over Time
# =====================================================================

def plot_error_analysis(scenario_prefix, sigmas=[10, 20, 30, 40, 50]):
    """Analyze tracking error over time"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigmas)))
    
    for i, sigma in enumerate(sigmas):
        filename = f'svpf_stress_{scenario_prefix}_{sigma:.0f}sigma.csv'
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            error = df['h_mean'] - df['true_h']
            
            # Signed error
            axes[0].plot(df['t'], error, color=colors[i], linewidth=1, 
                        label=f'{sigma:.0f}σ', alpha=0.8)
            
            # Cumulative absolute error
            cum_abs_error = error.abs().cumsum() / (df['t'] + 1)
            axes[1].plot(df['t'], cum_abs_error, color=colors[i], linewidth=1.5,
                        label=f'{sigma:.0f}σ', alpha=0.8)
        except FileNotFoundError:
            print(f"Not found: {filepath}")
    
    axes[0].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_ylabel('Error (h_est - h_true)')
    axes[0].set_title(f'{scenario_prefix.replace("_", " ")}: Tracking Error Over Time')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(-1.5, 1.5)
    
    axes[1].set_ylabel('Cumulative MAE')
    axes[1].set_xlabel('Timestep')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Running Average Absolute Error')
    
    plt.tight_layout()
    return fig

try:
    fig = plot_error_analysis('Single_Spike', [10, 20, 30, 40, 50])
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_error_analysis.png'), dpi=150)
    plt.show()
except Exception as e:
    print(f"Error analysis failed: {e}")

# =====================================================================
# 4. Find Breaking Points
# =====================================================================

print("\n" + "="*60)
print("BREAKING POINT ANALYSIS")
print("="*60)
print(f"\nNote: mu = -4.5, so recovery means h_mean < {-4.5 + 0.5} = -4.0")

# Find scenarios where filter failed
failed = summary[(summary['had_nan'] == 1) | (summary['had_inf'] == 1)]
if len(failed) > 0:
    print("\n❌ FAILED (NaN/Inf):")
    print(failed[['scenario', 'sigma', 'min_ess', 'max_h']].to_string(index=False))
else:
    print("\n✓ No numerical failures (NaN/Inf)")

# Find scenarios with particle collapse
collapsed = summary[summary['min_ess'] < 10]
if len(collapsed) > 0:
    print("\n⚠ PARTICLE COLLAPSE (ESS < 10):")
    print(collapsed[['scenario', 'sigma', 'min_ess', 'max_h']].to_string(index=False))
else:
    print("\n✓ No particle collapse")

# Find scenarios with recovery failure
no_recovery = summary[summary['failed_recovery'] == 1]
if len(no_recovery) > 0:
    print(f"\n⚠ FAILED TO RECOVER (h didn't return to within 0.5 of mu):")
    print(f"   Count: {len(no_recovery)} / {len(summary)} scenarios")
    # Show worst cases
    worst = no_recovery.nlargest(5, 'max_h')
    print(worst[['scenario', 'sigma', 'max_h', 'final_h']].to_string(index=False))
else:
    print("\n✓ All scenarios recovered")

# Breaking point by scenario
print("\n" + "-"*60)
print("BREAKING POINT (first sigma where ESS < 50):")
for scenario in summary['scenario'].unique():
    subset = summary[(summary['scenario'] == scenario) & (summary['min_ess'] < 50)]
    if len(subset) > 0:
        breaking_sigma = subset['sigma'].min()
        print(f"  {scenario}: {breaking_sigma:.0f}σ")
    else:
        print(f"  {scenario}: > 50σ (robust)")

# =====================================================================
# 4b. Tracking Accuracy (RMSE of h_est vs h_true)
# =====================================================================

print("\n" + "="*70)
print("TRACKING ACCURACY ANALYSIS")
print("="*70)

metrics_data = []
scenarios = ['Single_Spike', 'Double_Spike', 'Flash_Crash', 'Sustained_Chaos', 'Gradual_Build']
sigmas = [5, 10, 15, 20, 25, 30, 40, 50]

for scenario in scenarios:
    for sigma in sigmas:
        filename = f'svpf_stress_{scenario}_{sigma:.0f}sigma.csv'
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            error = df['h_mean'] - df['true_h']
            
            metrics_data.append({
                'scenario': scenario.replace('_', ' '),
                'sigma': sigma,
                'RMSE': np.sqrt((error**2).mean()),
                'Bias': error.mean(),
                'MAE': error.abs().mean(),
                'MaxErr': error.abs().max(),
                'Corr': df['h_mean'].corr(df['true_h']),
                'ESS_min': df['ess'].min()
            })
        except FileNotFoundError:
            pass

if metrics_data:
    metrics_df = pd.DataFrame(metrics_data)
    
    # Print full table
    print("\nFull Metrics Table:")
    print("-"*70)
    print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Summary by scenario (averaged across sigmas)
    print("\n" + "-"*70)
    print("Average Metrics by Scenario:")
    print("-"*70)
    scenario_avg = metrics_df.groupby('scenario')[['RMSE', 'Bias', 'MAE', 'Corr']].mean()
    print(scenario_avg.round(4).to_string())
    
    # Summary by sigma (averaged across scenarios)
    print("\n" + "-"*70)
    print("Average Metrics by Sigma:")
    print("-"*70)
    sigma_avg = metrics_df.groupby('sigma')[['RMSE', 'Bias', 'MAE', 'Corr']].mean()
    print(sigma_avg.round(4).to_string())
    
    # Plot RMSE heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # RMSE heatmap
    pivot_rmse = metrics_df.pivot_table(values='RMSE', index='scenario', columns='sigma')
    im1 = axes[0].imshow(pivot_rmse.values, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(len(pivot_rmse.columns)))
    axes[0].set_xticklabels([f'{s}σ' for s in pivot_rmse.columns])
    axes[0].set_yticks(range(len(pivot_rmse.index)))
    axes[0].set_yticklabels(pivot_rmse.index)
    axes[0].set_title('RMSE (lower = better)')
    plt.colorbar(im1, ax=axes[0])
    for i in range(len(pivot_rmse.index)):
        for j in range(len(pivot_rmse.columns)):
            val = pivot_rmse.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color='white' if val > pivot_rmse.values[~np.isnan(pivot_rmse.values)].mean() else 'black', fontsize=8)
    
    # Bias heatmap
    pivot_bias = metrics_df.pivot_table(values='Bias', index='scenario', columns='sigma')
    im2 = axes[1].imshow(pivot_bias.values, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[1].set_xticks(range(len(pivot_bias.columns)))
    axes[1].set_xticklabels([f'{s}σ' for s in pivot_bias.columns])
    axes[1].set_yticks(range(len(pivot_bias.index)))
    axes[1].set_yticklabels(pivot_bias.index)
    axes[1].set_title('Bias (0 = unbiased, + = overestimate)')
    plt.colorbar(im2, ax=axes[1])
    for i in range(len(pivot_bias.index)):
        for j in range(len(pivot_bias.columns)):
            val = pivot_bias.values[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=8)
    
    # Correlation heatmap
    pivot_corr = metrics_df.pivot_table(values='Corr', index='scenario', columns='sigma')
    im3 = axes[2].imshow(pivot_corr.values, cmap='Greens', aspect='auto', vmin=0.5, vmax=1.0)
    axes[2].set_xticks(range(len(pivot_corr.columns)))
    axes[2].set_xticklabels([f'{s}σ' for s in pivot_corr.columns])
    axes[2].set_yticks(range(len(pivot_corr.index)))
    axes[2].set_yticklabels(pivot_corr.index)
    axes[2].set_title('Correlation (higher = better)')
    plt.colorbar(im3, ax=axes[2])
    for i in range(len(pivot_corr.index)):
        for j in range(len(pivot_corr.columns)):
            val = pivot_corr.values[i, j]
            if not np.isnan(val):
                axes[2].text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color='white' if val > 0.8 else 'black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_metrics_heatmap.png'), dpi=150)
    plt.show()
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"  Mean RMSE:        {metrics_df['RMSE'].mean():.4f}")
    print(f"  Mean Bias:        {metrics_df['Bias'].mean():+.4f}")
    print(f"  Mean Correlation: {metrics_df['Corr'].mean():.4f}")
    print(f"  Worst RMSE:       {metrics_df['RMSE'].max():.4f} ({metrics_df.loc[metrics_df['RMSE'].idxmax(), 'scenario']} @ {metrics_df.loc[metrics_df['RMSE'].idxmax(), 'sigma']:.0f}σ)")
    print(f"  Best Correlation: {metrics_df['Corr'].max():.4f}")
    print(f"  Worst Correlation:{metrics_df['Corr'].min():.4f}")
    print("="*70)
    
else:
    print("No CSV files found - run test_svpf_extreme.exe first")

# =====================================================================
# 5. Plot All Scenarios at 20σ
# =====================================================================

def plot_all_scenarios_at_sigma(sigma=20):
    """Compare all scenario types at a fixed sigma"""
    scenarios = ['Single_Spike', 'Double_Spike', 'Flash_Crash', 'Gradual_Build']
    
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3*len(scenarios)), sharex=False)
    
    for i, scenario in enumerate(scenarios):
        filename = f'svpf_stress_{scenario}_{sigma:.0f}sigma.csv'
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            
            ax1 = axes[i]
            ax2 = ax1.twinx()
            
            # Returns (gray background)
            ax1.fill_between(df['t'], df['y_t'], alpha=0.2, color='blue')
            
            # Vol (left axis)
            ax1.plot(df['t'], df['vol'] * 100, 'g-', linewidth=1.5, label='Vol %')
            ax1.set_ylabel('Vol (%)', color='green')
            ax1.set_ylim(bottom=0)
            
            # ESS (right axis)
            ax2.plot(df['t'], df['ess'], 'purple', linewidth=1, alpha=0.7, label='ESS')
            ax2.set_ylabel('ESS', color='purple')
            ax2.set_ylim(0, 550)
            
            ax1.set_title(f'{scenario.replace("_", " ")} @ {sigma:.0f}σ')
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f'File not found: {filepath}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    axes[-1].set_xlabel('Timestep')
    plt.tight_layout()
    return fig

try:
    fig = plot_all_scenarios_at_sigma(20)
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_all_scenarios_20sigma.png'), dpi=150)
    plt.show()
except:
    print("Files not found")

print("\n✓ Visualization complete")