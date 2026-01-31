# SVPF Extreme Stress Test Visualization
# Run test_svpf_extreme.exe first to generate CSV files

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Set paths - script is in root/scripts, CSVs are in root/build/bin
DATA_DIR = "../build/bin"
OUTPUT_DIR = "../build/bin"  # Save plots alongside CSVs

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

print("\n" + "-"*60)
print("TRACKING ACCURACY (RMSE of h_estimated vs h_true):")
print("-"*60)

rmse_data = []
for scenario in ['Single_Spike', 'Double_Spike', 'Flash_Crash', 'Gradual_Build']:
    for sigma in [10, 20, 30, 40, 50]:
        filename = f'svpf_stress_{scenario}_{sigma:.0f}sigma.csv'
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            rmse = np.sqrt(((df['h_mean'] - df['true_h'])**2).mean())
            mae = (df['h_mean'] - df['true_h']).abs().mean()
            rmse_data.append({
                'scenario': scenario,
                'sigma': sigma,
                'RMSE': rmse,
                'MAE': mae
            })
        except FileNotFoundError:
            pass

if rmse_data:
    rmse_df = pd.DataFrame(rmse_data)
    pivot = rmse_df.pivot_table(values='RMSE', index='scenario', columns='sigma')
    print("\nRMSE by Scenario and Sigma:")
    print(pivot.round(3).to_string())
    
    # Plot RMSE heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{s:.0f}σ' for s in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title('Tracking Error (RMSE of h_est vs h_true)')
    plt.colorbar(im, ax=ax, label='RMSE')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color='white' if val > pivot.values.mean() else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stress_rmse_heatmap.png'), dpi=150)
    plt.show()
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
