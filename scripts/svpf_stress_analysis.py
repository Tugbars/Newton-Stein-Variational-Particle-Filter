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
    """Plot a single stress test scenario timeseries"""
    filepath = os.path.join(DATA_DIR, filename) if not os.path.isabs(filename) else filename
    df = pd.read_csv(filepath)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
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
    
    # Panel 2: Estimated volatility
    axes[1].plot(df['t'], df['vol'] * 100, 'g-', linewidth=1.5, label='Estimated Vol (%)')
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Base Vol (1%)')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(bottom=0)
    
    # Panel 3: ESS
    axes[2].fill_between(df['t'], df['ess'], alpha=0.3, color='purple')
    axes[2].plot(df['t'], df['ess'], 'purple', linewidth=1)
    axes[2].axhline(10, color='red', linestyle='--', alpha=0.7, label='Collapse threshold (ESS=10)')
    axes[2].axhline(512, color='green', linestyle='--', alpha=0.7, label='Max (N=512)')
    axes[2].set_ylabel('ESS')
    axes[2].set_xlabel('Timestep')
    axes[2].legend(loc='upper right')
    axes[2].set_ylim(0, 550)
    
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
    """Compare filter behavior across different sigma levels"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigmas)))
    
    for i, sigma in enumerate(sigmas):
        filename = f'svpf_stress_{scenario_prefix}_{sigma:.0f}sigma.csv'
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            axes[0].plot(df['t'], df['vol'] * 100, color=colors[i], 
                        linewidth=1.5, label=f'{sigma:.0f}σ', alpha=0.8)
            axes[1].plot(df['t'], df['ess'], color=colors[i], 
                        linewidth=1.5, label=f'{sigma:.0f}σ', alpha=0.8)
        except FileNotFoundError:
            print(f"Not found: {filepath}")
    
    axes[0].set_ylabel('Volatility (%)')
    axes[0].set_title(f'{scenario_prefix.replace("_", " ")}: Vol Tracking by Spike Magnitude')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(bottom=0)
    
    axes[1].set_ylabel('ESS')
    axes[1].set_xlabel('Timestep')
    axes[1].axhline(10, color='red', linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower right')
    axes[1].set_ylim(0, 550)
    
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
