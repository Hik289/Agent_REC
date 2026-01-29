#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Single Agent Learning-to-Rank Model Results
Reads single agent LTR results from the output directory and generates visualizations
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def safe_savefig(output_path, dpi=300, **kwargs):
    """Safely save figure with error handling for large images"""
    try:
        plt.savefig(output_path, dpi=dpi, **kwargs)
        return True
    except ValueError as e:
        if 'Image size' in str(e):
            print(f"⚠ Warning: Image size too large at DPI={dpi}, trying with lower DPI...")
            try:
                # Try with lower DPI
                plt.savefig(output_path, dpi=150, **kwargs)
                print(f"✓ Saved with DPI=150")
                return True
            except ValueError:
                # Try without bbox_inches='tight'
                print(f"⚠ Still too large, trying without bbox_inches='tight'...")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('bbox_inches', None)
                plt.savefig(output_path, dpi=150, **kwargs_copy)
                print(f"✓ Saved with DPI=150, no tight bbox")
                return True
        else:
            raise


def find_output_dir():
    """Smart path resolution to find output directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Locations to try (in order)
    search_paths = [
        # Relative to parent directory (../output from script location)
        os.path.join(parent_dir, 'output'),
        # Relative to current working directory
        'output',
        # Relative to script directory
        os.path.join(script_dir, 'output'),
        # Absolute path
        os.path.abspath('output')
    ]
    
    for path in search_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    
    # Default to parent/output
    return os.path.join(parent_dir, 'output')


def find_figure_dir():
    """Smart path resolution to find figure directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Default to parent/figure
    figure_dir = os.path.join(parent_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def load_results(output_dir):
    """Load result files from output directory"""
    results = {}
    
    # Load LTR model weights
    model_weights_path = os.path.join(output_dir, 'ltr_model_weights.json')
    if os.path.exists(model_weights_path):
        with open(model_weights_path, 'r', encoding='utf-8') as f:
            results['model_weights'] = json.load(f)
        print(f"✓ Loaded: {model_weights_path}")
    else:
        print(f"⚠ Not found: {model_weights_path}")
        results['model_weights'] = None
    
    # Load LTR test results
    test_results_path = os.path.join(output_dir, 'ltr_test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r', encoding='utf-8') as f:
            results['test_results'] = json.load(f)
        print(f"✓ Loaded: {test_results_path}")
    else:
        print(f"⚠ Not found: {test_results_path}")
        results['test_results'] = None
    
    # Load tool selection results
    selection_results_path = os.path.join(output_dir, 'tool_selection_results.json')
    if os.path.exists(selection_results_path):
        with open(selection_results_path, 'r', encoding='utf-8') as f:
            results['selection_results'] = json.load(f)
        print(f"✓ Loaded: {selection_results_path}")
    else:
        print(f"⚠ Not found: {selection_results_path}")
        results['selection_results'] = None
    
    return results


def visualize_weights(weights, interpretation, figure_dir):
    """Visualize feature weights with enhanced aesthetics"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    feature_names = [
        'φ_rel\n(Tool-Query\nRelevance)',
        'φ_hist\n(Historical\nReliability)',
        'φ_coop\n(Graph-Aware\nCompatibility)',
        'φ_struct\n(Structural\nUtility)'
    ]
    
    # Enhanced color palette
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    bars = ax.bar(feature_names, weights, color=colors, alpha=0.85, 
                  edgecolor='#2C3E50', linewidth=2, width=0.6)
    
    # Add gradient effect to bars
    for bar, color in zip(bars, colors):
        bar.set_linewidth(2)
        bar.set_edgecolor('#2C3E50')
    
    # Add value labels with better styling (handle both positive and negative weights)
    weight_range = max(weights) - min(weights)
    offset = max(0.1, weight_range * 0.02)  # Ensure reasonable offset
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        if height >= 0:
            y_pos = height + offset
            va = 'bottom'
        else:
            y_pos = height - offset
            va = 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{weight:.3f}',
                ha='center', va=va, fontsize=13, fontweight='bold',
                color='#2C3E50')
    
    ax.set_ylabel('Weight Value', fontsize=15, fontweight='bold', color='#2C3E50')
    ax.set_title('Single Agent LTR - Feature Weight Distribution', 
                 fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    
    # Set y-axis limits to handle both positive and negative weights
    y_min = min(0, min(weights) * 1.2)
    y_max = max(0, max(weights) * 1.2)
    if abs(y_max - y_min) < 0.1:  # Avoid too small range
        y_min, y_max = min(weights) - 0.5, max(weights) + 0.5
    ax.set_ylim(y_min, y_max)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='#34495E', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax.tick_params(colors='#2C3E50', labelsize=11)
    
    # Add annotation box with better styling (most influential by absolute value)
    abs_weights = [abs(w) for w in weights]
    max_idx = abs_weights.index(max(abs_weights))
    textstr = f'Most Influential Feature\n'
    textstr += f'{feature_names[max_idx].split()[0]}\n'
    textstr += f'Weight: {weights[max_idx]:.4f}'
    props = dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1', 
                edgecolor='#34495E', alpha=0.9, linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold', color='#2C3E50')
    
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'single_agent_weights.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Weights visualization saved to: {output_path}")
    plt.close()


def visualize_test_results(test_data, figure_dir):
    """Visualize test results with comparison of all methods including Oracle"""
    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor('white')
    
    # Check if we have all methods
    has_direct = 'direct_retrieval' in test_data
    has_oracle = 'oracle_retrieval' in test_data
    has_baseline = 'baseline' in test_data
    has_trained = 'trained_model' in test_data
    
    # Fallback to old format if new format not available
    if not has_trained and 'summary' in test_data:
        summary = test_data['summary']
        has_trained = True
        test_data['trained_model'] = summary
    
    if has_direct and has_oracle and has_baseline and has_trained:
        # New format with four methods comparison including Oracle
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Four methods comparison
        methods = ['Direct\nRetrieval', 'Oracle\n(Top-5)', 'LTR\nBaseline', 'LTR\nTrained']
        accuracies = [
            test_data['direct_retrieval']['top1_accuracy'] * 100,
            test_data['oracle_retrieval']['oracle_accuracy_top5'] * 100,
            test_data['baseline']['top1_accuracy'] * 100,
            test_data['trained_model']['top1_accuracy'] * 100
        ]
        colors = ['#95A5A6', '#9B59B6', '#E67E22', '#27AE60']  # Added purple for Oracle
        
        x = np.arange(len(methods))
        bars = ax1.bar(x, accuracies, color=colors, alpha=0.85, 
                      edgecolor='#2C3E50', linewidth=2.5, width=0.65)
        
        # Add special styling for Oracle (dashed pattern to indicate upper bound)
        bars[1].set_hatch('//')
        
        for bar, acc, method in zip(bars, accuracies, methods):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=13, fontweight='bold', color='#2C3E50')
        
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
        ax1.set_title('Method Comparison (Including Oracle Upper Bound)', fontsize=16, fontweight='bold', 
                      pad=15, color='#2C3E50')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
        ax1.set_axisbelow(True)
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.5)
        ax1.tick_params(colors='#2C3E50', labelsize=11)
        
        # Performance metrics display
        direct_acc = test_data['direct_retrieval']['top1_accuracy'] * 100
        oracle_acc = test_data['oracle_retrieval']['oracle_accuracy_top5'] * 100
        trained_acc = test_data['trained_model']['top1_accuracy'] * 100
        
        improvement = trained_acc - direct_acc
        gap_to_oracle = oracle_acc - trained_acc
        oracle_utilization = (trained_acc / oracle_acc * 100) if oracle_acc > 0 else 0
        
        ax2.text(0.5, 0.88, 'Performance Metrics', 
                ha='center', fontsize=15, fontweight='bold', 
                transform=ax2.transAxes, color='#2C3E50')
        
        # Improvement vs Direct Retrieval
        ax2.text(0.5, 0.72, f'+{improvement:.1f}%', 
                ha='center', fontsize=32, fontweight='bold', 
                color='#27AE60' if improvement > 0 else '#E74C3C',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.62, 'vs Direct Retrieval', 
                ha='center', fontsize=11, transform=ax2.transAxes, 
                color='#7F8C8D', style='italic')
        
        # Gap to Oracle
        ax2.text(0.5, 0.48, f'{gap_to_oracle:.1f}%', 
                ha='center', fontsize=24, fontweight='bold', 
                color='#E67E22',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.38, 'Gap to Oracle', 
                ha='center', fontsize=11, transform=ax2.transAxes, 
                color='#7F8C8D', style='italic')
        
        # Oracle Utilization
        ax2.text(0.5, 0.24, f'{oracle_utilization:.1f}%', 
                ha='center', fontsize=20, fontweight='bold', 
                color='#9B59B6',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.14, 'Oracle Utilization', 
                ha='center', fontsize=11, transform=ax2.transAxes, 
                color='#7F8C8D', style='italic')
        
        # Test info
        ax2.text(0.5, 0.02, f'Test Queries: {test_data["trained_model"]["num_test_queries"]} | '
                f'MRR: {test_data["trained_model"]["mrr"]:.4f}', 
                ha='center', fontsize=10, transform=ax2.transAxes, 
                color='#34495E')
        
        ax2.axis('off')
        
        plt.suptitle('Single Agent Tool Selection - Complete Method Comparison', 
                    fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
    
    elif has_direct and has_baseline and has_trained:
        # Fallback to three methods if Oracle not available
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Three methods comparison
        methods = ['Direct\nRetrieval', 'LTR\nBaseline', 'LTR\nTrained']
        accuracies = [
            test_data['direct_retrieval']['top1_accuracy'] * 100,
            test_data['baseline']['top1_accuracy'] * 100,
            test_data['trained_model']['top1_accuracy'] * 100
        ]
        colors = ['#95A5A6', '#E67E22', '#27AE60']
        
        x = np.arange(len(methods))
        bars = ax1.bar(x, accuracies, color=colors, alpha=0.85, 
                      edgecolor='#2C3E50', linewidth=2.5, width=0.6)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2C3E50')
        
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
        ax1.set_title('Method Comparison', fontsize=16, fontweight='bold', 
                      pad=15, color='#2C3E50')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
        ax1.set_axisbelow(True)
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.5)
        ax1.tick_params(colors='#2C3E50', labelsize=11)
        
        # Improvement display
        direct_acc = test_data['direct_retrieval']['top1_accuracy'] * 100
        trained_acc = test_data['trained_model']['top1_accuracy'] * 100
        improvement = trained_acc - direct_acc
        
        ax2.text(0.5, 0.85, 'Performance Gain', 
                ha='center', fontsize=15, fontweight='bold', 
                transform=ax2.transAxes, color='#2C3E50')
        
        ax2.text(0.5, 0.60, f'+{improvement:.1f}%', 
                ha='center', fontsize=42, fontweight='bold', 
                color='#27AE60' if improvement > 0 else '#E74C3C',
                transform=ax2.transAxes)
        
        ax2.text(0.5, 0.40, 'vs Direct Retrieval', 
                ha='center', fontsize=13, transform=ax2.transAxes, 
                color='#7F8C8D', style='italic')
        
        ax2.text(0.5, 0.20, f'Test Queries: {test_data["trained_model"]["num_test_queries"]}', 
                ha='center', fontsize=12, transform=ax2.transAxes, 
                fontweight='bold', color='#34495E')
        
        ax2.text(0.5, 0.08, f'MRR: {test_data["trained_model"]["mrr"]:.4f}', 
                ha='center', fontsize=11, transform=ax2.transAxes, 
                color='#7F8C8D')
        ax2.axis('off')
        
        plt.suptitle('Single Agent Tool Selection - Method Comparison', 
                    fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
    else:
        # Old format fallback
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        summary = test_data.get('trained_model', test_data.get('summary', {}))
        
        metrics = ['Top-1\nAccuracy', 'Top-3\nAccuracy']
        accuracies = [summary['top1_accuracy'] * 100, summary['top3_accuracy'] * 100]
        colors_acc = ['#27AE60', '#3498DB']
        
        bars = ax1.bar(metrics, accuracies, color=colors_acc, alpha=0.85, 
                       edgecolor='#2C3E50', linewidth=2.5, width=0.5)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=16, fontweight='bold', color='#2C3E50')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
        ax1.set_title('Test Set Accuracy', fontsize=16, fontweight='bold', 
                      pad=15, color='#2C3E50')
        ax1.set_ylim(0, 115)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
        ax1.set_axisbelow(True)
        ax1.axhline(y=100, color='#E74C3C', linestyle='--', linewidth=2.5, 
                    alpha=0.6, label='Perfect Score')
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='#BDC3C7')
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.5)
        ax1.tick_params(colors='#2C3E50', labelsize=12)
        
        ax2.text(0.5, 0.7, f'MRR: {summary["mrr"]:.4f}', 
                ha='center', fontsize=32, fontweight='bold', color='#E74C3C',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.4, f'Test Queries: {summary["num_test_queries"]}', 
                ha='center', fontsize=13, transform=ax2.transAxes, 
                fontweight='bold', color='#34495E')
        ax2.axis('off')
        
        plt.suptitle('Single Agent LTR - Test Set Performance', 
                    fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
    
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'single_agent_test_performance.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Test performance visualization saved to: {output_path}")
    plt.close()


def visualize_ground_truth_retrieval(selection_data, figure_dir):
    """Visualize ground truth retrieval statistics (Stage 1 success rate)"""
    if not selection_data or 'ground_truth_stats' not in selection_data:
        print("⚠ No ground truth statistics to visualize")
        return
    
    stats = selection_data['ground_truth_stats']
    if stats['total_with_ground_truth'] == 0:
        print("⚠ No ground truth data available")
        return
    
    # Analyze ground truth ranks
    ground_truth_ranks = []
    for sel in selection_data['selections']:
        if sel.get('ground_truth'):
            # Support both old and new formats
            candidates = sel.get('retrieved_candidates', sel.get('selected_tools', []))
            for i, tool in enumerate(candidates, 1):
                if tool.get('is_ground_truth'):
                    ground_truth_ranks.append(i)
                    break
    
    if not ground_truth_ranks:
        print("⚠ No ground truth rank data to visualize")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # Pie chart: Ground truth in candidates vs added
    in_candidates = stats.get('ground_truth_in_candidates', stats.get('ground_truth_in_top3', 0))
    not_in_candidates = stats.get('ground_truth_not_in_candidates', stats.get('ground_truth_not_in_top3', 0))
    total = stats['total_with_ground_truth']
    
    # Get n_candidates from results metadata if available
    n_candidates = 5  # default value
    
    sizes = [in_candidates, not_in_candidates]
    labels = [f'In Candidates\n({in_candidates}, {in_candidates/total*100:.1f}%)',
              f'Added Later\n({not_in_candidates}, {not_in_candidates/total*100:.1f}%)']
    colors = ['#2ECC71', '#E74C3C']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', shadow=True, startangle=90, 
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f'Ground Truth Retrieval\n(Stage 1: Retrieve {n_candidates} Candidates)', 
                  fontsize=15, fontweight='bold', pad=20)
    
    # Histogram: Ground truth rank distribution
    rank_counts = {}
    for rank in ground_truth_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]
    
    # Color bars: green for within candidates, red for added later
    bar_colors = ['#2ECC71' if r <= n_candidates else '#E74C3C' for r in ranks]
    
    bars = ax2.bar(ranks, counts, color=bar_colors, alpha=0.85, 
                   edgecolor='#2C3E50', linewidth=2, width=0.6)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Ground Truth Rank', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Ground Truth Rank Distribution', fontsize=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticks(ranks)
    ax2.set_ylim(0, max(counts) * 1.15)
    
    # Add success rate annotation
    success_rate = stats.get('candidate_retrieval_rate', stats.get('retrieval_success_rate', 0))
    textstr = f'Stage 1 Success Rate\n{success_rate:.1%}'
    props = dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1', 
                edgecolor='#34495E', alpha=0.9, linewidth=2)
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props, 
            fontweight='bold', color='#2C3E50')
    
    # Style the spines
    for spine in ax2.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax2.tick_params(colors='#2C3E50', labelsize=11)
    
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'single_agent_ground_truth_retrieval.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Ground truth retrieval statistics saved to: {output_path}")
    plt.close()


def visualize_selection_stats(selection_data, figure_dir):
    """Visualize tool selection statistics with enhanced aesthetics"""
    if not selection_data or 'selections' not in selection_data:
        print("⚠ No selection data to visualize")
        return
    
    selections = selection_data['selections']
    if not selections:
        return
    
    # Count top-ranked tools (selected as top-1)
    tool_counts = {}
    for sel in selections:
        # Support both new and old formats
        if 'selected_tool' in sel and sel['selected_tool']:
            # New format: selected_tool is a single dict
            top_tool = sel['selected_tool']['tool_name']
            tool_counts[top_tool] = tool_counts.get(top_tool, 0) + 1
        elif 'selected_tools' in sel and sel['selected_tools']:
            # Old format: selected_tools is a list
            top_tool = sel['selected_tools'][0]['tool_name']
            tool_counts[top_tool] = tool_counts.get(top_tool, 0) + 1
    
    # Get top 15 most selected tools
    sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if not sorted_tools:
        return
    
    tools, counts = zip(*sorted_tools)
    
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('white')
    
    # Enhanced color palette - gradient from dark to light
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tools)))
    bars = ax.barh(range(len(tools)), counts, color=colors, alpha=0.85, 
                   edgecolor='#2C3E50', linewidth=1.8)
    
    # Add value labels with better styling
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2.,
                f'{count}',
                ha='left', va='center', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='#34495E', alpha=0.9, linewidth=1.5),
                color='#2C3E50')
    
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools, fontsize=11, fontweight='500', color='#2C3E50')
    ax.set_xlabel('Selection Count', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_title('Top 15 Most Frequently Selected Tools\n(Embedding-based Similarity)', 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax.tick_params(colors='#2C3E50', labelsize=11)
    ax.invert_yaxis()  # Highest on top
    
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'single_agent_tool_selection_stats.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Tool selection statistics saved to: {output_path}")
    plt.close()


def print_summary(results):
    """Print results summary"""
    print("\n" + "="*80)
    print("Single Agent Learning-to-Rank - Results Summary")
    print("="*80)
    
    # Tool Selection Results
    if results['selection_results']:
        sel_data = results['selection_results']
        print("\n📋 Tool Selection Results (Step 1: Embedding-based):")
        print("-" * 60)
        print(f"  Total traces:        {sel_data.get('total_traces', 'N/A')}")
        print(f"  Total decisions:     {sel_data.get('total_decisions', 'N/A')}")
        print(f"  Method:              {sel_data.get('method', 'N/A')}")
        
        # Ground truth retrieval statistics
        if 'ground_truth_stats' in sel_data:
            gt_stats = sel_data['ground_truth_stats']
            print("\n🎯 Ground Truth Retrieval (Stage 1: Retrieve Candidates):")
            print("-" * 60)
            print(f"  Decisions with GT:      {gt_stats.get('total_with_ground_truth', 'N/A')}")
            
            # Support both old and new field names
            gt_in_candidates = gt_stats.get('ground_truth_in_candidates', gt_stats.get('ground_truth_in_top3', 'N/A'))
            gt_not_in_candidates = gt_stats.get('ground_truth_not_in_candidates', gt_stats.get('ground_truth_not_in_top3', 'N/A'))
            candidate_rate = gt_stats.get('candidate_retrieval_rate', gt_stats.get('retrieval_success_rate', 0))
            
            print(f"  GT in candidates:       {gt_in_candidates} ({candidate_rate*100:.1f}%)")
            print(f"  GT added later:         {gt_not_in_candidates} ({(1-candidate_rate)*100:.1f}%)")
            print(f"  Candidate retrieval:    {candidate_rate:.2%}")
            
            # Show selection accuracy if available
            if 'ground_truth_is_selected' in gt_stats:
                selection_acc = gt_stats.get('selection_accuracy', 0)
                gt_selected = gt_stats.get('ground_truth_is_selected', 'N/A')
                print(f"\n  GT is selected (top-1): {gt_selected} ({selection_acc*100:.1f}%)")
                print(f"  Selection accuracy:     {selection_acc:.2%}")
    
    # Model Weights
    if results['model_weights']:
        model_data = results['model_weights']
        weights = model_data['weights']
        print("\n📊 Learned Feature Weights (Step 2: Learning-to-Rank):")
        print("-" * 60)
        for i, (name, weight) in enumerate(zip(
            ['φ_rel (Relevance)', 'φ_hist (Reliability)', 'φ_coop (Compatibility)', 'φ_struct (Structure)'],
            weights
        )):
            marker = "⭐" if weight == max(weights) else "  "
            print(f"  {marker} w[{i}] = {weight:7.4f}  →  {name}")
    
    # Test Performance
    if results['test_results']:
        test_data = results['test_results']
        print("\n🎯 Test Set Performance Comparison:")
        print("-" * 60)
        
        # Check for three methods comparison
        if 'direct_retrieval' in test_data and 'baseline' in test_data and 'trained_model' in test_data:
            print(f"  Test queries: {test_data['trained_model']['num_test_queries']}")
            print(f"\n  {'Method':<45} {'Top-1 Acc':<15}")
            print(f"  {'-'*60}")
            print(f"  {'1. Direct Retrieval (no candidates)':<45} {test_data['direct_retrieval']['top1_accuracy']:.2%}")
            print(f"  {'2. LTR Baseline (initial weights, 5 cand.)':<45} {test_data['baseline']['top1_accuracy']:.2%}")
            print(f"  {'3. LTR Trained (trained weights, 5 cand.)':<45} {test_data['trained_model']['top1_accuracy']:.2%}")
            
            print(f"\n  {'Trained Model Details:':<45}")
            print(f"  {'  Top-3 accuracy:':<45} {test_data['trained_model']['top3_accuracy']:.2%}")
            print(f"  {'  MRR:':<45} {test_data['trained_model']['mrr']:.4f}")
            
            # Show improvements
            direct_acc = test_data['direct_retrieval']['top1_accuracy']
            baseline_acc = test_data['baseline']['top1_accuracy']
            trained_acc = test_data['trained_model']['top1_accuracy']
            
            print(f"\n  {'Improvements:':<45}")
            print(f"  {'  vs Direct Retrieval:':<45} {(trained_acc - direct_acc)*100:+.2f}%")
            print(f"  {'  vs LTR Baseline:':<45} {(trained_acc - baseline_acc)*100:+.2f}%")
            
            # Check for perfect performance
            if results['model_weights']:
                weights = results['model_weights']['weights']
                print("\n💡 Key Findings:")
                print("-" * 60)
                max_weight_idx = weights.index(max(weights))
                feature_names = ['Relevance', 'Reliability', 'Compatibility', 'Structure']
                print(f"  • Most important feature: φ_{['rel', 'hist', 'coop', 'struct'][max_weight_idx]} ({feature_names[max_weight_idx]})")
                print(f"  • Weight value: {max(weights):.4f}")
                
                if trained_acc == 1.0:
                    print(f"  • Perfect performance on test set!")
                    print("\n✅ Perfect! All test samples correctly predicted!")
        else:
            # Old format fallback
            summary = test_data.get('trained_model', test_data.get('summary', {}))
            print(f"  Test queries:        {summary['num_test_queries']}")
            print(f"  Top-1 accuracy:      {summary['top1_accuracy']:.2%}")
            print(f"  Top-3 accuracy:      {summary['top3_accuracy']:.2%}")
            print(f"  MRR:                 {summary['mrr']:.4f}")
            
            if results['model_weights']:
                weights = results['model_weights']['weights']
                print("\n💡 Key Findings:")
                print("-" * 60)
                max_weight_idx = weights.index(max(weights))
                feature_names = ['Relevance', 'Reliability', 'Compatibility', 'Structure']
                print(f"  • Most important feature: φ_{['rel', 'hist', 'coop', 'struct'][max_weight_idx]} ({feature_names[max_weight_idx]})")
                print(f"  • Weight value: {max(weights):.4f}")
                
                if summary['top1_accuracy'] == 1.0:
                    print(f"  • Perfect performance on test set!")
                    print("\n✅ Perfect! All test samples correctly predicted!")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Visualize Single Agent Learning-to-Rank Model Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default output directory (auto-detect ../output)
  python visualize_results.py
  
  # Specify custom output directory
  python visualize_results.py --output_dir /path/to/output
  
  # Only generate specific visualizations
  python visualize_results.py --skip-selection
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory containing results (default: auto-detect)'
    )
    
    parser.add_argument(
        '--skip-selection',
        action='store_true',
        help='Skip tool selection statistics visualization'
    )
    
    args = parser.parse_args()
    
    # Find output directory
    if args.output_dir is None:
        output_dir = find_output_dir()
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        print(f"✗ Error: Output directory not found: {output_dir}")
        print(f"  Please ensure the output directory exists or run the LTR model first.")
        return
    
    # Get figure directory
    figure_dir = find_figure_dir()
    
    print("="*80)
    print("Single Agent LTR Results Visualization")
    print("="*80)
    print(f"\nData directory:   {output_dir}")
    print(f"Figure directory: {figure_dir}\n")
    
    # Load results
    print("Loading results...")
    results = load_results(output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    if results['model_weights']:
        visualize_weights(
            results['model_weights']['weights'], 
            results['model_weights']['interpretation'],
            figure_dir
        )
    else:
        print("⚠ Skipping weights visualization (no model weights found)")
    
    if results['test_results']:
        visualize_test_results(results['test_results'], figure_dir)
    else:
        print("⚠ Skipping test results visualization (no test results found)")
    
    if results['selection_results'] and not args.skip_selection:
        visualize_ground_truth_retrieval(results['selection_results'], figure_dir)
        visualize_selection_stats(results['selection_results'], figure_dir)
    elif args.skip_selection:
        print("⊘ Skipping selection statistics (--skip-selection flag)")
    else:
        print("⚠ Skipping selection statistics (no selection results found)")
    
    # Print summary
    print_summary(results)
    
    print("✓ All visualizations complete!")
    print(f"  Visualizations saved to: {figure_dir}")


if __name__ == '__main__':
    # Set font for better compatibility
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()

