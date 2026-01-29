#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Multi-Agent System Learning-to-Rank Model Results
Reads graph-level LTR results from the output directory and generates visualizations
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Non-interactive backend


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
    
    # Load Graph LTR model weights
    model_weights_path = os.path.join(output_dir, 'graph_ltr_model_weights.json')
    if os.path.exists(model_weights_path):
        with open(model_weights_path, 'r', encoding='utf-8') as f:
            results['model_weights'] = json.load(f)
        print(f"✓ Loaded: {model_weights_path}")
    else:
        print(f"⚠ Not found: {model_weights_path}")
        results['model_weights'] = None
    
    # Load Graph LTR test results
    test_results_path = os.path.join(output_dir, 'graph_ltr_test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r', encoding='utf-8') as f:
            results['test_results'] = json.load(f)
        print(f"✓ Loaded: {test_results_path}")
    else:
        print(f"⚠ Not found: {test_results_path}")
        results['test_results'] = None
    
    # Load graph selection results
    selection_results_path = os.path.join(output_dir, 'graph_selection_results.json')
    if os.path.exists(selection_results_path):
        with open(selection_results_path, 'r', encoding='utf-8') as f:
            results['selection_results'] = json.load(f)
        print(f"✓ Loaded: {selection_results_path}")
    else:
        print(f"⚠ Not found: {selection_results_path}")
        results['selection_results'] = None
    
    return results


def visualize_weights(weights, interpretation, figure_dir):
    """Visualize feature weights for graph-level LTR with enhanced aesthetics"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    feature_names = [
        'φ_rel\n(Semantic\nAlignment)',
        'φ_hist\n(System\nReliability)',
        'φ_coop\n(Internal\nCooperation)',
        'φ_struct\n(Structural\nUtility)'
    ]
    
    # Enhanced colors - differentiate positive and negative
    colors = []
    for w in weights:
        if w > 0:
            colors.append('#2ECC71')  # Green for positive
        else:
            colors.append('#E74C3C')  # Red for negative
    
    # Handle negative weights for visualization
    y_min = min(0, min(weights) * 1.3)
    y_max = max(weights) * 1.3
    
    bars = ax.bar(feature_names, weights, color=colors, alpha=0.85, 
                  edgecolor='#2C3E50', linewidth=2, width=0.6)
    
    # Add value labels with better styling
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            y_pos = height + (y_max * 0.02)
        else:
            va = 'top'
            y_pos = height - (abs(y_min) * 0.02)
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{weight:.3f}',
                ha='center', va=va, fontsize=13, fontweight='bold', color='#2C3E50')
    
    ax.set_ylabel('Weight Value', fontsize=15, fontweight='bold', color='#2C3E50')
    ax.set_title('Multi-Agent System LTR - Feature Weight Distribution', 
                 fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color='#34495E', linestyle='-', linewidth=2)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax.tick_params(colors='#2C3E50', labelsize=11)
    
    # Add annotation for most important feature (by absolute value)
    abs_weights = [abs(w) for w in weights]
    max_idx = abs_weights.index(max(abs_weights))
    sign = '+' if weights[max_idx] > 0 else ''
    textstr = f'Most Important Feature\n'
    textstr += f'{feature_names[max_idx].split()[0]}\n'
    textstr += f'Weight: {sign}{weights[max_idx]:.4f}'
    props = dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1', 
                edgecolor='#34495E', alpha=0.9, linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold', color='#2C3E50')
    
    plt.tight_layout()
    output_path = os.path.join(figure_dir, 'multi_agent_system_weights.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Weights visualization saved to: {output_path}")
    plt.close()


def visualize_test_results(trained_summary, figure_dir, baseline_summary=None, direct_retrieval_summary=None, oracle_retrieval_summary=None):
    """Visualize test results for graph-level LTR with all baselines including Oracle"""
    if baseline_summary and direct_retrieval_summary and oracle_retrieval_summary:
        # Create comparison visualization with four methods
        fig = plt.figure(figsize=(20, 8))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Four-method comparison bar chart (Top-1 Accuracy only for clarity)
        methods = ['Direct\nRetrieval', 'Oracle\nRetrieval', 'LTR\nBaseline', 'LTR\nTrained']
        accuracies = [
            direct_retrieval_summary['top1_accuracy'] * 100,
            oracle_retrieval_summary.get('oracle_accuracy_all', 0) * 100,  # Use 'all' for fair comparison with LTR
            baseline_summary['top1_accuracy'] * 100,
            trained_summary['top1_accuracy'] * 100
        ]
        colors = ['#95A5A6', '#3498DB', '#E67E22', '#27AE60']
        
        x = np.arange(len(methods))
        bars = ax1.bar(x, accuracies, color=colors, alpha=0.85,
                      edgecolor='#2C3E50', linewidth=2.5, width=0.6)
        
        # Add diagonal lines for Oracle Retrieval to indicate it's an upper bound
        bars[1].set_hatch('///')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2C3E50')
        
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
        ax1.set_title('Multi-Agent System Method Comparison', fontsize=16, fontweight='bold',
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
        
        # Performance metrics panel
        direct_acc = direct_retrieval_summary['top1_accuracy'] * 100
        trained_acc = trained_summary['top1_accuracy'] * 100
        oracle_acc = oracle_retrieval_summary.get('oracle_accuracy_all', 0) * 100  # Use 'all' for fair comparison
        
        improvement_vs_direct = trained_acc - direct_acc
        gap_to_oracle = oracle_acc - trained_acc
        oracle_utilization = (trained_acc / oracle_acc) * 100 if oracle_acc > 0 else 0
        
        ax2.text(0.5, 0.90, 'Performance Analysis',
                ha='center', fontsize=15, fontweight='bold',
                transform=ax2.transAxes, color='#2C3E50')
        
        ax2.text(0.5, 0.75, f'Gain vs Direct: {improvement_vs_direct:.1f}pp',
                ha='center', fontsize=14, fontweight='bold',
                color='#27AE60' if improvement_vs_direct > 0 else '#E74C3C',
                transform=ax2.transAxes)
        
        ax2.text(0.5, 0.60, f'Gap to Oracle: {gap_to_oracle:.1f}pp',
                ha='center', fontsize=14, fontweight='bold',
                color='#E67E22' if gap_to_oracle > 0 else '#27AE60',
                transform=ax2.transAxes)
        
        ax2.text(0.5, 0.45, f'Oracle Utilization: {oracle_utilization:.1f}%',
                ha='center', fontsize=14, fontweight='bold',
                color='#3498DB',
                transform=ax2.transAxes)
        
        ax2.text(0.5, 0.25, f'Test Queries: {trained_summary["num_test_queries"]}',
                ha='center', fontsize=12, transform=ax2.transAxes,
                fontweight='bold', color='#34495E')
        
        ax2.text(0.5, 0.15, f'MRR (Trained): {trained_summary["mrr"]:.4f}',
                ha='center', fontsize=11, transform=ax2.transAxes,
                color='#7F8C8D')
        ax2.axis('off')
        
        plt.suptitle('Multi-Agent System LTR - Performance Comparison', 
                    fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
        
    elif baseline_summary:
        # Two-method comparison (old format, for compatibility)
        fig = plt.figure(figsize=(18, 7))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Grouped bar chart comparing baseline vs trained
        metrics = ['Top-1', 'Top-3']
        baseline_acc = [baseline_summary['top1_accuracy'] * 100, baseline_summary['top3_accuracy'] * 100]
        trained_acc = [trained_summary['top1_accuracy'] * 100, trained_summary['top3_accuracy'] * 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline (Untrained)',
                       color='#95A5A6', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, trained_acc, width, label='Trained Model',
                       color=['#2ECC71', '#3498DB'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Baseline vs Trained Model', fontsize=15, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11)
        
        # Improvement bar chart
        improvements = [(trained_summary['top1_accuracy'] - baseline_summary['top1_accuracy']) * 100,
                       (trained_summary['top3_accuracy'] - baseline_summary['top3_accuracy']) * 100]
        colors_imp = ['#27AE60', '#2980B9']
        
        bars = ax2.bar(metrics, improvements, color=colors_imp, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{imp:.1f}%',
                    ha='center', va='bottom' if imp > 0 else 'top', 
                    fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Performance Gain', fontsize=15, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # MRR comparison and stats
        ax3.text(0.5, 0.85, 'MRR Comparison', 
                ha='center', fontsize=13, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.5, 0.68, f'Baseline: {baseline_summary["mrr"]:.4f}', 
                ha='center', fontsize=11, color='#95A5A6', transform=ax3.transAxes)
        ax3.text(0.5, 0.55, f'Trained: {trained_summary["mrr"]:.4f}', 
                ha='center', fontsize=11, color='#E74C3C', fontweight='bold', transform=ax3.transAxes)
        
        mrr_improvement = trained_summary["mrr"] - baseline_summary["mrr"]
        ax3.text(0.5, 0.40, f'Gain: +{mrr_improvement:.4f}', 
                ha='center', fontsize=11, color='#27AE60', fontweight='bold', transform=ax3.transAxes)
        
        ax3.text(0.5, 0.20, f'Test Queries: {trained_summary["num_test_queries"]}', 
                ha='center', fontsize=10, transform=ax3.transAxes)
        ax3.axis('off')
        
        plt.suptitle('Multi-Agent System LTR - Training Impact', 
                    fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
        
    else:
        # Original visualization without baseline
        fig = plt.figure(figsize=(16, 7))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Accuracy bar chart
        metrics = ['Top-1', 'Top-3']
        accuracies = [trained_summary['top1_accuracy'] * 100, trained_summary['top3_accuracy'] * 100]
        colors_acc = ['#2ECC71', '#3498DB']
        
        bars = ax1.bar(metrics, accuracies, color=colors_acc, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Test Set Accuracy', fontsize=15, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                    alpha=0.5, label='Perfect Performance')
        ax1.legend()
        
        # MRR and test sample count
        ax2.text(0.5, 0.7, f'MRR (Mean Reciprocal Rank)', 
                 ha='center', fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.5, 0.5, f'{trained_summary["mrr"]:.4f}', 
                 ha='center', fontsize=40, fontweight='bold', color='#E74C3C',
                 transform=ax2.transAxes)
        ax2.text(0.5, 0.25, f'Test Queries: {trained_summary["num_test_queries"]}', 
                 ha='center', fontsize=12, transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.suptitle('Multi-Agent System LTR - Test Set Performance', 
                     fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
    
    plt.tight_layout()
    output_path = os.path.join(figure_dir, 'multi_agent_system_test_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Test performance visualization saved to: {output_path}")
    plt.close()


def visualize_graph_selection_stats(selection_data, figure_dir):
    """Visualize graph selection statistics"""
    if not selection_data or 'traces' not in selection_data:
        print("⚠ No selection data to visualize")
        return
    
    # Get top_k value from metadata
    metadata = selection_data.get('metadata', {})
    top_k = metadata.get('top_k', 3)
    
    # Analyze selection data
    ground_truth_ranks = []
    topk_count = 0
    added_count = 0
    total_nodes = 0
    
    for trace in selection_data['traces']:
        for node_sel in trace.get('node_selections', []):
            total_nodes += 1
            # Support both old (top3) and new (topk) field names
            in_topk = node_sel.get('ground_truth_in_topk', node_sel.get('ground_truth_in_top3', False))
            if in_topk:
                topk_count += 1
            if node_sel.get('ground_truth_added', False):
                added_count += 1
            
            # Find ground truth rank
            for i, cand in enumerate(node_sel.get('selected_candidates', []), 1):
                if cand.get('type') == 'ground_truth':
                    ground_truth_ranks.append(i)
                    break
    
    if not ground_truth_ranks:
        print("⚠ No ground truth data to visualize")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # Pie chart: Ground truth in top-k vs added
    # Only include non-zero parts
    sizes = []
    labels = []
    colors_pie = []
    explode_list = []
    
    if topk_count > 0:
        sizes.append(topk_count)
        labels.append(f'In Top-{top_k}\n({topk_count}, {topk_count/total_nodes*100:.1f}%)')
        colors_pie.append('#2ECC71')
        explode_list.append(0.05)
    
    if added_count > 0:
        sizes.append(added_count)
        labels.append(f'Added\n({added_count}, {added_count/total_nodes*100:.1f}%)')
        colors_pie.append('#E74C3C')
        explode_list.append(0.05)
    
    if sizes:
        ax1.pie(sizes, explode=explode_list, labels=labels, colors=colors_pie,
                autopct='', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
    
    ax1.set_title(f'Ground Truth Distribution (Top-{top_k})', fontsize=15, fontweight='bold', pad=20)
    
    # Histogram: Ground truth rank distribution
    rank_counts = {}
    for rank in ground_truth_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]
    
    colors_bar = ['#2ECC71' if r <= top_k else '#E74C3C' for r in ranks]
    bars = ax2.bar(ranks, counts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Ground Truth Rank', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Ground Truth Rank Distribution', fontsize=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticks(ranks)
    
    # Add a vertical line at top_k position
    if top_k in ranks or (len(ranks) > 0 and top_k <= max(ranks)):
        ax2.axvline(x=top_k + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                   label=f'Top-{top_k} cutoff')
        ax2.legend()
    
    plt.suptitle(f'Graph Retrieval Statistics (Top-{top_k})', fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
    plt.tight_layout()
    output_path = os.path.join(figure_dir, 'multi_agent_system_graph_selection_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph selection statistics saved to: {output_path}")
    plt.close()


def print_summary(results):
    """Print results summary"""
    print("\n" + "="*80)
    print("Multi-Agent System Learning-to-Rank - Results Summary")
    print("="*80)
    
    # Graph Selection Results
    if results['selection_results']:
        sel_data = results['selection_results']
        metadata = sel_data.get('metadata', {})
        print("\n📋 Graph Selection Results (Step 1: Embedding-based Retrieval):")
        print("-" * 60)
        print(f"  Total traces:            {metadata.get('total_traces', 'N/A')}")
        print(f"  Total nodes:             {metadata.get('total_nodes', 'N/A')}")
        print(f"  Method:                  {metadata.get('method', 'N/A')}")
        print(f"  Top-k retrieved:         {metadata.get('top_k', 'N/A')}")
        
        # Support both old (top3) and new (topk) field names
        gt_in_topk = metadata.get('ground_truth_in_topk', metadata.get('ground_truth_in_top3', 'N/A'))
        total_nodes = metadata.get('total_nodes', 1)
        print(f"  Ground truth in top-k:   {gt_in_topk} "
              f"({gt_in_topk/total_nodes*100:.1f}%)")
        print(f"  Ground truth added:      {metadata.get('ground_truth_added', 'N/A')} "
              f"({metadata.get('ground_truth_added', 0)/total_nodes*100:.1f}%)")
    
    # Model Weights
    if results['model_weights']:
        model_data = results['model_weights']
        weights = model_data['weights']
        print("\n📊 Learned Feature Weights (Step 2: Graph-Level LTR):")
        print("-" * 60)
        feature_names = [
            'φ_rel (Semantic Alignment)',
            'φ_hist (System Reliability)',
            'φ_coop (Internal Cooperation)',
            'φ_struct (Structural Utility)'
        ]
        
        abs_weights = [abs(w) for w in weights]
        max_abs_idx = abs_weights.index(max(abs_weights))
        
        for i, (name, weight) in enumerate(zip(feature_names, weights)):
            marker = "⭐" if i == max_abs_idx else "  "
            print(f"  {marker} w[{i}] = {weight:7.4f}  →  {name}")
    
    # Test Performance
    if results['test_results']:
        test_data = results['test_results']
        
        # Check for new format with four methods (including Oracle)
        if 'trained_model' in test_data and 'baseline' in test_data and 'direct_retrieval' in test_data:
            print("\n🎯 Test Set Performance Comparison:")
            print("-" * 60)
            direct = test_data['direct_retrieval']
            baseline = test_data['baseline']
            trained = test_data['trained_model']
            oracle = test_data.get('oracle_retrieval')
            
            print(f"  Test queries: {trained['num_test_queries']}")
            print(f"\n  {'Method':<55} {'Top-1 Acc':<15}")
            print(f"  {'-'*70}")
            print(f"  {'1. Direct Retrieval (20 cand., no filter)':<55} {direct['top1_accuracy']:.2%}")
            if oracle:
                oracle_acc = oracle.get('oracle_accuracy_all', 0)
                print(f"  {'2. Oracle Retrieval (best from retrieved cand.)':<55} {oracle_acc:.2%}  [Upper Bound]")
            print(f"  {'3. LTR Baseline (10 cand., init weights)':<55} {baseline['top1_accuracy']:.2%}")
            print(f"  {'4. LTR Trained (10 cand., trained weights)':<55} {trained['top1_accuracy']:.2%}")
            
            print(f"\n  {'Improvements:':<55}")
            print(f"  {'  vs Direct Retrieval:':<55} {(trained['top1_accuracy']-direct['top1_accuracy'])*100:+.2f}%")
            print(f"  {'  vs LTR Baseline:':<55} {(trained['top1_accuracy']-baseline['top1_accuracy'])*100:+.2f}%")
            if oracle:
                oracle_acc = oracle.get('oracle_accuracy_all', 0)
                print(f"  {'  Gap to Oracle (room for improvement):':<55} {(oracle_acc-trained['top1_accuracy'])*100:.2f}%")
            
            summary = trained  # For compatibility with rest of the code
        elif 'trained_model' in test_data and 'baseline' in test_data:
            # Two-method comparison (old format, for compatibility)
            print("\n🎯 Test Set Performance:")
            print("-" * 60)
            baseline = test_data['baseline']
            trained = test_data['trained_model']
            improvement = test_data.get('improvement', {})
            
            print(f"  Test queries:        {trained['num_test_queries']}")
            print(f"\n  📊 Baseline (Untrained):")
            print(f"     Top-1 accuracy:   {baseline['top1_accuracy']:.2%}")
            print(f"     Top-3 accuracy:   {baseline['top3_accuracy']:.2%}")
            print(f"     MRR:              {baseline['mrr']:.4f}")
            
            print(f"\n  🎓 Trained Model:")
            print(f"     Top-1 accuracy:   {trained['top1_accuracy']:.2%}")
            print(f"     Top-3 accuracy:   {trained['top3_accuracy']:.2%}")
            print(f"     MRR:              {trained['mrr']:.4f}")
            
            if improvement:
                print(f"\n  📈 Improvement:")
                print(f"     Top-1 accuracy:   +{improvement['top1_accuracy']*100:.2f}%")
                print(f"     Top-3 accuracy:   +{improvement['top3_accuracy']*100:.2f}%")
                print(f"     MRR:              +{improvement['mrr']:.4f}")
            
            summary = trained  # For compatibility with rest of the code
        else:
            # Old format without baseline
            print("\n🎯 Test Set Performance:")
            print("-" * 60)
            summary = test_data.get('summary', test_data.get('trained_model', {}))
            print(f"  Test queries:        {summary['num_test_queries']}")
            print(f"  Top-1 accuracy:      {summary['top1_accuracy']:.2%}")
            print(f"  Top-3 accuracy:      {summary['top3_accuracy']:.2%}")
            print(f"  MRR:                 {summary['mrr']:.4f}")
        
        if results['model_weights']:
            weights = results['model_weights']['weights']
            print("\n💡 Key Findings:")
            print("-" * 60)
            abs_weights = [abs(w) for w in weights]
            max_abs_idx = abs_weights.index(max(abs_weights))
            feature_names_short = ['Semantic Alignment', 'System Reliability', 
                                  'Internal Cooperation', 'Structural Utility']
            print(f"  • Most important feature: φ_{['rel', 'hist', 'coop', 'struct'][max_abs_idx]} "
                  f"({feature_names_short[max_abs_idx]})")
            print(f"  • Weight value: {weights[max_abs_idx]:.4f}")
            
            # Identify negative weights
            negative_weights = [(i, w) for i, w in enumerate(weights) if w < 0]
            if negative_weights:
                print(f"  • Negative weights indicate reverse preference:")
                for idx, w in negative_weights:
                    print(f"    - φ_{['rel', 'hist', 'coop', 'struct'][idx]}: {w:.4f}")
            
            if summary['top1_accuracy'] == 1.0:
                print(f"  • Perfect performance on test set!")
                print("\n✅ Perfect! All test samples correctly predicted!")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Agent System Learning-to-Rank Model Results',
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
        help='Skip graph selection statistics visualization'
    )
    
    args = parser.parse_args()
    
    # Find output directory
    if args.output_dir is None:
        output_dir = find_output_dir()
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        print(f"✗ Error: Output directory not found: {output_dir}")
        print(f"  Please ensure the output directory exists or run the graph LTR model first.")
        return
    
    # Get figure directory
    figure_dir = find_figure_dir()
    
    print("="*80)
    print("Multi-Agent System LTR Results Visualization")
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
        # Extract trained model summary, baseline, direct retrieval, and oracle retrieval if available
        trained_summary = results['test_results'].get('trained_model')
        baseline_summary = results['test_results'].get('baseline')
        direct_retrieval_summary = results['test_results'].get('direct_retrieval')
        oracle_retrieval_summary = results['test_results'].get('oracle_retrieval')
        
        # Fall back to old format if new format not available
        if trained_summary is None:
            trained_summary = results['test_results'].get('summary')
        
        if trained_summary:
            visualize_test_results(trained_summary, figure_dir, baseline_summary, 
                                 direct_retrieval_summary, oracle_retrieval_summary)
        else:
            print("⚠ Skipping test results visualization (no valid summary found)")
    else:
        print("⚠ Skipping test results visualization (no test results found)")
    
    if results['selection_results'] and not args.skip_selection:
        visualize_graph_selection_stats(results['selection_results'], figure_dir)
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
