#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def find_output_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    search_paths = [
        os.path.join(parent_dir, 'output'),
        'output',
        os.path.join(script_dir, 'output'),
        os.path.abspath('output')
    ]
    
    for path in search_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    
    return os.path.join(parent_dir, 'output')


def find_figure_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    figure_dir = os.path.join(parent_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def load_results(output_dir):
    results = {}
    
    model_weights_path = os.path.join(output_dir, 'graph_ltr_model_weights.json')
    if os.path.exists(model_weights_path):
        with open(model_weights_path, 'r', encoding='utf-8') as f:
            results['model_weights'] = json.load(f)
        print(f"✓ Loaded: {model_weights_path}")
    else:
        print(f"⚠ Not found: {model_weights_path}")
        results['model_weights'] = None
    
    test_results_path = os.path.join(output_dir, 'graph_ltr_test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r', encoding='utf-8') as f:
            results['test_results'] = json.load(f)
        print(f"✓ Loaded: {test_results_path}")
    else:
        print(f"⚠ Not found: {test_results_path}")
        results['test_results'] = None
    
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
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    feature_names = [
        'φ_rel\n(Semantic\nAlignment)',
        'φ_hist\n(System\nReliability)',
        'φ_coop\n(Internal\nCooperation)',
        'φ_struct\n(Structural\nUtility)'
    ]
    
    colors = []
    for w in weights:
        if w > 0:
            colors.append('#2ECC71')
        else:
            colors.append('#E74C3C')
    
    y_min = min(0, min(weights) * 1.3)
    y_max = max(weights) * 1.3
    
    bars = ax.bar(feature_names, weights, color=colors, alpha=0.85, 
                  edgecolor='#2C3E50', linewidth=2, width=0.6)
    
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
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax.tick_params(colors='#2C3E50', labelsize=11)
    
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


def visualize_test_results(summary, figure_dir):
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    metrics = ['Top-1', 'Top-3']
    accuracies = [summary['top1_accuracy'] * 100, summary['top3_accuracy'] * 100]
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
    
    ax2.text(0.5, 0.7, f'MRR (Mean Reciprocal Rank)', 
             ha='center', fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, f'{summary["mrr"]:.4f}', 
             ha='center', fontsize=40, fontweight='bold', color='#E74C3C',
             transform=ax2.transAxes)
    ax2.text(0.5, 0.25, f'Test Queries: {summary["num_test_queries"]}', 
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
    if not selection_data or 'traces' not in selection_data:
        print("⚠ No selection data to visualize")
        return
    
    ground_truth_ranks = []
    top3_count = 0
    added_count = 0
    total_nodes = 0
    
    for trace in selection_data['traces']:
        for node_sel in trace.get('node_selections', []):
            total_nodes += 1
            if node_sel.get('ground_truth_in_top3', False):
                top3_count += 1
            if node_sel.get('ground_truth_added', False):
                added_count += 1
            
            for i, cand in enumerate(node_sel.get('selected_candidates', []), 1):
                if cand.get('type') == 'ground_truth':
                    ground_truth_ranks.append(i)
                    break
    
    if not ground_truth_ranks:
        print("⚠ No ground truth data to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sizes = [top3_count, added_count]
    labels = [f'In Top-3\n({top3_count}, {top3_count/total_nodes*100:.1f}%)',
              f'Added\n({added_count}, {added_count/total_nodes*100:.1f}%)']
    colors = ['#2ECC71', '#E74C3C']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Ground Truth Distribution', fontsize=15, fontweight='bold', pad=20)
    
    rank_counts = {}
    for rank in ground_truth_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]
    
    colors_bar = ['#2ECC71' if r <= 3 else '#E74C3C' for r in ranks]
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
    
    plt.suptitle('Graph Retrieval Statistics', fontsize=19, fontweight='bold', y=0.98, color='#2C3E50')
    plt.tight_layout()
    output_path = os.path.join(figure_dir, 'multi_agent_system_graph_selection_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph selection statistics saved to: {output_path}")
    plt.close()


def print_summary(results):
    print("\n" + "="*80)
    print("Multi-Agent System Learning-to-Rank - Results Summary")
    print("="*80)
    
    if results['selection_results']:
        sel_data = results['selection_results']
        metadata = sel_data.get('metadata', {})
        print("\n📋 Graph Selection Results (Step 1: Embedding-based Retrieval):")
        print("-" * 60)
        print(f"  Total traces:            {metadata.get('total_traces', 'N/A')}")
        print(f"  Total nodes:             {metadata.get('total_nodes', 'N/A')}")
        print(f"  Method:                  {metadata.get('method', 'N/A')}")
        print(f"  Ground truth in top-3:   {metadata.get('ground_truth_in_top3', 'N/A')} "
              f"({metadata.get('ground_truth_in_top3', 0)/metadata.get('total_nodes', 1)*100:.1f}%)")
        print(f"  Ground truth added:      {metadata.get('ground_truth_added', 'N/A')} "
              f"({metadata.get('ground_truth_added', 0)/metadata.get('total_nodes', 1)*100:.1f}%)")
    
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
    
    if results['test_results']:
        test_data = results['test_results']
        print("\n🎯 Test Set Performance:")
        print("-" * 60)
        summary = test_data['summary']
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
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Agent System Learning-to-Rank Model Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py
  
  python visualize_results.py --output_dir /path/to/output
  
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
    
    if args.output_dir is None:
        output_dir = find_output_dir()
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        print(f"✗ Error: Output directory not found: {output_dir}")
        print(f"  Please ensure the output directory exists or run the graph LTR model first.")
        return
    
    figure_dir = find_figure_dir()
    
    print("="*80)
    print("Multi-Agent System LTR Results Visualization")
    print("="*80)
    print(f"\nData directory:   {output_dir}")
    print(f"Figure directory: {figure_dir}\n")
    
    print("Loading results...")
    results = load_results(output_dir)
    
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
        visualize_test_results(results['test_results']['summary'], figure_dir)
    else:
        print("⚠ Skipping test results visualization (no test results found)")
    
    if results['selection_results'] and not args.skip_selection:
        visualize_graph_selection_stats(results['selection_results'], figure_dir)
    elif args.skip_selection:
        print("⊘ Skipping selection statistics (--skip-selection flag)")
    else:
        print("⚠ Skipping selection statistics (no selection results found)")
    
    print_summary(results)
    
    print("✓ All visualizations complete!")
    print(f"  Visualizations saved to: {figure_dir}")


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
