#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def safe_savefig(output_path, dpi=300, **kwargs):
    try:
        plt.savefig(output_path, dpi=dpi, **kwargs)
        return True
    except ValueError as e:
        if 'Image size' in str(e):
            print(f"⚠ Warning: Image size too large at DPI={dpi}, trying with lower DPI...")
            try:
                plt.savefig(output_path, dpi=150, **kwargs)
                print(f"✓ Saved with DPI=150")
                return True
            except ValueError:
                print(f"⚠ Still too large, trying without bbox_inches='tight'...")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('bbox_inches', None)
                plt.savefig(output_path, dpi=150, **kwargs_copy)
                print(f"✓ Saved with DPI=150, no tight bbox")
                return True
        else:
            raise


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
    
    model_weights_path = os.path.join(output_dir, 'ltr_model_weights.json')
    if os.path.exists(model_weights_path):
        with open(model_weights_path, 'r', encoding='utf-8') as f:
            results['model_weights'] = json.load(f)
        print(f"✓ Loaded: {model_weights_path}")
    else:
        print(f"⚠ Not found: {model_weights_path}")
        results['model_weights'] = None
    
    test_results_path = os.path.join(output_dir, 'ltr_test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r', encoding='utf-8') as f:
            results['test_results'] = json.load(f)
        print(f"✓ Loaded: {test_results_path}")
    else:
        print(f"⚠ Not found: {test_results_path}")
        results['test_results'] = None
    
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
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    feature_names = [
        'φ_rel\n(Tool-Query\nRelevance)',
        'φ_hist\n(Historical\nReliability)',
        'φ_coop\n(Graph-Aware\nCompatibility)',
        'φ_struct\n(Structural\nUtility)'
    ]
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    bars = ax.bar(feature_names, weights, color=colors, alpha=0.85, 
                  edgecolor='#2C3E50', linewidth=2, width=0.6)
    
    for bar, color in zip(bars, colors):
        bar.set_linewidth(2)
        bar.set_edgecolor('#2C3E50')
    
    weight_range = max(weights) - min(weights)
    offset = max(0.1, weight_range * 0.02)
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
    
    y_min = min(0, min(weights) * 1.2)
    y_max = max(0, max(weights) * 1.2)
    if abs(y_max - y_min) < 0.1:
        y_min, y_max = min(weights) - 0.5, max(weights) + 0.5
    ax.set_ylim(y_min, y_max)
    
    ax.axhline(y=0, color='#34495E', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax.tick_params(colors='#2C3E50', labelsize=11)
    
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


def visualize_test_results(summary, figure_dir):
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
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
    
    ax2.add_patch(plt.Circle((0.5, 0.55), 0.25, color='#ECF0F1', 
                             transform=ax2.transAxes, zorder=0))
    ax2.text(0.5, 0.82, 'Mean Reciprocal Rank', 
             ha='center', fontsize=14, fontweight='bold', 
             transform=ax2.transAxes, color='#2C3E50')
    ax2.text(0.5, 0.55, f'{summary["mrr"]:.4f}', 
             ha='center', fontsize=48, fontweight='bold', color='#E74C3C',
             transform=ax2.transAxes)
    
    box = plt.Rectangle((0.15, 0.35), 0.7, 0.35, transform=ax2.transAxes,
                        fill=False, edgecolor='#BDC3C7', linewidth=2.5, 
                        linestyle='-', zorder=1)
    ax2.add_patch(box)
    
    ax2.text(0.5, 0.2, f'Test Queries: {summary["num_test_queries"]}', 
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
    if not selection_data or 'ground_truth_stats' not in selection_data:
        print("⚠ No ground truth statistics to visualize")
        return
    
    stats = selection_data['ground_truth_stats']
    if stats['total_with_ground_truth'] == 0:
        print("⚠ No ground truth data available")
        return
    
    ground_truth_ranks = []
    for sel in selection_data['selections']:
        if sel.get('ground_truth'):
            for i, tool in enumerate(sel['selected_tools'], 1):
                if tool.get('is_ground_truth'):
                    ground_truth_ranks.append(i)
                    break
    
    if not ground_truth_ranks:
        print("⚠ No ground truth rank data to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    in_top3 = stats['ground_truth_in_top3']
    not_in_top3 = stats['ground_truth_not_in_top3']
    total = stats['total_with_ground_truth']
    
    sizes = [in_top3, not_in_top3]
    labels = [f'In Top-3\n({in_top3}, {in_top3/total*100:.1f}%)',
              f'Added as 4th\n({not_in_top3}, {not_in_top3/total*100:.1f}%)']
    colors = ['#2ECC71', '#E74C3C']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', shadow=True, startangle=90, 
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Ground Truth Retrieval\n(Stage 1: Embedding Selection)', 
                  fontsize=15, fontweight='bold', pad=20)
    
    rank_counts = {}
    for rank in ground_truth_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]
    
    bar_colors = ['#2ECC71' if r <= 3 else '#E74C3C' for r in ranks]
    
    bars = ax2.bar(ranks, counts, color=bar_colors, alpha=0.85, 
                   edgecolor='#2C3E50', linewidth=2, width=0.6)
    
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
    
    success_rate = stats['retrieval_success_rate']
    textstr = f'Stage 1 Success Rate\n{success_rate:.1%}'
    props = dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1', 
                edgecolor='#34495E', alpha=0.9, linewidth=2)
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props, 
            fontweight='bold', color='#2C3E50')
    
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
    if not selection_data or 'selections' not in selection_data:
        print("⚠ No selection data to visualize")
        return
    
    selections = selection_data['selections']
    if not selections:
        return
    
    tool_counts = {}
    for sel in selections:
        if sel['selected_tools']:
            top_tool = sel['selected_tools'][0]['tool_name']
            tool_counts[top_tool] = tool_counts.get(top_tool, 0) + 1
    
    sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if not sorted_tools:
        return
    
    tools, counts = zip(*sorted_tools)
    
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('white')
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tools)))
    bars = ax.barh(range(len(tools)), counts, color=colors, alpha=0.85, 
                   edgecolor='#2C3E50', linewidth=1.8)
    
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
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    ax.tick_params(colors='#2C3E50', labelsize=11)
    ax.invert_yaxis()
    
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'single_agent_tool_selection_stats.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Tool selection statistics saved to: {output_path}")
    plt.close()


def print_summary(results):
    print("\n" + "="*80)
    print("Single Agent Learning-to-Rank - Results Summary")
    print("="*80)
    
    if results['selection_results']:
        sel_data = results['selection_results']
        print("\n📋 Tool Selection Results (Step 1: Embedding-based):")
        print("-" * 60)
        print(f"  Total traces:        {sel_data.get('total_traces', 'N/A')}")
        print(f"  Total decisions:     {sel_data.get('total_decisions', 'N/A')}")
        print(f"  Method:              {sel_data.get('method', 'N/A')}")
        
        if 'ground_truth_stats' in sel_data:
            gt_stats = sel_data['ground_truth_stats']
            print("\n🎯 Ground Truth Retrieval (Stage 1 Success Rate):")
            print("-" * 60)
            print(f"  Decisions with GT:   {gt_stats.get('total_with_ground_truth', 'N/A')}")
            print(f"  GT in top-3:         {gt_stats.get('ground_truth_in_top3', 'N/A')} "
                  f"({gt_stats.get('retrieval_success_rate', 0)*100:.1f}%)")
            print(f"  GT added as 4th:     {gt_stats.get('ground_truth_not_in_top3', 'N/A')} "
                  f"({(1-gt_stats.get('retrieval_success_rate', 0))*100:.1f}%)")
            print(f"  Success rate:        {gt_stats.get('retrieval_success_rate', 0):.2%}")
    
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
            max_weight_idx = weights.index(max(weights))
            feature_names = ['Relevance', 'Reliability', 'Compatibility', 'Structure']
            print(f"  • Most important feature: φ_{['rel', 'hist', 'coop', 'struct'][max_weight_idx]} ({feature_names[max_weight_idx]})")
            print(f"  • Weight value: {max(weights):.4f}")
            
            if summary['top1_accuracy'] == 1.0:
                print(f"  • Perfect performance on test set!")
                print("\n✅ Perfect! All test samples correctly predicted!")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Single Agent Learning-to-Rank Model Results',
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
        help='Skip tool selection statistics visualization'
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        output_dir = find_output_dir()
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        print(f"✗ Error: Output directory not found: {output_dir}")
        print(f"  Please ensure the output directory exists or run the LTR model first.")
        return
    
    figure_dir = find_figure_dir()
    
    print("="*80)
    print("Single Agent LTR Results Visualization")
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
        visualize_ground_truth_retrieval(results['selection_results'], figure_dir)
        visualize_selection_stats(results['selection_results'], figure_dir)
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
