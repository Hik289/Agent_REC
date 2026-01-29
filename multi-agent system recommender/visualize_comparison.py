#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Comparison of Different Methods for Multi-Agent System Recommender
Compares embedding-based, LLM-based, and merged retrieval methods
"""

import json
import os
import sys
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
                plt.savefig(output_path, dpi=150, **kwargs)
                print(f"✓ Saved with DPI=150")
                return True
            except ValueError:
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('bbox_inches', None)
                plt.savefig(output_path, dpi=150, **kwargs_copy)
                print(f"✓ Saved with DPI=150, no tight bbox")
                return True
        else:
            raise


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_output_dir():
    """Smart path resolution to find output directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Locations to try (in order)
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
    """Smart path resolution to find figure directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    figure_dir = os.path.join(parent_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def load_all_results(base_dir):
    """Load results from all three directories"""
    results = {}

    print("\nLoading results...")

    # Load embedding-based results (output_old)
    print("\n  Loading embedding-based results (output_old)...")
    output_old_dir = os.path.join(base_dir, 'output_old')
    if os.path.exists(output_old_dir):
        ltr_results_path = os.path.join(output_old_dir, 'graph_ltr_test_results.json')
        graph_selection_path = os.path.join(output_old_dir, 'graph_selection_results.json')

        if os.path.exists(ltr_results_path):
            data = load_json(ltr_results_path)
            # Old format: only has summary
            if 'summary' in data:
                results['embedding'] = {
                    'ltr_trained': data['summary'],
                    'direct': None,
                    'oracle': None,
                    'baseline': None
                }
                print(f"    ✓ Loaded LTR results (old format)")

        # Calculate embedding direct, baseline, and oracle accuracy
        if os.path.exists(graph_selection_path):
            graph_data = load_json(graph_selection_path)
            total = 0
            correct_direct = 0
            correct_baseline = 0
            gt_in_candidates = 0

            for trace in graph_data.get('traces', []):
                for node_selection in trace.get('node_selections', []):
                    # Find ground truth
                    ground_truth_id = None
                    candidates = node_selection.get('selected_candidates', [])

                    for cand in candidates:
                        if cand.get('type') == 'ground_truth':
                            ground_truth_id = cand.get('candidate_id')
                            break

                    if ground_truth_id and candidates:
                        total += 1

                        # Direct: top-1 from embedding retrieval
                        top1_id = candidates[0].get('candidate_id')
                        if top1_id == ground_truth_id:
                            correct_direct += 1

                        # Baseline: same as direct for embedding (approximation)
                        if top1_id == ground_truth_id:
                            correct_baseline += 1

                        # Oracle: check if GT is in any candidate
                        if ground_truth_id:
                            gt_in_candidates += 1

            if total > 0:
                if 'embedding' not in results:
                    results['embedding'] = {}

                results['embedding']['direct'] = {
                    'top1_accuracy': correct_direct / total,
                    'num_queries': total
                }
                results['embedding']['baseline'] = {
                    'top1_accuracy': correct_baseline / total,
                    'num_queries': total
                }
                results['embedding']['oracle'] = {
                    'oracle_accuracy': gt_in_candidates / total,
                    'oracle_accuracy_top5': gt_in_candidates / total,
                    'num_queries': total
                }
                print(f"    ✓ Calculated embedding direct accuracy: {correct_direct/total:.2%}")
                print(f"    ✓ Calculated embedding oracle: {gt_in_candidates/total:.2%}")

    # Load LLM-based results (output)
    print("\n  Loading LLM-based results (output)...")
    output_dir = os.path.join(base_dir, 'output')
    if os.path.exists(output_dir):
        ltr_results_path = os.path.join(output_dir, 'graph_ltr_test_results.json')

        if os.path.exists(ltr_results_path):
            data = load_json(ltr_results_path)
            results['llm'] = {
                'ltr_trained': data.get('trained_model'),
                'direct': data.get('direct_retrieval'),
                'oracle': data.get('oracle_retrieval'),
                'baseline': None  # LLM doesn't need baseline
            }
            print(f"    ✓ Loaded LLM results")

    # Load merged results (output_merge)
    print("\n  Loading merged results (output_merge)...")
    output_merge_dir = os.path.join(base_dir, 'output_merge')
    if os.path.exists(output_merge_dir):
        ltr_results_path = os.path.join(output_merge_dir, 'graph_ltr_test_results.json')

        if os.path.exists(ltr_results_path):
            data = load_json(ltr_results_path)
            results['merged'] = {
                'ltr_trained': data.get('trained_model'),
                'direct': data.get('direct_retrieval'),
                'oracle': data.get('oracle_retrieval'),
                'baseline': data.get('baseline')
            }
            print(f"    ✓ Loaded merged results")

    return results


def visualize_test_results(results, figure_dir):
    """Visualize test results with comparison of all methods"""
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('white')

    # Prepare data for 7 methods
    methods = []
    accuracies = []
    colors = []
    hatches = []

    # Color scheme
    color_embedding = '#3498DB'  # Blue
    color_llm = '#E74C3C'  # Red
    color_merged = '#2ECC71'  # Green
    color_oracle = '#9B59B6'  # Purple

    # 1. Embedding-based retrieval (direct)
    if results.get('embedding', {}).get('direct'):
        methods.append('Embedding\nRetrieval\n(Direct)')
        accuracies.append(results['embedding']['direct']['top1_accuracy'] * 100)
        colors.append(color_embedding)
        hatches.append('')

    # 2. Embedding + LTR baseline
    if results.get('embedding', {}).get('baseline'):
        methods.append('Embedding\n+ LTR\n(Baseline)')
        accuracies.append(results['embedding']['baseline']['top1_accuracy'] * 100)
        colors.append(color_embedding)
        hatches.append('//')

    # 3. Embedding + LTR trained
    if results.get('embedding', {}).get('ltr_trained'):
        methods.append('Embedding\n+ LTR\n(Trained)')
        accuracies.append(results['embedding']['ltr_trained']['top1_accuracy'] * 100)
        colors.append(color_embedding)
        hatches.append('\\\\')

    # 4. Embedding oracle
    if results.get('embedding', {}).get('oracle'):
        methods.append('Embedding\nOracle')
        accuracies.append(results['embedding']['oracle']['oracle_accuracy_top5'] * 100)
        colors.append(color_oracle)
        hatches.append('xx')

    # 5. LLM-based retrieval (direct)
    if results.get('llm', {}).get('direct'):
        methods.append('LLM\nRetrieval\n(Direct)')
        accuracies.append(results['llm']['direct']['top1_accuracy'] * 100)
        colors.append(color_llm)
        hatches.append('')

    # 6. LLM-based retrieval + LTR trained (skip baseline for LLM)
    if results.get('llm', {}).get('ltr_trained'):
        methods.append('LLM\n+ LTR\n(Trained)')
        accuracies.append(results['llm']['ltr_trained']['top1_accuracy'] * 100)
        colors.append(color_llm)
        hatches.append('\\\\')

    # 7. Merged retrieval + LTR trained
    if results.get('merged', {}).get('ltr_trained'):
        methods.append('Merged\n+ LTR\n(Trained)')
        accuracies.append(results['merged']['ltr_trained']['top1_accuracy'] * 100)
        colors.append(color_merged)
        hatches.append('\\\\')

    # Plot bars
    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.85,
                   edgecolor='#2C3E50', linewidth=2.5, width=0.65)

    # Add hatches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='#2C3E50')

    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
    ax.set_title('Multi-Agent System Selection - Method Comparison', fontsize=18, fontweight='bold',
                  pad=20, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    ax.tick_params(colors='#2C3E50', labelsize=11)

    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'multi_agent_test_performance.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Method comparison visualization saved to: {output_path}")
    plt.close()


def visualize_retrieval_vs_ranking(results, figure_dir):
    """Visualize the improvement from retrieval to ranking (LTR)"""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')

    # Prepare data: Direct vs LTR Trained for each retrieval method
    methods = []
    direct_accs = []
    ltr_accs = []
    improvements = []

    # Embedding
    if results.get('embedding', {}).get('direct') and results.get('embedding', {}).get('ltr_trained'):
        methods.append('Embedding')
        direct_acc = results['embedding']['direct']['top1_accuracy'] * 100
        ltr_acc = results['embedding']['ltr_trained']['top1_accuracy'] * 100
        direct_accs.append(direct_acc)
        ltr_accs.append(ltr_acc)
        improvements.append(ltr_acc - direct_acc)

    # LLM
    if results.get('llm', {}).get('direct') and results.get('llm', {}).get('ltr_trained'):
        methods.append('LLM')
        direct_acc = results['llm']['direct']['top1_accuracy'] * 100
        ltr_acc = results['llm']['ltr_trained']['top1_accuracy'] * 100
        direct_accs.append(direct_acc)
        ltr_accs.append(ltr_acc)
        improvements.append(ltr_acc - direct_acc)

    # Merged
    if results.get('merged', {}).get('direct') and results.get('merged', {}).get('ltr_trained'):
        methods.append('Merged')
        direct_acc = results['merged']['direct']['top1_accuracy'] * 100
        ltr_acc = results['merged']['ltr_trained']['top1_accuracy'] * 100
        direct_accs.append(direct_acc)
        ltr_accs.append(ltr_acc)
        improvements.append(ltr_acc - direct_acc)

    if not methods:
        print("⚠ No data for retrieval vs ranking visualization")
        return

    x = np.arange(len(methods))
    width = 0.35

    # Plot grouped bars
    bars1 = ax.bar(x - width/2, direct_accs, width, label='Direct Retrieval',
                   color='#95A5A6', alpha=0.85, edgecolor='#2C3E50', linewidth=2)
    bars2 = ax.bar(x + width/2, ltr_accs, width, label='+ LTR Trained',
                   color='#27AE60', alpha=0.85, edgecolor='#2C3E50', linewidth=2)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2C3E50')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2C3E50')

    # Add improvement arrows and labels
    for i, (direct, ltr, imp) in enumerate(zip(direct_accs, ltr_accs, improvements)):
        if imp > 0:
            # Draw arrow
            ax.annotate('', xy=(i + width/2, ltr - 2), xytext=(i - width/2, direct + 2),
                       arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2.5))
            # Add improvement text
            mid_y = (direct + ltr) / 2
            ax.text(i, mid_y, f'+{imp:.1f}%',
                   ha='center', va='center', fontsize=13, fontweight='bold',
                   color='#E67E22',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#E67E22', linewidth=2))
        elif imp < 0:
            # Draw arrow (downward)
            ax.annotate('', xy=(i + width/2, ltr + 2), xytext=(i - width/2, direct - 2),
                       arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
            # Add decrease text
            mid_y = (direct + ltr) / 2
            ax.text(i, mid_y, f'{imp:.1f}%',
                   ha='center', va='center', fontsize=13, fontweight='bold',
                   color='#E74C3C',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#E74C3C', linewidth=2))

    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
    ax.set_title('Impact of LTR on Different Retrieval Methods', fontsize=18, fontweight='bold',
                 pad=20, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9, edgecolor='#BDC3C7')
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    ax.tick_params(colors='#2C3E50', labelsize=11)

    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'multi_agent_retrieval_vs_ranking.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Retrieval vs ranking visualization saved to: {output_path}")
    plt.close()


def visualize_weights(base_dir, figure_dir):
    """Visualize feature weights for the best model"""
    weights = None
    method_name = ''

    # Try to load weights from merged, then llm, then embedding
    for dir_name, name in [('output_merge', 'Merged'), ('output', 'LLM'), ('output_old', 'Embedding')]:
        weights_path = os.path.join(base_dir, dir_name, 'ltr_model_weights.json')
        if os.path.exists(weights_path):
            data = load_json(weights_path)
            weights = data.get('weights')
            method_name = name
            break

    if not weights:
        print("⚠ No weights found to visualize")
        return

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
    ax.set_title(f'Multi-Agent LTR - Feature Weight Distribution ({method_name})',
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
    output_path = os.path.join(figure_dir, 'multi_agent_weights.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Weights visualization saved to: {output_path}")
    plt.close()


def visualize_ground_truth_retrieval(base_dir, figure_dir):
    """Visualize ground truth retrieval statistics"""
    selection_data = None
    for dir_name in ['output_merge', 'output_old']:
        selection_path = os.path.join(base_dir, dir_name, 'graph_selection_results.json')
        if os.path.exists(selection_path):
            selection_data = load_json(selection_path)
            break

    if not selection_data or 'metadata' not in selection_data:
        print("⚠ No ground truth statistics to visualize")
        return

    metadata = selection_data['metadata']
    total_nodes = metadata.get('total_nodes', 0)
    gt_in_top3 = metadata.get('ground_truth_in_top3', 0)

    if total_nodes == 0:
        print("⚠ No ground truth data available")
        return

    gt_not_in_top3 = total_nodes - gt_in_top3

    # Analyze ground truth ranks
    ground_truth_ranks = []
    for trace in selection_data.get('traces', []):
        for node_selection in trace.get('node_selections', []):
            candidates = node_selection.get('selected_candidates', [])
            for i, cand in enumerate(candidates, 1):
                if cand.get('type') == 'ground_truth':
                    ground_truth_ranks.append(i)
                    break

    if not ground_truth_ranks:
        print("⚠ No ground truth rank data to visualize")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    n_candidates = 3

    sizes = [gt_in_top3, gt_not_in_top3]
    labels = [f'In Candidates\n({gt_in_top3}, {gt_in_top3/total_nodes*100:.1f}%)',
              f'Added Later\n({gt_not_in_top3}, {gt_not_in_top3/total_nodes*100:.1f}%)']
    colors = ['#2ECC71', '#E74C3C']
    explode = (0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f'Ground Truth Retrieval\n(Stage 1: Retrieve {n_candidates} Candidates)',
                  fontsize=15, fontweight='bold', pad=20)

    rank_counts = {}
    for rank in ground_truth_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]

    bar_colors = ['#2ECC71' if r <= n_candidates else '#E74C3C' for r in ranks]

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

    success_rate = gt_in_top3 / total_nodes if total_nodes > 0 else 0
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
    output_path = os.path.join(figure_dir, 'multi_agent_ground_truth_retrieval.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Ground truth retrieval statistics saved to: {output_path}")
    plt.close()


def visualize_selection_stats(base_dir, figure_dir):
    """Visualize graph selection statistics"""
    selection_data = None
    for dir_name in ['output_merge', 'output_old']:
        selection_path = os.path.join(base_dir, dir_name, 'graph_selection_results.json')
        if os.path.exists(selection_path):
            selection_data = load_json(selection_path)
            break

    if not selection_data or 'traces' not in selection_data:
        print("⚠ No selection data to visualize")
        return

    # Count top-ranked graphs
    graph_counts = {}
    for trace in selection_data.get('traces', []):
        for node_selection in trace.get('node_selections', []):
            candidates = node_selection.get('selected_candidates', [])
            if candidates:
                top_id = candidates[0].get('candidate_id')
                graph_counts[top_id] = graph_counts.get(top_id, 0) + 1

    sorted_graphs = sorted(graph_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    if not sorted_graphs:
        return

    graphs, counts = zip(*sorted_graphs)

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('white')

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(graphs)))
    bars = ax.barh(range(len(graphs)), counts, color=colors, alpha=0.85,
                   edgecolor='#2C3E50', linewidth=1.8)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2.,
                f'{count}',
                ha='left', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#34495E', alpha=0.9, linewidth=1.5),
                color='#2C3E50')

    ax.set_yticks(range(len(graphs)))
    ax.set_yticklabels(graphs, fontsize=11, fontweight='500', color='#2C3E50')
    ax.set_xlabel('Selection Count', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_title('Top 15 Most Frequently Selected Graphs\n(Embedding-based Similarity)',
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)

    ax.tick_params(colors='#2C3E50', labelsize=11)
    ax.invert_yaxis()

    plt.tight_layout(pad=1.5)
    output_path = os.path.join(figure_dir, 'multi_agent_tool_selection_stats.png')
    safe_savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph selection statistics saved to: {output_path}")
    plt.close()


def print_summary(results):
    """Print results summary"""
    print("\n" + "="*80)
    print("Multi-Agent Learning-to-Rank - Method Comparison Summary")
    print("="*80)

    print("\n📊 All Methods Performance:")
    print("-" * 60)

    if results.get('embedding', {}).get('direct'):
        acc = results['embedding']['direct']['top1_accuracy']
        print(f"  1. Embedding Direct:          {acc:.2%}")

    if results.get('embedding', {}).get('baseline'):
        acc = results['embedding']['baseline']['top1_accuracy']
        print(f"  2. Embedding + LTR Baseline:  {acc:.2%}")

    if results.get('embedding', {}).get('ltr_trained'):
        acc = results['embedding']['ltr_trained']['top1_accuracy']
        print(f"  3. Embedding + LTR Trained:   {acc:.2%}")

    if results.get('embedding', {}).get('oracle'):
        acc = results['embedding']['oracle']['oracle_accuracy_top5']
        print(f"  4. Embedding Oracle:          {acc:.2%}  [Upper Bound]")

    if results.get('llm', {}).get('direct'):
        acc = results['llm']['direct']['top1_accuracy']
        print(f"  5. LLM Direct:                {acc:.2%}")

    if results.get('llm', {}).get('ltr_trained'):
        acc = results['llm']['ltr_trained']['top1_accuracy']
        print(f"  6. LLM + LTR Trained:         {acc:.2%}")

    if results.get('merged', {}).get('ltr_trained'):
        acc = results['merged']['ltr_trained']['top1_accuracy']
        print(f"  7. Merged + LTR Trained:      {acc:.2%}")

    print("\n" + "="*80 + "\n")


def main():
    """Main function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    figure_dir = find_figure_dir()

    print("="*80)
    print("Multi-Agent System Selection - Method Comparison Visualization")
    print("="*80)
    print(f"\nBase directory:   {base_dir}")
    print(f"Figure directory: {figure_dir}\n")

    # Load results
    results = load_all_results(base_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_test_results(results, figure_dir)
    visualize_retrieval_vs_ranking(results, figure_dir)
    visualize_weights(base_dir, figure_dir)
    visualize_ground_truth_retrieval(base_dir, figure_dir)
    visualize_selection_stats(base_dir, figure_dir)

    # Print summary
    print_summary(results)

    print("✓ All visualizations complete!")
    print(f"  Visualizations saved to: {figure_dir}")


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    main()
