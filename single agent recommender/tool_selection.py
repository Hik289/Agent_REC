#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util


class ToolSelectorWithEmbedding:
    
    def __init__(self, tool_pool_path: str, calling_graph_path: str, 
                 model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("Loading data files...")
        self.tool_pool = self._load_json(tool_pool_path)
        self.calling_graph = self._load_json(calling_graph_path)
        self.tools = self.tool_pool.get('tools', {})
        
        print("Precomputing tool embeddings...")
        self._precompute_tool_embeddings()
        
    def _load_json(self, filepath: str) -> dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_tool_text(self, tool_name: str, tool_info: dict) -> str:
        description = tool_info.get('description', '')
        inputs = tool_info.get('inputs', '')
        
        tool_text = f"Tool: {tool_name}. Description: {description}"
        
        if inputs and inputs != "no parameters":
            tool_text += f" Inputs: {inputs}"
            
        return tool_text
    
    def _precompute_tool_embeddings(self):
        self.tool_names = []
        self.tool_texts = []
        
        for tool_name, tool_info in self.tools.items():
            self.tool_names.append(tool_name)
            tool_text = self._create_tool_text(tool_name, tool_info)
            self.tool_texts.append(tool_text)
        
        self.tool_embeddings = self.model.encode(
            self.tool_texts, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        print(f"Computed embeddings for {len(self.tool_names)} tools")
    
    def _calculate_similarity(self, task: str, tool_indices: List[int]) -> np.ndarray:
        task_embedding = self.model.encode(task, convert_to_tensor=True)
        
        candidate_embeddings = self.tool_embeddings[tool_indices]
        
        similarities = util.cos_sim(task_embedding, candidate_embeddings)[0]
        
        return similarities.cpu().numpy()
    
    def select_top_tools(self, task: str, candidates: List[str], 
                        top_k: int = 3) -> List[Tuple[str, float]]:
        tool_name_to_idx = {name: idx for idx, name in enumerate(self.tool_names)}
        
        candidate_indices = []
        valid_candidates = []
        
        for tool_name in candidates:
            if tool_name in tool_name_to_idx:
                candidate_indices.append(tool_name_to_idx[tool_name])
                valid_candidates.append(tool_name)
        
        if not candidate_indices:
            return []
        
        similarities = self._calculate_similarity(task, candidate_indices)
        
        results = list(zip(valid_candidates, similarities))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def process_all_decisions(self, output_path: str = None) -> Dict:
        results = {
            'method': 'embedding_similarity',
            'model': self.model.get_sentence_embedding_dimension(),
            'total_traces': 0,
            'total_decisions': 0,
            'selections': []
        }
        
        traces = self.calling_graph.get('traces', [])
        results['total_traces'] = len(traces)
        
        print(f"\nProcessing decision points in {len(traces)} traces...")
        
        for trace_idx, trace in enumerate(traces):
            trace_id = trace.get('trace_id', 'unknown')
            nodes = trace.get('nodes', {})
            decisions = trace.get('decisions', [])
            
            if (trace_idx + 1) % 10 == 0:
                print(f"  Processed {trace_idx + 1}/{len(traces)} traces...")
            
            for decision in decisions:
                node_id = decision.get('node', '')
                candidates = decision.get('candidates', [])
                ground_truth = decision.get('chosen', '')
                
                if not candidates:
                    continue
                
                task_desc = nodes.get(node_id, {}).get('task', '')
                
                if not task_desc:
                    continue
                
                top_tools = self.select_top_tools(task_desc, candidates, top_k=3)
                
                selected_tool_names = [tool for tool, score in top_tools]
                ground_truth_in_top3 = ground_truth in selected_tool_names if ground_truth else None
                
                if ground_truth and not ground_truth_in_top3 and ground_truth in candidates:
                    gt_tools = self.select_top_tools(task_desc, [ground_truth], top_k=1)
                    if gt_tools:
                        top_tools.append(gt_tools[0])
                
                selection = {
                    'trace_id': trace_id,
                    'node_id': node_id,
                    'task': task_desc,
                    'total_candidates': len(candidates),
                    'ground_truth': ground_truth if ground_truth else None,
                    'ground_truth_in_top3': ground_truth_in_top3,
                    'selected_tools': [
                        {
                            'tool_name': tool,
                            'similarity_score': round(float(score), 4),
                            'is_ground_truth': (tool == ground_truth) if ground_truth else False
                        }
                        for tool, score in top_tools
                    ]
                }
                
                results['selections'].append(selection)
                results['total_decisions'] += 1
        
        print(f"Processing complete! Processed {results['total_decisions']} decision points")
        
        ground_truth_stats = {
            'total_with_ground_truth': 0,
            'ground_truth_in_top3': 0,
            'ground_truth_not_in_top3': 0,
            'retrieval_success_rate': 0.0
        }
        
        for selection in results['selections']:
            if selection.get('ground_truth'):
                ground_truth_stats['total_with_ground_truth'] += 1
                if selection.get('ground_truth_in_top3'):
                    ground_truth_stats['ground_truth_in_top3'] += 1
                else:
                    ground_truth_stats['ground_truth_not_in_top3'] += 1
        
        if ground_truth_stats['total_with_ground_truth'] > 0:
            ground_truth_stats['retrieval_success_rate'] = (
                ground_truth_stats['ground_truth_in_top3'] / 
                ground_truth_stats['total_with_ground_truth']
            )
        
        results['ground_truth_stats'] = ground_truth_stats
        
        print(f"\nGround Truth Retrieval Statistics:")
        print(f"  Total decisions with ground truth: {ground_truth_stats['total_with_ground_truth']}")
        print(f"  Ground truth in top-3: {ground_truth_stats['ground_truth_in_top3']}")
        print(f"  Ground truth not in top-3 (added as 4th): {ground_truth_stats['ground_truth_not_in_top3']}")
        print(f"  Retrieval success rate: {ground_truth_stats['retrieval_success_rate']:.2%}")
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        
        return results
    
    def print_sample_results(self, results: Dict = None, num_samples: int = 5):
        if results is None:
            results = self.process_all_decisions()
        
        print(f"\n{'='*80}")
        print(f"Tool Selection Results Based on Word Embedding Similarity")
        print(f"{'='*80}")
        print(f"Method: Sentence-BERT Embedding Similarity")
        print(f"Embedding Dimension: {results.get('model', 'N/A')}")
        print(f"Total Traces: {results['total_traces']}")
        print(f"Total Decisions: {results['total_decisions']}")
        print(f"\n{'='*80}")
        print(f"Selection Results for First {num_samples} Decision Points:")
        print(f"{'='*80}\n")
        
        for i, selection in enumerate(results['selections'][:num_samples], 1):
            print(f"Sample {i}:")
            print(f"  Trace ID: {selection['trace_id']}")
            print(f"  Node ID: {selection['node_id']}")
            print(f"  Task Description: {selection['task']}")
            print(f"  Number of Candidates: {selection['total_candidates']}")
            print(f"  Selected Top-3 Tools (based on semantic similarity):")
            for j, tool in enumerate(selection['selected_tools'], 1):
                print(f"    {j}. {tool['tool_name']} (similarity: {tool['similarity_score']:.4f})")
            print()
    
    def compare_with_ground_truth(self, results: Dict = None) -> Dict:
        if results is None:
            results = self.process_all_decisions()
        
        metrics = {
            'total_decisions': results['total_decisions'],
            'top1_match': 0,
            'top3_match': 0,
            'mrr': 0.0
        }
        
        traces = self.calling_graph.get('traces', [])
        
        for trace in traces:
            nodes = trace.get('nodes', {})
            decisions = trace.get('decisions', [])
            
            for decision in decisions:
                node_id = decision.get('node', '')
                ground_truth = decision.get('ground_truth', None)
                
                if not ground_truth:
                    continue
                
                for selection in results['selections']:
                    if (selection['trace_id'] == trace.get('trace_id') and 
                        selection['node_id'] == node_id):
                        
                        selected = [t['tool_name'] for t in selection['selected_tools']]
                        
                        if selected and selected[0] == ground_truth:
                            metrics['top1_match'] += 1
                        
                        if ground_truth in selected:
                            metrics['top3_match'] += 1
                            rank = selected.index(ground_truth) + 1
                            metrics['mrr'] += 1.0 / rank
                        
                        break
        
        if metrics['total_decisions'] > 0:
            metrics['top1_accuracy'] = metrics['top1_match'] / metrics['total_decisions']
            metrics['top3_accuracy'] = metrics['top3_match'] / metrics['total_decisions']
            metrics['mrr'] = metrics['mrr'] / metrics['total_decisions']
        
        return metrics


def _find_data_file(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        filename,
        os.path.join(script_dir, '..', filename),
        os.path.join(script_dir, filename),
        os.path.abspath(filename)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return search_paths[0]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_parent_dir = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(
        description='Tool Selection using Word Embedding Similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tool_selection.py
  
  python tool_selection.py --tool_pool /path/to/tool_pool.json \\
                          --calling_graph /path/to/tool_calling_graphs.json \\
                          --output_dir ./results
  
  python tool_selection.py --model all-mpnet-base-v2
        """
    )
    
    parser.add_argument(
        '--tool_pool',
        type=str,
        default=None,
        help='Path to tool_pool.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--calling_graph',
        type=str,
        default=None,
        help='Path to tool_calling_graphs.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(default_parent_dir, 'output'),
        help='Output directory for results (default: ../output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-multilingual-MiniLM-L12-v2'],
        help='Sentence-BERT model to use (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, 'tool_selection_results.json')
    
    print(f"\n{'='*80}")
    print(f"Tool Selection Configuration")
    print(f"{'='*80}")
    print(f"  Tool Pool:      {args.tool_pool}")
    print(f"  Calling Graph:  {args.calling_graph}")
    print(f"  Output Dir:     {args.output_dir}")
    print(f"  Model:          {args.model}")
    print(f"{'='*80}\n")
    
    selector = ToolSelectorWithEmbedding(
        args.tool_pool, 
        args.calling_graph,
        model_name=args.model
    )
    
    results = selector.process_all_decisions(output_path)
    
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"  - Total Traces: {results['total_traces']}")
    print(f"  - Total Decisions: {results['total_decisions']}")
    print(f"  - Results saved to: {output_path}")
    print(f"{'='*80}")
    
    selector.print_sample_results(results, num_samples=5)


if __name__ == '__main__':
    main()
