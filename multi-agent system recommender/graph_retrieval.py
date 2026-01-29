#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Retrieval using BERT Embedding Similarity
Select top-k most relevant candidate trees from candidates using cosine similarity
"""

import json
import os
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


class GraphRetriever:
    """Graph retrieval using BERT embedding similarity"""
    
    def __init__(self, node_candidates_path: str, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 top_k: int = 10):
        """
        Initialize graph retriever
        
        Args:
            node_candidates_path: Path to node_candidates.json
            model_name: BERT model name from Hugging Face
            top_k: Number of top candidates to retrieve (default: 10)
        """
        print(f"\n{'='*80}")
        print("Graph Retrieval using BERT Embedding Similarity")
        print(f"{'='*80}\n")
        
        print(f"[1/3] Loading BERT model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            print("  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            raise
        
        print(f"\n[2/3] Loading node candidates...")
        self.node_candidates_data = self._load_json(node_candidates_path)
        total_traces = len(self.node_candidates_data.get('traces', []))
        print(f"  ✓ Loaded {total_traces} traces")
        
        self.top_k = top_k
        print(f"\n[3/3] Configuration:")
        print(f"  Top-k candidates to retrieve: {top_k}")
        print(f"  Model: {model_name}")
        print(f"  Top-K candidates: 3")
        print(f"  Total candidates per node: 11 (1 ground truth + 10 random)")
        print(f"\n{'='*80}\n")
    
    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_json(self, data: dict, filepath: str):
        """Save JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to: {filepath}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for text using mean pooling
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Get model output
            output = self.model(**encoded)
            
            # Mean pooling
            embeddings = output.last_hidden_state
            attention_mask = encoded['attention_mask']
            
            # Mask and mean
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding[0].cpu().numpy()
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _tree_to_text(self, tree: Dict) -> str:
        """
        Convert a tree structure to text representation for embedding
        
        Args:
            tree: Tree dictionary with 'nodes' and 'edges'
            
        Returns:
            Text representation of the tree
        """
        nodes = tree.get('nodes', [])
        edges = tree.get('edges', [])
        
        if not nodes:
            return ""
        
        # Extract tasks from nodes
        tasks = [node.get('task', '') for node in nodes]
        
        # Create a text representation
        text_parts = []
        
        # Add root task (first node)
        if tasks:
            text_parts.append(f"Root: {tasks[0]}")
        
        # Add child tasks
        if len(tasks) > 1:
            children = tasks[1:]
            text_parts.append(f"Children: {' -> '.join(children)}")
        
        # Add structural information
        text_parts.append(f"Depth: {len(nodes)} nodes, {len(edges)} edges")
        
        return ". ".join(text_parts)
    
    def _select_top_candidates(self, node_task: str, candidates: List[Dict], 
                               top_k: int = 10) -> Tuple[List[Dict], bool]:
        """
        Select top-k candidates based on similarity to node task
        
        Args:
            node_task: The task description of the current node
            candidates: List of candidate dictionaries (ground_truth + random_candidates)
            top_k: Number of top candidates to select
            
        Returns:
            Tuple of (selected_candidates, ground_truth_in_top_k)
        """
        # Get embedding for node task
        node_embedding = self._get_embedding(node_task)
        
        # Compute similarities for all candidates
        candidate_scores = []
        
        for candidate in candidates:
            tree = candidate.get('tree', {})
            tree_text = self._tree_to_text(tree)
            
            if tree_text:
                tree_embedding = self._get_embedding(tree_text)
                similarity = self._compute_cosine_similarity(node_embedding, tree_embedding)
            else:
                similarity = 0.0
            
            candidate_scores.append({
                'candidate_id': candidate.get('candidate_id', ''),
                'type': candidate.get('type', ''),
                'tree': tree,
                'similarity': similarity,
                'tree_size': len(tree.get('nodes', []))
            })
        
        # Sort by similarity (descending)
        candidate_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Select top-k
        top_candidates = candidate_scores[:top_k]
        
        # Check if ground truth is in top-k
        ground_truth_in_top_k = any(c['type'] == 'ground_truth' for c in top_candidates)
        
        return top_candidates, ground_truth_in_top_k
    
    def retrieve_for_all_nodes(self, output_path: str = None) -> Dict:
        """
        Retrieve top candidates for all nodes
        
        Returns:
            Dictionary with retrieval results
        """
        print(f"Retrieving top candidates for all nodes...")
        print(f"{'='*80}\n")
        
        results = {
            'metadata': {
                'method': 'bert_embedding_similarity',
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'top_k': self.top_k,
                'total_traces': 0,
                'total_nodes': 0,
                'ground_truth_in_topk': 0,
                'ground_truth_added': 0
            },
            'traces': []
        }
        
        traces = self.node_candidates_data.get('traces', [])
        results['metadata']['total_traces'] = len(traces)
        
        total_nodes = 0
        gt_in_topk = 0
        gt_added = 0
        
        for trace_idx, trace in enumerate(traces):
            trace_id = trace.get('trace_id', '')
            node_candidates_list = trace.get('node_candidates', [])
            
            if (trace_idx + 1) % 10 == 0 or trace_idx == 0:
                print(f"Processing trace {trace_idx + 1}/{len(traces)}: {trace_id}")
            
            trace_results = {
                'trace_id': trace_id,
                'node_selections': []
            }
            
            for node_cand in node_candidates_list:
                node_id = node_cand.get('node_id', '')
                node_task = node_cand.get('node_task', '')
                
                # Combine ground truth and random candidates
                all_candidates = [node_cand['ground_truth']]
                all_candidates.extend(node_cand.get('random_candidates', []))
                
                # Select top-k candidates
                top_candidates, gt_in_top_k = self._select_top_candidates(
                    node_task, all_candidates, top_k=self.top_k
                )
                
                # Prepare final selection
                final_selection = []
                for cand in top_candidates:
                    final_selection.append({
                        'candidate_id': cand['candidate_id'],
                        'type': cand['type'],
                        'similarity': round(cand['similarity'], 4),
                        'tree_size': cand['tree_size']
                    })
                
                # Add ground truth if not in top-k
                ground_truth_added_flag = False
                if not gt_in_top_k:
                    # Find ground truth candidate
                    gt_candidate = node_cand['ground_truth']
                    gt_tree = gt_candidate.get('tree', {})
                    gt_tree_text = self._tree_to_text(gt_tree)
                    
                    if gt_tree_text:
                        gt_embedding = self._get_embedding(gt_tree_text)
                        node_embedding = self._get_embedding(node_task)
                        gt_similarity = self._compute_cosine_similarity(node_embedding, gt_embedding)
                    else:
                        gt_similarity = 0.0
                    
                    final_selection.append({
                        'candidate_id': gt_candidate.get('candidate_id', ''),
                        'type': 'ground_truth',
                        'similarity': round(gt_similarity, 4),
                        'tree_size': len(gt_tree.get('nodes', [])),
                        'added_manually': True
                    })
                    ground_truth_added_flag = True
                    gt_added += 1
                else:
                    gt_in_topk += 1
                
                node_selection = {
                    'node_id': node_id,
                    'node_task': node_task,
                    'selected_candidates': final_selection,
                    'total_selected': len(final_selection),
                    'ground_truth_in_topk': gt_in_top_k,
                    'ground_truth_added': ground_truth_added_flag
                }
                
                trace_results['node_selections'].append(node_selection)
                total_nodes += 1
            
            results['traces'].append(trace_results)
        
        # Update metadata
        results['metadata']['total_nodes'] = total_nodes
        results['metadata']['ground_truth_in_topk'] = gt_in_topk
        results['metadata']['ground_truth_added'] = gt_added
        
        print(f"\n{'='*80}")
        print(f"✓ Retrieval complete!")
        print(f"  Total traces: {len(traces)}")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Ground truth in top-{self.top_k}: {gt_in_topk} ({gt_in_topk/total_nodes*100:.1f}%)")
        print(f"  Ground truth added: {gt_added} ({gt_added/total_nodes*100:.1f}%)")
        print(f"{'='*80}\n")
        
        # Save to file if output path is provided
        if output_path:
            self._save_json(results, output_path)
        
        return results
    
    def print_sample_results(self, results: Dict, num_samples: int = 3):
        """Print sample retrieval results"""
        print(f"\n{'='*80}")
        print(f"Sample Retrieval Results (first {num_samples} nodes)")
        print(f"{'='*80}\n")
        
        sample_count = 0
        for trace in results['traces']:
            if sample_count >= num_samples:
                break
            
            trace_id = trace['trace_id']
            for node_sel in trace['node_selections']:
                if sample_count >= num_samples:
                    break
                
                print(f"Sample {sample_count + 1}:")
                print(f"  Trace: {trace_id}")
                print(f"  Node: {node_sel['node_id']}")
                print(f"  Task: {node_sel['node_task'][:80]}...")
                print(f"  Ground truth in top-k: {'Yes' if node_sel['ground_truth_in_topk'] else 'No'}")
                print(f"  Total selected: {node_sel['total_selected']}")
                print(f"  Selected candidates:")
                
                for i, cand in enumerate(node_sel['selected_candidates'], 1):
                    added_flag = " [ADDED]" if cand.get('added_manually', False) else ""
                    print(f"    {i}. {cand['candidate_id']} ({cand['type']})")
                    print(f"       Similarity: {cand['similarity']:.4f}, Size: {cand['tree_size']} nodes{added_flag}")
                
                print()
                sample_count += 1


def _find_data_file(filename: str) -> str:
    """Smart path resolution to find data files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        filename,
        os.path.join(script_dir, '..', 'output', filename),
        os.path.join(script_dir, filename),
        os.path.abspath(filename)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return search_paths[0]


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Graph Retrieval using BERT Embedding Similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detect node_candidates.json)
  python graph_retrieval.py
  
  # Specify custom paths
  python graph_retrieval.py --node_candidates /path/to/node_candidates.json \\
                           --output_dir ./output
  
  # Use different BERT model
  python graph_retrieval.py --model sentence-transformers/all-mpnet-base-v2
  
  # Show sample results
  python graph_retrieval.py --show_samples
        """
    )
    
    parser.add_argument(
        '--node_candidates',
        type=str,
        default=None,
        help='Path to node_candidates.json (default: auto-detect in ../output/)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: ../output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='BERT model from Hugging Face (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top candidates to retrieve per node (default: 10)'
    )
    
    parser.add_argument(
        '--show_samples',
        action='store_true',
        help='Show sample results after retrieval'
    )
    
    args = parser.parse_args()
    
    # Auto-detect node_candidates file if not specified
    if args.node_candidates is None:
        args.node_candidates = _find_data_file('node_candidates.json')
    
    # Set output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.output_dir = os.path.join(parent_dir, 'output')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'graph_selection_results.json')
    
    # Initialize retriever
    retriever = GraphRetriever(args.node_candidates, model_name=args.model, top_k=args.top_k)
    
    # Retrieve candidates
    results = retriever.retrieve_for_all_nodes(output_path)
    
    # Show samples if requested
    if args.show_samples:
        retriever.print_sample_results(results, num_samples=5)


if __name__ == '__main__':
    main()

