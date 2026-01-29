#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Selector Based on Word Embedding Similarity
Uses Sentence-BERT to calculate semantic similarity between task descriptions and tool descriptions
"""

import json
import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util


class ToolSelectorWithEmbedding:
    """Tool Selector Based on Word Embedding Similarity"""
    
    def __init__(self, tool_pool_path: str, calling_graph_path: str, 
                 model_name: str = 'all-MiniLM-L6-v2', n_candidates: int = 5):
        """
        Initialize tool selector
        
        Args:
            tool_pool_path: Path to tool_pool.json file
            calling_graph_path: Path to tool_calling_graphs.json file
            model_name: Sentence-BERT model name
                       - 'all-MiniLM-L6-v2': Lightweight and fast (default)
                       - 'all-mpnet-base-v2': Higher accuracy but slower
                       - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual support
            n_candidates: Number of candidate tools to retrieve from pool (default: 5)
        """
        self.n_candidates = n_candidates
        print(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("Loading data files...")
        self.tool_pool = self._load_json(tool_pool_path)
        self.calling_graph = self._load_json(calling_graph_path)
        self.tools = self.tool_pool.get('tools', {})
        
        # Precompute embeddings for all tools
        print("Precomputing tool embeddings...")
        self._precompute_tool_embeddings()
        
    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_tool_text(self, tool_name: str, tool_info: dict) -> str:
        """
        Create text representation of a tool (for embedding)
        
        Args:
            tool_name: Tool name
            tool_info: Tool information dictionary
            
        Returns:
            Complete text description of the tool
        """
        description = tool_info.get('description', '')
        inputs = tool_info.get('inputs', '')
        
        # Combine tool name, description, and input information
        tool_text = f"Tool: {tool_name}. Description: {description}"
        
        if inputs and inputs != "no parameters":
            tool_text += f" Inputs: {inputs}"
            
        return tool_text
    
    def _precompute_tool_embeddings(self):
        """Precompute embeddings for all tools"""
        self.tool_names = []
        self.tool_texts = []
        
        for tool_name, tool_info in self.tools.items():
            self.tool_names.append(tool_name)
            tool_text = self._create_tool_text(tool_name, tool_info)
            self.tool_texts.append(tool_text)
        
        # Batch compute embeddings for efficiency
        self.tool_embeddings = self.model.encode(
            self.tool_texts, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        print(f"Computed embeddings for {len(self.tool_names)} tools")
    
    def _calculate_similarity(self, task: str, tool_indices: List[int]) -> np.ndarray:
        """
        Calculate cosine similarity between task and specified tools
        
        Args:
            task: Task description
            tool_indices: List of tool indices
            
        Returns:
            Similarity array
        """
        # Compute task embedding
        task_embedding = self.model.encode(task, convert_to_tensor=True)
        
        # Get candidate tool embeddings
        candidate_embeddings = self.tool_embeddings[tool_indices]
        
        # Calculate cosine similarity
        similarities = util.cos_sim(task_embedding, candidate_embeddings)[0]
        
        return similarities.cpu().numpy()
    
    def select_top_tools(self, task: str, candidates: List[str], 
                        top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Select top-k most relevant tools from candidates based on embedding similarity
        
        Args:
            task: Task description
            candidates: List of candidate tool names
            top_k: Return top k tools
            
        Returns:
            [(tool_name, similarity_score), ...] sorted by similarity in descending order
        """
        # Find candidate tools in precomputed list
        tool_name_to_idx = {name: idx for idx, name in enumerate(self.tool_names)}
        
        candidate_indices = []
        valid_candidates = []
        
        for tool_name in candidates:
            if tool_name in tool_name_to_idx:
                candidate_indices.append(tool_name_to_idx[tool_name])
                valid_candidates.append(tool_name)
        
        if not candidate_indices:
            return []
        
        # Calculate similarity
        similarities = self._calculate_similarity(task, candidate_indices)
        
        # Combine tool names and similarity scores
        results = list(zip(valid_candidates, similarities))
        
        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def process_all_decisions(self, output_path: str = None) -> Dict:
        """
        Process all decision points in the calling graph
        
        Args:
            output_path: Output file path, if None then don't save file
            
        Returns:
            Processing results dictionary
        """
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
                ground_truth = decision.get('chosen', '')  # Get ground truth
                
                if not candidates:
                    continue
                
                # Get task description
                task_desc = nodes.get(node_id, {}).get('task', '')
                
                if not task_desc:
                    continue
                
                # Step 1: Select top-n_candidates from all tools in pool
                all_tool_candidates = self.select_top_tools(task_desc, self.tool_names, top_k=self.n_candidates)
                candidate_tool_names = [tool for tool, score in all_tool_candidates]
                
                # Step 2: Select top-1 from the n_candidates
                top_tool = self.select_top_tools(task_desc, candidate_tool_names, top_k=1)
                
                # Check if ground truth is in candidates
                ground_truth_in_candidates = ground_truth in candidate_tool_names if ground_truth else None
                
                # Check if ground truth is the selected tool (top-1)
                selected_tool_name = top_tool[0][0] if top_tool else None
                ground_truth_is_selected = (selected_tool_name == ground_truth) if ground_truth and selected_tool_name else None
                
                # If ground truth exists and not in candidates, add it
                if ground_truth and not ground_truth_in_candidates:
                    # Calculate similarity score for ground truth
                    gt_tools = self.select_top_tools(task_desc, [ground_truth], top_k=1)
                    if gt_tools:
                        all_tool_candidates.append(gt_tools[0])
                
                selection = {
                    'trace_id': trace_id,
                    'node_id': node_id,
                    'task': task_desc,
                    'total_tools_in_pool': len(self.tool_names),
                    'n_candidates_retrieved': self.n_candidates,
                    'ground_truth': ground_truth if ground_truth else None,
                    'ground_truth_in_candidates': ground_truth_in_candidates,
                    'ground_truth_is_selected': ground_truth_is_selected,
                    'retrieved_candidates': [
                        {
                            'tool_name': tool,
                            'similarity_score': round(float(score), 4),
                            'is_ground_truth': (tool == ground_truth) if ground_truth else False
                        }
                        for tool, score in all_tool_candidates
                    ],
                    'selected_tool': {
                        'tool_name': top_tool[0][0],
                        'similarity_score': round(float(top_tool[0][1]), 4),
                        'is_ground_truth': ground_truth_is_selected if ground_truth_is_selected is not None else False
                    } if top_tool else None
                }
                
                results['selections'].append(selection)
                results['total_decisions'] += 1
        
        print(f"Processing complete! Processed {results['total_decisions']} decision points")
        
        # Calculate ground truth statistics
        ground_truth_stats = {
            'total_with_ground_truth': 0,
            'ground_truth_in_candidates': 0,
            'ground_truth_not_in_candidates': 0,
            'ground_truth_is_selected': 0,
            'candidate_retrieval_rate': 0.0,
            'selection_accuracy': 0.0
        }
        
        for selection in results['selections']:
            if selection.get('ground_truth'):
                ground_truth_stats['total_with_ground_truth'] += 1
                if selection.get('ground_truth_in_candidates'):
                    ground_truth_stats['ground_truth_in_candidates'] += 1
                else:
                    ground_truth_stats['ground_truth_not_in_candidates'] += 1
                
                if selection.get('ground_truth_is_selected'):
                    ground_truth_stats['ground_truth_is_selected'] += 1
        
        if ground_truth_stats['total_with_ground_truth'] > 0:
            ground_truth_stats['candidate_retrieval_rate'] = (
                ground_truth_stats['ground_truth_in_candidates'] / 
                ground_truth_stats['total_with_ground_truth']
            )
            ground_truth_stats['selection_accuracy'] = (
                ground_truth_stats['ground_truth_is_selected'] / 
                ground_truth_stats['total_with_ground_truth']
            )
        
        results['ground_truth_stats'] = ground_truth_stats
        
        print(f"\nGround Truth Retrieval Statistics:")
        print(f"  Total decisions with ground truth: {ground_truth_stats['total_with_ground_truth']}")
        print(f"  Ground truth in candidates: {ground_truth_stats['ground_truth_in_candidates']}")
        print(f"  Ground truth not in candidates (added later): {ground_truth_stats['ground_truth_not_in_candidates']}")
        print(f"  Ground truth is selected (top-1): {ground_truth_stats['ground_truth_is_selected']}")
        print(f"  Candidate retrieval rate: {ground_truth_stats['candidate_retrieval_rate']:.2%}")
        print(f"  Selection accuracy (top-1): {ground_truth_stats['selection_accuracy']:.2%}")
        
        # Save results to file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        
        return results
    
    def print_sample_results(self, results: Dict = None, num_samples: int = 5):
        """
        Print some sample results
        
        Args:
            results: Processing results dictionary, if None then reprocess
            num_samples: Number of samples to display
        """
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
            print(f"  Total tools in pool: {selection.get('total_tools_in_pool', 'N/A')}")
            print(f"  Retrieved candidates: {selection.get('n_candidates_retrieved', 'N/A')}")
            
            # Show ground truth info
            if selection.get('ground_truth'):
                print(f"  Ground Truth: {selection['ground_truth']}")
                print(f"  GT in candidates: {selection.get('ground_truth_in_candidates', 'N/A')}")
                print(f"  GT is selected: {selection.get('ground_truth_is_selected', 'N/A')}")
            
            # Show retrieved candidates
            print(f"  Retrieved Candidates (Top-{selection.get('n_candidates_retrieved', 5)}):")
            for j, tool in enumerate(selection.get('retrieved_candidates', []), 1):
                gt_marker = " ✓ [GT]" if tool.get('is_ground_truth') else ""
                print(f"    {j}. {tool['tool_name']} (similarity: {tool['similarity_score']:.4f}){gt_marker}")
            
            # Show selected tool
            if selection.get('selected_tool'):
                selected = selection['selected_tool']
                gt_marker = " ✓ [GT]" if selected.get('is_ground_truth') else ""
                print(f"  Selected Tool (Top-1): {selected['tool_name']} (similarity: {selected['similarity_score']:.4f}){gt_marker}")
            
            print()
    
    def compare_with_ground_truth(self, results: Dict = None) -> Dict:
        """
        Compare selection results with actually called tools (if annotated)
        
        Args:
            results: Processing results dictionary
            
        Returns:
            Evaluation metrics dictionary
        """
        if results is None:
            results = self.process_all_decisions()
        
        metrics = {
            'total_decisions': results['total_decisions'],
            'top1_match': 0,
            'top3_match': 0,
            'mrr': 0.0  # Mean Reciprocal Rank
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
                
                # Find corresponding selection in results
                for selection in results['selections']:
                    if (selection['trace_id'] == trace.get('trace_id') and 
                        selection['node_id'] == node_id):
                        
                        selected = [t['tool_name'] for t in selection['selected_tools']]
                        
                        # Top-1 accuracy
                        if selected and selected[0] == ground_truth:
                            metrics['top1_match'] += 1
                        
                        # Top-3 accuracy
                        if ground_truth in selected:
                            metrics['top3_match'] += 1
                            # Calculate MRR
                            rank = selected.index(ground_truth) + 1
                            metrics['mrr'] += 1.0 / rank
                        
                        break
        
        # Calculate ratios
        if metrics['total_decisions'] > 0:
            metrics['top1_accuracy'] = metrics['top1_match'] / metrics['total_decisions']
            metrics['top3_accuracy'] = metrics['top3_match'] / metrics['total_decisions']
            metrics['mrr'] = metrics['mrr'] / metrics['total_decisions']
        
        return metrics


def _find_data_file(filename: str) -> str:
    """
    Smart path resolution to find data files
    Tries multiple locations relative to script and working directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locations to try (in order)
    search_paths = [
        # Relative to current working directory
        filename,
        # Relative to script directory (parent)
        os.path.join(script_dir, '..', filename),
        # Relative to script directory (same level)
        os.path.join(script_dir, filename),
        # Absolute path if provided
        os.path.abspath(filename)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    # If not found, return the first attempt (will raise error with clear message)
    return search_paths[0]


def main():
    """Main function"""
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_parent_dir = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(
        description='Tool Selection using Word Embedding Similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detect tool_pool.json and tool_calling_graphs.json)
  python tool_selection.py
  
  # Specify custom input files and output directory
  python tool_selection.py --tool_pool /path/to/tool_pool.json \\
                          --calling_graph /path/to/tool_calling_graphs.json \\
                          --output_dir ./results
  
  # Use a different BERT model
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
    
    parser.add_argument(
        '--n_candidates',
        type=int,
        default=5,
        help='Number of candidates to retrieve from tool pool (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect data files if not specified
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file path
    output_path = os.path.join(args.output_dir, 'tool_selection_results.json')
    
    print(f"\n{'='*80}")
    print(f"Tool Selection Configuration")
    print(f"{'='*80}")
    print(f"  Tool Pool:      {args.tool_pool}")
    print(f"  Calling Graph:  {args.calling_graph}")
    print(f"  Output Dir:     {args.output_dir}")
    print(f"  Model:          {args.model}")
    print(f"  N Candidates:   {args.n_candidates}")
    print(f"  Strategy:       Retrieve {args.n_candidates} from pool → Select top-1")
    print(f"{'='*80}\n")
    
    # Create tool selector
    selector = ToolSelectorWithEmbedding(
        args.tool_pool, 
        args.calling_graph,
        model_name=args.model,
        n_candidates=args.n_candidates
    )
    
    # Process all decision points
    results = selector.process_all_decisions(output_path)
    
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"  - Total Traces: {results['total_traces']}")
    print(f"  - Total Decisions: {results['total_decisions']}")
    print(f"  - Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print sample results
    selector.print_sample_results(results, num_samples=5)


if __name__ == '__main__':
    main()
