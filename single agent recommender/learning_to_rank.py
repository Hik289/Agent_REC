#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import torch
import os
import argparse
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


class LinearLearningToRank:
    
    def __init__(self, tool_pool_path: str, calling_graph_path: str,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print("=" * 80)
        print("Initializing Linear Learning-to-Rank Model (BERT Embeddings)")
        print("=" * 80)
        
        print(f"\n[1/4] Loading BERT model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            print("  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            raise
        
        print("\n[2/4] Loading data...")
        self.tool_pool = self._load_json(tool_pool_path)
        self.tools = self.tool_pool.get('tools', {})
        print(f"  ✓ Loaded {len(self.tools)} tools")
        
        print("\n[3/4] Precomputing tool embeddings...")
        self._precompute_tool_embeddings()
        
        print("\n[4/4] Analyzing tool usage statistics...")
        self._analyze_tool_statistics(calling_graph_path)
        
        self.w = np.array([1.0, 1.0, 1.0, 1.0])
        
        print("\n✓ Initialization complete!\n")
    
    def _load_json(self, filepath: str) -> dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            output = self.model(**encoded)
            
            embeddings = output.last_hidden_state
            attention_mask = encoded['attention_mask']
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding[0].cpu().numpy()
    
    def _precompute_tool_embeddings(self):
        self.tool_names = []
        self.tool_embeddings_list = []
        
        for tool_name, tool_info in self.tools.items():
            self.tool_names.append(tool_name)
            
            desc = tool_info.get('description', '')
            inputs = tool_info.get('inputs', '')
            tool_text = f"Tool: {tool_name}. Description: {desc}"
            if inputs and inputs != "no parameters":
                tool_text += f" Inputs: {inputs}"
            
            embedding = self._get_embedding(tool_text)
            self.tool_embeddings_list.append(embedding)
        
        self.tool_embeddings = np.vstack(self.tool_embeddings_list)
        
        self.tool_name_to_idx = {
            name: idx for idx, name in enumerate(self.tool_names)
        }
        
        print(f"  ✓ Computed embeddings for {len(self.tool_names)} tools")
        print(f"    Embedding dimension: {self.tool_embeddings.shape[1]}")
    
    def _analyze_tool_statistics(self, filepath: str):
        data = self._load_json(filepath)
        traces = data.get('traces', [])
        
        tool_freq = {tool: 0 for tool in self.tools.keys()}
        
        max_traces_for_stats = min(30, len(traces))
        for trace in traces[:max_traces_for_stats]:
            nodes = trace.get('nodes', {})
            for node_info in nodes.values():
                task = node_info.get('task', '')
                if task.startswith('Call '):
                    tool_name = task.replace('Call ', '').strip()
                    if tool_name in tool_freq:
                        tool_freq[tool_name] += 1
        
        total = sum(tool_freq.values())
        self.tool_reliability = {
            tool: (freq / max(total, 1)) * 0.5 + 0.5
            for tool, freq in tool_freq.items()
        }
        
        print(f"  ✓ Analyzed {max_traces_for_stats} traces for tool usage statistics")
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def phi_rel(self, query: str, tool_name: str) -> float:
        if tool_name not in self.tool_name_to_idx:
            return 0.0
        
        try:
            query_embedding = self._get_embedding(query)
            tool_idx = self.tool_name_to_idx[tool_name]
            tool_embedding = self.tool_embeddings[tool_idx]
            
            similarity = self._compute_cosine_similarity(query_embedding, tool_embedding)
            return (similarity + 1) / 2
        except Exception as e:
            print(f"Warning: Similarity calculation error: {e}")
            return 0.0
    
    def phi_hist(self, tool_name: str) -> float:
        return self.tool_reliability.get(tool_name, 0.5)
    
    def phi_coop(self, query: str, tool_name: str) -> float:
        tool_name_lower = tool_name.lower()
        query_lower = query.lower()
        
        if tool_name_lower in query_lower:
            return 1.0
        elif any(word in query_lower for word in tool_name_lower.split('_')):
            return 0.7
        else:
            return 0.3
    
    def phi_struct(self, query: str, tool_name: str) -> float:
        phi_rel_val = self.phi_rel(query, tool_name)
        return np.sqrt(phi_rel_val + 0.1)
    
    def compute_features(self, query: str, tool_name: str) -> np.ndarray:
        if tool_name not in self.tool_name_to_idx:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        return np.array([
            self.phi_rel(query, tool_name),
            self.phi_hist(tool_name),
            self.phi_coop(query, tool_name),
            self.phi_struct(query, tool_name)
        ])
    
    def score(self, query: str, tool_name: str) -> float:
        features = self.compute_features(query, tool_name)
        return np.dot(self.w, features)
    
    def prepare_dataset(self, calling_graph_path: str = None, 
                       tool_selection_path: str = None,
                       max_traces: int = None) -> Tuple[List[Dict], List[Dict]]:
        if tool_selection_path is None:
            tool_selection_path = _find_data_file('tool_selection_results.json')
        
        print("=" * 80)
        print("Preparing Training and Test Datasets")
        print("=" * 80)
        print(f"Loading tool selection results from: {tool_selection_path}")
        
        selection_data = self._load_json(tool_selection_path)
        all_selections = selection_data.get('selections', [])
        selections = all_selections if max_traces is None else all_selections[:max_traces]
        
        dataset = []
        
        print(f"\nProcessing {len(selections)} decision points...")
        for i, selection in enumerate(selections):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(selections)}")
            
            trace_id = selection.get('trace_id', '')
            node_id = selection.get('node_id', '')
            query = selection.get('task', '')
            ground_truth = selection.get('ground_truth', '')
            selected_tools = selection.get('selected_tools', [])
            
            if not ground_truth or not selected_tools:
                continue
            
            if ground_truth not in self.tools:
                continue
            
            for tool_info in selected_tools:
                candidate = tool_info.get('tool_name', '')
                
                if candidate not in self.tools:
                    continue
                
                features = self.compute_features(query, candidate)
                label = 1 if candidate == ground_truth else 0
                
                dataset.append({
                    'trace_id': trace_id,
                    'node_id': node_id,
                    'query': query,
                    'tool': candidate,
                    'features': features,
                    'label': label
                })
        
        print(f"\n✓ Dataset size: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("\n" + "="*80)
            print("ERROR: No valid samples found in the dataset!")
            print("="*80)
            print("\nPossible reasons:")
            print("  1. The calling graph doesn't contain ground truth labels")
            print("     - Check if nodes have tasks starting with 'Call [tool_name]'")
            print("  2. Tool names in the calling graph don't match tool_pool.json")
            print("  3. No valid decision points with candidates found")
            print("\nSuggestions:")
            print("  - Verify the format of your calling graph matches the expected structure")
            print("  - Check that tool names are consistent across files")
            print("  - Ensure nodes have proper 'task' fields")
            print("="*80)
            raise ValueError("Dataset preparation failed: No valid samples found. See error message above.")
        
        trace_ids = list(set([d['trace_id'] for d in dataset]))
        
        if len(trace_ids) < 2:
            print(f"\n⚠ Warning: Only {len(trace_ids)} trace(s) found. Using all data for both train and test.")
            return dataset, dataset
        
        if len(trace_ids) < 5:
            test_size = 1
            print(f"\n⚠ Warning: Only {len(trace_ids)} traces. Using 1 trace for test, rest for training.")
            train_traces = trace_ids[:-1]
            test_traces = [trace_ids[-1]]
        else:
            test_size = max(0.2, 1.0 / len(trace_ids))
            train_traces, test_traces = train_test_split(
                trace_ids, 
                test_size=test_size, 
                random_state=42
            )
        
        train_data = [d for d in dataset if d['trace_id'] in train_traces]
        test_data = [d for d in dataset if d['trace_id'] in test_traces]
        
        print(f"  - Training set: {len(train_data)} samples ({len(train_traces)} traces)")
        print(f"  - Test set: {len(test_data)} samples ({len(test_traces)} traces)")
        
        return train_data, test_data
    
    def train(self, train_data: List[Dict], learning_rate: float = 0.01,
              num_epochs: int = 100, lambda_reg: float = 0.001) -> List[float]:
        print(f"\n{'='*80}")
        print(f"Training Model")
        print(f"{'='*80}")
        print(f"Parameters: LR={learning_rate}, Epochs={num_epochs}, L2={lambda_reg}\n")
        
        query_groups = {}
        for sample in train_data:
            key = (sample['trace_id'], sample['node_id'])
            if key not in query_groups:
                query_groups[key] = []
            query_groups[key].append(sample)
        
        print(f"Training samples grouped into {len(query_groups)} queries\n")
        
        loss_history = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_groups = 0
            
            for group_samples in query_groups.values():
                if len(group_samples) < 2:
                    continue
                
                features_matrix = np.array([s['features'] for s in group_samples])
                labels = np.array([s['label'] for s in group_samples])
                
                if labels.sum() == 0:
                    continue
                
                num_groups += 1
                
                scores = features_matrix @ self.w
                probs = softmax(scores)
                
                loss = -np.sum(labels * np.log(probs + 1e-10))
                loss += lambda_reg * np.sum(self.w ** 2)
                total_loss += loss
                
                grad = features_matrix.T @ (probs - labels) + 2 * lambda_reg * self.w
                self.w -= learning_rate * grad
            
            avg_loss = total_loss / max(num_groups, 1)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"w=[{self.w[0]:.3f}, {self.w[1]:.3f}, "
                      f"{self.w[2]:.3f}, {self.w[3]:.3f}]")
        
        print(f"\n✓ Training complete!")
        self._print_learned_weights()
        
        return loss_history
    
    def _print_learned_weights(self):
        print(f"\n{'='*60}")
        print("Learned Feature Weights:")
        print(f"{'='*60}")
        print(f"  w[0] = {self.w[0]:7.4f}  →  φ_rel   (tool-query relevance)")
        print(f"  w[1] = {self.w[1]:7.4f}  →  φ_hist  (historical reliability)")
        print(f"  w[2] = {self.w[2]:7.4f}  →  φ_coop  (graph-aware compatibility)")
        print(f"  w[3] = {self.w[3]:7.4f}  →  φ_struct (structural utility)")
        print(f"{'='*60}")
    
    def evaluate(self, test_data: List[Dict], verbose: bool = True) -> Dict[str, any]:
        print(f"\n{'='*80}")
        print("Evaluating Model on Test Set")
        print(f"{'='*80}\n")
        
        query_groups = {}
        for sample in test_data:
            key = (sample['trace_id'], sample['node_id'])
            if key not in query_groups:
                query_groups[key] = []
            query_groups[key].append(sample)
        
        top1_correct = 0
        top3_correct = 0
        mrr_sum = 0.0
        total_queries = 0
        
        detailed_results = []
        
        for group_samples in query_groups.values():
            if len(group_samples) < 2:
                continue
            
            features_matrix = np.array([s['features'] for s in group_samples])
            labels = np.array([s['label'] for s in group_samples])
            tools = [s['tool'] for s in group_samples]
            query = group_samples[0]['query']
            trace_id = group_samples[0]['trace_id']
            node_id = group_samples[0]['node_id']
            
            if labels.sum() == 0:
                continue
            
            scores = features_matrix @ self.w
            ranked_indices = np.argsort(scores)[::-1]
            
            ranked_tools = [tools[i] for i in ranked_indices]
            ranked_scores = [scores[i] for i in ranked_indices]
            
            true_idx = np.where(labels == 1)[0]
            if len(true_idx) == 0:
                continue
            
            true_idx = true_idx[0]
            true_tool = tools[true_idx]
            rank = np.where(ranked_indices == true_idx)[0][0] + 1
            
            total_queries += 1
            is_top1 = (rank == 1)
            is_top3 = (rank <= 3)
            
            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1
            mrr_sum += 1.0 / rank
            
            detailed_results.append({
                'trace_id': trace_id,
                'node_id': node_id,
                'query': query,
                'true_tool': true_tool,
                'predicted_tool': ranked_tools[0],
                'top3_tools': ranked_tools[:3],
                'top3_scores': [float(s) for s in ranked_scores[:3]],
                'rank': int(rank),
                'is_top1_correct': bool(is_top1),
                'is_top3_correct': bool(is_top3)
            })
        
        metrics = {
            'top1_accuracy': top1_correct / total_queries if total_queries > 0 else 0,
            'top3_accuracy': top3_correct / total_queries if total_queries > 0 else 0,
            'mrr': mrr_sum / total_queries if total_queries > 0 else 0,
            'num_queries': total_queries,
            'detailed_results': detailed_results
        }
        
        print(f"{'='*80}")
        print("Test Results Summary:")
        print(f"{'='*80}")
        print(f"  Test queries:     {metrics['num_queries']}")
        print(f"  Top-1 accuracy:   {metrics['top1_accuracy']:.2%}  "
              f"({top1_correct}/{total_queries})")
        print(f"  Top-3 accuracy:   {metrics['top3_accuracy']:.2%}  "
              f"({top3_correct}/{total_queries})")
        print(f"  MRR:              {metrics['mrr']:.4f}")
        print(f"{'='*80}")
        
        if verbose and detailed_results:
            print(f"\nDetailed Test Results (first 10 samples):")
            print(f"{'='*80}\n")
            
            for i, result in enumerate(detailed_results[:10], 1):
                status = "✓" if result['is_top1_correct'] else "✗"
                query_preview = result['query'][:80] + "..." if len(result['query']) > 80 else result['query']
                print(f"{status} Sample {i}:")
                print(f"   Query: {query_preview}")
                print(f"   True tool:      {result['true_tool']}")
                print(f"   Predicted tool: {result['predicted_tool']} (rank: {result['rank']})")
                print(f"   Top-3: {', '.join(result['top3_tools'])}")
                print()
        
        return metrics
    
    def save_model(self, output_dir: str, test_metrics: Dict = None):
        filepath = os.path.join(output_dir, 'ltr_model_weights.json')
        
        model_data = {
            'weights': self.w.tolist(),
            'tool_reliability': self.tool_reliability,
            'interpretation': {
                'w[0]': 'φ_rel - tool-query relevance',
                'w[1]': 'φ_hist - historical reliability',
                'w[2]': 'φ_coop - graph-aware compatibility',
                'w[3]': 'φ_struct - structural utility'
            }
        }
        
        if test_metrics:
            model_data['test_performance'] = {
                'top1_accuracy': test_metrics['top1_accuracy'],
                'top3_accuracy': test_metrics['top3_accuracy'],
                'mrr': test_metrics['mrr'],
                'num_test_queries': test_metrics['num_queries']
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Model saved to: {filepath}")
    
    def save_test_results(self, metrics: Dict, output_dir: str):
        filepath = os.path.join(output_dir, 'ltr_test_results.json')
        
        results_data = {
            'summary': {
                'num_test_queries': metrics['num_queries'],
                'top1_accuracy': metrics['top1_accuracy'],
                'top3_accuracy': metrics['top3_accuracy'],
                'mrr': metrics['mrr']
            },
            'detailed_results': metrics.get('detailed_results', [])
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed test results saved to: {filepath}")


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
        description='Linear Learning-to-Rank Model for Tool Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_to_rank.py
  
  python learning_to_rank.py --tool_pool /path/to/tool_pool.json \\
                             --calling_graph /path/to/tool_calling_graphs.json \\
                             --output_dir ./results
  
  python learning_to_rank.py --lr 0.05 --epochs 200 --lambda_reg 0.01
  
  python learning_to_rank.py --model sentence-transformers/all-mpnet-base-v2
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
        help='Path to tool_calling_graphs.json (for compatibility, not used in dataset preparation)'
    )
    
    parser.add_argument(
        '--tool_selection',
        type=str,
        default=None,
        help='Path to tool_selection_results.json (default: auto-detect from output dir)'
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
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='BERT model from Hugging Face (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--lambda_reg',
        type=float,
        default=0.001,
        help='L2 regularization coefficient (default: 0.001)'
    )
    
    parser.add_argument(
        '--max_traces',
        type=int,
        default=None,
        help='Maximum number of traces to use (default: None, use all)'
    )
    
    args = parser.parse_args()
    
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')
    if args.tool_selection is None:
        tool_selection_path = os.path.join(args.output_dir, 'tool_selection_results.json')
        if not os.path.exists(tool_selection_path):
            tool_selection_path = _find_data_file('tool_selection_results.json')
        args.tool_selection = tool_selection_path
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Linear Learning-to-Rank for Tool Selection")
    print("Using BERT Embeddings (transformers library)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Tool Pool:         {args.tool_pool}")
    print(f"  Calling Graph:     {args.calling_graph}")
    print(f"  Tool Selection:    {args.tool_selection}")
    print(f"  Output Dir:        {args.output_dir}")
    print(f"  Model:             {args.model}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  L2 Reg:            {args.lambda_reg}")
    print(f"  Max Traces:        {args.max_traces if args.max_traces else 'All'}")
    print("="*80 + "\n")
    
    ltr = LinearLearningToRank(args.tool_pool, args.calling_graph, model_name=args.model)
    
    train_data, test_data = ltr.prepare_dataset(
        calling_graph_path=args.calling_graph,
        tool_selection_path=args.tool_selection,
        max_traces=args.max_traces
    )
    
    if len(train_data) == 0:
        print("\nError: No training data available!")
        return
    
    loss_history = ltr.train(
        train_data,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        lambda_reg=args.lambda_reg
    )
    
    metrics = ltr.evaluate(test_data, verbose=True)
    
    ltr.save_model(args.output_dir, test_metrics=metrics)
    ltr.save_test_results(metrics, args.output_dir)
    
    print(f"\n{'='*80}")
    print("✓ Pipeline Complete!")
    print(f"  All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
