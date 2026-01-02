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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class GraphLearningToRank:
    
    def __init__(self, graph_selection_path: str, node_candidates_path: str,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print("=" * 80)
        print("Graph-Level Linear Learning-to-Rank Model (BERT Embeddings)")
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
        
        print("\n[2/4] Loading graph selection results...")
        self.graph_selection = self._load_json(graph_selection_path)
        print(f"  ✓ Loaded graph selection results")
        
        print("\n[3/4] Loading node candidates...")
        self.node_candidates = self._load_json(node_candidates_path)
        print(f"  ✓ Loaded node candidates")
        
        print("\n[4/4] Analyzing graph statistics...")
        self._analyze_graph_statistics()
        
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
                max_length=512,
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
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _analyze_graph_statistics(self):
        self.graph_frequency = defaultdict(int)
        self.graph_size_stats = defaultdict(list)
        
        for trace in self.graph_selection.get('traces', []):
            for node_sel in trace.get('node_selections', []):
                for cand in node_sel.get('selected_candidates', []):
                    cand_id = cand.get('candidate_id', '')
                    cand_type = cand.get('type', '')
                    tree_size = cand.get('tree_size', 0)
                    
                    self.graph_frequency[cand_type] += 1
                    self.graph_size_stats[cand_type].append(tree_size)
        
        total = sum(self.graph_frequency.values())
        self.graph_reliability = {
            cand_type: (freq / max(total, 1)) * 0.5 + 0.5
            for cand_type, freq in self.graph_frequency.items()
        }
        
        print(f"  ✓ Analyzed graph statistics")
        print(f"    - Ground truth frequency: {self.graph_frequency.get('ground_truth', 0)}")
        print(f"    - Random candidates frequency: {self.graph_frequency.get('random', 0)}")
    
    def _get_candidate_tree(self, trace_id: str, node_id: str, candidate_id: str) -> Dict:
        for trace in self.node_candidates.get('traces', []):
            if trace['trace_id'] == trace_id:
                for node_cand in trace.get('node_candidates', []):
                    if node_cand['node_id'] == node_id:
                        gt = node_cand.get('ground_truth', {})
                        if gt.get('candidate_id') == candidate_id:
                            return gt.get('tree', {})
                        
                        for rand_cand in node_cand.get('random_candidates', []):
                            if rand_cand.get('candidate_id') == candidate_id:
                                return rand_cand.get('tree', {})
        
        return {}
    
    def _tree_to_text(self, tree: Dict) -> str:
        nodes = tree.get('nodes', [])
        edges = tree.get('edges', [])
        
        if not nodes:
            return ""
        
        tasks = [node.get('task', '') for node in nodes]
        text_parts = []
        
        if tasks:
            text_parts.append(f"Root: {tasks[0]}")
        
        if len(tasks) > 1:
            children = tasks[1:]
            text_parts.append(f"Children: {' -> '.join(children[:5])}")
        
        text_parts.append(f"Structure: {len(nodes)} nodes, {len(edges)} edges")
        
        return ". ".join(text_parts)
    
    def phi_rel(self, query: str, candidate_id: str, trace_id: str, node_id: str) -> float:
        try:
            tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
            tree_text = self._tree_to_text(tree)
            
            if not tree_text or not query:
                return 0.0
            
            query_embedding = self._get_embedding(query)
            tree_embedding = self._get_embedding(tree_text)
            
            similarity = self._compute_cosine_similarity(query_embedding, tree_embedding)
            return (similarity + 1) / 2
        except Exception as e:
            return 0.0
    
    def phi_hist(self, candidate_type: str) -> float:
        return self.graph_reliability.get(candidate_type, 0.5)
    
    def phi_coop(self, query: str, candidate_id: str, trace_id: str, node_id: str) -> float:
        tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
        num_nodes = len(tree.get('nodes', []))
        num_edges = len(tree.get('edges', []))
        
        query_complexity = len(query.split())
        
        if 'ground_truth' in candidate_id:
            return 0.9
        
        if query_complexity < 5:
            size_score = 1.0 / (1.0 + np.log1p(num_nodes))
        else:
            size_score = min(1.0, np.log1p(num_nodes) / 5.0)
        
        return size_score * 0.8 + 0.2
    
    def phi_struct(self, candidate_id: str, trace_id: str, node_id: str, 
                   all_candidate_ids: List[str]) -> float:
        tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
        num_nodes = len(tree.get('nodes', []))
        num_edges = len(tree.get('edges', []))
        
        depth = num_edges / max(num_nodes, 1) if num_nodes > 0 else 0
        
        if num_nodes > 0:
            balance_score = min(1.0, depth)
        else:
            balance_score = 0.0
        
        sizes = []
        for cand_id in all_candidate_ids:
            if cand_id != candidate_id:
                other_tree = self._get_candidate_tree(trace_id, node_id, cand_id)
                sizes.append(len(other_tree.get('nodes', [])))
        
        if sizes:
            avg_size = np.mean(sizes)
            uniqueness = abs(num_nodes - avg_size) / max(avg_size, 1)
            uniqueness_score = min(1.0, uniqueness / 2.0)
        else:
            uniqueness_score = 0.5
        
        return balance_score * 0.6 + uniqueness_score * 0.4
    
    def compute_features(self, query: str, candidate_id: str, candidate_type: str,
                        trace_id: str, node_id: str, all_candidate_ids: List[str]) -> np.ndarray:
        return np.array([
            self.phi_rel(query, candidate_id, trace_id, node_id),
            self.phi_hist(candidate_type),
            self.phi_coop(query, candidate_id, trace_id, node_id),
            self.phi_struct(candidate_id, trace_id, node_id, all_candidate_ids)
        ])
    
    def score(self, features: np.ndarray) -> float:
        return np.dot(self.w, features)
    
    def prepare_dataset(self, max_traces: int = None) -> Tuple[List[Dict], List[Dict]]:
        print("=" * 80)
        print("Preparing Training and Test Datasets")
        print("=" * 80)
        
        dataset = []
        traces = self.graph_selection.get('traces', [])
        
        if max_traces:
            traces = traces[:max_traces]
        
        print(f"\nProcessing {len(traces)} traces...")
        
        for i, trace in enumerate(traces):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(traces)}")
            
            trace_id = trace.get('trace_id', '')
            
            for node_sel in trace.get('node_selections', []):
                node_id = node_sel.get('node_id', '')
                node_task = node_sel.get('node_task', '')
                selected_candidates = node_sel.get('selected_candidates', [])
                
                if not selected_candidates:
                    continue
                
                all_candidate_ids = [c['candidate_id'] for c in selected_candidates]
                
                ground_truth_id = None
                for cand in selected_candidates:
                    if cand['type'] == 'ground_truth':
                        ground_truth_id = cand['candidate_id']
                        break
                
                if not ground_truth_id:
                    continue
                
                for candidate in selected_candidates:
                    candidate_id = candidate['candidate_id']
                    candidate_type = candidate['type']
                    
                    features = self.compute_features(
                        node_task, candidate_id, candidate_type,
                        trace_id, node_id, all_candidate_ids
                    )
                    
                    label = 1 if candidate_id == ground_truth_id else 0
                    
                    dataset.append({
                        'trace_id': trace_id,
                        'node_id': node_id,
                        'query': node_task,
                        'candidate_id': candidate_id,
                        'candidate_type': candidate_type,
                        'features': features,
                        'label': label
                    })
        
        print(f"\n✓ Dataset size: {len(dataset)} samples")
        
        trace_ids = list(set([d['trace_id'] for d in dataset]))
        train_traces, test_traces = train_test_split(
            trace_ids,
            test_size=0.2,
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
        print(f"  w[0] = {self.w[0]:7.4f}  →  φ_rel   (semantic alignment)")
        print(f"  w[1] = {self.w[1]:7.4f}  →  φ_hist  (system reliability)")
        print(f"  w[2] = {self.w[2]:7.4f}  →  φ_coop  (internal cooperation)")
        print(f"  w[3] = {self.w[3]:7.4f}  →  φ_struct (structural utility)")
        print(f"{'='*60}")
    
    def evaluate(self, test_data: List[Dict], verbose: bool = True) -> Dict:
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
            candidate_ids = [s['candidate_id'] for s in group_samples]
            query = group_samples[0]['query']
            trace_id = group_samples[0]['trace_id']
            node_id = group_samples[0]['node_id']
            
            if labels.sum() == 0:
                continue
            
            scores = features_matrix @ self.w
            ranked_indices = np.argsort(scores)[::-1]
            
            ranked_candidates = [candidate_ids[i] for i in ranked_indices]
            ranked_scores = [scores[i] for i in ranked_indices]
            
            true_idx = np.where(labels == 1)[0]
            if len(true_idx) == 0:
                continue
            
            true_idx = true_idx[0]
            true_candidate = candidate_ids[true_idx]
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
                'true_candidate': true_candidate,
                'predicted_candidate': ranked_candidates[0],
                'top3_candidates': ranked_candidates[:3],
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
                query_preview = result['query'][:60] + "..." if len(result['query']) > 60 else result['query']
                print(f"{status} Sample {i}:")
                print(f"   Query: {query_preview}")
                print(f"   True candidate:      {result['true_candidate']}")
                print(f"   Predicted candidate: {result['predicted_candidate']} (rank: {result['rank']})")
                print(f"   Top-3: {', '.join(result['top3_candidates'])}")
                print()
        
        return metrics
    
    def save_model(self, output_dir: str, test_metrics: Dict = None):
        filepath = os.path.join(output_dir, 'graph_ltr_model_weights.json')
        
        model_data = {
            'weights': self.w.tolist(),
            'graph_reliability': dict(self.graph_reliability),
            'interpretation': {
                'w[0]': 'φ_rel - semantic alignment (query-system)',
                'w[1]': 'φ_hist - system reliability',
                'w[2]': 'φ_coop - internal cooperation',
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
        filepath = os.path.join(output_dir, 'graph_ltr_test_results.json')
        
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
        os.path.join(script_dir, '..', 'output', filename),
        os.path.join(script_dir, filename),
        os.path.abspath(filename)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return search_paths[0]


def main():
    parser = argparse.ArgumentParser(
        description='Graph-Level Linear Learning-to-Rank Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_to_rank.py
  
  python learning_to_rank.py --graph_selection /path/to/graph_selection_results.json \\
                             --node_candidates /path/to/node_candidates.json \\
                             --output_dir ./output
  
  python learning_to_rank.py --lr 0.05 --epochs 200
        """
    )
    
    parser.add_argument(
        '--graph_selection',
        type=str,
        default=None,
        help='Path to graph_selection_results.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--node_candidates',
        type=str,
        default=None,
        help='Path to node_candidates.json (default: auto-detect)'
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
        help='BERT model from Hugging Face'
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
    
    if args.graph_selection is None:
        args.graph_selection = _find_data_file('graph_selection_results.json')
    if args.node_candidates is None:
        args.node_candidates = _find_data_file('node_candidates.json')
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.output_dir = os.path.join(parent_dir, 'output')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Graph-Level Linear Learning-to-Rank for Agent System Selection")
    print("Using BERT Embeddings (transformers library)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Graph Selection:  {args.graph_selection}")
    print(f"  Node Candidates:  {args.node_candidates}")
    print(f"  Output Dir:       {args.output_dir}")
    print(f"  Model:            {args.model}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  L2 Reg:           {args.lambda_reg}")
    print(f"  Max Traces:       {args.max_traces if args.max_traces else 'All'}")
    print("="*80 + "\n")
    
    ltr = GraphLearningToRank(
        args.graph_selection,
        args.node_candidates,
        model_name=args.model
    )
    
    train_data, test_data = ltr.prepare_dataset(max_traces=args.max_traces)
    
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
