"""
LLM-based Tool Selection Agent
Uses OpenAI API to select tools based on task descriptions
"""

import json
import os
import argparse
from typing import List, Dict, Tuple, Optional
import time
import asyncio
from openai import OpenAI, AsyncOpenAI


class ToolSelectAgent:
    """Tool Selection Agent"""

    def __init__(self, tool_pool_path: str, calling_graph_path: str,
                 model_name: str = 'gpt-4.1',
                 n_candidates: int = 5,
                 temperature: float = 1.0,
                 max_concurrent: int = 10):
        """
        Initialize Tool Selection Agent

        Args:
            tool_pool_path: Path to tool_pool.json file
            calling_graph_path: Path to tool_calling_graphs.json file
            model_name: model name (default: gpt-4.1)
            n_candidates: Number of candidate tools to retrieve (default: 5)
            temperature: Temperature for LLM sampling (default: 1.0)
            max_concurrent: Maximum number of concurrent API calls (default: 10)
        """
        self.n_candidates = n_candidates
        self.model_name = model_name
        self.temperature = temperature
        self.max_concurrent = max_concurrent

        print(f"Initializing ToolSelectAgent with {self.model_name}...")
        self.client = OpenAI()  # Synchronous client
        self.async_client = AsyncOpenAI()  # Async client for concurrent calls

        print("Loading data files...")
        self.tool_pool = self._load_json(tool_pool_path)
        self.calling_graph = self._load_json(calling_graph_path)
        self.tools = self.tool_pool.get('tools', {})

        print(f"Loaded {len(self.tools)} tools from tool pool")

        # Statistics
        self.total_llm_calls = 0
        self.total_tokens = 0

    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_tool_description(self, tool_name: str, tool_info: dict) -> str:
        """Create a concise tool description for LLM prompt"""
        description = tool_info.get('description', '')
        inputs = tool_info.get('inputs', '')

        tool_desc = f"- {tool_name}: {description}"
        if inputs and inputs != "no parameters":
            tool_desc += f" (Inputs: {inputs})"

        return tool_desc

    def _build_prompt(self, task: str, tool_names: List[str], top_k: int) -> str:
        """
        Build prompt for LLM to select top-k tools

        Args:
            task: Task description
            tool_names: List of available tool names
            top_k: Number of tools to select

        Returns:
            Prompt string
        """
        # Create tool catalog
        tool_catalog = []
        for tool_name in tool_names:
            if tool_name in self.tools:
                tool_desc = self._create_tool_description(tool_name, self.tools[tool_name])
                tool_catalog.append(tool_desc)

        prompt = f"""You are an expert tool selection agent. Your task is to select the most relevant tools for a given task.

Task Description:
{task}

Available Tools:
{chr(10).join(tool_catalog)}

Instructions:
1. Analyze the task description carefully
2. Select the top {top_k} most relevant tools from the list above
3. Rank them by relevance (most relevant first)
4. Return ONLY a JSON array of tool names in order, nothing else

Example output format:
["tool1", "tool2", "tool3"]

Your selection (JSON array only):"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI API and return response (synchronous)

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        self.total_llm_calls += 1

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            # Track token usage
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM API call failed: {e}")
            return "[]"

    async def _call_llm_async(self, prompt: str) -> str:
        """
        Call OpenAI API and return response (asynchronous)

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            # Track token usage
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM API call failed: {e}")
            return "[]"

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract tool names

        Args:
            response: LLM response string

        Returns:
            List of tool names
        """
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            tools = json.loads(response)

            if isinstance(tools, list):
                return [str(t).strip() for t in tools]
            else:
                print(f"Warning: Expected list, got {type(tools)}")
                return []

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print(f"  Response: {response[:200]}...")
            return []

    def select_top_tools(self, task: str, candidates: List[str],
                        top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Select top-k most relevant tools from candidates using LLM (synchronous)

        Args:
            task: Task description
            candidates: List of candidate tool names
            top_k: Return top k tools

        Returns:
            [(tool_name, confidence_score), ...] sorted by relevance
        """
        if not candidates:
            return []

        # Build prompt
        prompt = self._build_prompt(task, candidates, top_k)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        selected_tools = self._parse_llm_response(response)

        # Validate and filter to only valid candidates
        valid_tools = []
        for tool in selected_tools:
            if tool in candidates and tool not in [t[0] for t in valid_tools]:
                valid_tools.append(tool)

        # Assign confidence scores (decreasing from 1.0)
        # Since LLM doesn't give scores, we assign based on rank
        results = []
        for i, tool in enumerate(valid_tools[:top_k]):
            # Score decreases linearly: 1.0, 0.9, 0.8, ...
            score = 1.0 - (i * 0.1)
            results.append((tool, max(0.1, score)))  # Minimum score of 0.1

        return results

    async def select_top_tools_async(self, task: str, candidates: List[str],
                                     top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Select top-k most relevant tools from candidates using LLM (asynchronous)

        Args:
            task: Task description
            candidates: List of candidate tool names
            top_k: Return top k tools

        Returns:
            [(tool_name, confidence_score), ...] sorted by relevance
        """
        if not candidates:
            return []

        # Build prompt
        prompt = self._build_prompt(task, candidates, top_k)

        # Call LLM asynchronously
        response = await self._call_llm_async(prompt)

        # Parse response
        selected_tools = self._parse_llm_response(response)

        # Validate and filter to only valid candidates
        valid_tools = []
        for tool in selected_tools:
            if tool in candidates and tool not in [t[0] for t in valid_tools]:
                valid_tools.append(tool)

        # Assign confidence scores (decreasing from 1.0)
        results = []
        for i, tool in enumerate(valid_tools[:top_k]):
            score = 1.0 - (i * 0.1)
            results.append((tool, max(0.1, score)))

        return results

    def process_all_decisions(self, output_path: str = None,
                             max_decisions: int = None) -> Dict:
        """
        Process all decision points in the calling graph using LLM

        Args:
            output_path: Output file path, if None then don't save file
            max_decisions: Maximum number of decisions to process (for testing)

        Returns:
            Processing results dictionary (same format as tool_selection.py)
        """
        results = {
            'method': 'llm_tool_selection',
            'model': self.model_name,
            'total_traces': 0,
            'total_decisions': 0,
            'selections': []
        }

        traces = self.calling_graph.get('traces', [])
        results['total_traces'] = len(traces)

        print(f"\nProcessing decision points in {len(traces)} traces using ToolSelectAgent...")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print(f"Candidates per decision: {self.n_candidates}")
        print(f"{'='*80}\n")

        decision_count = 0
        start_time = time.time()

        for trace_idx, trace in enumerate(traces):
            trace_id = trace.get('trace_id', 'unknown')
            nodes = trace.get('nodes', {})
            decisions = trace.get('decisions', [])

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

                # Check max_decisions limit
                if max_decisions and decision_count >= max_decisions:
                    print(f"\n✓ Reached max_decisions limit ({max_decisions}), stopping...")
                    break

                # Print progress
                decision_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / decision_count if decision_count > 0 else 0

                print(f"[{decision_count}/{max_decisions or '?'}] Trace {trace_idx + 1}/{len(traces)} | "
                      f"Node: {node_id} | LLM calls: {self.total_llm_calls} | "
                      f"Avg time: {avg_time:.1f}s/decision")
                print(f"  Task: {task_desc[:80]}..." if len(task_desc) > 80 else f"  Task: {task_desc}")

                # Step 1: Select top-n_candidates from all tools in pool
                print(f"Step 1: Selecting top-{self.n_candidates} from {len(self.tools)} tools...")
                all_tool_names = list(self.tools.keys())
                candidate_tools = self.select_top_tools(
                    task_desc,
                    all_tool_names,
                    top_k=self.n_candidates
                )
                candidate_tool_names = [tool for tool, score in candidate_tools]
                print(f"Step 2: Selecting top-1 from {len(candidate_tool_names)} candidates...")

                # Step 2: Select top-1 from the n_candidates
                if candidate_tool_names:
                    top_tool = self.select_top_tools(
                        task_desc,
                        candidate_tool_names,
                        top_k=1
                    )
                else:
                    top_tool = []

                # Check if ground truth is in candidates
                ground_truth_in_candidates = ground_truth in candidate_tool_names if ground_truth else None

                # Check if ground truth is the selected tool (top-1)
                selected_tool_name = top_tool[0][0] if top_tool else None
                ground_truth_is_selected = (selected_tool_name == ground_truth) if ground_truth and selected_tool_name else None

                # Show result
                if ground_truth:
                    gt_status = "✓" if ground_truth_is_selected else ("○" if ground_truth_in_candidates else "✗")
                    print(f"  Result: Selected '{selected_tool_name}' | GT: '{ground_truth}' {gt_status}")
                else:
                    print(f"  Result: Selected '{selected_tool_name}'")
                print()

                # If ground truth exists and not in candidates, add it
                if ground_truth and not ground_truth_in_candidates:
                    # Calculate "score" for ground truth (assign lowest score)
                    gt_score = min([score for _, score in candidate_tools]) - 0.1 if candidate_tools else 0.5
                    candidate_tools.append((ground_truth, max(0.1, gt_score)))

                selection = {
                    'trace_id': trace_id,
                    'node_id': node_id,
                    'task': task_desc,
                    'total_tools_in_pool': len(all_tool_names),
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
                        for tool, score in candidate_tools
                    ],
                    'selected_tool': {
                        'tool_name': top_tool[0][0],
                        'similarity_score': round(float(top_tool[0][1]), 4),
                        'is_ground_truth': ground_truth_is_selected if ground_truth_is_selected is not None else False
                    } if top_tool else None
                }

                results['selections'].append(selection)
                results['total_decisions'] += 1

                # Rate limiting (optional, adjust as needed)
                time.sleep(0.1)  # 100ms delay between calls

            # Break outer loop if max_decisions reached
            if max_decisions and decision_count >= max_decisions:
                break

        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"  Processed {results['total_decisions']} decision points")
        print(f"  Total LLM calls: {self.total_llm_calls}")
        print(f"  Total tokens used: {self.total_tokens}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Avg time per decision: {total_time/results['total_decisions']:.1f}s" if results['total_decisions'] > 0 else "")
        print(f"{'='*80}")

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
        print(f"  Ground truth not in candidates: {ground_truth_stats['ground_truth_not_in_candidates']}")
        print(f"  Ground truth is selected (top-1): {ground_truth_stats['ground_truth_is_selected']}")
        print(f"  Candidate retrieval rate: {ground_truth_stats['candidate_retrieval_rate']:.2%}")
        print(f"  Selection accuracy (top-1): {ground_truth_stats['selection_accuracy']:.2%}")

        # Save results to file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")

        return results

    async def _process_single_decision_async(self, decision_info: Dict) -> Dict:
        """
        Process a single decision point asynchronously

        Args:
            decision_info: Dictionary containing decision information

        Returns:
            Selection result dictionary
        """
        trace_id = decision_info['trace_id']
        node_id = decision_info['node_id']
        task_desc = decision_info['task']
        ground_truth = decision_info['ground_truth']
        all_tool_names = decision_info['all_tool_names']

        # Step 1: Select top-n_candidates from all tools
        candidate_tools = await self.select_top_tools_async(
            task_desc,
            all_tool_names,
            top_k=self.n_candidates
        )
        candidate_tool_names = [tool for tool, score in candidate_tools]

        # Step 2: Select top-1 from the n_candidates
        if candidate_tool_names:
            top_tool = await self.select_top_tools_async(
                task_desc,
                candidate_tool_names,
                top_k=1
            )
        else:
            top_tool = []

        # Check ground truth statistics
        ground_truth_in_candidates = ground_truth in candidate_tool_names if ground_truth else None
        selected_tool_name = top_tool[0][0] if top_tool else None
        ground_truth_is_selected = (selected_tool_name == ground_truth) if ground_truth and selected_tool_name else None

        # If ground truth not in candidates, add it for analysis
        if ground_truth and not ground_truth_in_candidates:
            gt_score = min([score for _, score in candidate_tools]) - 0.1 if candidate_tools else 0.5
            candidate_tools.append((ground_truth, max(0.1, gt_score)))

        # Build selection result
        selection = {
            'trace_id': trace_id,
            'node_id': node_id,
            'task': task_desc,
            'total_tools_in_pool': len(all_tool_names),
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
                for tool, score in candidate_tools
            ],
            'selected_tool': {
                'tool_name': top_tool[0][0],
                'similarity_score': round(float(top_tool[0][1]), 4),
                'is_ground_truth': ground_truth_is_selected if ground_truth_is_selected is not None else False
            } if top_tool else None
        }

        return selection

    async def _process_batch_async(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of decisions concurrently

        Args:
            batch: List of decision info dictionaries

        Returns:
            List of selection results
        """
        tasks = [self._process_single_decision_async(decision_info) for decision_info in batch]
        return await asyncio.gather(*tasks)

    def process_all_decisions_concurrent(self, output_path: str = None,
                                         max_decisions: int = None) -> Dict:
        """
        Process all decision points using concurrent LLM calls (much faster!)

        Args:
            output_path: Output file path, if None then don't save file
            max_decisions: Maximum number of decisions to process (for testing)

        Returns:
            Processing results dictionary
        """
        results = {
            'method': 'llm_tool_selection_concurrent',
            'model': self.model_name,
            'total_traces': 0,
            'total_decisions': 0,
            'selections': []
        }

        traces = self.calling_graph.get('traces', [])
        results['total_traces'] = len(traces)

        print(f"\n{'='*80}")
        print(f"Processing decision points using CONCURRENT mode")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print(f"Candidates per decision: {self.n_candidates}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Total traces: {len(traces)}")
        print(f"{'='*80}\n")

        # Collect all decision info
        all_decisions = []
        for trace in traces:
            trace_id = trace.get('trace_id', 'unknown')
            nodes = trace.get('nodes', {})
            decisions = trace.get('decisions', [])

            for decision in decisions:
                node_id = decision.get('node', '')
                candidates = decision.get('candidates', [])
                ground_truth = decision.get('chosen', '')

                if not candidates:
                    continue

                task_desc = nodes.get(node_id, {}).get('task', '')
                if not task_desc:
                    continue

                decision_info = {
                    'trace_id': trace_id,
                    'node_id': node_id,
                    'task': task_desc,
                    'ground_truth': ground_truth,
                    'all_tool_names': list(self.tools.keys())
                }
                all_decisions.append(decision_info)

                if max_decisions and len(all_decisions) >= max_decisions:
                    break

            if max_decisions and len(all_decisions) >= max_decisions:
                break

        print(f"Total decisions to process: {len(all_decisions)}")
        print(f"Processing in batches of {self.max_concurrent}...\n")

        start_time = time.time()

        # Process in batches
        for i in range(0, len(all_decisions), self.max_concurrent):
            batch = all_decisions[i:i + self.max_concurrent]
            batch_num = i // self.max_concurrent + 1
            total_batches = (len(all_decisions) + self.max_concurrent - 1) // self.max_concurrent

            print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch)} decisions concurrently...")

            # Run async batch processing
            batch_results = asyncio.run(self._process_batch_async(batch))
            results['selections'].extend(batch_results)

            # Update statistics
            for selection in batch_results:
                self.total_llm_calls += 2  # Two calls per decision

            results['total_decisions'] += len(batch_results)

            elapsed = time.time() - start_time
            avg_time = elapsed / results['total_decisions'] if results['total_decisions'] > 0 else 0

            # Show progress
            correct = sum(1 for s in batch_results if s.get('ground_truth_is_selected'))
            print(f"  Completed: {results['total_decisions']}/{len(all_decisions)} | "
                  f"Batch accuracy: {correct}/{len(batch_results)} | "
                  f"Avg: {avg_time:.2f}s/decision")
            print()

        total_time = time.time() - start_time

        print(f"{'='*80}")
        print(f"Processing complete!")
        print(f"  Processed {results['total_decisions']} decision points")
        print(f"  Total LLM calls: {self.total_llm_calls}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Avg time per decision: {total_time/results['total_decisions']:.2f}s")
        print(f"  Speedup vs sequential: ~{2.0/avg_time:.1f}x (estimated)")
        print(f"{'='*80}")

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

        print(f"\nGround Truth Statistics:")
        print(f"  Total decisions with ground truth: {ground_truth_stats['total_with_ground_truth']}")
        print(f"  Ground truth in candidates: {ground_truth_stats['ground_truth_in_candidates']}")
        print(f"  Ground truth is selected (top-1): {ground_truth_stats['ground_truth_is_selected']}")
        print(f"  Candidate retrieval rate: {ground_truth_stats['candidate_retrieval_rate']:.2%}")
        print(f"  Selection accuracy (top-1): {ground_truth_stats['selection_accuracy']:.2%}")

        # Save results
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")

        return results

    def print_sample_results(self, results: Dict = None, num_samples: int = 3):
        """
        Print some sample results

        Args:
            results: Processing results dictionary
            num_samples: Number of samples to display
        """
        if results is None:
            return

        print(f"\n{'='*80}")
        print(f"ToolSelectAgent Results")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Total Traces: {results['total_traces']}")
        print(f"Total Decisions: {results['total_decisions']}")
        print(f"\n{'='*80}")
        print(f"Sample Results (First {num_samples} Decision Points):")
        print(f"{'='*80}\n")

        for i, selection in enumerate(results['selections'][:num_samples], 1):
            print(f"Sample {i}:")
            print(f"  Trace ID: {selection['trace_id']}")
            print(f"  Node ID: {selection['node_id']}")
            print(f"  Task: {selection['task'][:100]}...")

            # Show ground truth info
            if selection.get('ground_truth'):
                print(f"  Ground Truth: {selection['ground_truth']}")
                print(f"  GT in candidates: {selection.get('ground_truth_in_candidates')}")
                print(f"  GT is selected: {selection.get('ground_truth_is_selected')}")

            # Show retrieved candidates
            print(f"  Retrieved Candidates (Top-{selection.get('n_candidates_retrieved', 5)}):")
            for j, tool in enumerate(selection.get('retrieved_candidates', []), 1):
                gt_marker = " ✓ [GT]" if tool.get('is_ground_truth') else ""
                print(f"    {j}. {tool['tool_name']} (score: {tool['similarity_score']:.4f}){gt_marker}")

            # Show selected tool
            if selection.get('selected_tool'):
                selected = selection['selected_tool']
                gt_marker = " ✓ [GT]" if selected.get('is_ground_truth') else ""
                print(f"  Selected Tool (Top-1): {selected['tool_name']} (score: {selected['similarity_score']:.4f}){gt_marker}")

            print()


def _find_data_file(filename: str) -> str:
    """Smart path resolution to find data files"""
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
    """Main function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_parent_dir = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description='LLM-based Tool Selection Agent (OpenAI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (GPT-4.1, sequential mode)
  python llm_tool_selection.py

  # Enable concurrent processing (MUCH FASTER!)
  python llm_tool_selection.py --concurrent

  # Concurrent mode with 20 parallel requests
  python llm_tool_selection.py --concurrent --max_concurrent 20

  # Use GPT-4o model with concurrent processing
  python llm_tool_selection.py --model gpt-4o --concurrent

  # Process only first 10 decisions (for testing)
  python llm_tool_selection.py --max_decisions 10 --concurrent

  # Full example
  python llm_tool_selection.py --model gpt-4o --concurrent --max_concurrent 15 --max_decisions 100

Note: Set OPENAI_API_KEY environment variable before running
Performance: Concurrent mode can be 5-10x faster than sequential!
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
        default='gpt-4.1',
        help='OpenAI model name (default: gpt-4.1)'
    )

    parser.add_argument(
        '--n_candidates',
        type=int,
        default=5,
        help='Number of candidates to retrieve (default: 5)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='LLM temperature (default: 1.0 for deterministic)'
    )

    parser.add_argument(
        '--max_decisions',
        type=int,
        default=None,
        help='Maximum number of decisions to process (default: None, process all)'
    )

    parser.add_argument(
        '--concurrent',
        action='store_true',
        help='Enable concurrent processing (much faster!)'
    )

    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=10,
        help='Maximum number of concurrent API calls (default: 10)'
    )

    args = parser.parse_args()

    # Auto-detect data files if not specified
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output file path
    output_filename = 'tool_selection_results.json'
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"\n{'='*80}")
    print(f"ToolSelectAgent Configuration")
    print(f"{'='*80}")
    print(f"  Tool Pool:       {args.tool_pool}")
    print(f"  Calling Graph:   {args.calling_graph}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  Model:           {args.model}")
    print(f"  N Candidates:    {args.n_candidates}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Max Decisions:   {args.max_decisions or 'all'}")
    print(f"  Concurrent Mode: {args.concurrent}")
    if args.concurrent:
        print(f"  Max Concurrent:  {args.max_concurrent}")
    print(f"{'='*80}\n")

    # Create ToolSelectAgent
    agent = ToolSelectAgent(
        args.tool_pool,
        args.calling_graph,
        model_name=args.model,
        n_candidates=args.n_candidates,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent
    )

    # Process all decision points
    if args.concurrent:
        results = agent.process_all_decisions_concurrent(
            output_path=output_path,
            max_decisions=args.max_decisions
        )
    else:
        results = agent.process_all_decisions(
            output_path=output_path,
            max_decisions=args.max_decisions
        )

    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"  Total Traces: {results['total_traces']}")
    print(f"  Total Decisions: {results['total_decisions']}")
    print(f"  Total LLM Calls: {agent.total_llm_calls}")
    print(f"  Total Tokens: {agent.total_tokens}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()