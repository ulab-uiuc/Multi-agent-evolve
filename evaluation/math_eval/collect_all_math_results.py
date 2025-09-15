import os
import csv
import json
from collections import OrderedDict, defaultdict

def insert_sorted_acc_fields(result_dict):
    # Extract and remove all *_acc keys except "model"
    acc_fields = {
        k: result_dict.pop(k) for k in list(result_dict.keys())
        if k != "model" and k.endswith("_acc")
    }

    # Sort acc keys
    sorted_acc_fields = dict(sorted(acc_fields.items()))

    # Rebuild the OrderedDict with model first, then sorted accs, then the rest
    reordered = OrderedDict()
    reordered["model"] = result_dict["model"]
    reordered.update(sorted_acc_fields)
    reordered.update(result_dict)  # remaining keys are details

    return reordered

def convert_latex_table(data, selected_data=None):
    """
    Convert a list of dicts into a LaTeX table, sorted by descending average accuracy.

    Args:
        data (List[Dict]): your JSON‐like list.
        selected_data (List[str], optional):
            List of metric names _without_ the '_acc' suffix to include.
            E.g. ['aime24', 'amc23', 'hmmt_2024'].
            Defaults to all metrics found in data except 'avg_acc'.
    Returns:
        str: the LaTeX code for a table.
    """
    # 1. Infer all available metrics (minus the pre‐computed avg_acc) if none specified
    if selected_data is None:
        selected_data = sorted(
            k[:-4] for k in data[0].keys()
            if k.endswith('_acc') and k != "avg_acc"
        )

    # 2. Build rows: clean model name, grab each metric, compute new average
    rows = []
    for item in data:
        model_name = item["model"].replace("_temp0_n1_seed2", "")
        vals = []
        for metric in selected_data:
            key = f"{metric}_acc"
            vals.append(float(item.get(key, 0.0)))
        avg_selected = sum(vals) / len(vals) if vals else 0.0
        if model_name != "Qwen2.5-7B":
            model_name = model_name.replace("Qwen2.5-7B", "7B").replace("_stage1", "").replace("qwen2.5-7b", "7B")
            model_name = model_name.replace("_", "\\_")
        rows.append((model_name, vals, avg_selected))

    # 3. Sort rows by avg_selected descending
    rows.sort(key=lambda x: x[2], reverse=True)

    # 4. Start LaTeX
    col_spec = "l" + "r" * (len(selected_data) + 1)
    header = ["Model"] + [m.replace("_", r"\_") for m in selected_data] + ["Avg"]
    header = " & ".join(header) + r" \\"
    header = header.replace("livemathbench", "livemath").replace("olympiadbench", "olympiad").replace("minerva\\_math", "minerva").replace("hmmt\\_2024", "hmmt24")

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    for model, vals, avg in rows:
        formatted = [f"{v:.1f}" for v in vals] + [f"{avg:.1f}"]
        lines.append(" & ".join([model] + formatted) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Model accuracies on selected benchmarks, sorted by average}")
    lines.append(r"\label{tab:acc_sorted}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
def compute_method_ranks(data, selected_models=None, selected_data=None):
    """
    Compute, for each metric, the rank of each model (1 = best accuracy).

    Args:
        data (List[Dict]): your JSON‐like list of dicts.
        selected_models (List[str], optional):
            List of clean model names (with "_temp0_n1_seed2" already stripped)
            whose ranks you care about.  If None, returns ranks for _all_ models.
        selected_data (List[str], optional):
            List of metric names _without_ the "_acc" suffix.  If None,
            defaults to all keys ending in "_acc" except "avg_acc".

    Returns:
        Dict[str, Dict[str,int]]:  
            Outer: metric →  
            Inner: model_name → rank (1 = highest accuracy)
    """
    # 1. Determine which metrics to rank
    if selected_data is None:
        selected_data = sorted(
            k[:-4] for k in data[0].keys()
            if k.endswith("_acc") and k != "avg_acc"
        )

    # 2. Prepare clean model names + parsed accuracies
    models = []
    for item in data:
        clean_name = item["model"].replace("_temp0_n1_seed2", "")
        models.append((clean_name, item))

    # 3. For each metric, sort and assign ranks
    all_ranks = {}
    for metric in selected_data:
        key = f"{metric}_acc"
        # build list of (model, float(acc))
        vals = [
            (name, float(item.get(key, 0.0)))
            for name, item in models
        ]
        # sort desc by accuracy
        vals.sort(key=lambda x: x[1], reverse=True)
        # assign ranks (1-based). Ties get the same rank.
        ranks = {}
        prev_score = None
        prev_rank = 0
        for idx, (name, score) in enumerate(vals, start=1):
            if score == prev_score:
                rank = prev_rank
            else:
                rank = idx
            ranks[name] = rank
            prev_score, prev_rank = score, rank

        # if user only wants a subset, filter
        if selected_models is not None:
            ranks = {m: ranks[m] for m in selected_models if m in ranks}

        all_ranks[metric] = ranks

    return all_ranks

def collect_eval_results_by_prefix(root):
    all_results = []

    for model_dir in os.listdir(root):
        model_path = os.path.join(root, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Look for the eval_results directory and its subdirectories
        eval_results_dir = os.path.join(model_path, "eval_results")
        if not os.path.isdir(eval_results_dir):
            print(f"⚠️ Missing eval_results directory for: {model_dir}")
            continue
            
        # Find the global_step directory (assuming there might be only one)
        global_step_dirs = [d for d in os.listdir(eval_results_dir) if os.path.isdir(os.path.join(eval_results_dir, d))]
        if not global_step_dirs:
            print(f"⚠️ No global step directories found in: {eval_results_dir}")
            continue
            
        # Use the first global step directory (usually global_step_0)
        global_step_dir = os.path.join(eval_results_dir, global_step_dirs[0])
        
        # Create a new result entry for this model
        result = OrderedDict()
        result["model"] = model_dir
        
        # Collect accuracies from each benchmark directory
        benchmark_dirs = [d for d in os.listdir(global_step_dir) if os.path.isdir(os.path.join(global_step_dir, d))]
        
        for benchmark in benchmark_dirs:
            if "livemath" in benchmark :
                # skip livemathbench or "aime25" in benchmark
                continue
            benchmark_path = os.path.join(global_step_dir, benchmark)
            
            # Look for the metrics json file
            metrics_files = [f for f in os.listdir(benchmark_path) if f.endswith('_metrics.json')]
            if not metrics_files:
                print(f"⚠️ No metrics file found for {model_dir}/{benchmark}")
                continue
                
            # Use the first metrics file found
            metrics_file = os.path.join(benchmark_path, metrics_files[0])
            
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    
                # Extract the accuracy value
                if 'acc' in metrics_data:
                    result[f"{benchmark}_acc"] = metrics_data['acc']
                else:
                    print(f"⚠️ No accuracy found in {metrics_file}")
            except Exception as e:
                print(f"⚠️ Error reading {metrics_file}: {e}")
        
        # Only add results if we have some accuracies
        if len(result) > 1:  # More than just the "model" key
            # Calculate average accuracy
            acc_values = [v for k, v in result.items() if k.endswith('_acc')]
            if acc_values:
                avg_acc = sum(acc_values) / len(acc_values)
                result["avg_acc"] = round(avg_acc, 1)
                
                # Add metadata about how many benchmarks were averaged
                result["avg_metadata"] = {
                    "num_benchmarks": len(acc_values),
                    "benchmarks": [k[:-4] for k in result.keys() if k.endswith('_acc') and k != "avg_acc"]
                }
            
            result = insert_sorted_acc_fields(result)
            all_results.append(result)
        else:
            print(f"⚠️ No accuracies found for {model_dir}")

    # sort by model name
    all_results.sort(key=lambda x: x["model"])
    output_path = os.path.join(root, "combined_eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved structured JSON to {output_path}")
# Example usage
collect_eval_results_by_prefix("./EVAL/checkpoints")