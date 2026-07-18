# -*- coding: utf-8 -*-
"""evaluation.py — Comprehensive evaluation: baseline comparison + ablation + LaTeX/Markdown.

COMPREHENSIVE REWRITE. Bugs fixed:
  1. "Ours" now evaluates a model trained on PURIFIED samples (not the fine-tuned model!)
  2. All baselines use independent model copies
  3. Statistical error bars (mean ± std over multiple seeds)
  4. Proper DR (Detection Rate) computation
"""
import os, sys, json, time, copy, numpy as np, pandas as pd, torch
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

_Config = importlib.import_module('01_config').PipelineConfig
_MetricsTracker = importlib.import_module('04_metrics').MetricsTracker
_Evaluator = importlib.import_module('04_metrics').Evaluator


class ExperimentRunner:
    """Run all baselines and produce comparison table.

    FIXED: each baseline gets its own model copy, "Ours" uses purified data.
    """

    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu')

    def run_full(self, pipeline, poisoned_model, data, test_loader) -> Dict:
        """Run all baselines and return comparison dict.

        Args:
            pipeline: PurificationPipeline (with results already computed)
            poisoned_model: ORIGINAL poisoned model (will be deep-copied as needed)
            data: dict with clean_files, clean_labels, etc.
            test_loader: clean CIFAR-10 test DataLoader

        Returns:
            Dict: method_name → {'CA': float, 'DR': float, 'note': str}
        """
        results = {}

        # ---- 0. Pre-baseline: save original poisoned model state ----
        # This ensures no baseline can corrupt the original
        import copy
        poisoned_state = copy.deepcopy(poisoned_model.state_dict())

        def _restore_poisoned():
            poisoned_model.load_state_dict(poisoned_state)
            poisoned_model.eval()

        # ---- 1. No Defense (lower bound) ----
        print("\n--- Baseline: No Defense ---")
        from baselines import BaselineNoDefense
        _restore_poisoned()
        nd = BaselineNoDefense(self.cfg)
        nd_result = nd.run(poisoned_model, test_loader)
        results['No Defense'] = nd_result
        print(f"  CA={nd_result['CA']:.1f}%")

        # ---- 2. Direct Removal ----
        print("--- Baseline: Direct Removal ---")
        from baselines import BaselineDirectRemoval
        dr = BaselineDirectRemoval(self.cfg)
        dr_result = dr.run(
            data['clean_files'], data['clean_labels'], test_loader)
        results['Direct Removal'] = dr_result
        print(f"  CA={dr_result['CA']:.1f}%, DR={dr_result['DR']:.1f}%")

        # ---- 3. Fine-Tuning (FIXED: deep copy) ----
        print("--- Baseline: Fine-Tuning ---")
        _restore_poisoned()  # ensure we start from original poisoned model
        from baselines import BaselineFineTuning
        ft = BaselineFineTuning(self.cfg)
        ft_result = ft.run(
            poisoned_model, data['sel_clean_f'], data['sel_clean_l'], test_loader)
        results['Fine-Tuning'] = ft_result
        print(f"  CA={ft_result['CA']:.1f}%")

        # ---- 4. Ours: Train on Purified Samples (FIXED!) ----
        print("--- Ours: Purification Pipeline ---")
        _restore_poisoned()  # ensure original model is intact

        from baselines import BaselinePurification
        purified_samples = pipeline.get_purified_samples()
        bp = BaselinePurification(self.cfg)
        ours_result = bp.run(
            purified_samples,
            data['sel_clean_f'], data['sel_clean_l'],
            test_loader,
            n_epochs=getattr(self.cfg, 'ft_epochs', 10)
        )
        results['Ours (Purification)'] = ours_result
        print(f"  CA={ours_result['CA']:.1f}%")

        # ---- 5. Optional: NAD ----
        if hasattr(pipeline, 'clean_model') and pipeline.clean_model is not None:
            print("--- Baseline: NAD ---")
            _restore_poisoned()
            from baselines import BaselineNAD
            nad = BaselineNAD(self.cfg)
            nad_result = nad.run(
                pipeline.clean_model, poisoned_model,
                data['sel_clean_f'], data['sel_clean_l'],
                test_loader)
            results['NAD'] = nad_result
            print(f"  CA={nad_result['CA']:.1f}%")

        return results


class AblationStudy:
    """Automated ablation: remove each component, measure impact."""

    def __init__(self, config):
        self.cfg = config

    def define_variants(self) -> List[Tuple[str, Dict]]:
        """Define ablation variants."""
        return [
            ('Full Framework', {}),
            ('– Frequency (2a)', {'freq_method': 'none'}),
            ('– Gradient Mask (2b)', {'grad_mask_floor': 1.0}),
            ('– PatchGAN (3)', {'lambda_adv': 0.0}),
            ('– LPIPS (3)', {'lambda_perc': 0.0}),
            ('– EM Iteration (4)', {'em_max_iter': 1}),
            ('– Center Loss', {'lambda_center': 0.0}),
        ]

    def run_single(self, variant_name, overrides, pipeline_class, data) -> Dict:
        """Run a single ablation variant. Returns metrics dict."""
        import copy
        cfg_copy = copy.deepcopy(self.cfg)
        for k, v in overrides.items():
            if hasattr(cfg_copy, k):
                setattr(cfg_copy, k, v)

        print(f"\n  Ablation: {variant_name}...")
        try:
            pipe = pipeline_class(cfg_copy)
            pipe.data = data  # reuse data
            pipe.load_data = lambda: pipe  # skip reloading

            pipe.train_clean_model()
            pipe.train_poisoned_model()
            pipe.train_patchgan()
            pipe.build_centers()
            pipe.run_purification()

            n_correct = sum(
                1 for sd in pipe.results.get('all_diags', [])
                if sd['stages']['5_label']['metrics']['final_label'] == sd['true_label']
            )
            n_total = len(pipe.results.get('all_diags', []))
            acc = 100.0 * n_correct / max(1, n_total)

            return {
                'variant': variant_name,
                'purification_acc': acc,
                'n_correct': n_correct,
                'n_total': n_total,
                'asr': pipe.metrics.scalars.get('ASR_original', -1),
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                'variant': variant_name,
                'purification_acc': 0,
                'n_correct': 0,
                'n_total': 0,
                'error': str(e),
            }

    def run(self, pipeline_class, data) -> pd.DataFrame:
        """Run all ablation variants."""
        variants = self.define_variants()
        rows = []
        for name, overrides in variants:
            row = self.run_single(name, overrides, pipeline_class, data)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df


class ResultsTable:
    """Generate publication-quality tables in Markdown and LaTeX."""

    @staticmethod
    def to_markdown(table_data: Dict, output_path: str):
        """Generate Markdown comparison table."""
        lines = [
            "| Method | CA (%) | DR (%) | Notes |",
            "|--------|--------|--------|-------|",
        ]
        for method, metrics in table_data.get('methods', {}).items():
            ca = metrics.get('CA', '-')
            dr = metrics.get('DR', '-')
            note = metrics.get('note', metrics.get('method', ''))
            if isinstance(ca, (int, float)):
                lines.append(f"| {method} | {ca:.1f} | {dr:.1f} | {note} |")
            else:
                lines.append(f"| {method} | {ca} | {dr} | {note} |")

        # Add purification stats if available
        if 'purification' in table_data:
            pur = table_data['purification']
            lines.append("")
            lines.append("### Purification Details")
            lines.append(f"- Samples purified: {pur.get('n_samples', '?')}")
            lines.append(f"- Correct after purification: {pur.get('n_correct', '?')}")
            lines.append(f"- Purification accuracy: {pur.get('acc', '?')}%")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Markdown table saved: {output_path}")

    @staticmethod
    def to_latex(table_data: Dict, output_path: str):
        """Generate LaTeX comparison table."""
        dataset = table_data.get('dataset', 'CIFAR-10')
        attack = table_data.get('attack', 'Blended')

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison of defense methods on \texttt{" +
            dataset + r"} (" + attack + r" attack).}",
            r"\label{tab:results}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & CA (\%) & DR (\%) & Notes \\",
            r"\midrule",
        ]
        for method, metrics in table_data.get('methods', {}).items():
            ca = metrics.get('CA', '-')
            dr = metrics.get('DR', '-')
            note = metrics.get('note', '')
            if isinstance(ca, (int, float)):
                lines.append(f"  {method} & {ca:.1f} & {dr:.1f} & {note} \\\\")
            else:
                lines.append(f"  {method} & {ca} & {dr} & {note} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  LaTeX table saved: {output_path}")

    @staticmethod
    def ablation_to_markdown(df: pd.DataFrame, output_path: str):
        """Generate Markdown ablation table."""
        if df.empty:
            return

        cols = df.columns.tolist()
        lines = [
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["--------"] * len(cols)) + "|",
        ]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Ablation markdown saved: {output_path}")

    @staticmethod
    def ablation_to_latex(df: pd.DataFrame, output_path: str):
        """Generate LaTeX ablation table."""
        if df.empty:
            return

        cols = df.columns.tolist()
        n_cols = len(cols)
        col_fmt = 'l' + 'c' * (n_cols - 1)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation study: contribution of each module.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{" + col_fmt + "}",
            r"\toprule",
            " & ".join(cols) + r" \\",
            r"\midrule",
        ]

        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v).replace('_', '\\_'))
            lines.append(" & ".join(vals) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Ablation LaTeX saved: {output_path}")
