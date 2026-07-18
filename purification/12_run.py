# -*- coding: utf-8 -*-
"""12_run.py — Complete experiment entry point with baselines + evaluation + tables.

Usage:
  python 12_run.py --attack blended --dataset cifar10
  python 12_run.py --attack blended --dataset cifar10 --eval_only
  python 12_run.py --attack badnets --dataset cifar10 --backbone resnet18
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib, torch, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def main():
    parser = argparse.ArgumentParser(description='Backdoor Purification — Full Experiment Pipeline')
    parser.add_argument('--attack', type=str, default='blended',
                       choices=['badnets','blended','sig','wanet','refool','inputaware'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist','cifar10','gtsrb'])
    parser.add_argument('--target', type=int, default=7)
    parser.add_argument('--demo', type=int, default=6, help='Demo samples to visualize')
    parser.add_argument('--em_iters', type=int, default=5)
    parser.add_argument('--backbone', type=str, default='cnn',
                       choices=['cnn','resnet18'], help='Feature extractor backbone')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true', help='Skip purification, only baselines')
    parser.add_argument('--no_baselines', action='store_true', help='Skip baseline comparisons')
    args = parser.parse_args()

    PipelineConfig = importlib.import_module('01_config').PipelineConfig
    config = PipelineConfig(
        dataset=args.dataset, attack=args.attack, target_class=args.target,
        n_demo_samples=args.demo, em_max_iter=args.em_iters,
    )
    if args.output: config.output_root = args.output

    print(f"\n{'='*60}")
    print(f"Backdoor Sample Purification — Full Experiment Pipeline")
    print(f"  Dataset:   {config.dataset.upper()}")
    print(f"  Attack:    {config.attack}  |  Target:  class {config.target_class}")
    print(f"  Backbone:  {args.backbone}")
    print(f"  Output:    {config.exp_dir}")
    print(f"{'='*60}")

    # ===== Step 1: Run purification pipeline =====
    PurificationPipeline = importlib.import_module('11_pipeline').PurificationPipeline

    if not args.eval_only:
        pipeline = (PurificationPipeline(config)
                    .load_data()
                    .train_clean_model()       # clean CIFAR-10 → good features
                    .train_poisoned_model()    # poisoned data → gradient only
                    .train_patchgan()
                    .build_centers()
                    .run_purification()
                    .visualize()
                    .save())
        print(f"\n  Pipeline complete → {config.exp_dir}")
    else:
        # In eval-only mode, just load data + train model
        pipeline = PurificationPipeline(config)
        pipeline.load_data()
        pipeline.train_clean_model()
        pipeline.train_poisoned_model()
        pipeline.train_patchgan()
        pipeline.build_centers()
        print("  Eval-only mode: model ready, skipping purification.")

    # ===== Step 2: Baselines =====
    if not args.no_baselines:
        print(f"\n{'='*60}")
        print("BASELINE COMPARISONS")
        print(f"{'='*60}")

        # Prepare test loader
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ]))
        test_loader = DataLoader(cifar_test, batch_size=config.batch_size, shuffle=False)

        ExperimentRunner = importlib.import_module('evaluation').ExperimentRunner
        runner = ExperimentRunner(config)
        results = runner.run_full(pipeline, pipeline.poisoned_model, pipeline.data, test_loader)

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"{'Method':<25} {'CA (%)':>8} {'DR (%)':>8}  Notes")
        print(f"{'-'*70}")
        for method, metrics in results.items():
            notes = metrics.get('note', metrics.get('method', ''))
            print(f"{method:<25} {metrics['CA']:>8.1f} {metrics['DR']:>8.1f}  {notes}")

        # Generate tables
        ResultsTable = importlib.import_module('evaluation').ResultsTable
        table_data = {
            'dataset': config.dataset, 'attack': config.attack,
            'methods': results
        }
        ResultsTable.to_markdown(table_data, os.path.join(config.exp_dir, 'results_table.md'))
        ResultsTable.to_latex(table_data, os.path.join(config.exp_dir, 'results_table.tex'))

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Results: {config.exp_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
