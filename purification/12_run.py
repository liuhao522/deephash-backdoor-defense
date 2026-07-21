# -*- coding: utf-8 -*-
"""12_run.py — Complete experiment entry point with baselines + evaluation + tables.

Usage:
  python 12_run.py --attack blended --dataset cifar10
  python 12_run.py --attack blended --dataset cifar10 --backbone resnet18 --demo 10
  python 12_run.py --attack badnets --dataset cifar10 --em_iters 10
  python 12_run.py --attack blended --dataset cifar10 --ablation

New flags:
  --ablation       Run ablation study (removes each module and measures impact)
  --skip_freq      Skip frequency filter (for debugging)
  --freq_method    Frequency method: channel_fft, dct
  --em_init        EM initialization: nearest, true_label
  --clean_epochs   Override clean model training epochs
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib, torch, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def main():
    parser = argparse.ArgumentParser(
        description='Backdoor Purification — Full Experiment Pipeline (v2.0)')

    # ---- Core settings ----
    parser.add_argument('--attack', type=str, default='blended',
                        choices=['badnets', 'blended', 'sig', 'wanet', 'refool', 'inputaware'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'gtsrb'])
    parser.add_argument('--target', type=int, default=7)
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['cnn', 'resnet18', 'mobilenet'])

    # ---- Pipeline control ----
    parser.add_argument('--demo', type=int, default=6,
                        help='Demo samples to visualize')
    parser.add_argument('--em_iters', type=int, default=8,
                        help='Max EM iterations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory override')
    parser.add_argument('--seed', type=int, default=42)

    # ---- Module overrides ----
    parser.add_argument('--skip_freq', action='store_true',
                        help='Skip frequency filter (identity pass-through)')
    parser.add_argument('--freq_method', type=str, default='dct',
                        choices=['dct', 'channel_fft', 'grayscale_fft', 'none'])
    parser.add_argument('--freq_z', type=float, default=3.0,
                        help='Frequency z-score threshold')
    parser.add_argument('--freq_attenuation', type=float, default=0.35,
                        help='Frequency attenuation factor')
    parser.add_argument('--grad_method', type=str, default='smoothgrad',
                        choices=['vanilla', 'smoothgrad', 'integrated'])
    parser.add_argument('--grad_floor', type=float, default=0.15,
                        help='Gradient mask floor')
    parser.add_argument('--em_init', type=str, default='nearest',
                        choices=['nearest', 'true_label', 'top3_nearest'])
    parser.add_argument('--coarse_steps', type=int, default=None,
                        help='EM coarse optimization steps per iter')
    parser.add_argument('--fine_steps', type=int, default=None,
                        help='EM fine optimization steps per iter')
    parser.add_argument('--clean_epochs', type=int, default=None,
                        help='Override clean model epochs')
    parser.add_argument('--lpips_size', type=int, default=64,
                        help='LPIPS resize target size')
    parser.add_argument('--n_purify', type=int, default=20,
                        help='Number of poisoned samples to purify (rest use as-is with true labels)')

    # ---- Modes ----
    parser.add_argument('--ablation', action='store_true',
                        help='Run full ablation study')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip purification, only baselines')
    parser.add_argument('--no_baselines', action='store_true',
                        help='Skip baseline comparisons')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visualizations')

    args = parser.parse_args()

    # ---- Build config with overrides ----
    PipelineConfig = importlib.import_module('01_config').PipelineConfig
    config = PipelineConfig(
        dataset=args.dataset,
        attack=args.attack,
        target_class=args.target,
        n_demo_samples=args.demo,
        em_max_iter=args.em_iters,
        backbone=args.backbone,
        seed=args.seed,
    )

    if args.output:
        config.output_root = args.output

    # Apply module overrides
    config.freq_method = args.freq_method
    config.freq_z_threshold = args.freq_z
    config.freq_attenuation = args.freq_attenuation
    config.grad_method = args.grad_method
    config.grad_mask_floor = args.grad_floor
    config.em_init_mode = args.em_init
    config.lpips_resize = args.lpips_size
    config.n_purify = args.n_purify

    if args.clean_epochs is not None:
        config.model_epochs_clean = args.clean_epochs
    if args.coarse_steps is not None:
        config.opt_steps_coarse = args.coarse_steps
    if args.fine_steps is not None:
        config.opt_steps_fine = args.fine_steps

    if args.skip_freq:
        config.freq_method = 'none'
        print("⚠ Frequency filter DISABLED (--skip_freq)")

    # ---- Print configuration ----
    print(f"\n{'='*60}")
    print(f"Backdoor Sample Purification — Full Experiment Pipeline v2.0")
    print(f"  Dataset:   {config.dataset.upper()}")
    print(f"  Attack:    {config.attack}  |  Target:  class {config.target_class}")
    print(f"  Backbone:  {args.backbone}")
    print(f"  EM init:   {config.em_init_mode}  |  Max iters: {config.em_max_iter}")
    print(f"  Freq:      {config.freq_method} (z>{config.freq_z_threshold})")
    print(f"  Gradient:  {config.grad_method} (floor={config.grad_mask_floor})")
    print(f"  LPIPS:     resize to {config.lpips_resize}×{config.lpips_resize}")
    print(f"  Output:    {config.exp_dir}")
    print(f"{'='*60}")

    t_total_start = time.time()

    # ================================================================
    # Step 1: Purification Pipeline
    # ================================================================
    PurificationPipeline = importlib.import_module('11_pipeline').PurificationPipeline

    if not args.eval_only:
        pipeline = (PurificationPipeline(config)
                    .load_data()
                    .train_clean_model()
                    .train_poisoned_model()
                    .train_patchgan()
                    .build_centers()
                    .run_purification())

        if not args.no_viz:
            pipeline.visualize()

        pipeline.save()
        print(f"\n  Pipeline complete → {config.exp_dir}")
    else:
        pipeline = PurificationPipeline(config)
        pipeline.load_data()
        pipeline.train_clean_model()
        pipeline.train_poisoned_model()
        pipeline.train_patchgan()
        pipeline.build_centers()
        pipeline.run_purification()
        print("  Eval-only mode: purification complete, skipping viz.")

    # ================================================================
    # Step 2: Baselines
    # ================================================================
    if not args.no_baselines:
        print(f"\n{'='*60}")
        print("BASELINE COMPARISONS")
        print(f"{'='*60}")

        # Prepare test loader — resize depends on backbone
        backbone = getattr(config, 'backbone', 'mobilenet')
        if backbone in ('resnet18', 'mobilenet'):
            test_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ])
        else:
            test_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ])
        cifar_test = datasets.CIFAR10(
            root=r'D:\deephash_original\data', train=False, download=False,
            transform=test_tf)
        test_loader = DataLoader(
            cifar_test, batch_size=config.batch_size, shuffle=False)

        ExperimentRunner = importlib.import_module('evaluation').ExperimentRunner
        runner = ExperimentRunner(config)
        results = runner.run_full(
            pipeline, pipeline.poisoned_model, pipeline.data, test_loader)

        # ---- Print comparison table ----
        print(f"\n{'='*70}")
        print(f"{'Method':<25} {'CA (%)':>8} {'DR (%)':>8}  Notes")
        print(f"{'-'*70}")
        for method, metrics in results.items():
            notes = metrics.get('note', metrics.get('method', ''))
            ca = metrics['CA']
            dr = metrics['DR']
            if isinstance(ca, (int, float)):
                print(f"{method:<25} {ca:>8.1f} {dr:>8.1f}  {notes}")
            else:
                print(f"{method:<25} {str(ca):>8} {str(dr):>8}  {notes}")

        # Add purification-specific stats
        n_correct = sum(
            1 for s in pipeline.metrics.per_sample if s.get('correct', False))
        n_total = len(pipeline.metrics.per_sample)
        print(f"\n  Purification accuracy: {n_correct}/{n_total} "
              f"({100*n_correct/max(1,n_total):.1f}%)")

        # ---- Generate tables ----
        ResultsTable = importlib.import_module('evaluation').ResultsTable
        table_data = {
            'dataset': config.dataset,
            'attack': config.attack,
            'methods': results,
            'purification': {
                'n_samples': n_total,
                'n_correct': n_correct,
                'acc': 100 * n_correct / max(1, n_total),
            }
        }
        ResultsTable.to_markdown(
            table_data, os.path.join(config.exp_dir, 'results_table.md'))
        ResultsTable.to_latex(
            table_data, os.path.join(config.exp_dir, 'results_table.tex'))

    # ================================================================
    # Step 3: Ablation Study (optional)
    # ================================================================
    if args.ablation:
        print(f"\n{'='*60}")
        print("ABLATION STUDY")
        print(f"{'='*60}")

        AblationStudy = importlib.import_module('evaluation').AblationStudy
        ablation = AblationStudy(config)
        df = ablation.run(PurificationPipeline, pipeline.data)

        print("\n" + df.to_string(index=False))

        # Save
        df.to_csv(os.path.join(config.exp_dir, 'ablation.csv'), index=False)
        ResultsTable.ablation_to_markdown(
            df, os.path.join(config.exp_dir, 'ablation.md'))
        ResultsTable.ablation_to_latex(
            df, os.path.join(config.exp_dir, 'ablation.tex'))

    # ---- Done ----
    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE ({t_total:.0f}s total)")
    print(f"Results: {config.exp_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
