#!/usr/bin/env python3
"""
Master Pipeline Script

Runs the complete analysis pipeline end-to-end:
  1. Data preparation (preprocessing, QC)
  2. Model training (AutoEncoders + GNN)
  3. Results analysis (visualization, correlations)

Usage:
    python scripts/run_pipeline.py [--steps STEPS]

Options:
    --steps STEPS    Comma-separated list of steps to run (default: all)
                     Options: prep, train, analyze, all
    --skip-qc        Skip QC plot generation (faster)
    --help           Show this help message

Examples:
    # Run complete pipeline
    python scripts/run_pipeline.py

    # Run only preparation
    python scripts/run_pipeline.py --steps prep

    # Run training and analysis
    python scripts/run_pipeline.py --steps train,analyze

    # Skip QC plots for faster execution
    python scripts/run_pipeline.py --skip-qc
"""

import sys
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


class PipelineRunner:
    """
    Master pipeline orchestrator

    Runs all analysis steps in sequence with error handling and logging.
    """

    def __init__(self, skip_qc: bool = False):
        self.skip_qc = skip_qc
        self.start_time = None
        self.step_times = {}

    def run_step(self, step_name: str, script_path: Path) -> bool:
        """
        Run a pipeline step

        Args:
            step_name: Name of the step
            script_path: Path to Python script

        Returns:
            True if successful, False otherwise
        """
        print("")
        print("=" * 80)
        print(f"RUNNING: {step_name}")
        print("=" * 80)
        print(f"Script: {script_path}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")

        step_start = time.time()

        try:
            # Run script as subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=False  # Show output in real-time
            )

            step_time = time.time() - step_start
            self.step_times[step_name] = step_time

            print("")
            print(f"✓ {step_name} completed successfully")
            print(f"  Time: {step_time:.1f} seconds ({step_time/60:.1f} minutes)")
            print("=" * 80)

            return True

        except subprocess.CalledProcessError as e:
            step_time = time.time() - step_start
            self.step_times[step_name] = step_time

            print("")
            print(f"✗ {step_name} FAILED")
            print(f"  Error code: {e.returncode}")
            print(f"  Time: {step_time:.1f} seconds")
            print("=" * 80)

            return False

    def run_preparation(self) -> bool:
        """Run data preparation step"""
        return self.run_step(
            "Step 1: Data Preparation",
            PROJECT_ROOT / "scripts" / "01_prepare_data.py"
        )

    def run_training(self) -> bool:
        """Run model training step"""
        return self.run_step(
            "Step 2: Model Training",
            PROJECT_ROOT / "scripts" / "02_train_models.py"
        )

    def run_analysis(self) -> bool:
        """Run results analysis step"""
        return self.run_step(
            "Step 3: Results Analysis",
            PROJECT_ROOT / "scripts" / "03_analyze_results.py"
        )

    def print_summary(self):
        """Print pipeline execution summary"""
        total_time = time.time() - self.start_time

        print("")
        print("=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print("")
        print("Step timings:")

        for step_name, step_time in self.step_times.items():
            mins = step_time / 60
            pct = (step_time / total_time) * 100
            print(f"  {step_name:.<50} {mins:>6.1f} min ({pct:>5.1f}%)")

        print(f"  {'Total':.<50} {total_time/60:>6.1f} min")
        print("")
        print("=" * 80)
        print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def run(self, steps: list = None):
        """
        Run pipeline

        Args:
            steps: List of steps to run. If None, runs all steps.
                   Options: ['prep', 'train', 'analyze']
        """
        self.start_time = time.time()

        # Default to all steps
        if steps is None or 'all' in steps:
            steps = ['prep', 'train', 'analyze']

        print("")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 15 + "HUPERZIA ANALYSIS PIPELINE" + " " * 37 + "║")
        print("╚" + "=" * 78 + "╝")
        print("")
        print("Multi-Omics Integration Pipeline for Alkaloid Biosynthesis Gene Discovery")
        print("")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: {PROJECT_ROOT}")
        print(f"Steps to run: {', '.join(steps)}")
        print("")

        success = True

        # Step 1: Data Preparation
        if 'prep' in steps:
            if not self.run_preparation():
                print("\n✗ Pipeline aborted: Data preparation failed")
                success = False
                return success

        # Step 2: Model Training
        if 'train' in steps:
            if not self.run_training():
                print("\n✗ Pipeline aborted: Model training failed")
                success = False
                return success

        # Step 3: Results Analysis
        if 'analyze' in steps:
            if not self.run_analysis():
                print("\n✗ Pipeline aborted: Results analysis failed")
                success = False
                return success

        # Print summary
        self.print_summary()

        if success:
            print("")
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print("")
            print("Next steps:")
            print("  1. Review visualizations in: outputs/figures/")
            print("  2. Check correlations in: outputs/tables/")
            print("  3. Examine logs in: outputs/logs/")
            print("")

        return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run multi-omics integration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run (prep,train,analyze,all)'
    )

    parser.add_argument(
        '--skip-qc',
        action='store_true',
        help='Skip QC plot generation (faster)'
    )

    args = parser.parse_args()

    # Parse steps
    if args.steps == 'all':
        steps = ['all']
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]

    # Validate steps
    valid_steps = {'prep', 'train', 'analyze', 'all'}
    invalid = set(steps) - valid_steps
    if invalid:
        print(f"Error: Invalid steps: {', '.join(invalid)}")
        print(f"Valid steps: {', '.join(valid_steps)}")
        return 1

    # Run pipeline
    runner = PipelineRunner(skip_qc=args.skip_qc)
    success = runner.run(steps=steps)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
