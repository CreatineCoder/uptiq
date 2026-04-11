"""
Phase 4.4 — CLI Entry Point for the Benchmark Pipeline.

Usage:
  # Full benchmark (1,500 queries)
  python evaluation/scripts/run_benchmark.py --config configs/default.yaml

  # Pilot run (10 queries for testing)
  python evaluation/scripts/run_benchmark.py --config configs/default.yaml --pilot 10

  # Evaluate only (on existing results, no agent runs)
  python evaluation/scripts/run_benchmark.py --evaluate-only --results-dir data/results
"""
import os
import sys
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Manual .env loader
def load_env():
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--pilot", type=int, default=None, help="Run only N queries (for testing/cost estimation)")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip agent runs, only evaluate existing results")
    parser.add_argument("--results-dir", type=str, default="data/results", help="Results directory for evaluate-only mode")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load env
    load_env()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY is missing. Please add it to your .env file.")
        sys.exit(1)
    
    if args.evaluate_only:
        print("📊 Evaluate-only mode is not yet implemented. Run the full benchmark first.")
        sys.exit(0)
    
    # Run the benchmark
    from src.pipeline.benchmark_runner import BenchmarkRunner
    from src.analysis.analyzer import generate_analysis_report
    
    runner = BenchmarkRunner(config_path=args.config, pilot=args.pilot)
    runner.run()
    
    print("\n✅ Benchmark complete! Running failure analyzer...")
    try:
        generate_analysis_report()
        print("✅ Analysis report generated. Check the Streamlit dashboard!")
    except Exception as e:
        print(f"⚠️ Analysis failed: {e}")

if __name__ == "__main__":
    main()
