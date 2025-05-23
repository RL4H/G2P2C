import argparse
from pathlib import Path
import subprocess
import pandas as pd


def run_dmms(exe: Path, cfg: Path, log: Path, results_dir: Path) -> None:
    """Run DMMS.R simulator once using CLI and print path to generated CSV files."""
    exe = Path(exe)
    cfg = Path(cfg)
    log = Path(log)
    results_dir = Path(results_dir)

    cmd = [str(exe), str(cfg), str(log), str(results_dir)]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"DMMS.R failed with code {e.returncode}")
        raise

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    csv_files = list(results_dir.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
    else:
        print(f"Generated CSV files:")
        for f in csv_files:
            print(f" - {f}")

        # Example of loading the first CSV file
        try:
            df = pd.read_csv(csv_files[0])
            print(f"Loaded {csv_files[0]} with {len(df)} rows")
        except Exception as e:
            print(f"Failed to load {csv_files[0]}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run DMMS.R via CLI")
    parser.add_argument("exe", type=Path, help="Path to DMMS.R executable")
    parser.add_argument("cfg", type=Path, help="Path to config XML")
    parser.add_argument("log", type=Path, help="Path to log file")
    parser.add_argument("results", type=Path, help="Directory for results")
    args = parser.parse_args()

    run_dmms(args.exe, args.cfg, args.log, args.results)


if __name__ == "__main__":
    main()
