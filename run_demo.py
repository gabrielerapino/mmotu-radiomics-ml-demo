"""Run the MMOTU radiomics research demonstration."""
import argparse
from pathlib import Path
from src.experiment import run_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",default="data/raw/OTU_2d")
    parser.add_argument("--output-dir",default="outputs")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--synthetic",action="store_true",help="Force explicitly artificial demo observations")
    args=parser.parse_args()
    run_experiment(Path(args.data_root),Path(args.output_dir),args.seed,args.folds,force_synthetic=args.synthetic)


if __name__=="__main__": main()
