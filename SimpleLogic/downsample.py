import argparse
import pandas as pd
import random

def main(args):
    # Load the data
    print(f"Loading data from {args.input_path}")
    data = pd.read_csv(args.input_path)
    
    print(f"Original data shape: {data.shape}")
    print(f"Original k distribution:\n{data['k'].value_counts().sort_index()}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Downsample according to specifications
    sampled_dfs = []
    
    for k_val in sorted(data['k'].unique()):
        k_data = data[data['k'] == k_val]
        
        if k_val == 1:
            # Keep 1000 k=1 problems
            if len(k_data) > args.k1_sample:
                sampled = k_data.sample(n=args.k1_sample, random_state=args.seed)
            else:
                sampled = k_data
            print(f"k={k_val}: {len(k_data)} -> {len(sampled)}")
        elif k_val == 2:
            # Keep 1000 k=2 problems
            if len(k_data) > args.k2_sample:
                sampled = k_data.sample(n=args.k2_sample, random_state=args.seed)
            else:
                sampled = k_data
            print(f"k={k_val}: {len(k_data)} -> {len(sampled)}")
        elif k_val in [3, 4]:
            # Keep all k=3, k=4 problems
            sampled = k_data
            print(f"k={k_val}: {len(k_data)} -> {len(sampled)} (keeping all)")
        else:
            # for now drop k>4
            sampled = k_data.iloc[0:0]  # empty dataframe
            print(f"k={k_val}: {len(k_data)} -> {len(sampled)} (dropping all)")

        sampled_dfs.append(sampled)
    
    # Combine and shuffle
    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # Shuffle
    
    print(f"\nFinal data shape: {result.shape}")
    print(f"Final k distribution:\n{result['k'].value_counts().sort_index()}")
    
    # Save
    print(f"\nSaving to {args.output_path}")
    result.to_csv(args.output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_11_500k/simplelogic_heldout_k_sufficient_data_new.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output_path",
        default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_11_500k/simplelogic_heldout_k_sufficient_data_new_sampled.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--k1_sample",
        type=int,
        default=1000,
        help="Number of k=1 samples to keep.",
    )
    parser.add_argument(
        "--k2_sample",
        type=int,
        default=1000,
        help="Number of k=2 samples to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    main(parser.parse_args())
    