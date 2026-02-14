import argparse
import random
import sys

def reservoir_sample(input_file, n):
    """
    Samples n lines from a file using reservoir sampling algorithm.
    This works efficiently for very large files without loading everything into memory.
    """
    reservoir = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < n:
                    reservoir.append(line)
                else:
                    j = random.randint(0, i)
                    if j < n:
                        reservoir[j] = line
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
        
    return reservoir

def main():
    parser = argparse.ArgumentParser(description="Sample n lines from a large JSONL file.")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    parser.add_argument("--n", type=int, default=500, help="Number of samples to take (default: 500)")
    
    args = parser.parse_args()
    
    print(f"Sampling {args.n} lines from {args.input_file}...")
    sampled_lines = reservoir_sample(args.input_file, args.n)
    
    if not sampled_lines:
        print("Warning: No lines were read from the input file.")
        return

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)

    print(f"Successfully saved {len(sampled_lines)} samples to {args.output_file}")

if __name__ == "__main__":
    main()
