import argparse
from jiayan import load_lm, CRFPunctuator

def main():
    parser = argparse.ArgumentParser(description="Quick inference of a single sample using Jiayan")
    parser.add_argument("--text", type=str, required=True, help="Input raw text without punctuation")
    parser.add_argument("--lm", type=str, default="jiayan.klm", help="Path to language model")
    parser.add_argument("--cut_model", type=str, default="cut_model", help="Path to segmentation model")
    parser.add_argument("--punc_model", type=str, default="punc_model", help="Path to punctuation model")
    args = parser.parse_args()

    print("Loading language model...")
    lm = load_lm(args.lm)
    
    print("Loading punctuation model...")
    punctuator = CRFPunctuator(lm, args.cut_model)
    punctuator.load(args.punc_model)
    
    print("Inference...")
    result = punctuator.punctuate(args.text)
    
    print("-" * 50)
    print(f"Input : {args.text}")
    print(f"Output: {result}")
    print("-" * 50)

if __name__ == "__main__":
    main()
