import json
import tqdm
from math_utils.grader import grade_answer
from pathlib import Path
import random
import os
import scipy.stats as stats
import numpy as np
from scipy.stats import beta

if "__main__" == __name__:    
    import argparse
    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--problem_file",
        type=Path,
        default="downloads/math_splits/test.json",
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--result_folder",
        type=Path,
        default="TE_result/TE_result/",
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--answer_name",
        type=Path,
        default="math-shepherd-mistral-7b-sft",
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Maximum number of samples to consider.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Whether using best-of-N strategy.",
    )
    parser.add_argument(
        "--problem_range",
        type=int,
        default=-1,
        help="Problem range.",
    )
    args = parser.parse_args()
    if args.problem_range == -1:
        # read from the problem file
        with open(args.problem_file, "r") as f:
            if args.problem_file.suffix == ".json":
                problems = json.load(f)
            elif args.problem_file.suffix == ".jsonl":
                problems = []
                for line in f:
                    problems.append(json.loads(line))
            args.problem_range = len(problems)

    total_entropies = []
    correct_entropies = []
    wrong_entropies = []

    answer_file = os.path.join(args.result_folder, f"{args.answer_name}_{args.num_samples}.json")
    record_file = os.path.join(args.result_folder, f"entropy_record.json")
    with open(answer_file, "r") as f:
        answers = [json.loads(line) for line in f]
    if not os.path.exists(record_file):
        records = {}
    else:
        try:
            with open(record_file, "r") as f:
                records = json.load(f)
        except:
            records = {}
    for i in range(args.problem_range):
        cal_entropy = 0
        for j in range(len(answers[i]['output'])):
            cal_entropy += answers[i]['entropy'][j]
        total_entropies.append(cal_entropy / len(answers[i]['output']))
    print(f"Averaged entropy: {np.mean(total_entropies)}")
    records[str(args.answer_name)] = np.mean(total_entropies)
    if len(answers) != args.problem_range:
        print("Answer length does not match problem length.")
        exit(0)
    with open(record_file, "w") as f:
        json.dump(records, f)