import json
import tqdm
from math_utils.grader import grade_answer
from pathlib import Path
import random
import os
import scipy.stats as stats
import numpy as np
from scipy.stats import beta

def main(problem_file: Path, result_folder: Path, answer_name: str, greedy = False, reward = False, max_samples = 32, best = False, output_reward = False, seed = 0, problem_range = 0, level = -1) -> None:
    r"""
        The default strategy is majority voting.
        Args:
            greedy: whether using greedy decoding strategy. If set to True,
                    only file with "greedy" in the name will be used
            reward: whether reward is used as weight. If set to True,
                    only file with "reward" in the name will be used
            max_samples: maximum number of samples to consider
            best: whether using best-of-N strategy, only valid when reward is used
            output_reward: deprecated
        Returns:
            accuracy: accuracy of the model
            upper_bound_accuracy: upper bound accuracy of the model
            entropy: entropy of outputs (answers)
            majority_upper_bound_accuracy: majority upper bound accuracy, only valid when using majority voting with reward
    """
    # load the problems and set the random seed
    with open(problem_file, "r") as f:
        if problem_file.suffix == ".json":
            problems = json.load(f)
        elif problem_file.suffix == ".jsonl":
            problems = [json.loads(line) for line in f]
        if problem_range > 0:
            # -1 means all problems
            problems = problems[:problem_range]
        id_set = set()
        for problem in problems:
            if level != -1 and problem["level"] != level:
                continue
            id_set.add(problem["unique_id"])
    random.seed(seed)

    # load the answers
    if str(answer_name).endswith(".json"):
        all_answer_files = [str(result_folder / answer_name)]
    else:
        all_answer_files = list(result_folder.glob(f"{answer_name}*.json"))
        all_answer_files = [all_answer_files[i] for i in range(len(all_answer_files)) \
                        if "acc" not in str(all_answer_files[i]) and "copy" not in str(all_answer_files[i])]
    print("all_answer_files: ", all_answer_files)
    if greedy == False:
        all_answer_files = [str(file) for file in all_answer_files if "greedy" not in str(file)]
    else:
        all_answer_files = [str(file) for file in all_answer_files if "greedy" in str(file)]
        assert len(all_answer_files) == 1
    if reward == False:
        all_answer_files = [str(file) for file in all_answer_files if "reward" not in str(file)]
        weighted = False
    else:
        all_answer_files = [str(file) for file in all_answer_files if "reward" in str(file)]
        weighted = True

    if best == True:
        assert weighted == True
        strategy = "Best-of-N"
    elif weighted == True:
        strategy = "Weighted Majority Voting"
    elif greedy == True:
        strategy = "Greedy"
    else:
        strategy = "Majority Voting"

    datas = []
    filtered_answer_files = []
    for file in all_answer_files:
        file_data = []
        with open(file, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                file_data.append(json_obj)
            """
            if output_reward:
                if "raw_score" not in file_data[0]:
                    continue
                filter_data = []
                for i in range(len(file_data)):
                    for j in range(len(file_data[i]["raw_score"])):
                        if type(file_data[i]["raw_score"][j]) != list:
                            file_data[i]["raw_score"][j] = [file_data[i]["raw_score"][j]]
                        file_data[i]["score"][j] = file_data[i]["raw_score"][j][-1]
            """
            filter_data = []
            for i in range(len(file_data)):
                if "unique_id" in file_data[i] and file_data[i]["unique_id"] in id_set:
                    filter_data.append(file_data[i])
            if len(filter_data) == len(id_set):
                #sort by unique_id
                filter_data = sorted(filter_data, key=lambda x: x["unique_id"])
                # assert difference
                for i in range(1, len(filter_data)):
                    assert filter_data[i]["unique_id"] != filter_data[i - 1]["unique_id"]
                datas.append(filter_data)
                filtered_answer_files.append(file)
            else:
                print(f"file {file} has answered {len(filter_data)} samples, {len(id_set)} samples are expected")
    print("filtered answer files: ", filtered_answer_files)

    # calculate the accuracy
    total = len(datas[0])
    print("total: ", total)
    acc_count = 0
    upper_bound_acc_count = 0
    majority_upper_bound_acc_count = 0
    result = []
    if True:
        total_entropy = 0
        for idx, answer in tqdm.tqdm(enumerate(datas[0]), total=total, desc="Calculating accuracy"):
            parsed_answer = answer["parsed_answer"]
            entropy = 0
            score = answer["score"] if "score" in answer else [1 for _ in range(len(parsed_answer))]
            parsed_answer = parsed_answer[:len(score)]
            assert len(parsed_answer) == len(score)
            gt_answer = answer["answer"]
            right_flag = False
            if strategy == "Greedy":
                final_answer = parsed_answer
            else:
                for i in range(1, len(datas)):
                    parsed_answer += datas[i][idx]["parsed_answer"]
                    score += datas[i][idx]["score"] if "score" in datas[i][idx] else [1 for _ in range(len(datas[i][idx]["parsed_answer"]))]
                if idx == 0: print("total samples: ", len(parsed_answer), "selected samples: ", max_samples)
                if len(parsed_answer) >= max_samples:
                    #parsed_answer = parsed_answer[:max_samples]
                    #random sampling
                    if (seed + 1) * max_samples > len(parsed_answer):
                        print("Not enough samples")
                        return -1, -1
                    else:
                        idxs = range(seed * max_samples, (seed + 1) * max_samples)
                    parsed_answer = [parsed_answer[_] for _ in idxs]
                    score = [score[_] for _ in idxs]
                else:
                    print("Not enough samples")
                    return -1, -1
                if strategy == "Majority Voting" or strategy == "Weighted Majority Voting":
                    answer_dict = dict()
                    no_answer_count = 0
                    for j in range(len(parsed_answer)):
                        flag = False
                        if parsed_answer[j] == "no answer":
                            no_answer_count += 1
                            continue
                        for key in answer_dict:
                            if grade_answer(key, parsed_answer[j]):
                                answer_dict[key] += score[j]
                                flag = True
                                break
                        if not flag:
                            answer_dict[parsed_answer[j]] = score[j]
                        if right_flag == False and grade_answer(parsed_answer[j], gt_answer):
                            right_flag = True
                    count = 0
                    correct_answer_count = 0
                    final_answer = "no answer"
                    for each in answer_dict:
                        if grade_answer(each, gt_answer):
                            correct_answer_count = answer_dict[each]
                        if answer_dict[each] > count and each != "no answer":
                            count = answer_dict[each]
                            final_answer = each
                        p_each = answer_dict[each] / len(parsed_answer)
                        entropy -= p_each * np.log(p_each)
                    # Miller-Maddow correction
                    # Only caclulate the entropy of the parsed answers
                    if len(parsed_answer) - no_answer_count == 0:
                        entropy = 0
                    else:
                        entropy = entropy + (len(answer_dict) - 1) / (2 * (len(parsed_answer) - no_answer_count))
                else: # best of N
                    final_answer = parsed_answer[0]
                    reward = answer["score"][0]
                    for i in range(1, len(parsed_answer)):
                        if answer["score"][i] > reward:
                            final_answer = parsed_answer[i]
                            reward = answer["score"][i]
                        if right_flag == False and grade_answer(each, gt_answer):
                            right_flag = True
            if grade_answer(gt_answer, final_answer):
                acc_count += 1
                right_flag = True
                result.append(1)
            else:
                result.append(0)
            if right_flag == True:
                upper_bound_acc_count += 1
            majority_upper_bound_acc_count += 1
            if count > correct_answer_count:
                # confidence interval, prior is uniform
                cdf_val = beta.cdf(0.5, count + 1, correct_answer_count + 1)
                # 0.25 is the threshold for the confidence interval (Beta(2, 1) = 0.25)
                if cdf_val <= 0.1:
                    # wrong answer has a higher probability
                    majority_upper_bound_acc_count -= 1

            total_entropy += entropy
    print(f"Total: {total}, Correct: {acc_count}, Accuracy: {acc_count/total}, Upper Bound: {upper_bound_acc_count/total}, Entropy: {total_entropy/total}, Majority Upper Bound: {majority_upper_bound_acc_count/total}")
    return acc_count/total, upper_bound_acc_count/total, total_entropy/total, majority_upper_bound_acc_count/total, result

if "__main__" == __name__:    
    import argparse
    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "--problem_file",
        type=Path,
        default="downloads/math_splits/test.json",
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--result_folder",
        type=Path,
        default="results/",
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--answer_name",
        type=str,
        default="math-shepherd-mistral-7b-sft",
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Whether using greedy decoding strategy.",
    )
    parser.add_argument(
        "--reward",
        action="store_true",
        help="Whether using reward decoding strategy.",
    )
    parser.add_argument(
        "--max_samples",
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
        "--output_reward",
        action="store_true",
        help="Whether use output reward.",
    )
    parser.add_argument(
        "--sample_times",
        type=int,
        default=10,
        help="Number of samples to average",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Output file for results.",
    )
    parser.add_argument(
        "--problem_range",
        type=int,
        default=-1,
        help="Problem range.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the previous results.",
        default=False,
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Output the detailed results.",
        default=False,
    )
    args = parser.parse_args()
    print(args)
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
    if args.output_file is None:
        if args.answer_name.endswith(".json"):
            output_file = os.path.join(args.result_folder, f"{'.'.join(args.answer_name.split('.')[:-1])}_{args.problem_range}_acc.json")
        else:
            output_file = os.path.join(args.result_folder, f"{args.answer_name}_{args.problem_range}_acc.json")
    else:
        output_file = os.path.join(args.result_folder, args.output_file)

    if not os.path.exists(output_file):
        acc = dict()
    else:
        with open(output_file, "r") as f:
            acc = json.load(f)

    args_str = f"{args.max_samples}_{args.greedy}_{args.reward}_{args.best}_{args.sample_times}"
    if args_str in acc and args.resume:
        print(acc[args_str])
        exit()
    total_accuracy = 0
    upper_bound_accuracy = 0
    total_entropy = 0
    majority_upper_bound_accuracy = 0
    num = []
    sample_times = args.sample_times
    for i in range(args.sample_times):
        num.append(main(
            problem_file=args.problem_file,
            result_folder=args.result_folder,
            answer_name=args.answer_name,
            greedy=args.greedy,
            reward=args.reward,
            max_samples=args.max_samples,
            best=args.best,
            output_reward=args.output_reward,
            problem_range=args.problem_range,
            seed = i,
        ))
        if num[-1][0] == -1:
            num = num[:-1]
            sample_times = i
            break
            # all samples are used
        total_accuracy += num[-1][0]
        upper_bound_accuracy += num[-1][1]
        total_entropy += num[-1][2]
        majority_upper_bound_accuracy += num[-1][3]
    print("Average accuracy: ", total_accuracy / sample_times)
    print("Upper bound accuracy: ", upper_bound_accuracy / sample_times)
    print("Average entropy: ", total_entropy / sample_times)
    print("Majority upper bound accuracy: ", majority_upper_bound_accuracy / sample_times)

    if len(num) > 1:
        standard_error = stats.sem([x[0] for x in num])
    else:
        standard_error = 0

    if args.detail:
        acc[args_str] = {
            "average": total_accuracy / sample_times,
            "upper_bound": upper_bound_accuracy / sample_times,
            "entropy": total_entropy / sample_times,
            "majority_upper_bound": majority_upper_bound_accuracy / sample_times,
            "individual": num,
            "standard_error": standard_error
        }
    else:
        acc[args_str] = {
            "average": total_accuracy / sample_times,
            "upper_bound": upper_bound_accuracy / sample_times,
            "entropy": total_entropy / sample_times,
            "majority_upper_bound": majority_upper_bound_accuracy / sample_times,
            "standard_error": standard_error
        }
    with open(output_file, "w") as f:
        json.dump(acc, f)