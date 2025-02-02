from vllm import LLM, SamplingParams
from transformers.generation.logits_process import TypicalLogitsWarper
import json, os
from tqdm import tqdm
from typing import List
import torch
import numpy as np

def calc_turning_point(x_values, log_y_values):
    # Calculate the first derivative
    dy_dx = np.gradient(log_y_values, x_values)

    # Calculate the second derivative
    d2y_dx2 = np.gradient(dy_dx, x_values)

    print("dy_dx: ", dy_dx)
    print("d2y_dx2: ", d2y_dx2)
    
    # The first time the second derivative is greater than zero is the turning point
    mask = d2y_dx2[1:-1] > 0
    if not np.any(mask):
        # handle "no positive second derivative" case
        mask[-1] = True
    turning_point_index = np.where(mask)[0][0]  # first True
    turning_point_index += 1
    return x_values[turning_point_index]

AGGREGATION_ADAPTOR = {
    "BofN": 0.1,
    "MJ": 0.0,
}

class TypicalLogitsWarperWrapper:
    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)
        
    def __call__(self, token_ids: List[int], logits: torch.tensor) -> torch.tensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(input_ids=None, scores=logits.reshape((1, -1)))

def main(
    data: List[dict],
    model: LLM,
    batch_size: int,
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    typical_sampling: bool,
    logprobs: int,
    ) -> float:
    sampling_params = SamplingParams(
            temperature=temperature, 
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_new_tokens, 
            n = min(batch_size, num_samples),
            logprobs = logprobs,
    )
    if typical_sampling:
        assert top_k == -1 and top_p == 1.0 and min_p == 0.0, \
            "top_k and top_p should be -1 and 1.0 and 0.0 respectively"
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_new_tokens, 
            n = min(batch_size, num_samples),
            logits_processors=[TypicalLogitsWarperWrapper(mass=0.8)],
            logprobs=logprobs,
        )

    problem_per_batch = 1
    if batch_size > num_samples:
        problem_per_batch = batch_size // num_samples
    
    print("problem_per_batch: ", problem_per_batch)
    print("sampling_params: ")
    print(sampling_params)

    answered_length = 0

    instruct_prompt = ""
    #TODO: add instruction prompt

    problems = data

    total_entropy = 0
    for id in tqdm(range(0, len(problems), problem_per_batch), desc="sampling"):
        output = []
        entropy = []
        if problem_per_batch == 1:
            for i in range((num_samples - 1) // batch_size + 1):
                batch_input = problems[id]["problem"] + instruct_prompt
                batch_output = model.generate(batch_input, sampling_params)
                for j in range(sampling_params.n):
                    output.append(batch_output[0].outputs[j].text)
                    logprobs = batch_output[0].outputs[j].logprobs
                    sum_entropy = 0
                    for k in range(len(logprobs)):
                        cal_entropy = 0
                        cal_cum_prob = 0
                        for item in logprobs[k]:
                            cal_entropy += logprobs[k][item].logprob * np.exp(logprobs[k][item].logprob)
                            cal_cum_prob += np.exp(logprobs[k][item].logprob)
                        sum_entropy += -cal_entropy / cal_cum_prob
                    sum_entropy /= len(logprobs)
                    entropy.append(sum_entropy)
            problems[id]["output"] = output
            problems[id]["entropy"] = entropy
        # I am not sure if this part accerates the process
        else:
            batch_input = []
            for i in range(min(problem_per_batch, len(problems) - id)):
                batch_input.append(problems[id + i]["problem"] + instruct_prompt)
            batch_output = model.generate(batch_input, sampling_params)
            for i in range(min(problem_per_batch, len(problems) - id)):
                output = []
                entropy = []
                for j in range(sampling_params.n):
                    output.append(batch_output[i].outputs[j].text)
                    logprobs = batch_output[i].outputs[j].logprobs
                    sum_entropy = 0
                    for k in range(len(logprobs)):
                        cal_entropy = 0
                        cal_cum_prob = 0
                        for item in logprobs[k]:
                            cal_entropy += logprobs[k][item].logprob * np.exp(logprobs[k][item].logprob)
                            cal_cum_prob += np.exp(logprobs[k][item].logprob)
                        sum_entropy += -cal_entropy / cal_cum_prob
                    sum_entropy /= len(logprobs)
                    entropy.append(sum_entropy)
                problems[id + i]["output"] = output
                problems[id + i]["entropy"] = entropy
        answered_length += min(problem_per_batch, len(problems) - id)
        for i in range(min(problem_per_batch, len(problems) - id)):
            total_entropy += np.mean(problems[id + i]["entropy"])
    print(f"Mean entropy at temperature {temperature}: ", total_entropy / len(problems))
    return total_entropy / len(problems)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="entropy.json",
        help="Output file path.",
    )
    parser.add_argument(
        "--instruction_file",
        type=str,
        default=None,
        help="File containing instructions for prompts.",
    )
    parser.add_argument(
        "--t_interval",
        type=float,
        default=0.1,
        help="Temperature interval for entropy calculation.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=1.5,
        help="Maximum temperature for entropy calculation.",
    )
    parser.add_argument("--gpu_id", type=int, help="GPU ID.", default=0)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer checkpoint path.",
    )
    parser.add_argument(
        "--resume_generation", action="store_true", help="Whether to resume generation."
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use few-shot learning.",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Whether to reparse the answers.",
    )
    # sampling parameters
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=-1, help="Top-k for sampling, -1 means no top-k."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--typical_sampling",
        action="store_true",
        help="Whether to use typical sampling.",
        default=False,
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1000,
        help="Number of logprobs to store.",
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        default="MJ",
        help="Aggregation strategy.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    all_test_tempertures = np.arange(0.1, args.t_max, args.t_interval)
    # load data
    with open(args.data_path, "r") as f:
        problems = [json.loads(line) for line in f]
    if len(problems) > args.num_samples:
        problems = np.random.choice(problems, args.num_samples, replace=False)
        problems = problems[:args.num_samples]
        sample_per_problem = 1
    else:
        sample_per_problem = args.num_samples // len(problems)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    model = LLM(model=str(args.model_path), tokenizer = args.tokenizer_path, max_logprobs=args.logprobs, swap_space=8)

    entropy = []
    temp = []

    pre_entropy_dict = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            output = json.load(f)
            if "entropy" in output:
                pre_entropy_dict = output["entropy"]
                pre_entropy_dict = {f"{float(k):.2f}": v for k, v in pre_entropy_dict.items()}
                print("Pre-entropy: ", pre_entropy_dict)

    for t in all_test_tempertures:
        args.temperature = t
        args.num_samples = sample_per_problem
        if f"{t:.2f}" in pre_entropy_dict:
            average_e = pre_entropy_dict[f"{t:.2f}"]
        else:
            average_e = main(
                data=problems,
                model=model,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                typical_sampling=args.typical_sampling,
                temperature=args.temperature,
                logprobs=args.logprobs,
            )
        entropy.append(average_e)
        temp.append(f"{t:.2f}")
    print("Entropy: ", entropy)
    print("Temperature: ", temp)
    entropy_dict = dict(zip(temp, entropy))
    turning_point = calc_turning_point(all_test_tempertures, np.log(entropy))
    predicted_temperature = turning_point + AGGREGATION_ADAPTOR[args.aggregation_strategy]
    print("Predicted temperature: ", predicted_temperature)
    output = {
        "entropy": entropy_dict,
        "predicted_temperature": predicted_temperature,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f)