from vllm import LLM, SamplingParams
from transformers.generation.logits_process import TypicalLogitsWarper
import json, os
from tqdm import tqdm
from pathlib import Path
from typing import List
import torch
import numpy as np

class TypicalLogitsWarperWrapper:
    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)
        
    def __call__(self, token_ids: List[int], logits: torch.tensor) -> torch.tensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(input_ids=None, scores=logits.reshape((1, -1)))

def main(
    seed: int,
    prompt_file: Path,
    output_file: Path,
    gpu_id: int,
    batch_size: int,
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    checkpoint_path: Path,
    tokenizer_path: Path,
    resume_generation: bool,
    typical_sampling: bool,
    few_shot: bool,
    reparse: bool,
    infer_entropy: bool = False,
    max_parse_length: int = 100
    ) -> None:
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if infer_entropy:
        logprobs = 1000
    else:
        logprobs = 0

    sampling_params = SamplingParams(
            temperature=temperature, 
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_new_tokens, 
            n = min(batch_size, num_samples),
            stop= "Question 6:",
            logprobs = logprobs,
    )
    if typical_sampling:
        assert top_k == -1 and top_p == 1.0 and min_p == 0.0, \
            "top_k and top_p should be -1 and 1.0 and 0.0 respectively"
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_new_tokens, 
            n = min(batch_size, num_samples),
            stop= "Question 6:",
            logits_processors=[TypicalLogitsWarperWrapper(mass=0.8)],
            logprobs=logprobs,
        )

    problem_per_batch = 1
    if batch_size > num_samples:
        problem_per_batch = batch_size // num_samples
    
    print("problem_per_batch: ", problem_per_batch)
    print("sampling_params: ")
    print(sampling_params)

    output_file = str(output_file)
    answered_length = 0
    if os.path.exists(output_file):
        # resume inference
        if resume_generation:
            with open(output_file, "r") as f:
                answered_problems = [json.loads(line) for line in f]
                answered_length = len(answered_problems)
            print(f"Answered {answered_length} problems")
        else:
            assert False, f"Output file {output_file} already exists"

    if few_shot:
        few_shot_file = "math_utils/few_shot_example_for_math.txt"
        with open(few_shot_file, "r") as f:
            few_shot_prompt = f.read()
    else:
        few_shot_prompt = ""

    instruct_file = "math_utils/instruct_prompt.json"
    with open(instruct_file, "r") as f:
        instruct_prompts = json.load(f)
    if str(checkpoint_path).split("/")[-1] in instruct_prompts:
        instruct_prompt = instruct_prompts[str(checkpoint_path).split("/")[-1]]['instruct']
        parse_sentence = instruct_prompts[str(checkpoint_path).split("/")[-1]]['parse_sentence']
    else:
        instruct_prompt = ""
        parse_sentence = ["The answer is:", "ки"]

    print("instruct_prompt: ", instruct_prompt, "parse_sentence: ", parse_sentence)

    output_writer = open(output_file, "a")
    problem_file = prompt_file
    with open(problem_file, "r") as f:
        problems = [json.loads(line) for line in f]
        total_problems = len(problems)
        for i in range(len(problems)):
            problems[i]["idx"] = i

    if answered_length < len(problems):
        model = LLM(model=str(checkpoint_path), tokenizer = tokenizer_path, max_logprobs=logprobs, swap_space=8)

    for id in tqdm(range(len(problems)), desc="sampling"):
        if id < answered_length:
            continue
        output = []
        entropy = []
        if problem_per_batch == 1:
            for i in range((num_samples - 1) // batch_size + 1):
                batch_input = few_shot_prompt + problems[id]["problem"] + instruct_prompt
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
                batch_input.append(few_shot_prompt + problems[id + i]["problem"] + instruct_prompt)
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
            output_writer.write(json.dumps(problems[id + i]) + "\n")
        output_writer.flush()
    
    output_writer.close()
    output_writer = open(output_file, "r")
    problems = [json.loads(line) for line in output_writer]
    output_writer.close()
    output_writer = open(output_file, "w")

    with tqdm(total=len(problems),desc="parsing") as pbar:
        for problem in problems:
            pbar.update(1)
            if "parsed_answer" in problem and not reparse:
                continue
            problem["parsed_answer"] = []
            for i in range(num_samples):
                try:
                    if few_shot:
                        # we have 4 questions in the few-shot example
                        problem["output"][i] = problem["output"][i].split("Question 6")[0].strip()
                    if parse_sentence[0] not in problem["output"][i]:
                        problem["parsed_answer"].append("no answer")
                        continue
                    front_parsed_answer = problem["output"][i].split(parse_sentence[0])[-1].strip().split("\n")[0].strip()
                    if parse_sentence[1] != "" and parse_sentence[1] in front_parsed_answer:
                        parsed_answer = f"{parse_sentence[1]}".join(front_parsed_answer.split(parse_sentence[1])[:-1]).strip()
                    else:
                        parsed_answer = front_parsed_answer
                    assert len(parsed_answer) < max_parse_length, f"parsed answer too long: {len(parsed_answer)}"
                    problem['parsed_answer'].append(parsed_answer)
                except:
                    problem["parsed_answer"].append("no answer")
            output_writer.write(json.dumps(problem) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File to write generated samples to.",
    )
    parser.add_argument("--gpu_id", type=int, help="GPU ID.", default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k for sampling, -1 means no top-k.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/tokenizer.json"),
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
        "--infer_entropy",
        action="store_true",
        help="Whether to infer entropy.",
        default=False,
    )
    args = parser.parse_args()
    print(args)
    main(
        seed=args.seed,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        typical_sampling=args.typical_sampling,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        resume_generation=args.resume_generation,
        few_shot=args.few_shot,
        reparse=args.reparse,
        infer_entropy=args.infer_entropy,
    )