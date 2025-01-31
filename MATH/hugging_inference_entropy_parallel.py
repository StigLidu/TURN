from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json, os
import numpy as np
import tqdm
from pathlib import Path

def process_problems_on_gpu(
    problems_subset,
    gpu_id,
    output_file,
    batch_size,
    num_samples,
    max_new_tokens,
    top_k,
    checkpoint_path,
    tokenizer_path,
    resume_generation,
    few_shot,
    temp_alpha,
    temp_beta,
    temp_min=0.001,
    temp_mode = "binary"
    ):
    # Model and tokenizer loading
    print(f"Loading model on GPU {gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).eval().to(f"cuda:{gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    calculator_tokens = [
    "$", "$$",
    "\\[", "\\]",
    "\\(", "\\)",
    "\\begin{equation}", "\\end{equation}",
    "\\begin{equation*}", "\\end{equation*}",
    "\\begin{align}", "\\end{align}",
    "\\begin{align*}", "\\end{align*}",
    "\\begin{gather}", "\\end{gather}",
    "\\begin{gather*}", "\\end{gather*}",
    "\\begin{multline}", "\\end{multline}",
    "\\begin{multline*}", "\\end{multline*}"
    ]
    step_tag = 'ки'
    temp_for_thought = temp_beta
    temp_for_calculation = temp_alpha

    answered_length = 0
    if os.path.exists(output_file):
        if resume_generation:
            with open(output_file, "r") as f:
                answered_problems = [json.loads(line) for line in f]
                answered_length = len(answered_problems)
            print(f"GPU {gpu_id}: Answered {answered_length} problems")
        else:
            assert False, f"Output file {output_file} already exists"

    if few_shot:
        few_shot_file = "math_utils/few_shot_example_for_math.txt"
        with open(few_shot_file, "r") as f:
            few_shot_prompt = f.read()
    else:
        few_shot_prompt = ""

    entropy_measurements = [1.0]
    total_entropy = np.zeros(len(entropy_measurements))
    total_samples = 0
    with open(output_file, 'a') as output_writer:
        with torch.no_grad():
            id = 0
            calculation_entropy = 0
            thought_entropy = 0
            calculation_tokens = 0
            thought_tokens = 0
            for problem in tqdm.tqdm(problems_subset, desc=f"GPU {gpu_id}"):
                if id < answered_length:
                    id += 1
                    continue
                id += 1
                output = []
                for i in tqdm.tqdm(range((num_samples - 1) // batch_size + 1), desc="Generating samples"):
                    current_batch_size = min(batch_size, num_samples - i * batch_size)
                    input_texts = [few_shot_prompt + problem["problem"]] * current_batch_size
                    input_ids = tokenizer.batch_encode_plus(
                        input_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )['input_ids'].to(f"cuda:{gpu_id}")
                    generated_tokens = [[] for _ in range(current_batch_size)]
                    done = torch.zeros(current_batch_size, dtype=torch.bool).to(f"cuda:{gpu_id}")
                    past_key_values = None
                    calculation_flags = torch.zeros(current_batch_size, dtype=torch.bool).to(f"cuda:{gpu_id}")
                    entropy_length = np.zeros(batch_size)
                    entropies = np.zeros((len(entropy_measurements), current_batch_size))
                    for j in range(0, max_new_tokens):
                        if j == 0:
                            next_output = model(input_ids=input_ids, use_cache=True)
                        else:
                            next_input_ids = next_token_ids.unsqueeze(-1)
                            next_output = model(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)
                        logits = next_output.logits
                        past_key_values = next_output.past_key_values
                        next_token_logits = logits[:, -1, :]
                        if temp_mode == "decay":
                            temp_value = max((temp_beta * np.exp(-temp_alpha * j)), temp_min)
                            temp = torch.full((current_batch_size,), temp_value, device=f"cuda:{gpu_id}")
                        elif temp_mode == "binary":
                            temp = torch.zeros(current_batch_size).to(f"cuda:{gpu_id}")
                            temp[calculation_flags] = temp_for_calculation
                            temp[~calculation_flags] = temp_for_thought
                        else:
                            raise ValueError("Invalid temp_mode")
                        for id, ttt in enumerate(entropy_measurements):
                            next_token_probs = torch.nn.functional.log_softmax(next_token_logits / ttt, dim=-1)
                            next_token_probs, _ = torch.topk(next_token_probs, 1000, dim=-1)
                            total_probs = np.array(torch.sum(torch.exp(next_token_probs), dim=-1).cpu())
                            entropy = -np.nansum(np.array(next_token_probs.cpu()) * np.exp(np.array(next_token_probs.cpu())), axis=-1)
                            entropy = entropy / total_probs
                            for k in range(current_batch_size):
                                if not done[k]:
                                    entropies[id, k] += entropy[k]
                                    if id == 0:
                                        entropy_length[k] += 1

                        next_token_probs = torch.nn.functional.softmax(next_token_logits / temp.unsqueeze(-1), dim=-1)
                        if top_k != -1:
                            topk_probs, topk_indices = torch.topk(next_token_probs, top_k, dim=-1)
                        else:
                            topk_probs, topk_indices = next_token_probs, torch.arange(next_token_probs.size(-1)).to(f"cuda:{gpu_id}").unsqueeze(0).repeat(current_batch_size, 1)
                        topk_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
                        sampled_indices = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
                        next_token_ids = topk_indices.gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)
                        for k in range(current_batch_size):
                            if not done[k]:
                                token_id = next_token_ids[k].item()
                                generated_tokens[k].append(token_id)
                                decode_str = tokenizer.decode(generated_tokens[k], skip_special_tokens=True)
                                decode_str = decode_str.replace("$$", "$")
                                count = np.zeros(len(calculator_tokens), dtype=int)
                                if calculation_flags[k]:
                                    calculation_entropy += entropy[k].item()
                                    calculation_tokens += 1
                                else:
                                    thought_entropy += entropy[k].item()
                                    thought_tokens += 1
                                calculation_flags[k] = False
                                for t, calculator_token in enumerate(calculator_tokens):
                                    count[t] = decode_str.count(calculator_token)
                                for t in range(2, len(calculator_tokens) - 1, 2):
                                 #   assert count[t] >= count[t + 1], f"Error: {tokenizer.decode(generated_tokens[k], skip_special_tokens=True)}"
                                    if count[t] > count[t + 1]:
                                        calculation_flags[k] = True
                                if (count[0] + count[1]) % 2 == 1:
                                    calculation_flags[k] = True
                                if token_id == tokenizer.eos_token_id:
                                    done[k] = True
                        if done.all():
                            break
                    outputs = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
                    output.extend(outputs)
                    print(f"Calculation entropy: {calculation_entropy / calculation_tokens}")
                    print(f"Thought entropy: {thought_entropy / thought_tokens}")

                problem["output"] = output
                problem["parsed_answer"] = []

                for i in range(num_samples):
                    try:
                        problem["parsed_answer"].append(problem["output"][i].split("The answer is:")[1].split("ки")[0].strip())
                        if few_shot:
                            problem["parsed_answer"][-1] = problem["parsed_answer"][-1].split("Question 6")[0].strip()
                    except:
                        problem["parsed_answer"].append("no answer")
                    
                output_writer.write(json.dumps(problem) + "\n")
                output_writer.flush()
                entropies = (entropies / entropy_length).mean(axis=-1)
                total_entropy += entropies
                total_samples += 1
                measured_entropy = {entropy_measurements[i]: total_entropy[i] / total_samples for i in range(len(entropy_measurements))}
                print(f"Sample Temperature: {temp_for_thought}")
                print(f"Measured Entropy: {measured_entropy}")

def main(
    seed: int,
    prompt_file: Path,
    output_file: Path,
    gpu_ids: list[int],
    batch_size: int,
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    checkpoint_path: Path,
    tokenizer_path: Path,
    resume_generation: bool,
    few_shot: bool,
    temp_alpha: float = 0.1,
    temp_beta: float = 1.0,
    temp_mode = "binary",
    test_subset = -1
    ) -> None:

    import multiprocessing

    # Parse GPU IDs
    num_gpus = len(gpu_ids)

    output_file = str(output_file)
    print("resume_generation: ", resume_generation)
    if os.path.exists(output_file):
        if resume_generation:
            print(f"Resuming generation. Output file {output_file} already exists.")
        else:
            assert False, f"Output file {output_file} already exists"

    problem_file = prompt_file
    with open(problem_file, "r") as f:
        problems = [json.loads(line) for line in f]
        if test_subset != -1:
            problems = problems[:test_subset]
        for i, problem in enumerate(problems):
            problem["idx"] = i
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                answered_problems = [json.loads(line) for line in f]
                answered_problems_idx = {problem["idx"] for problem in answered_problems}
            problems = [problem for problem in problems if problem["idx"] not in answered_problems_idx]

    # Split problems among GPUs
    problems_per_gpu = len(problems) // num_gpus
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * problems_per_gpu
        if i == num_gpus - 1:
            # Last GPU takes the rest
            problems_subset = problems[start_idx:]
        else:
            problems_subset = problems[start_idx : start_idx + problems_per_gpu]
        output_file_gpu = f"{output_file}_gpu{gpu_id}"
        p = multiprocessing.Process(
            target=process_problems_on_gpu,
            args=(
                problems_subset,
                gpu_id,
                output_file_gpu,
                batch_size,
                num_samples,
                max_new_tokens,
                top_k,
                checkpoint_path,
                tokenizer_path,
                resume_generation,
                few_shot,
                temp_alpha,
                temp_beta,
                0.001, # temp_min
                temp_mode
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Combine output files
    with open(output_file, 'a') as outfile:
        for gpu_id in gpu_ids:
            output_file_gpu = f"{output_file}_gpu{gpu_id}"
            with open(output_file_gpu, 'r') as infile:
                outfile.write(infile.read())
            os.remove(output_file_gpu)

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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
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
        "--temp_alpha",
        type=float,
        default=0.1,
        help="Temperature decay rate.",
    )
    parser.add_argument(
        "--temp_beta",
        type=float,
        default=1.0,
        help="Temperature decay rate.",
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs='+', default=[0], help="List of GPU ids."
    )
    parser.add_argument(
        "--temp_mode", type=str, default="binary", help="Temperature mode."
    )
    parser.add_argument(
        "--test_subset", type=int, default=-1, help="Test subset."
    )
    args = parser.parse_args()
    print(args)
    main(
        args.seed,
        args.prompt_file,
        args.output_file,
        args.gpu_ids,
        args.batch_size,
        args.num_samples,
        args.max_new_tokens,
        args.top_k,
        args.checkpoint_path,
        args.tokenizer_path,
        args.resume_generation,
        args.few_shot,
        args.temp_alpha,
        args.temp_beta,
        temp_mode = args.temp_mode,
        test_subset = args.test_subset
    )