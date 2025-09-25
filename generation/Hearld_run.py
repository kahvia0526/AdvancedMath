

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import argparse
import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as numpy
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOCAL_MODEL_PATH ="./model"  

def run_eval(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    temperature,

):
    print('##################' + str(torch.cuda.is_available()))
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ['RANK'])
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = '<unk>'
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = '</s>'
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = '<s>'
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = '<unk>'
    if len(special_tokens_dict) > 0 and model_path.find('Qwen') == -1:
        tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Output to {answer_file}")
    print(f"Num Questions: {len(questions)}")

   

    try:
        # model = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.7
        )
    except RecursionError:
        model = LLM(model=model_path, tokenizer_mode='slow', trust_remote_code=True, dtype="bfloat16",gpu_memory_utilization=0.95)
    # sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token,
    #                                  stop=['[UNUSED_TOKEN_146]', '[UNUSED_TOKEN_145]', 'by', 'sorry'])

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        n=10,  
        stop=['[UNUSED_TOKEN_146]', '[UNUSED_TOKEN_145]', 'by', 'sorry']
    )
   
    def get_query(example):
            return "[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n" + str(example[
                                                                                                'natural_language_statement']) + "[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"

    def get_query_newprompt(example):
        instruction = (
            "[UNUSED_TOKEN_146]user\n"
            "Task: Convert the following problem into LEAN 4 code. "
            "Use the mathematical definitions provided below for reference, "
            "but DO NOT translate the definitions themselves.\n\n"
            "Problem to translate:\n"
            f"{str(example['natural_language_statement'])}\n\n"
            "Reference definitions (for context only):\n"
        )


        context_entries = []
        for ctx in example.get('context', []):
        
            filtered_ctx = {
                k: v
                for k, v in ctx.items()
                if k not in ['score', 'explan']
            }

            context_entry = (
                f"- {filtered_ctx.get('def_name', 'Unknown Concept')}: "
                f"{filtered_ctx.get('def_content', 'No content available.')}"
            )
            context_entries.append(context_entry)

        context_str = "\n".join(context_entries) if context_entries else "No additional context needed."

 
        full_prompt = (
            f"{instruction}{context_str}\n\n"
            "[UNUSED_TOKEN_145]\n"
            "[UNUSED_TOKEN_146]assistant\n"
            "Here is the formal statement in LEAN 4:\n"
            "```lean4\ntheorem"
        )

        return full_prompt

    prompts = [get_query_newprompt(example) for example in questions]

    prompt_id_map = {prompt: idx for idx, prompt in enumerate(prompts)}

    outputs = model.generate(prompts, sampling_params)

   
    outputs = model.generate(prompts, sampling_params)

    for output in outputs:
    
        output_texts = []
        for candidate in output.outputs: 

            decoded_text = model.get_tokenizer().decode(
                candidate.token_ids,
                spaces_between_special_tokens=False,
            )


            for special_token in model.get_tokenizer().special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        decoded_text = decoded_text.replace(special_tok, "")
                else:
                    decoded_text = decoded_text.replace(special_token, "")
            decoded_text = decoded_text.strip()
            decoded_text = "theorem " + decoded_text
            output_texts.append(decoded_text)


        question = questions[prompt_id_map[output.prompt]]


        question['output'] = output_texts[0]  
        question['outputs'] = output_texts  
        question['generator'] = model_id

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(question, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=LOCAL_MODEL_PATH,
        help="Local model directory path"
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="The output answer file.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    args.model_path = LOCAL_MODEL_PATH


    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]

    if args.repeat_times > 1:
        questions = questions * args.repeat_times


    run_eval(
        args.model_path,
        args.model_path,
        questions,
        args.answer_file,
        args.max_new_token,
        args.temperature,
    )