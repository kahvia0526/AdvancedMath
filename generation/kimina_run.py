from click import prompt
from huggingface_hub.utils.tqdm import progress_bar_states
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = " "  
from outlines.serve.serve import tokenizer
from vllm import LLM,SamplingParams
from  transformers import AutoTokenizer
from tqdm import tqdm
import json
model_path=""
model=LLM(
    model=model_path,
    tensor_parallel_size=1,
gpu_memory_utilization=0.9)

tokenizer=AutoTokenizer.from_pretrained(model_path)
sampling_params=SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048,
    n=10,
    stop=['[UNUSED_TOKEN_145]', '</s>'],
    repetition_penalty=1.2
)


def getquery_new(question):
    problem = question['natural_language_statement']

    instruction = (
        "Task:Please autoformalize the following problem in Lean 4 with a header.  Use the following theorem names: my_favorite_theorem"
        "Use the mathematical definitions provided below for reference, "
        "but DO NOT translate the definitions themselves.\n\n"
        "Problem to translate:\n"
        f"{problem}\n\n"
        "Reference definitions (for context only):\n"
    )


    context_entries = []
    for ctx in question.get('context', []):

        concept = ctx.get('def_name', 'Unknown Concept')
        def_content = ctx.get('def_content', 'No content available.')
        context_entries.append(f"- {concept}: {def_content}")

    context_str = "\n".join(context_entries) if context_entries else "No additional context needed."


    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": f"{instruction}{context_str}"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    text += "\nHere is the formal statement in LEAN 4:\n"

    return text

def getquery(question):
    problem=question['natural_language_statement']
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt+=problem
    messages=[
        {"role":"system","content":"You are an expert in mathematics and Lean 4."},
        {"role":"user","content":prompt}
        ]
    text=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def process_dataset(input_path, output_path):
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    
    max_batch_size = 32 
    total_items = len(dataset)

    progress_bar = tqdm(total=total_items, desc="Processing", unit="sample")

    try:
        for i in range(0, total_items, max_batch_size):
            batch = dataset[i:i + max_batch_size]
            prompts = [getquery_new(item) for item in batch]  

         
            outputs = model.generate(prompts, sampling_params)

            for idx, output in enumerate(outputs):
                item = dataset[i + idx]
                candidates = [
                    candidate.text.split("```")[0].replace(tokenizer.eos_token, "").strip()
                    for candidate in output.outputs
                ]
                item['outputs'] = candidates

       
                progress_bar.update(1)
                progress_bar.set_postfix({
                                             "current_batch": f"{i // max_batch_size + 1}/{(total_items + max_batch_size - 1) // max_batch_size}"})

  
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

    finally:
        progress_bar.close()


if __name__ == "__main__":
    input_file = ""
    output_file = ""


    with open(output_file, 'w') as f:
        json.dump([], f)

    process_dataset(input_file, output_file)
    print(f"Processing completed. Results saved to {output_file}")