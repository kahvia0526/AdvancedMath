import argparse
import json
import os
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class BackTranslator:
    def __init__(self, model_path: str, gpus: int = 1):
        self.model_path = model_path
        self.gpus = gpus
        self.llm = None
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        special_tokens = {}
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = "<unk>"
        if self.tokenizer.eos_token is None:
            special_tokens["eos_token"] = "</s>"
        if self.tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<s>"
        if self.tokenizer.unk_token is None:
            special_tokens["unk_token"] = "<unk>"
        if special_tokens and 'Qwen' not in self.model_path:
            self.tokenizer.add_special_tokens(special_tokens)

    def _init_model(self):
        if self.llm is None:
            try:
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.gpus,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.5
                )
            except RecursionError:
                self.llm = LLM(
                    model=self.model_path,
                    tokenizer_mode='slow',
                    tensor_parallel_size=self.gpus,
                    trust_remote_code=True,
                    dtype="float16"
                )

    def build_prompt(self, code: str) -> str:
        return (
            '[UNUSED_TOKEN_146]user\nConvert the formal statement into natural language:\n'
            f'```lean\n{code}\n```[UNUSED_TOKEN_145]\n'
            '[UNUSED_TOKEN_146]assistant\n'
        )

    def postprocess(self, text: str) -> str:
        for token in self.tokenizer.special_tokens_map.values():
            if isinstance(token, list):
                for t in token:
                    text = text.replace(t, "")
            else:
                text = text.replace(token, "")
        return text.strip()

    def batch_translate(self, codes: List[str], sampling_params: dict) -> List[str]:
        self._init_model()
        prompts = [self.build_prompt(code) for code in codes]
        vllm_params = SamplingParams(
            temperature=sampling_params.get('temperature', 0.1),
            max_tokens=sampling_params.get('max_tokens', 1024),
            stop=['[UNUSED_TOKEN_146]', '[UNUSED_TOKEN_145]']
        )
        outputs = self.llm.generate(prompts, vllm_params)
        return [
            self.postprocess(output.outputs[0].text.replace('[UNUSED_TOKEN_145]', ''))
            for output in outputs
        ]


def process_questions(questions, translator, sampling_params):
    code_list = []
    position_map = []

    for q_idx, q in enumerate(questions):
        q['back_translate'] = [""] * len(q.get('outputs', []))
        for c_idx, code in enumerate(q.get('outputs', [])):
            if q['compile'][c_idx] == "pass":
                code_list.append(code)
                position_map.append((q_idx, c_idx))

    translations = translator.batch_translate(code_list, sampling_params)

    for (q_idx, c_idx), trans in zip(position_map, translations):
        questions[q_idx]['back_translate'][c_idx] = trans

    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repeat-times", type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open(args.question_file) as f:
        questions = [json.loads(line) for line in f]

    if args.repeat_times > 1:
        questions = questions * args.repeat_times

    translator = BackTranslator(model_path=args.model_path, gpus=1)

    sampling_params = {
        'temperature': args.temperature,
        'max_tokens': args.max_new_token
    }

    processed_questions = process_questions(
        questions=questions,
        translator=translator,
        sampling_params=sampling_params
    )

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    with open(args.answer_file, 'w', encoding='utf-8') as fout:
        for q in processed_questions:
            fout.write(json.dumps(q, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
