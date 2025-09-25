import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from openai import OpenAI

# Configuration for the NLI model (replace with your actual settings)
NLI_MODEL = 'your-nli-model-name'  # e.g., 'deepseek-chat'
NLI_API_BASE_URL = 'https://your-api-base-url.com'
NLI_API_KEY = 'your-api-key'

# Sampling parameters for generation
NLI_SAMPLING_PARAMS = {
    'max_tokens': 2048,
    'temperature': 0.01,
    'top_p': 0.7,
    'frequency_penalty': 1
}


class NLInferencer:
    def __init__(self, model_name: str, base_url: str, api_key: str, sampling_params: dict):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.sampling_params = sampling_params
        self.client = None

    def _init_client(self):
        if self.client is None:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=30,
                max_retries=3
            )

    def build_messages(self, original: str, translation: str) -> List[Dict]:
        system_prompt = (
            "Please check whether the following two math problems are the same or different. "
            "Compare each statement individually. If any statement differs, they are considered different. "
            "Please explain any differences. End your reply with ||same|| or ||different||."
        )
        user_content = f"Problem 1:\n{original}\nProblem 2:\n{translation}\nFinal verdict:"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def generate(self, original: str, translation: str, max_retries: int = 3) -> str:
        self._init_client()
        messages = self.build_messages(original, translation)

        for _ in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.sampling_params
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API request failed: {str(e)}")
        return ""


def run_eval(questions, answer_file):
    inferencer = NLInferencer(
        model_name=NLI_MODEL,
        base_url=NLI_API_BASE_URL,
        api_key=NLI_API_KEY,
        sampling_params=NLI_SAMPLING_PARAMS
    )

    request_queue = []
    position_mapping = []

    # Build request queue
    for q_idx, question in enumerate(questions):
        original = question['natural_language_statement']
        translations = question.get('back_translate', [])

        if not translations:
            continue

        question['nli_output'] = [{} for _ in translations]

        for t_idx, translation in enumerate(translations):
            if question['compile'][t_idx] == "pass":
                request_queue.append((original, translation))
                position_mapping.append((q_idx, t_idx))

    print(f"Total requests to process: {len(request_queue)}")

    # Execute requests in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for original, translation in request_queue:
            futures.append(executor.submit(inferencer.generate, original, translation))

        for future, (q_idx, t_idx) in zip(futures, position_mapping):
            decoded_text = future.result()

            # Parse judgment
            final_judgment = "unknown"
            if "||same||" in decoded_text.lower():
                final_judgment = "same"
            elif "||different||" in decoded_text.lower():
                final_judgment = "different"

            questions[q_idx]['nli_output'][t_idx] = {
                "full_response": decoded_text.strip(),
                "judgment": final_judgment
            }

    # Save results
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, 'w', encoding='utf-8') as fout:
        for question in questions:
            fout.write(json.dumps(question, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.question_file, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    run_eval(questions, args.answer_file)
