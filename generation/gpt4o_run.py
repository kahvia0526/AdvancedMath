from openai import OpenAI
import json
from tqdm import tqdm
import time
import os

# ====================== Configuration ======================
OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://your.api.endpoint")

INPUT_FILE = "data/input.json"
OUTPUT_FILE = "data/output_gpt4o.json"

# ====================== Prompt Template ======================
SYSTEM_PROMPT = """   
You are an expert mathematician and Lean 4 programmer.
Translate the following mathematical theorem from natural language
into formal Lean 4 code with a theorem header.
Provide only the Lean 4 code without any additional explanation.

Important instructions:
1. ONLY translate the theorem statement. DO NOT provide any proof or implementation.
2. End the translated theorem with `sorry` to indicate the proof is omitted.
"""

USER_TEMPLATE = """Translate this mathematical statement into Lean 4:

{description}

Ensure the formal statement includes all necessary assumptions and conclusions.
Use standard mathematical definitions from Mathlib where appropriate.
Provide only the Lean 4 code without any additional explanation.
"""

# ====================== API Call Function ======================
def call_gpt4o_multiple(description, num_calls=10):
    translations = []
    for i in range(num_calls):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(description=description)}
                ],
                top_p=0.95,
                max_tokens=2048,
                temperature=0.3
            )
            translations.append(response.choices[0].message.content)
            time.sleep(0.5)
        except Exception as e:
            print(f"API call failed (attempt {i+1}): {str(e)}")
            translations.append("translation_failed")
    return translations

# ====================== Data Processing ======================
def process_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {item['id'] for item in results}
    else:
        results = []
        processed_ids = set()

    for item in tqdm(data, desc="Processing with GPT-4o"):
        if item['id'] in processed_ids:
            continue

        description = item.get("natural_language_statement", "")
        if not description:
            continue

        try:
            translations = call_gpt4o_multiple(description, num_calls=10)
            item["outputs"] = translations
        except Exception:
            item["outputs"] = ["translation_failed"] * 10

        results.append(item)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)} items. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
