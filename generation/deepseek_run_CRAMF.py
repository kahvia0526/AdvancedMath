from openai import OpenAI
import json
from tqdm import tqdm
import time
import os

# ====================== Configuration ======================
DEEPSEEK_API_KEY = "YOUR_API_KEY"
API_ENDPOINT = "https://your.api.endpoint"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=API_ENDPOINT)

INPUT_FILE = "data/input_with_context.json"
OUTPUT_FILE = "data/output_deepseek_context.json"

# ====================== Prompt Template ======================
SYSTEM_PROMPT = """   
You are an expert mathematician and Lean 4 programmer.
Translate the following mathematical theorem from natural language
into formal Lean 4 code with a theorem header.
Use the mathematical definitions provided in the context for reference.

Important instructions:
1. ONLY translate the theorem statement. DO NOT provide any proof or implementation.
2. End the translated theorem with `sorry` to indicate the proof is omitted.
3. Provide only the Lean 4 code without any additional explanation.
"""

def build_user_prompt(description, context):
    instruction = (
        "Task: Convert the following problem into a Lean 4 theorem statement ONLY.\n\n"
        "Important Constraints:\n"
        "- Translate ONLY the theorem statement.\n"
        "- End the theorem with `sorry`.\n"
        "- Use the mathematical definitions below for reference, "
        "but DO NOT translate the definitions themselves.\n\n"
        f"Problem to translate:\n{description}\n\n"
        "Reference definitions:\n"
    )

    context_entries = [
        f"- {ctx.get('def_name', 'Unknown Concept')}: {ctx.get('def_content', 'No content available.')}"
        for ctx in context if 'def_name' in ctx and 'def_content' in ctx
    ]
    context_str = "\n".join(context_entries) if context_entries else "No context available."

    return f"{instruction}{context_str}\n\nDo not include explanations."

# ====================== API Call Function ======================
def call_deepseek_api_multiple(description, context, num_calls=10):
    translations = []
    user_prompt = build_user_prompt(description, context)

    for i in range(num_calls):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
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

    for item in tqdm(data, desc="Processing"):
        if item['id'] in processed_ids:
            continue

        description = item.get("natural_language_statement", "")
        context = item.get("context", [])
        if not description:
            continue

        try:
            translations = call_deepseek_api_multiple(description, context, num_calls=10)
            item["outputs"] = translations
        except Exception:
            item["outputs"] = ["translation_failed"] * 10

        results.append(item)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)} items. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
