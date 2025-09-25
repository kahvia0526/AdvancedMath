from openai import OpenAI
import json
from tqdm import tqdm
import time
import os


OPENAI_API_KEY = "xxx"
client =OpenAI(
      base_url="xxx",
      api_key="xxx"
  )

INPUT_FILE = ""
OUTPUT_FILE = ""

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
        "Task: Convert the following problem into LEAN 4 theorem statement ONLY.\n\n "
        "Important Constraints:\n"
        "- Translate ONLY the theorem statement, DO NOT include any proof\n"
        "- End the theorem with `sorry` to omit the proof\n"
        "- Use the mathematical definitions provided below for reference, "
        "but DO NOT translate the definitions themselves.\n\n"
        "Problem to translate:\n"
        f"{description}\n\n"
        "Reference definitions (for context only):\n"
    )
    

    context_entries = []
    for ctx in context:

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
    
    context_str = "\n".join(context_entries) if context_entries else "No additional context provided."
    
    full_prompt = (
        f"{instruction}{context_str}\n\n"
        "Important Notes:\n"
        "1. Use the provided definitions to understand the mathematical concepts.\n"
        "2. Ensure the formal statement includes all necessary assumptions and conclusions.\n"
        "3. Use standard mathematical definitions from Mathlib where appropriate.\n"
        "4. Provide only the Lean 4 code without any additional explanation."
    )
    
    return full_prompt

# ====================== API Call Function ======================
def call_deepseek_api_multiple(description, context, num_calls=10):
    """Call DeepSeek API multiple times for Lean 4 formalization with context"""
    translations = []
    

    user_prompt = build_user_prompt(description, context)
    
    for i in range(num_calls):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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
    
    for idx, item in enumerate(tqdm(data, desc="Processing")):

        if item['id'] in processed_ids:
            continue
            
        description = item.get("natural_language_statement", "")
        if not description:
            print(f"Skipping item {item['id']} with empty description")
            continue
        
        try:
    
            context = item.get("context", [])
            

            translations = call_deepseek_api_multiple(description, context, num_calls=10)
            item["outputs"] = translations
            print(f"Item {item['id']}: Generated {len(translations)} translations with context")
            

            results.append(item)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            processed_ids.add(item['id'])
            
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
            item["outputs"] = ["translation_failed"] * 10
            results.append(item)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(results)} items. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()