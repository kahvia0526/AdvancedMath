import json
import requests
from tqdm import tqdm
from time import sleep
from openai import OpenAI

# Configuration settings
DEEPSEEK_API_KEY = "YOUR_API_KEY"  # Replace with actual API key
API_ENDPOINT = "https://api.deepseek.com"
INPUT_FILE = "dataset.json"
OUTPUT_FILE = "abstract_concepts.json"

# Initialize API client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=API_ENDPOINT)

# Prompt template for concept extraction
SYSTEM_PROMPT = """Analyze the given mathematical theorem description and:
1. Extract fundamental mathematical concepts in English
2. Identify the most relevant mathematical domain in English
Return JSON format: {"concepts": ["...", ...], "domain": "..."}"""

USER_TEMPLATE = """Analyze this mathematical theorem description:
{description}
Provide response in exact JSON format:"""

def call_deepseek_api(description):
    """Call API for concept extraction"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(description=description)}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        
        # Validate response format
        if not isinstance(result.get("concepts", []), list):
            raise ValueError("Invalid concepts format")
        if not isinstance(result.get("domain", ""), str):
            raise ValueError("Invalid domain format")
            
        return result
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

def process_data():
    """Process dataset and extract concepts"""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        description = item.get("statement", "")
        if not description:
            continue
        
        response = call_deepseek_api(description)
        if not response:
            parsed = {"concepts": [], "domain": ""}
        else:
            parsed = {
                "concepts": [c.strip() for c in response.get("concepts", []) if c],
                "domain": response.get("domain", "").strip()
            }
        
        new_item = {
            **item,
            "abstract_concepts": parsed["concepts"],
            "mathematical_domain": parsed["domain"]
        }
        results.append(new_item)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(results)} items")

def deduplicate_concepts():
    """Create unique concept list"""
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    concept_set = set()
    for item in data:
        for concept in item['abstract_concepts']:
            concept_set.add(concept)
    
    with open('unique_concepts.json', 'w', encoding='utf-8') as f:
        json.dump({"concepts": list(concept_set)}, f, ensure_ascii=False, indent=2)

def generate_concept_explanation(concept):
    """Generate English explanation for concept"""
    system_msg = "You are a math expert specialized in concise explanations."
    user_msg = f"""Explain this mathematical concept in one professional English sentence:
{concept}"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Explanation Error: {str(e)}")
        return None

def process_concept_explanations():
    """Generate explanations for all concepts"""
    with open('unique_concepts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for i, concept in enumerate(tqdm(data["concepts"], desc="Generating explanations")):
        explanation = generate_concept_explanation(concept)
        if not explanation:
            explanation = "Explanation unavailable"
        
        results.append({
            "concept": concept,
            "explanation": explanation
        })
        sleep(1)  # Rate limiting
    
    with open('concept_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def map_to_mathlib(concept, explanation):
    """Find Mathlib4 references for concept"""
    system_prompt = """You are a Lean4 math library expert. Return JSON format:
{"names": ["def1", ...], "theorems": ["thm1", ...]}"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Concept: {concept}\nExplanation: {explanation}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Mapping Error: {str(e)}")
        return {"names": [], "theorems": []}

def create_mathlib_mappings():
    """Create Mathlib4 mappings for all concepts"""
    with open('concept_explanations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for item in tqdm(data, desc="Creating Mathlib mappings"):
        mapping = map_to_mathlib(item["concept"], item["explanation"])
        results.append({
            "concept": item["concept"],
            "mathlib_info": mapping
        })
    
    with open('mathlib_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_data()
    deduplicate_concepts()
    process_concept_explanations()
    create_mathlib_mappings()