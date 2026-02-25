import json
import requests
from tqdm import tqdm
from time import sleep
from openai import OpenAI

# Configuration settings
DEEPSEEK_API_KEY = "YOUR_API_KEY"  # Replace with actual API key
API_ENDPOINT = "https://api.deepseek.com"
INPUT_FILE = "problem_dataset.json"
OUTPUT_FILE = "rewritten_problems.json"

# Initialize API client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=API_ENDPOINT)

def rewrite_problems():
    """Rewrite math problems to explicit mathematical concepts"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    # Prompt for rewriting
    system_prompt = """Rewrite mathematical problems using explicit mathematical concepts"""
    user_template = """Rewrite this problem: {problem}"""
    
    for problem in tqdm(problems, desc="Rewriting problems"):
        if "rewritten" in problem:
            continue
            
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.format(problem=problem["text"])}
                ],
                temperature=0.2,
                max_tokens=256
            )
            problem["rewritten"] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Rewriting Error: {str(e)}")
            problem["rewritten"] = problem["text"]
        
        sleep(1)  # Rate limiting
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)

# Prompt template for concept extraction
SYSTEM_PROMPT = """Analyze mathematical descriptions and extract:
1. Fundamental concepts
2. Relevant mathematical domain
Return JSON: {"concepts": ["...", ...], "domain": "..."}"""

def extract_concepts(description):
    """Extract concepts from rewritten text"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Description: {description}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {"concepts": [], "domain": ""}

def process_rewritten_data():
    """Process rewritten problems for concept extraction"""
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    for item in tqdm(data, desc="Extracting concepts"):
        description = item.get("rewritten", "")
        if not description:
            continue
        
        response = extract_concepts(description)
        concepts = [c.strip() for c in response.get("concepts", []) if c]
        domain = response.get("domain", "").strip()
        
        item["abstract_concepts"] = concepts
        item["mathematical_domain"] = domain
        results.append(item)
    
    with open("final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def generate_concept_explanation(concept):
    """Generate English explanation for concept"""
    system_msg = "Provide concise mathematical explanations"
    user_msg = f"""Explain: {concept}"""
    
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
    with open('final_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    concept_set = set()
    for item in data:
        for concept in item['abstract_concepts']:
            concept_set.add(concept)
    
    results = []
    for concept in tqdm(concept_set, desc="Generating explanations"):
        explanation = generate_concept_explanation(concept)
        if not explanation:
            explanation = "Explanation unavailable"
        results.append({"concept": concept, "explanation": explanation})
        sleep(1)
    
    with open('concept_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def map_to_mathlib(concept, explanation):
    """Find Mathlib4 references for concept"""
    system_prompt = """Identify Mathlib4 definitions and theorems"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Concept: {concept}\n{explanation}"}
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
    rewrite_problems()
    process_rewritten_data()
    process_concept_explanations()
    create_mathlib_mappings()