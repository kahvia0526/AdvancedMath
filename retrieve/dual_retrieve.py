import json
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os

class DualRetriever:
    def __init__(self, device="cuda"):
        """
        Initialize dual-path retrieval system with semantic and symbolic components
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.semantic_model = self._load_semantic_model()
        self.symbolic_model = self._load_symbolic_model()
        
    def _load_semantic_model(self):
        """Load semantic similarity model"""
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(self.device)
        model.eval()
        return {"model": model, "tokenizer": tokenizer}
    
    def _load_symbolic_model(self):
        """Load symbolic reranking model"""
        model_name = 'BAAI/bge-reranker-v2-m3'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        model.eval()
        return {"model": model, "tokenizer": tokenizer}
    
    def calculate_semantic_similarity(self, text1, text2):
        """
        Compute cosine similarity between two text segments
        
        Args:
            text1 (str): First text segment
            text2 (str): Second text segment
            
        Returns:
            float: Cosine similarity score
        """
        if not text1 or not text2:
            return 0.0
            
        tokenizer = self.semantic_model["tokenizer"]
        model = self.semantic_model["model"]
        
        inputs = tokenizer([text1, text2], 
                          padding=True,
                          truncation=True,
                          max_length=512,
                          return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            similarity = torch.nn.functional.cosine_similarity(
                cls_embeddings[0], cls_embeddings[1], dim=0
            ).item()
            
        return similarity

    def perform_symbolic_retrieval(self, concept_explanation, candidates):
        """
        Retrieve top definitions using symbolic matching
        
        Args:
            concept_explanation (str): Explanation of mathematical concept
            candidates (list): List of candidate definitions
            
        Returns:
            list: Reranked candidates with scores
        """
        tokenizer = self.symbolic_model["tokenizer"]
        model = self.symbolic_model["model"]
        
        if not candidates:
            return []
            
        pairs = [[concept_explanation, cand["explan"]] for cand in candidates]
        
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
        
        for idx, cand in enumerate(candidates):
            cand["rerank_score"] = scores[idx].item()
            
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    def hybrid_retrieval(self, source_concepts, target_concepts, top_k=5):
        """
        Perform dual-path hybrid retrieval combining semantic and symbolic approaches
        
        Args:
            source_concepts (list): Concepts to be formalized
            target_concepts (list): Reference concepts from knowledge base
            top_k (int): Number of top matches to return
            
        Returns:
            list: Enriched concepts with retrieved definitions
        """
        results = []
        
        for source in tqdm(source_concepts, desc="Processing concepts"):
            source_explanation = source['explanation']
            matches = []
            
            # Semantic similarity matching
            for target in target_concepts:
                similarity = self.calculate_semantic_similarity(
                    source_explanation, 
                    target['explanation']
                )
                matches.append({
                    "concept": target['concept'],
                    "key": target['key'],
                    "score": similarity,
                    "explanation": target['explanation']
                })
            
            # Select top semantic matches
            matches.sort(key=lambda x: x['score'], reverse=True)
            semantic_matches = matches[:top_k]
            
            # Symbolic reranking
            reranked_matches = self.perform_symbolic_retrieval(
                source_explanation,
                semantic_matches
            )
            
            # Filter by threshold
            filtered_matches = [
                match for match in reranked_matches 
                if match.get("rerank_score", -1) >= 0.0
            ]
            
            result_entry = source.copy()
            result_entry['retrieved_definitions'] = filtered_matches[:top_k]
            results.append(result_entry)
            
        return results

    def build_context(self, concepts, knowledge_base):
        """
        Build formalization context from retrieved definitions
        
        Args:
            concepts (list): Concepts with retrieved definitions
            knowledge_base (list): Full knowledge base entries
            
        Returns:
            list: Concepts enriched with formal context
        """
        for item in concepts:
            context = []
            definitions = item.get('retrieved_definitions', [])
            
            for definition in definitions:
                concept_name = definition['concept']
                # Find corresponding knowledge base entry
                kb_entry = next(
                    (kb for kb in knowledge_base if kb['concept'] == concept_name), 
                    None
                )
                if kb_entry:
                    context.extend(kb_entry.get('def_info', []))
            
            # Remove duplicate definitions
            seen = set()
            unique_context = [
                ctx for ctx in context 
                if not (ctx['def_name'] in seen or seen.add(ctx['def_name']))
            ]
            
            item['formal_context'] = unique_context
            
        return concepts

    def save_results(self, data, output_path):
        """Save processed results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")

def main():
    # Initialize dual retriever
    retriever = DualRetriever()
    
    # Load source concepts (to be formalized)
    with open('source_concepts.json', 'r', encoding='utf-8') as f:
        source_concepts = json.load(f)
    
    # Load target concepts (knowledge base)
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        target_concepts = json.load(f)
    
    # Perform hybrid retrieval
    enriched_concepts = retriever.hybrid_retrieval(source_concepts, target_concepts)
    
    # Build formalization context
    final_concepts = retriever.build_context(enriched_concepts, target_concepts)
    
    # Save results
    retriever.save_results(final_concepts, 'formalization_context.json')

if __name__ == '__main__':
    main()