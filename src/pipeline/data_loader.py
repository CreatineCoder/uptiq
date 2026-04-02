import json
import logging
import os
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_nq_context(document_html):
    """
    NQ includes full HTML. Simple tag removal to simulate a retrieved document.
    """
    import re
    # Remove HTML tags
    clean_text = re.sub('<[^<]+>', ' ', document_html)
    # Remove excessive whitespace
    clean_text = ' '.join(clean_text.split())
    # Limit to 4000 characters
    return clean_text[:4000]

def build_benchmark_dataset():
    output_path = "data/processed/benchmark_dataset.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    benchmark_data = []

    # 1. HotpotQA (500 samples)
    logging.info("Downloading HotpotQA (fullwiki validation)...")
    hotpot = load_dataset('hotpot_qa', 'fullwiki', split='validation')
    hotpot_samples = hotpot.select(range(500))
    
    for item in tqdm(hotpot_samples, desc="Processing HotpotQA"):
        # Fix: HotpotQA context structure is a dictionary with 'title' and 'sentences' lists
        # item['context'] structure: {'title': ['Title1', 'Title2'], 'sentences': [['S1', 'S2'], ['S3', 'S4']]}
        context_str = ""
        context_dict = item["context"]
        for title, sentences in zip(context_dict["title"], context_dict["sentences"]):
            context_str += f"Title: {title}\n" + " ".join(sentences) + "\n\n"
            
        benchmark_data.append({
            "id": f"hotpot_{item['id']}",
            "question": item["question"],
            "gold_answer": item["answer"],
            "gold_context": context_str.strip(),
            "dataset": "hotpotqa",
            "difficulty": "multi-hop",
            "supporting_facts": {
                "titles": item["supporting_facts"]["title"], 
                "sent_ids": item["supporting_facts"]["sent_id"]
            }
        })

    # 2. Natural Questions (1000 samples)
    # Using 'nq_open' which is the pre-processed version for RAG (no HTML, clean answers)
    logging.info("Downloading Natural Questions (nq_open)...")
    try:
        nq = load_dataset('nq_open', split='validation[:1000]')
        
        for i, item in enumerate(tqdm(nq, desc="Processing Natural Questions")):
            # nq_open has 'question' and 'answer' (list of strings)
            benchmark_data.append({
                "id": f"nq_{i}",
                "question": item["question"],
                "gold_answer": item["answer"][0] if item["answer"] else "UNANSWERABLE",
                "gold_context": "Context not available in nq_open. This will be retrieved from Wikipedia during the benchmark.",
                "dataset": "nq",
                "difficulty": "single-hop",
                "supporting_facts": None
            })
    except Exception as e:
        logging.error(f"Failed to load NQ: {e}")

    # 3. Save to unified JSONL
    logging.info(f"Saving {len(benchmark_data)} queries to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in benchmark_data:
            f.write(json.dumps(entry) + "\n")
            
    logging.info("Data preparation complete! 🎉")

if __name__ == "__main__":
    build_benchmark_dataset()
