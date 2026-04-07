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

    # 2. SQuAD v2 (1000 samples with answers)
    # Filter to only questions that have at least one answer
    logging.info("Downloading and filtering SQuAD v2 (validation)...")
    try:
        squad = load_dataset('squad_v2', split='validation')
        # Filter for only answerable questions (where answers['text'] is not empty)
        squad_answerable = squad.filter(lambda x: len(x['answers']['text']) > 0)
        squad_samples = squad_answerable.select(range(min(1000, len(squad_answerable))))
        
        for item in tqdm(squad_samples, desc="Processing SQuAD"):
            # squad_v2 has 'question', 'context', and 'answers' (list of dicts)
            benchmark_data.append({
                "id": f"squad_{item['id']}",
                "question": item["question"],
                "gold_answer": item['answers']['text'][0],
                "gold_context": item["context"],
                "dataset": "squad_v2",
                "difficulty": "single-hop",
                "supporting_facts": None
            })
    except Exception as e:
        logging.error(f"Failed to load SQuAD: {e}")

    # 3. Save to unified JSONL
    logging.info(f"Saving {len(benchmark_data)} queries to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in benchmark_data:
            f.write(json.dumps(entry) + "\n")
            
    logging.info("Data preparation complete! 🎉")

if __name__ == "__main__":
    build_benchmark_dataset()
