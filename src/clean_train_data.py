import json
import os

def clean_squad_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    cleaned_data = []
    total_qas_before = 0
    total_qas_after = 0

    for article in dataset.get('data', []):
        cleaned_paragraphs = []
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '').strip()
            if not context:
                continue
                
            cleaned_qas = []
            for qa in paragraph.get('qas', []):
                total_qas_before += 1
                
                question = qa.get('question', '').strip()
                if not question:
                    continue
                    
                cleaned_answers = []
                for answer in qa.get('answers', []):
                    answer_text = answer.get('text', '').strip()
                    if answer_text:
                        answer['text'] = answer_text
                        cleaned_answers.append(answer)
                
                if cleaned_answers:
                    qa['question'] = question
                    qa['answers'] = cleaned_answers
                    cleaned_qas.append(qa)
                    total_qas_after += 1
            
            if cleaned_qas:
                paragraph['context'] = context
                paragraph['qas'] = cleaned_qas
                cleaned_paragraphs.append(paragraph)
                
        if cleaned_paragraphs:
            article['paragraphs'] = cleaned_paragraphs
            cleaned_data.append(article)
            
    dataset['data'] = cleaned_data

    print(f"Cleaning complete. QAs before: {total_qas_before}, QAs after: {total_qas_after}")
    print(f"Saving cleaned data to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == '__main__':
    # File paths based on workspace structure
    workspace_dir = os.path.join(os.path.dirname(__file__), '..')
    input_file = os.path.join(workspace_dir, 'archive (5)', 'train-v1.1.json')
    output_file = os.path.join(workspace_dir, 'archive (5)', 'cleaned-train-v1.1.json')
    
    clean_squad_data(input_file, output_file)
