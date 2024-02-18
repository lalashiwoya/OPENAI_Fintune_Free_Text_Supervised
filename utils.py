from langchain_community.document_loaders import PyPDFLoader
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
import json
from typing import Tuple, List, Dict, Any
import pandas as pd
import toml
from openai import FineTuningJob
import tomli_w

def load_split_pdf(pdf_path:str) -> str:
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    page_content = " "
    for page in pages:
        page_content += page.page_content
    return page_content


def clean_text(text:str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r' \.{2,}', '.', text)
    text = re.sub(r' \n', ' ', text)
    text = re.sub(r'•', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' . .', ' ', text)
    text = re.sub(r'\uf0b7', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def split_texts_to_chunks(text:str)->list:
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
    )
    return r_splitter.split_text(text)

# all_page_contents = [load_split_pdf(pdf_path) for pdf_path in glob.glob("datasets/*.pdf")]
# all_clean_texts = [clean_text(text)for text in all_page_contents]
# all_short_texts = []
# for text in all_clean_texts:
#     all_short_texts += split_texts_to_chunks(text)


def mask_text_randomly(text:str) -> Tuple[str, List[str]]:
    # Tokenize the text by words, keeping punctuations attached to the word
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    
    # Calculate how many tokens to mask (20% of total tokens, rounded)
    num_to_mask = round(len(tokens) * 0.2)
    
    # Randomly select indices of tokens to mask
    indices_to_mask = random.sample(range(len(tokens)), num_to_mask)
    
    # Initialize list for masked tokens
    masked_tokens = []
    
    # Create a copy of the tokens to modify
    masked_text_tokens = tokens[:]
    
    # Replace selected tokens with <mask> and collect masked tokens
    for index in indices_to_mask:
        masked_tokens.append(masked_text_tokens[index])
        masked_text_tokens[index] = '<mask>'
    
    # Reconstruct the text from the masked tokens
    masked_text = ' '.join(masked_text_tokens).replace(' <mask>', '<mask>')  # Adjust spacing around <mask>
    
    return masked_text, masked_tokens

def prepare_training_conversation(row: pd.Series, system_message:str) -> Dict[str, Any] :
    messages = []
    messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": row['human']})
    
    messages.append({"role": "assistant", "content": row['assistant']})
    
    return {"messages": messages}

# training_data = df.apply(lambda x: prepare_training_conversation(x, system_message), axis=1).tolist()


def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


def df_to_jsonl(df: pd.DataFrame, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            jout = json.dumps(row.to_dict()) + "\n"
            out.write(jout)
            
def load_toml(file_path: str) -> None:
    with open(file_path, 'r') as toml_file:
        data = toml.load(toml_file)
    return data

def num_tokens_from_string(string:str, encoder) -> int:
    return len(encoder.encode(string))

def print_fine_tune_status(response: FineTuningJob) -> None:
    print("Job ID:", response.id)
    print("Status:", response.status)
    print("Trained Tokens:", response.trained_tokens)

def print_training_process(response_id: str, client) -> None:
    response = client.fine_tuning.jobs.list_events(response_id)
    events = response.data
    events.reverse()

    for event in events:
        print(event.message)

def save_response_id_to_config(config, config_file_path, response_id):
    with open(config_file_path, "wb") as f:
        config['response id'] = response_id
        tomli_w.dump(config, f)