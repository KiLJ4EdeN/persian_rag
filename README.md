# persian_rag

simple rag application in persian

```bash
pip install -qq -U transformers
pip install -qq -U accelerate
pip install -qq -U bitsandbytes
pip install -qq -U langchain
pip install -qq -U sentence-transformers
huggingface-cli login --token <TOK>
mkdir part-model
huggingface-cli download PartAI/Dorna-Llama3-8B-Instruct --local-dir part-model --local-dir-use-symlinks False
```

```python
import json
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

config = AutoConfig.from_pretrained("part-model")
config.gradient_checkpointing = True

generator_model = AutoModelForCausalLM.from_pretrained(
    "part-model",
    device_map=device,
    trust_remote_code=True,
    quantization_config=quantization_config,
    config=config
)

generator_tokenizer = AutoTokenizer.from_pretrained("part-model")

sentence_transformer_model = SentenceTransformer('ahdsoft/persian-sentence-transformer-news-wiki-pairs-v3')

knowledge_base = [
    ("چه خدماتی در دانشگاه ارائه می‌شود؟", "دانشگاه خدمات آموزشی، پژوهشی و رفاهی به دانشجویان ارائه می‌دهد."),
    ("نحوه ثبت نام در دانشگاه چگونه است؟", "برای ثبت نام در دانشگاه، ابتدا باید فرم درخواست را پر کنید و مدارک مورد نیاز را ارائه دهید."),
    ("شرایط پذیرش در دانشگاه چیست؟", "پذیرش در دانشگاه بستگی به معدل دیپلم و نمرات کنکور دارد."),
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)

def split_documents(knowledge_base):
    split_knowledge_base = []
    for qa in knowledge_base:
        question, answer = qa
        chunks = text_splitter.split_text(answer)
        for chunk in chunks:
            split_knowledge_base.append((question, chunk))
    return split_knowledge_base

split_knowledge_base = split_documents(knowledge_base)

def get_embeddings(text, model):
    embeddings = model.encode(text)
    return embeddings

def retrieve_documents(query, knowledge_base, model, top_n=3, similarity_threshold=0.2):
    query_embedding = get_embeddings([query], model)
    kb_embeddings = [get_embeddings([qa[0]], model) for qa in knowledge_base]
    similarities = [cosine_similarity([query_embedding.reshape(1024)], [kb_embedding.reshape(1024)]).item() for kb_embedding in kb_embeddings]
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    if not filtered_indices:
        return []

    top_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)[:top_n]
    return [knowledge_base[i] for i in top_indices]

query = 'چگونه می‌توانم در دانشگاه ثبت نام کنم؟'
retrieved_documents = retrieve_documents(query=query, knowledge_base=split_knowledge_base, model=sentence_transformer_model, top_n=10, similarity_threshold=0.2)
retrieved_knowledge = "\n".join([doc[0].replace('\n', ' ') + '\n' + doc[1].replace('\n', ' ') for doc in retrieved_documents]) if retrieved_documents else ""
with open('knowledge.txt', 'w') as f:
    f.write(retrieved_knowledge)

messages = [
    {"role": "system", "content": f"You are provided with Knowledge, answer the question using it, say i do not know if its not in there, the answer must be in persian. Knowledge: {retrieved_knowledge}"},
    {"role": "user", "content": query},
]

input_ids = generator_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(generator_model.device)

terminators = [
    generator_tokenizer.eos_token_id,
    generator_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = generator_model.generate(
    input_ids,
    max_new_tokens=400,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]
resp_final = generator_tokenizer.decode(response, skip_special_tokens=True)
with open('out.txt', 'w') as f:
    f.write(str(resp_final))
```
