import torch
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", pad_token_id=tokenizer.eos_token_id)
res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])