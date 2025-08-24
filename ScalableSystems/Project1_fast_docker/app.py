from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import redis
import json
import hashlib

MODEL_PATH = "./models/multi-class-humanitarian_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
label_map = {0:'affected_individuals', 1: 'infrastructure_and_utility_damage', 2: 'injured_or_dead_people', 3:'missing_or_found_people', 4: 'not_humanitarian', 5:'other_relevant_information',6:'rescue_volunteering_or_donation_effort', 7:'vehicle_damage'}


#Redis Connection
redis_client = redis.Redis(host="redis", port=6379, db=0)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    #caching the key of input
    cache_key = hashlib.sha256(input_data.text.encode()).hexdigest()
    #checking redis if the input is already present or not
    cached_result = redis_client.get(cache_key)

    if cached_result:
        return {"cached": True, "prediction": json.loads(cached_result)}
    
    #tokenize
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)

    #predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        predicted_class_id = torch.argmax(logits, dim=1).item()
        

    predicted_label = label_map[predicted_class_id]

    result = {
        "class_id": predicted_class_id,
        "label": predicted_label
    }

   

    redis_client.set(cache_key, json.dumps(result))

    return {"cached": False, "predicton": result}