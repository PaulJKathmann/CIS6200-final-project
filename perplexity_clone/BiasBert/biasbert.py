from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BiasBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
    
    def classify(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,  # Ensure truncation to max length
            max_length=512    # Explicitly set max length to 512
        )
        labels = torch.tensor([0])
        outputs = self.model(**inputs, labels=labels)
        loss, logits = outputs[:2]
        # [0] -> left 
        # [1] -> center
        # [2] -> right
        return logits.softmax(dim=-1)[0].tolist()
    
    def classify_batch(self, texts):
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,  # Ensure truncation to max length
            max_length=512    # Explicitly set max length to 512
        )
        labels = torch.zeros(len(texts), dtype=torch.long)  # Ensure labels match batch size
        outputs = self.model(**inputs, labels=labels)
        return outputs.logits.softmax(dim=-1).tolist()