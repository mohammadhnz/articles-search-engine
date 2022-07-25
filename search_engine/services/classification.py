import os

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    TextClassificationPipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification


class ClassificationService:
    def __init__(self):
        self.model, self.tokenizer = self.load_transformer_based_model()

    def load_transformer_based_model(self):
        cl_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.dirname(__file__) + '/../../results/checkpoint-500',
            num_labels=3,
            label2id={"'ECCV'": 0, "CVPR": 1, "ICCV": 2},
            id2label={0: "ECCV", 1: "CVPR", 2: "ICCV"},
        )
        checkpoint = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return cl_model, tokenizer

    def predict_text_class(self, text):
        pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)
        prediction = pipe(text, return_all_scores=True)
        return max(prediction, key=lambda x: x['score'])


classification_service = ClassificationService()
