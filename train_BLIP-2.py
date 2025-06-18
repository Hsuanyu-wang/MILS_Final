import os
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import csv

# 1. Define dataset class
class AVADataset(Dataset):
    def __init__(self, main_dirs, csv_paths, processor, question):
        self.samples = []
        self.processor = processor
        self.question = question
        # Read CSVs for folder labels
        folder2label = {}
        for prefix, csv_path in csv_paths.items():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    folder2label[f"{prefix}_{row['file_name']}"] = int(row['risk'])
        # Collect image paths and labels
        for prefix, main_dir in main_dirs:
            for folder in sorted(os.listdir(main_dir)):
                folder_path = os.path.join(main_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                folder_name = f"{prefix}_{folder}"
                label = folder2label.get(folder_name, None)
                if label is None:
                    continue
                img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                for img_file in img_files:
                    img_path = os.path.join(folder_path, img_file)
                    self.samples.append((img_path, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        answer = str(label)  # "1" or "0"
        encoding = self.processor(
            images=image,
            text=self.question,
            return_tensors="pt",
            padding=True
        )
        # Tokenize the answer as the target
        target = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding=True
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = target['input_ids'].squeeze(0)
        return encoding

# 2. Set up paths and parameters
main_dirs = [
    ("road", "/home/MILS_Final/AVA_Dataset/road/train"),
    ("freeway", "/home/MILS_Final/AVA_Dataset/freeway/train")
]
csv_paths = {
    "road": "/home/MILS_Final/AVA_Dataset/road_train.csv",
    "freeway": "/home/MILS_Final/AVA_Dataset/freeway_train.csv"
}
question = "Question: Is this road risky?(answer 1:risky, answer 0:not risky) Answer:"

# 3. Load processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 4. Prepare dataset
train_dataset = AVADataset(main_dirs, csv_paths, processor, question)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./blip2-finetuned-ava",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to=[],
)

# 6. Define data collator
def data_collator(features):
    batch = {}
    for k in features[0]:
        batch[k] = torch.stack([f[k] for f in features])
    return batch

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 8. Train
trainer.train()

# 9. Save model
trainer.save_model("./blip2-finetuned-ava")