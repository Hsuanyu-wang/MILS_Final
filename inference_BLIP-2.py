import os
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import csv

# 1. Load BLIP-2 model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. Set main directories for road and freeway
main_dirs = [
    ("road", "/home/MILS_Final/AVA_Dataset/road/test"),
    ("freeway", "/home/MILS_Final/AVA_Dataset/freeway/test")
]

# 3. Set the question
question = "Question: Is this road risky?(answer 1:risky, answer 0:not risky) Answer:"

def extract_label(text):
    text = text.strip()
    if text == '1' or text.lower().startswith('1') or 'risky' in text.lower():
        return 1
    return 0

# 4. Prepare results
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, 'submission.csv')

submission_rows = [('file_name', 'risk')]

for prefix, main_dir in main_dirs:
    for folder in sorted(os.listdir(main_dir)):
        folder_path = os.path.join(main_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if not img_files:
            continue
        labels = []
        for img_file in img_files:
            image = Image.open(os.path.join(folder_path, img_file)).convert('RGB')
            inputs = processor(images=[image], text=[question], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            label = extract_label(generated_texts[0])
            labels.append(label)
            del image, inputs, generated_ids, generated_texts
            torch.cuda.empty_cache()
        # Show all image predictions in the folder
        print(f"Folder {prefix}_{folder} image predictions: {labels}")
        # If any image is risky, mark the folder as risky
        video_label = 1 if any(labels) else 0
        print(f"Folder {prefix}_{folder} prediction(1:risk, 0:safe): {video_label}")
        submission_rows.append((f"{prefix}_{folder}", video_label))

# 5. Write to CSV in the required format
with open(results_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(submission_rows)
print(f"Results saved to {results_path}")