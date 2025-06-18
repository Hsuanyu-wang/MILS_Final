import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from CLIP4CLIP_train import CLIP4ClipRiskPredictor, RiskCLIPDataset
from tqdm import tqdm

# 推論資料集
class InferenceRiskCLIPDataset(Dataset):
    def __init__(self, file_names, root_dir, sample_num=16):
        self.file_names = file_names
        self.root_dir = root_dir
        self.sample_num = sample_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_prompts = [
            "a safe traffic scene with no danger or risk",
            "a dangerous traffic scene with high risk and potential accidents"
        ]

    def load_and_preprocess_frames(self, video_folder):
        frames = []
        video_path = os.path.join(self.root_dir, video_folder)
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for img_file in image_files:
            img_path = os.path.join(video_path, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                frames.append(self.preprocess(image))
            except Exception:
                continue
        if len(frames) > self.sample_num:
            import numpy as np
            indices = np.linspace(0, len(frames) - 1, self.sample_num).astype(int)
            frames = [frames[i] for i in indices]
        while len(frames) < self.sample_num:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, 224, 224))
        return torch.stack(frames[:self.sample_num])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        frames = self.load_and_preprocess_frames(file_name)
        text_tokens = clip.tokenize(self.text_prompts).to(self.device)
        return {'frames': frames, 'file_name': file_name}

def get_test_folders(test_dir):
    return sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

def inference_and_save(test_dir, model_path, output_csv):
    file_names = get_test_folders(test_dir)
    dataset = InferenceRiskCLIPDataset(file_names, test_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictor = CLIP4ClipRiskPredictor(fusion_method='transformer')
    predictor.create_model()
    predictor.load_model(model_path)
    predictor.model.eval()
    results = []
    device = predictor.device
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Inference {os.path.basename(test_dir)}"):
            frames = batch['frames'].to(device)
            file_name = batch['file_name'][0]
            logits = predictor.model(frames)
            pred = torch.argmax(logits, dim=1).item()
            results.append({'file_name': file_name, 'risk': pred})
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved inference results to {output_csv}")

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'clip4clip_freeway_risk.pth')
    # 1. road
    test_dir_road = '/home/MILS_Final/AVA_Dataset/road/test'
    output_csv_road = 'road_submission.csv'
    inference_and_save(test_dir_road, model_path, output_csv_road)
    # 2. freeway
    test_dir_freeway = '/home/MILS_Final/AVA_Dataset/freeway/test'
    output_csv_freeway = 'freeway_submission.csv'
    inference_and_save(test_dir_freeway, model_path, output_csv_freeway)
    # # 合併成 sample_submission 格式
    # import pandas as pd
    # sample_path = os.path.join(os.path.dirname(__file__), 'AVA_Dataset', 'sample_submission.csv')
    # road_df = pd.read_csv(output_csv_road)
    # freeway_df = pd.read_csv(output_csv_freeway)
    # submission = pd.concat([road_df, freeway_df], ignore_index=True)
    # sample = pd.read_csv(sample_path)
    # submission = submission.set_index('file_name').reindex(sample['file_name']).reset_index()
    # submission.to_csv('final_submission.csv', index=False)
    # print('已產生 final_submission.csv') 