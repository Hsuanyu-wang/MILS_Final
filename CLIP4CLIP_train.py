# ============================================================================
# CLIP4Clip 風險預測完整實作
# ============================================================================

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 安裝必要套件
# !pip install ftfy regex tqdm
# !pip install git+https://github.com/openai/CLIP.git

class RiskCLIPDataset(Dataset):
    """CLIP4Clip 風險預測資料集"""
    
    def __init__(self, csv_file, root_dir, preprocess, sample_num=16, is_train=True):
        """
        Args:
            csv_file: CSV檔案路徑 (格式: file_name,risk)
            root_dir: 影像資料夾路徑
            preprocess: CLIP 的預處理方法
            sample_num: 每個影片採樣的幀數
            is_train: 是否為訓練模式
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.sample_num = sample_num
        self.is_train = is_train
        
        # 風險預測的文字提示
        self.text_prompts = [
            "a safe traffic scene with no danger or risk",
            "a dangerous traffic scene with high risk and potential accidents"
        ]
        
        print(f"載入CLIP資料集: {len(self.data)} 個樣本")
        print(f"Risk分布: {self.data['risk'].value_counts().to_dict()}")
    
    def load_and_preprocess_frames(self, video_folder):
        """載入並預處理影片幀"""
        frames = []
        video_path = os.path.join(self.root_dir, video_folder)
        
        if not os.path.exists(video_path):
            print(f"警告: 找不到資料夾 {video_path}")
            return torch.zeros(self.sample_num, 3, 224, 224)
        
        # 載入影像檔案
        image_files = sorted([f for f in os.listdir(video_path) 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(image_files) == 0:
            print(f"警告: 資料夾中沒有影像檔案 {video_path}")
            return torch.zeros(self.sample_num, 3, 224, 224)
        
        # 處理所有影像
        for img_file in image_files:
            img_path = os.path.join(video_path, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                frames.append(self.preprocess(image))
            except Exception as e:
                print(f"無法載入影像 {img_path}: {e}")
                continue
        
        # 採樣或填充到指定數量
        if len(frames) > self.sample_num:
            indices = np.linspace(0, len(frames) - 1, self.sample_num).astype(int)
            frames = [frames[i] for i in indices]
        
        while len(frames) < self.sample_num:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames[:self.sample_num])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['file_name']
        risk_label = int(row['risk'])
        
        # 載入影片幀
        frames = self.load_and_preprocess_frames(file_name)
        
        # 文字編碼
        text_tokens = clip.tokenize(self.text_prompts)
        
        return {
            'frames': frames,
            'text_tokens': text_tokens,
            'labels': torch.tensor(risk_label, dtype=torch.long),
            'file_name': file_name
        }

class CLIP4ClipRiskModel(torch.nn.Module):
    """CLIP4Clip 風險預測模型"""
    
    def __init__(self, clip_model, num_frames=16, fusion_method='mean'):
        super().__init__()
        self.clip_model = clip_model
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        
        # 凍結CLIP參數
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 時序融合層
        if fusion_method == 'transformer':
            self.temporal_transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
                num_layers=2
            )
        elif fusion_method == 'lstm':
            self.temporal_lstm = torch.nn.LSTM(512, 256, batch_first=True, bidirectional=True)
            self.lstm_fc = torch.nn.Linear(512, 512)
        
        # 分類頭
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2)  # 二元分類
        )
    
    def forward(self, frames, text_tokens=None):
        batch_size = frames.size(0)
        
        # 重塑為 (batch_size * num_frames, C, H, W)
        frames = frames.view(-1, *frames.shape[2:])
        
        # 使用CLIP編碼影像
        with torch.no_grad():
            image_features = self.clip_model.encode_image(frames)
        
        image_features = image_features.float()  # 保證 float32
        
        # 重塑回 (batch_size, num_frames, feature_dim)
        image_features = image_features.view(batch_size, self.num_frames, -1)
        
        # 時序融合
        if self.fusion_method == 'mean':
            video_features = torch.mean(image_features, dim=1)
        elif self.fusion_method == 'max':
            video_features, _ = torch.max(image_features, dim=1)
        elif self.fusion_method == 'transformer':
            video_features = self.temporal_transformer(image_features)
            video_features = torch.mean(video_features, dim=1)
        elif self.fusion_method == 'lstm':
            lstm_out, _ = self.temporal_lstm(image_features)
            video_features = self.lstm_fc(lstm_out[:, -1, :])
        
        # 分類
        logits = self.classifier(video_features)
        
        return logits

class CLIP4ClipRiskPredictor:
    """CLIP4Clip 風險預測器"""
    
    def __init__(self, clip_model_name="ViT-B/32", fusion_method='mean'):
        self.clip_model_name = clip_model_name
        self.fusion_method = fusion_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = None
        self.preprocess = None
        self.model = None
        
    def create_model(self):
        """建立模型"""
        # 載入CLIP
        self.clip_model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        
        # 建立CLIP4Clip模型
        self.model = CLIP4ClipRiskModel(
            self.clip_model, 
            fusion_method=self.fusion_method
        )
        self.model.to(self.device)
        
        return self.model
    
    def create_datasets(self, train_csv, test_csv, train_dir, test_dir):
        """建立訓練和測試資料集"""
        train_dataset = RiskCLIPDataset(
            csv_file=train_csv,
            root_dir=train_dir,
            preprocess=self.preprocess,
            is_train=True
        )
        
        test_dataset = RiskCLIPDataset(
            csv_file=test_csv,
            root_dir=test_dir,
            preprocess=self.preprocess,
            is_train=False
        )
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, test_dataset, batch_size=4):
        """建立資料載入器"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader=None, epochs=20, learning_rate=1e-4, save_path='clip4clip_risk_model.pth'):
        """只做訓練，不做驗證，支援 AMP"""
        if self.model is None:
            raise ValueError("請先建立模型")
        print(f"開始訓練CLIP4Clip，使用設備: {self.device}")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()
        train_losses = []
        best_accuracy = 0.0
        scaler = GradScaler()  # AMP scaler
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            # tqdm 進度條
            batch_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
            for batch_idx, batch in batch_iter:
                try:
                    frames = batch['frames'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    optimizer.zero_grad()
                    with autocast():
                        logits = self.model(frames)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    train_total += labels.size(0)
                    train_correct += (predictions == labels).sum().item()
                    if (batch_idx + 1) % 5 == 0:
                        batch_iter.set_postfix({"Loss": f"{loss.item():.4f}"})
                except Exception as e:
                    print(f"  訓練批次 {batch_idx} 發生錯誤: {e}")
                    continue
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            train_losses.append(avg_train_loss)
            scheduler.step()
            print(f"訓練損失: {avg_train_loss:.4f}")
            print(f"訓練準確率: {train_accuracy:.4f}")
            print(f"學習率: {optimizer.param_groups[0]['lr']:.6f}")
            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'fusion_method': self.fusion_method,
                }, save_path)
                print(f"  ✓ 儲存最佳模型，訓練準確率: {best_accuracy:.4f}")
        print(f"\n🎉 訓練完成！最佳訓練準確率: {best_accuracy:.4f}")
        return train_losses
    
    def evaluate(self, test_loader):
        """評估模型"""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    frames = batch['frames'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits = self.model(frames)
                    predictions = torch.argmax(logits, dim=1)
                    
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"評估批次發生錯誤: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0
        
        # 生成分類報告
        if len(all_predictions) > 0:
            report = classification_report(
                all_labels, all_predictions, 
                target_names=['Safe', 'Risk'],
                output_dict=True
            )
        else:
            report = {}
        
        return accuracy, report
    
    def predict_single_video(self, video_folder, model_path=None):
        """預測單個影片的風險"""
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            raise ValueError("請先載入模型")
        
        self.model.eval()
        
        # 建立臨時資料集
        temp_data = pd.DataFrame({'file_name': [os.path.basename(video_folder)], 'risk': [0]})
        temp_csv = 'temp_clip_predict.csv'
        temp_data.to_csv(temp_csv, index=False)
        
        try:
            dataset = RiskCLIPDataset(
                csv_file=temp_csv,
                root_dir=os.path.dirname(video_folder),
                preprocess=self.preprocess,
                is_train=False
            )
            
            # 獲取資料
            data = dataset[0]
            frames = data['frames'].unsqueeze(0).to(self.device)
            
            # 預測
            with torch.no_grad():
                logits = self.model(frames)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1)
            
            risk_prob = probabilities[0][1].item()
            safe_prob = probabilities[0][0].item()
            is_risky = prediction[0].item()
            
            # 清理臨時檔案
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            
            return {
                'prediction': 'Risk' if is_risky else 'Safe',
                'risk_probability': risk_prob,
                'safe_probability': safe_prob,
                'confidence': max(risk_prob, safe_prob)
            }
            
        except Exception as e:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            raise e
    
    def load_model(self, model_path):
        """載入訓練好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model is None:
            self.create_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"載入CLIP4Clip模型，最佳準確率: {checkpoint.get('best_accuracy', 'N/A')}")

# 使用範例
def run_clip4clip_pipeline():
    """只用 train set 訓練 CLIP4Clip"""
    predictor = CLIP4ClipRiskPredictor(fusion_method='transformer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    train_csv = 'AVA_Dataset/freeway_train.csv'
    train_dir = 'AVA_Dataset/freeway/train'
    train_dataset = RiskCLIPDataset(
        csv_file=train_csv,
        root_dir=train_dir,
        preprocess=preprocess,
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  # 增大 batch size
        shuffle=True,
        num_workers=8,  # 增加 workers
        pin_memory=True
    )
    predictor.create_model()  # 確保模型已建立
    predictor.train(
        train_loader, None, 
        epochs=20, 
        save_path='model/clip4clip_freeway_risk.pth'
    )
    print("\n訓練完成，模型已儲存。")
    return predictor

# 執行CLIP4Clip pipeline
if __name__ == "__main__":
    predictor = run_clip4clip_pipeline()