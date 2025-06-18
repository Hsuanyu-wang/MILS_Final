# ============================================================================
# VideoMAE 風險預測完整實作
# ============================================================================

import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 安裝必要套件
# !pip install transformers torch torchvision opencv-python pandas numpy pillow scikit-learn matplotlib

class RiskVideoDataset(Dataset):
    """VideoMAE 風險預測資料集"""
    
    def __init__(self, csv_file, root_dir, image_processor, sample_num=16, is_train=True):
        """
        Args:
            csv_file: CSV檔案路徑 (格式: file_name,risk)
            root_dir: 影像資料夾路徑
            image_processor: VideoMAE的影像處理器
            sample_num: 每個影片採樣的幀數
            is_train: 是否為訓練模式
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.sample_num = sample_num
        self.is_train = is_train
        
        print(f"載入資料集: {len(self.data)} 個樣本")
        print(f"Risk分布: {self.data['risk'].value_counts().to_dict()}")
    
    def load_video_frames(self, video_folder):
        """載入影片幀序列"""
        frames = []
        video_path = os.path.join(self.root_dir, video_folder)
        
        if not os.path.exists(video_path):
            print(f"警告: 找不到資料夾 {video_path}")
            return [Image.new('RGB', (224, 224), color='black')] * self.sample_num
        
        # 載入資料夾中的所有影像
        image_files = sorted([f for f in os.listdir(video_path) 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(image_files) == 0:
            print(f"警告: 資料夾中沒有影像檔案 {video_path}")
            return [Image.new('RGB', (224, 224), color='black')] * self.sample_num
        
        # 載入所有影像
        for img_file in image_files:
            img_path = os.path.join(video_path, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                frames.append(image)
            except Exception as e:
                print(f"無法載入影像 {img_path}: {e}")
                continue
        
        # 如果影像數量超過sample_num，進行均勻採樣
        if len(frames) > self.sample_num:
            indices = np.linspace(0, len(frames) - 1, self.sample_num).astype(int)
            frames = [frames[i] for i in indices]
        
        # 如果影像數量不足，重複最後一幀
        while len(frames) < self.sample_num:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new('RGB', (224, 224), color='black'))
        
        return frames[:self.sample_num]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['file_name']  # 例如: freeway_0000
        risk_label = int(row['risk'])  # 0=safe, 1=risk
        
        try:
            # 載入影片幀
            frames = self.load_video_frames(file_name)
            
            # 使用VideoMAE的影像處理器
            inputs = self.image_processor(frames, return_tensors="pt")
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(risk_label, dtype=torch.long),
                'file_name': file_name
            }
        except Exception as e:
            print(f"處理 {file_name} 時發生錯誤: {e}")
            # 返回空白資料
            blank_frames = [Image.new('RGB', (224, 224), color='black')] * self.sample_num
            inputs = self.image_processor(blank_frames, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(risk_label, dtype=torch.long),
                'file_name': file_name
            }

class VideoMAERiskPredictor:
    """VideoMAE 風險預測器"""
    
    def __init__(self, model_name="MCG-NJU/videomae-base-finetuned-kinetics"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = None
        self.model = None
        
    def create_model(self):
        """建立模型"""
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # 二元分類：0=safe, 1=risk
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        return self.model, self.image_processor
    
    def create_datasets(self, train_csv, test_csv, train_dir, test_dir):
        """建立訓練和測試資料集"""
        if self.image_processor is None:
            self.create_model()
            
        train_dataset = RiskVideoDataset(
            csv_file=train_csv,
            root_dir=train_dir,
            image_processor=self.image_processor,
            is_train=True
        )
        
        test_dataset = RiskVideoDataset(
            csv_file=test_csv,
            root_dir=test_dir,
            image_processor=self.image_processor,
            is_train=False
        )
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, test_dataset, batch_size=2):
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
    
    def train(self, train_loader, test_loader, epochs=15, learning_rate=5e-5, save_path='videomae_risk_model.pth'):
        """訓練模型"""
        if self.model is None:
            raise ValueError("請先建立模型")
            
        print(f"開始訓練，使用設備: {self.device}")
        
        # 優化器和調度器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # 記錄訓練過程
        train_losses = []
        val_accuracies = []
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 前向傳播
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                    # 反向傳播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 計算訓練準確率
                    predictions = torch.argmax(outputs.logits, dim=1)
                    train_total += labels.size(0)
                    train_correct += (predictions == labels).sum().item()
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"  訓練批次 {batch_idx} 發生錯誤: {e}")
                    continue
            
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            train_losses.append(avg_train_loss)
            
            # 驗證階段
            val_accuracy, val_report = self.evaluate(test_loader)
            val_accuracies.append(val_accuracy)
            
            # 學習率調度
            scheduler.step()
            
            print(f"訓練損失: {avg_train_loss:.4f}")
            print(f"訓練準確率: {train_accuracy:.4f}")
            print(f"驗證準確率: {val_accuracy:.4f}")
            print(f"學習率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 儲存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'image_processor': self.image_processor,
                }, save_path)
                print(f"  ✓ 儲存最佳模型，準確率: {best_accuracy:.4f}")
        
        print(f"\n🎉 訓練完成！最佳驗證準確率: {best_accuracy:.4f}")
        return train_losses, val_accuracies
    
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
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(pixel_values=pixel_values)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    
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
        
        if self.model is None or self.image_processor is None:
            raise ValueError("請先載入模型")
        
        self.model.eval()
        
        # 建立臨時資料集
        temp_data = pd.DataFrame({'file_name': [os.path.basename(video_folder)], 'risk': [0]})
        temp_csv = 'temp_predict.csv'
        temp_data.to_csv(temp_csv, index=False)
        
        try:
            dataset = RiskVideoDataset(
                csv_file=temp_csv,
                root_dir=os.path.dirname(video_folder),
                image_processor=self.image_processor,
                is_train=False
            )
            
            # 獲取資料
            data = dataset[0]
            pixel_values = data['pixel_values'].unsqueeze(0).to(self.device)
            
            # 預測
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(outputs.logits, dim=1)
            
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
        print(f"載入模型，最佳準確率: {checkpoint.get('best_accuracy', 'N/A')}")

# 使用範例
def run_videomae_pipeline():
    """執行VideoMAE完整pipeline"""
    
    # 初始化預測器
    predictor = VideoMAERiskPredictor()
    
    # 建立模型
    model, image_processor = predictor.create_model()
    
    # 設定資料路徑（請根據您的實際路徑修改）
    train_csv = 'freeway_train.csv'  # 您的訓練CSV
    test_csv = 'freeway_test.csv'    # 您的測試CSV  
    train_dir = 'freeway/train'      # 訓練影像資料夾
    test_dir = 'freeway/test'        # 測試影像資料夾
    
    # 建立資料集
    train_dataset, test_dataset = predictor.create_datasets(
        train_csv, test_csv, train_dir, test_dir
    )
    
    # 建立資料載入器
    train_loader, test_loader = predictor.create_dataloaders(
        train_dataset, test_dataset, batch_size=2
    )
    
    # 訓練模型
    train_losses, val_accuracies = predictor.train(
        train_loader, test_loader, 
        epochs=15, 
        save_path='videomae_freeway_risk.pth'
    )
    
    # 最終評估
    final_accuracy, final_report = predictor.evaluate(test_loader)
    print(f"\n最終測試準確率: {final_accuracy:.4f}")
    
    # 預測單個影片範例
    # result = predictor.predict_single_video('path/to/video/folder')
    # print(f"預測結果: {result}")
    
    return predictor, train_losses, val_accuracies

# 執行VideoMAE pipeline
if __name__ == "__main__":
    predictor, losses, accuracies = run_videomae_pipeline()
