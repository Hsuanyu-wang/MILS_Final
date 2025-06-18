# MILS_Final

本專案為多模態學習系統（MILS）的最終專案，包含資料集處理、模型訓練、推論與結果分析等功能。

## 目錄結構

- `AVA_Dataset/`：包含 freeway 與 road 兩種資料集及其訓練、測試資料夾與標註檔案
- `model/`：儲存訓練好的模型權重
- `logs/`：訓練與測試過程的日誌檔案
- `results/`：推論與評估結果
- `CLIP4CLIP_train.py`、`CLIP4CLIP_inference.py`：主要訓練與推論腳本
- 其他工具腳本與說明文件

## 安裝方式

1. 建議使用 Python 3.8 以上版本。
2. 安裝必要套件：
   ```bash
   pip install -r requirements.txt
   ```

## 使用說明

### 訓練模型
```bash
python CLIP4CLIP_train.py --config <config_path>
```

### 推論
```bash
python CLIP4CLIP_inference.py --config <config_path>
```

### 產生提交檔案
請參考 `TOOL_combine_csv.py`。

## 聯絡資訊

如有問題請聯絡：
- Hsuanyu Wang ([GitHub](https://github.com/Hsuanyu-wang)) 