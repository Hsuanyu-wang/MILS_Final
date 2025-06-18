import pandas as pd

# 讀取 road 與 freeway 預測結果
road_df = pd.read_csv('road_submission.csv')
freeway_df = pd.read_csv('freeway_submission.csv')

# 合併
submission = pd.concat([road_df, freeway_df], ignore_index=True)

# 依 sample_submission.csv 的順序排序（如果需要）
sample = pd.read_csv('AVA_Dataset/sample_submission.csv')
submission = submission.set_index('file_name').reindex(sample['file_name']).reset_index()

# 輸出
submission.to_csv('final_submission.csv', index=False)
print('已產生 final_submission.csv')