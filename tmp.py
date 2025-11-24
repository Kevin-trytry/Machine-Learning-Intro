import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

# 忽略警告訊息 (讓輸出乾淨一點)
warnings.filterwarnings("ignore")

# ==========================================
# 1. 資料讀取與清洗 (沿用你的邏輯)
# ==========================================
def prepare_mrt_data(path):
    files = sorted(glob.glob(os.path.join(path, "*.xlsx")))
    if not files:
        print(f"警告：在 {path} 找不到任何 .xlsx 檔案")
        return pd.DataFrame()

    df_list = []
    # 自動判斷年份以設定每月天數 (簡單防呆)
    year_str = path.split('mrt')[0][-4:] # 嘗試從路徑抓年份，例如 './2024mrt' -> '2024'
    is_leap = False
    if '2024' in path or '2028' in path: 
        is_leap = True
    
    days_in_month = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for i, file in enumerate(files):
        try:
            # 確保不會讀超過 months 陣列長度
            if i >= len(days_in_month): break
            
            # print(f"Reading: {os.path.basename(file)}")
            df = pd.read_excel(file, header=4, nrows=days_in_month[i], usecols=[0, 1, 2, 3, 4])
            df.columns = ['Date', 'Day_of_Week', 'Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
            df_list.append(df)
        except Exception as e:
            print(f"讀取 {file} 失敗: {e}")

    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df
    else:
        return pd.DataFrame()

def clean_num(df):
    if df.empty: return df
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 嘗試設定頻率
    df = df[~df.index.duplicated(keep='first')]
    try:
        df = df.asfreq('D')
    except:
        pass

    cols_to_fix = ['Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 補值邏輯：前後7天平均
        avg_neighbors = (df[col].shift(7) + df[col].shift(-7)) / 2
        df[col] = df[col].fillna(avg_neighbors)
        
        # 連鎖填補
        for _ in range(3):
            df[col] = df[col].fillna(df[col].shift(7))
            df[col] = df[col].fillna(df[col].shift(-7))
            
        df[col] = df[col].fillna(0) # 最後防線

    df['Day_of_Week'] = df.index.dayofweek
    return df

# ==========================================
# 2. 關鍵功能：加入演唱會特徵 (Feature Engineering)
# ==========================================
def add_concert_feature(df):
    """
    在這裡建立 'Is_Concert' 欄位。
    """
    df['Is_Concert'] = 0 # 初始化，預設為 0
    
    # ==============================================
    # ⚠️【請在這裡填入真實的演唱會日期】⚠️
    # 格式：'YYYY-MM-DD'
    # ==============================================
    concert_dates_list = [
        # --- 2024 範例資料 (請替換成真的) ---
        '2024-01-01', '2024-01-19', '2024-01-27', 
        '2024-02-03', 
        '2024-03-09', '2024-03-30',
        '2024-07-05',  
        '2024-08-10',
        '2024-09-07','2024-09-08','2024-09-14','2024-09-21','2024-09-28', '2024-09-29',
        '2024-10-05', 
        '2024-11-03', '2024-11-16', '2024-11-23',
        '2024-12-05',


        # --- 2025 範例資料 (請替換成真的) ---
        '2025-01-01', '2025-01-04', '2025-01-05', # 五月天 (範例)
        '2025-02-14', # 假設的情人節演唱會
        # ... 繼續填寫你們抓到的日期 ...
    ]
    
    # 轉換日期格式並標記
    concert_dates = pd.to_datetime(concert_dates_list)
    
    # 標記邏輯：如果是演唱會當天，設為 1
    # 進階技巧：如果你知道當天人數，可以填入預估人數 (例如 45000) 取代 1，效果會更好
    mask = df.index.isin(concert_dates)
    df.loc[mask, 'Is_Concert'] = 1 
    
    return df

# ==========================================
# 3. 主程式
# ==========================================

#設定路徑 (請依據你的資料夾名稱修改)
path_2024 = './dataset/高雄捷運113運量統計表' 
path_2025 = './dataset/高雄捷運114年運量統計表'

print("1. 正在讀取並清洗資料...")
print(f"【DEBUG檢查】path_2025 的值是: [{path_2025}]")
print(f"【DEBUG檢查】資料夾真的存在嗎?: {os.path.exists(path_2025)}")
# 讀取
raw_2024 = prepare_mrt_data(path_2024)
raw_2025 = prepare_mrt_data(path_2025)

# 清洗
df_2024 = clean_num(raw_2024)
df_2025 = clean_num(raw_2025)

if df_2024.empty or df_2025.empty:
    print("錯誤：資料讀取失敗，請檢查路徑或檔案。")
    exit()

print("2. 正在加入演唱會特徵 (Feature Engineering)...")
# 加入演唱會特徵
df_2024 = add_concert_feature(df_2024)
df_2025 = add_concert_feature(df_2025)

# === 設定訓練集與測試集 ===
# 訓練集：2024 全年
train_y = df_2024['Total_Count']
train_exog = df_2024[['Is_Concert']] # 外部變數 (X)

# 測試集：2025 (用來驗證)
test_y = df_2025['Total_Count']
test_exog = df_2025[['Is_Concert']]  # 外部變數 (X) - 這裡代表「未來的行程表」

print(f"訓練集長度 (2024): {len(train_y)} 天")
print(f"測試集長度 (2025): {len(test_y)} 天")

# ==========================================
# 4. SARIMAX 模型訓練 (含 Exog)
# ==========================================
print("\n3. 開始訓練 SARIMAX 模型 (含演唱會參數)...")

# 設定參數 (這裡填入你們 Grid Search 跑出來的最佳參數)
# 如果還沒跑過，可以先用這個常見組合試試：
#my_order = (2, 1, 1)          # (p, d, q)
#my_seasonal_order = (1, 1, 0, 7) # (P, D, Q, s) - s=7 很重要

# 修改這兩行
my_order = (1, 0, 1)          # 把中間的 d 改成 0 (假設平穩)
# 將 D 從 0 改為 1
my_seasonal_order = (1, 1, 1, 7)

model = SARIMAX(train_y, 
                exog=train_exog,        # <--- 關鍵：加入演唱會特徵
                order=my_order, 
                seasonal_order=my_seasonal_order,
                enforce_stationarity=False, 
                enforce_invertibility=False)

results = model.fit(disp=False)
print("模型訓練完成！")
print(results.summary().tables[1]) # 印出係數表，看 'Is_Concert' 的 P>|z| 是否小於 0.05

# ==========================================
# 5. 預測 2025 並評估
# ==========================================
print("\n4. 正在預測 2025 年流量...")

# 預測需要提供 steps (幾天) 和 exog (那幾天的演唱會狀況)
forecast_steps = len(test_y)
pred_obj = results.get_forecast(steps=forecast_steps, exog=test_exog)
pred_mean = pred_obj.predicted_mean
pred_ci = pred_obj.conf_int()

# --- 計算誤差 (RMSE) ---
rmse_total = sqrt(mean_squared_error(test_y, pred_mean))
print(f"\n=== 評估結果 ===")
print(f"2025 整體 RMSE: {rmse_total:.2f}")

# --- 進階評估：只看演唱會日子的誤差 ---
concert_indices = test_exog[test_exog['Is_Concert'] == 1].index
if not concert_indices.empty:
    # 找出那些日子的真實值與預測值
    y_true_concert = test_y[test_y.index.isin(concert_indices)]
    y_pred_concert = pred_mean[pred_mean.index.isin(concert_indices)]
    
    rmse_concert = sqrt(mean_squared_error(y_true_concert, y_pred_concert))
    print(f"2025 演唱會日 RMSE: {rmse_concert:.2f}")
    
    # 印出這幾天的細節給你看
    print("\n--- 演唱會日預測詳情 (前5筆) ---")
    detail_df = pd.DataFrame({'Actual': y_true_concert, 'Predicted': y_pred_concert})
    detail_df['Diff'] = detail_df['Actual'] - detail_df['Predicted']
    print(detail_df.head())
else:
    print("2025 測試資料中沒有標記任何演唱會，無法計算特定誤差。")

# ==========================================
# 6. 畫圖 (Visualization)
# ==========================================
plt.figure(figsize=(15, 7))

# 畫 2024 最後一季 (當作背景參考)
plt.plot(train_y.index[-90:], train_y[-90:], label='History (2024 Q4)', color='gray', alpha=0.4)

# 畫 2025 真實數據
plt.plot(test_y.index, test_y, label='Actual 2025', color='blue', linewidth=1.5)

# 畫 2025 預測數據
plt.plot(pred_mean.index, pred_mean, label='Predicted 2025 (SARIMAX+Exog)', color='red', linestyle='--', linewidth=2)

# 標記演唱會日期 (畫虛線)
if not concert_indices.empty:
    for date in concert_indices:
        plt.axvline(x=date, color='orange', linestyle=':', alpha=0.6, ymax=0.1) # 底部畫個小標記

# 畫信賴區間
plt.fill_between(pred_mean.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.2)

plt.title(f'Kaohsiung MRT Flow Prediction: 2024 Train -> 2025 Test\nRMSE: {rmse_total:.0f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()