import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

# --- 設定繪圖與警告 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 讓圖表顯示中文 (Windows)
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 資料讀取
# ==========================================
def prepare_mrt_data(path):
    files = sorted(glob.glob(os.path.join(path, "*.xlsx")))
    if not files:
        print(f"警告：在 {path} 找不到任何 .xlsx 檔案")
        return pd.DataFrame()

    df_list = []
    # 簡單判斷年份
    is_leap = '2024' in path or '2028' in path
    days_in_month = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for i, file in enumerate(files):
        if i >= len(days_in_month): break
        try:
            # 讀取 Excel (假設格式固定: 前4行Header, 取前5欄)
            df = pd.read_excel(file, header=4, nrows=days_in_month[i], usecols=[0, 1, 2, 3, 4])
            df.columns = ['Date', 'Day_of_Week', 'Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
            df_list.append(df)
        except Exception as e:
            print(f"讀取 {os.path.basename(file)} 失敗: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

# ==========================================
# 2. 資料清洗 (含颱風處理)
# ==========================================
def clean_and_impute(df):
    if df.empty: return df
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 設定頻率，移除重複
    df = df[~df.index.duplicated(keep='first')]
    try:
        df = df.asfreq('D')
    except:
        pass

    cols = ['Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
    
    # --- 關鍵：定義 2024 高雄颱風假日期 ---
    # 凱米(7/24-26), 山陀兒(10/1-4) -> 這些天運量極低，必須視為缺失值
    typhoon_dates = [
        '2024-07-24', '2024-07-25', '2024-07-26', 
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04'
    ]
    typhoon_dt = pd.to_datetime(typhoon_dates)

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. 把颱風天強制設為 NaN
        df.loc[df.index.isin(typhoon_dt), col] = np.nan
        
        # 2. 補值邏輯：前後 7 天平均 (修補颱風和一般缺失)
        avg_neighbors = (df[col].shift(7) + df[col].shift(-7)) / 2
        df[col] = df[col].fillna(avg_neighbors)
        
        # 3. 連鎖填補
        for _ in range(3):
            df[col] = df[col].fillna(df[col].shift(7)) # 用上週補
            df[col] = df[col].fillna(df[col].shift(-7)) # 用下週補
            
        df[col] = df[col].fillna(0) # 最後防線

    df['Day_of_Week'] = df.index.dayofweek
    return df

# ==========================================
# 3. 特徵工程 (實際人數 + 假日)
# ==========================================
def add_features_enhanced(df):
    # 初始化
    df['Concert_People'] = 0  # 實際人數 (Int)
    df['Is_Holiday'] = 0      # 0=平日, 1=假日/連假
    
    # --- A. 演唱會/活動 人數估算表 (請持續更新) ---
    concert_map = {
        # 2024 (歷史)
        '2024-01-27': 13000, # Super Junior
        '2024-01-28': 13000, 
        '2024-02-03': 50000, # Ed Sheeran
        '2024-03-23': 50000, # 五月天
        '2024-03-24': 50000,
        '2024-03-29': 50000,
        '2024-03-30': 50000,
        '2024-03-31': 50000,
        '2024-04-13': 40000, # Golden Wave
        '2024-09-07': 55000, # Bruno Mars
        '2024-09-08': 55000,
        '2024-09-21': 40000, # ONE OK ROCK
        '2024-11-02': 45000, # Stray Kids
        '2024-11-03': 45000,
        
        # 2025 (未來預測 - 務必填寫你知道的大型活動)
        '2025-01-01': 50000, # 五月天
        '2025-01-04': 50000,
        '2025-01-05': 50000,
        '2025-02-14': 50000, # Maroon 5 (世運)
        '2025-02-15': 50000, 
        '2025-04-01': 45000, # (範例) 阿妹演唱會?
    }
    
    for date_str, people in concert_map.items():
        date = pd.to_datetime(date_str)
        if date in df.index:
            df.loc[date, 'Concert_People'] = people

    # --- B. 國定假日 ---
    holidays = [
        # 2024
        '2024-01-01', 
        '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14',
        '2024-02-28',
        '2024-04-04', '2024-04-05',
        '2024-06-10', '2024-09-17', '2024-10-10',
        # 2025
        '2025-01-01',
        '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31',
        '2025-02-28',
        '2025-04-03', '2025-04-04',
        '2025-05-30', '2025-10-06', '2025-10-10',
    ]
    
    holiday_dt = pd.to_datetime(holidays)
    mask = df.index.isin(holiday_dt)
    df.loc[mask, 'Is_Holiday'] = 1 
    
    return df

# ==========================================
# 4. 主程式執行
# ==========================================

print("1. 讀取與清洗資料 (自動修補颱風天)...")
# 請確認這裡的資料夾路徑正確
path_2024 = './dataset/高雄捷運113運量統計表' 
path_2025 = './dataset/高雄捷運114年運量統計表'

df_2024 = clean_and_impute(prepare_mrt_data(path_2024))
df_2025 = clean_and_impute(prepare_mrt_data(path_2025))

if df_2024.empty or df_2025.empty:
    print("錯誤：資料讀取失敗，請檢查路徑或檔案。")
    exit()

print("2. 加入特徵 (實際人數 + 假日效應)...")
df_2024 = add_features_enhanced(df_2024)
df_2025 = add_features_enhanced(df_2025)

# --- 準備滾動預測需要的全資料 ---
full_df = pd.concat([df_2024, df_2025])

# ⚠️ 關鍵：取 Log 來穩定模型
full_y_log = np.log(full_df['Total_Count'])
full_exog = full_df[['Concert_People', 'Is_Holiday']]

# 訓練集 (只取 2024)
train_y_log = np.log(df_2024['Total_Count'])
train_exog = df_2024[['Concert_People', 'Is_Holiday']]

# --- 訓練模型 ---
print("\n3. 訓練模型 (僅使用 2024 數據學習參數)...")

# 經驗證實對 Log 數據最穩定的參數組合
# p=1 (考慮昨天), d=0 (因為 Log 後不用強烈差分), q=1 (考慮誤差)
# s=7 (週週期)
best_order = (1, 0, 1)
best_seasonal = (1, 1, 0, 7)

model_train = SARIMAX(train_y_log, 
                      exog=train_exog,
                      order=best_order, 
                      seasonal_order=best_seasonal,
                      enforce_stationarity=False, 
                      enforce_invertibility=False)
results_train = model_train.fit(disp=False)
print(f"模型訓練完成。AIC: {results_train.aic:.2f}")

# --- 滾動式預測 (Rolling Forecast) ---
print("\n4. 執行滾動式預測 (模擬每日更新數據)...")

# 建立一個包含 2024+2025 的全空殼模型
model_full = SARIMAX(full_y_log, 
                     exog=full_exog,
                     order=best_order, 
                     seasonal_order=best_seasonal,
                     enforce_stationarity=False, 
                     enforce_invertibility=False)

# 將 2024 練好的「智慧」(參數) 灌入全模型
results_full = model_full.filter(results_train.params)

# 開始預測 2025
# dynamic=False 代表：預測 1/2 時，使用 1/1 的「真實數據」，而不是使用 1/1 的「預測數據」
# 這就是讓紅線不會失憶的關鍵
split_date = df_2025.index[0]
pred_obj = results_full.get_prediction(start=split_date, dynamic=False)

# 取出預測值 (Log 狀態)
pred_mean_log = pred_obj.predicted_mean
pred_ci_log = pred_obj.conf_int()

# 轉回正常數值 (Exp)
pred_mean = np.exp(pred_mean_log)
pred_ci = np.exp(pred_ci_log)

# 計算 RMSE
rmse_total = sqrt(mean_squared_error(df_2025['Total_Count'], pred_mean))
print(f"2025 滾動預測 RMSE: {rmse_total:.0f}")

# ==========================================
# 5. 繪圖
# ==========================================
plt.figure(figsize=(15, 8))

# 畫歷史 (2024 Q4)
plt.plot(df_2024.index[-90:], df_2024['Total_Count'][-90:], label='歷史數據 (2024 Q4)', color='gray', alpha=0.5)

# 畫真實 2025
plt.plot(df_2025.index, df_2025['Total_Count'], label='真實運量 (2025)', color='blue', linewidth=1.5)

# 畫預測 2025
plt.plot(pred_mean.index, pred_mean, label='預測運量 (SARIMAX Rolling)', color='red', linestyle='--', linewidth=2)

# 標記有填入人數的日子 (人數 > 5000 才標，避免太亂)
concert_dates = df_2025[df_2025['Concert_People'] >= 5000].index
for date in concert_dates:
    people = df_2025.loc[date, 'Concert_People']
    # 畫橘色虛線，高度稍微隨人數變化
    ymax_val = 0.3 if people > 40000 else 0.15
    plt.axvline(x=date, color='orange', linestyle=':', alpha=0.8, ymax=ymax_val)

# 畫信賴區間
plt.fill_between(pred_mean.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.2)

plt.title(f'高雄捷運滾動預測 (實際人數+Log優化+颱風清洗)\nRMSE: {rmse_total:.0f}', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
