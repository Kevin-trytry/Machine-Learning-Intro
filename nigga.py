import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

# 忽略惱人的警告訊息
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
            # 讀取 Excel (假設格式固定)
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
    
    # --- A. 定義 2024 高雄颱風假日期 (非常重要！必須視為缺失值) ---
    # 凱米(7/24-26), 山陀兒(10/1-3) -> 這些天運量極低，不能讓模型學到
    typhoon_dates = [
        '2024-07-24', '2024-07-25', '2024-07-26', 
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04'
    ]
    typhoon_dt = pd.to_datetime(typhoon_dates)

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. 把颱風天強制設為 NaN (讓後面的補值邏輯去修補它，假裝那天正常營運)
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
# 3. 特徵工程 (演唱會權重 + 假日)
# ==========================================
def add_features_enhanced(df):
    # 初始化
    df['Concert_Weight'] = 0  # 0=無, 1=小, 2=中, 3=超大
    df['Is_Holiday'] = 0      # 0=平日, 1=假日/連假
    
    # --- A. 演唱會資料庫 (請務必根據真實情況擴充) ---
    concert_map = {
        # 2024 (部分範例)
        '2024-01-27': 2, # Super Junior (巨蛋)
        '2024-02-03': 3, # Ed Sheeran (世運)
        '2024-03-30': 3, '2024-03-31': 3, # 五月天 (世運)
        '2024-04-13': 3, # Golden Wave (世運)
        '2024-09-07': 3, '2024-09-08': 3, # Bruno Mars (世運)
        '2024-11-02': 3, '2024-11-03': 3, # Stray Kids (世運)
        
        # 2025 (未來預測用 - 請填入你知道的活動)
        '2025-01-01': 3, '2025-01-04': 3, '2025-01-05': 3, # 五月天 (世運)
        '2025-02-14': 2, # (假設) 情人節巨蛋演唱會
        '2025-04-05': 3, # (假設) 阿妹世運
    }
    
    for date_str, weight in concert_map.items():
        date = pd.to_datetime(date_str)
        if date in df.index:
            df.loc[date, 'Concert_Weight'] = weight

    # --- B. 國定假日 (Holiday) ---
    # 高捷假日人多，必須標記連假
    holidays = [
        # 2024
        '2024-01-01', 
        '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', # 春節
        '2024-04-04', '2024-04-05', # 清明
        '2024-06-10', # 端午
        '2024-09-17', # 中秋
        '2024-10-10', # 國慶
        
        # 2025
        '2025-01-01',
        '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', # 春節
        '2025-02-28',
        '2025-04-03', '2025-04-04',
        '2025-05-30', # 端午
        '2025-10-06', # 中秋
        '2025-10-10',
    ]
    
    holiday_dt = pd.to_datetime(holidays)
    mask = df.index.isin(holiday_dt)
    df.loc[mask, 'Is_Holiday'] = 1 # 國定假日設為 1
    
    return df

# ==========================================
# 4. Grid Search (含 Exog)
# ==========================================
def run_grid_search_exog(train_y, test_y, train_exog, test_exog, 
                         p_list, d_list, q_list, 
                         P_list, D_list, Q_list, s_list):
    
    total_combinations = len(p_list) * len(d_list) * len(q_list) * len(P_list) * len(D_list) * len(Q_list)
    print(f"啟動 Grid Search... 預計測試 {total_combinations} 種組合")
    
    results = []
    current_iter = 0
    forecast_steps = len(test_y)

    for p in p_list:
        for d in d_list:
            for q in q_list:
                for P in P_list:
                    for D in D_list:
                        for Q in Q_list:
                            for s in s_list:
                                current_iter += 1
                                param = (p, d, q)
                                seasonal_param = (P, D, Q, s)
                                
                                try:
                                    model = SARIMAX(train_y, 
                                                    exog=train_exog,
                                                    order=param, 
                                                    seasonal_order=seasonal_param,
                                                    enforce_stationarity=False, 
                                                    enforce_invertibility=False)
                                    
                                    model_fit = model.fit(disp=False)
                                    
                                    # 預測需要未來的 exog
                                    pred_res = model_fit.get_forecast(steps=forecast_steps, exog=test_exog)
                                    pred = pred_res.predicted_mean
                                    
                                    error = sqrt(mean_squared_error(test_y, pred))
                                    
                                    results.append({
                                        'p': p, 'd': d, 'q': q,
                                        'P': P, 'D': D, 'Q': Q, 's': s,
                                        'RMSE': error
                                    })
                                    
                                    if current_iter % 5 == 0:
                                        print(f"[{current_iter}/{total_combinations}] RMSE: {error:.0f} | {param}x{seasonal_param}")

                                except:
                                    continue
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by='RMSE')
        return result_df
    return pd.DataFrame()

# ... (前面 1~4 部分的函式都不用動) ...

# ==========================================
# 5. 主程式執行 (滾動預測版 - Rolling Forecast)
# ==========================================

# 1. 讀取與清洗
print("1. 讀取與清洗資料 (自動修補颱風天)...")
# 請確保你的路徑設定正確
path_2024 = './dataset/高雄捷運113運量統計表' 
path_2025 = './dataset/高雄捷運114年運量統計表'

df_2024 = clean_and_impute(prepare_mrt_data(path_2024))
df_2025 = clean_and_impute(prepare_mrt_data(path_2025))

if df_2024.empty or df_2025.empty:
    print("錯誤：資料讀取失敗。")
    exit()

# 2. 加入特徵
print("2. 加入特徵 (演唱會權重 + 假日效應)...")
df_2024 = add_features_enhanced(df_2024)
df_2025 = add_features_enhanced(df_2025)

# 3. 準備數據 (Log 轉換)
# 合併兩年的數據，這是滾動預測的關鍵
full_df = pd.concat([df_2024, df_2025])
full_y_log = np.log(full_df['Total_Count'])
full_exog = full_df[['Concert_Weight', 'Is_Holiday']]

# 切分索引位置
split_date = df_2025.index[0] # 2025 第一天
train_end_idx = len(df_2024)

# 4. 訓練模型 (只用 2024 練參數)
print("\n3. 訓練模型 (只用 2024 數據學習參數)...")

# 這裡先用 Grid Search 找出的最佳參數
# 如果你懶得跑 Grid Search，直接用這個經驗參數 (1, 0, 1) x (1, 1, 0, 7)
best_order = (1, 0, 1) 
best_seasonal = (1, 1, 0, 7)

# 建立模型 (只餵 2024)
model_train = SARIMAX(np.log(df_2024['Total_Count']), 
                      exog=df_2024[['Concert_Weight', 'Is_Holiday']],
                      order=best_order, 
                      seasonal_order=best_seasonal,
                      enforce_stationarity=False, 
                      enforce_invertibility=False)
results_train = model_train.fit(disp=False)

print(f"模型訓練完成。AIC: {results_train.aic:.2f}")

# 5. 滾動預測 (Rolling Forecast / One-Step-Ahead)
print("\n4. 執行滾動式預測 (模擬每日更新)...")

# 技巧：使用 apply 方法
# 我們把「訓練好的參數」套用到「包含 2025 的完整數據」上
# 這樣模型就會在預測 2025/1/2 時，自動「看見」2025/1/1 的真實數據
model_full = SARIMAX(full_y_log, 
                     exog=full_exog,
                     order=best_order, 
                     seasonal_order=best_seasonal,
                     enforce_stationarity=False, 
                     enforce_invertibility=False)

# 將 2024 練好的參數 (results_train.params) 套用到全數據模型
# 這樣就不用重新訓練，速度超快
results_full = model_full.filter(results_train.params)

# 取得 2025 年部分的預測值
# get_prediction(start=...) 會自動做 "One-step-ahead" 預測
pred_obj = results_full.get_prediction(start=split_date, dynamic=False)
pred_mean_log = pred_obj.predicted_mean
pred_ci_log = pred_obj.conf_int()

# 6. 轉回數值 (Exp)
pred_mean = np.exp(pred_mean_log)
pred_ci = np.exp(pred_ci_log)

# 計算誤差
rmse_total = sqrt(mean_squared_error(df_2025['Total_Count'], pred_mean))
print(f"2025 滾動預測 RMSE: {rmse_total:.0f}")

# 7. 繪圖
plt.figure(figsize=(15, 8))

# 畫 2024 (歷史)
plt.plot(df_2024.index[-90:], df_2024['Total_Count'][-90:], label='歷史數據 (2024 Q4)', color='gray', alpha=0.5)

# 畫 2025 (真實)
plt.plot(df_2025.index, df_2025['Total_Count'], label='真實運量 (2025)', color='blue', linewidth=1.5)

# 畫 2025 (預測 - 滾動版)
plt.plot(pred_mean.index, pred_mean, label='預測運量 (滾動式更新)', color='red', linestyle='--', linewidth=2)

# 標記演唱會
concert_dates = df_2025[df_2025['Concert_Weight'] >= 2].index
for date in concert_dates:
    plt.axvline(x=date, color='orange', linestyle=':', alpha=0.8, ymax=0.2)

# 畫信賴區間
plt.fill_between(pred_mean.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.2)

plt.title(f'高雄捷運滾動式預測 (One-Step-Ahead Forecast)\nRMSE: {rmse_total:.0f} (紅線現在會緊貼藍線)', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
