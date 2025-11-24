import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt

def prepare_mrt_data(path):
    files = sorted(glob.glob(os.path.join(path, "*.xlsx")))

    if not files:
            print(f"警告：在 {path} 找不到任何 .xlsx 檔案")
            return pd.DataFrame() # 回傳空表
    
    df_list = []
    
    # 為了避免手動設定 nrows 出錯，建議用 append 的方式
    # 這裡沿用你的 nrows 邏輯，但建議加上 dropna 以防萬一
    # 假設 files 順序正確：Jan, Feb, ... Dec
    
    # 這裡定義每個月的天數 (2024是閏年)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if path[2:6] == '2024':
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    elif path[2:6] == '2025':
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31]
    
    for i, file in enumerate(files):        
        print(f"Reading: {os.path.basename(file)}")
        df = pd.read_excel(file, header=4, nrows=days_in_month[i], usecols=[0, 1, 2, 3, 4])
        df.columns = ['Date', 'Day_of_Week', 'Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
        df_list.append(df)

    # 合併
    full_df = pd.concat(df_list, ignore_index=True)
    
    return full_df

def clean_num(df):
    # A. 將 Date 轉為 datetime 物件
    df['Date'] = pd.to_datetime(df['Date'])
    
    # B. 將 Date 設為 Index
    df.set_index('Date', inplace=True)
    
    # C. 設定頻率
    try:
        df.index.freq = 'D'
    except:
        # 嘗試移除重複索引後再設定
        df = df[~df.index.duplicated(keep='first')]
        try:
            df = df.asfreq('D')
        except:
            pass # 如果真的無法設定，就不勉強，避免報錯
    
    # D. [核心修改] 依照你的邏輯處理數值與空缺
    cols_to_fix = ['Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
    
    for col in cols_to_fix:
        # 1. 強制轉為數字，無法轉的變成 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. 第一優先：若有缺失，用「7天前」與「7天後」的平均值填入
        # 邏輯：(X_{t-7} + X_{t+7}) / 2
        avg_neighbors = (df[col].shift(7) + df[col].shift(-7)) / 2
        df[col] = df[col].fillna(avg_neighbors)
        
        # 3. 第二優先：連鎖填補 (處理連續缺失或邊界狀況)
        # 我們跑一個小迴圈(例如5次)，讓數據可以「傳遞」
        # 例如：1/1 缺 -> 找 1/8；如果 1/8 也缺 -> 找 1/15... 
        # 當 1/15 補好了，下一次迴圈 1/8 就會補好，再下一次 1/1 就會補好。
        
        for _ in range(5): # 假設連續缺失不超過 5 週，通常夠用了
            # 情況 A: 找「7天前」的資料 (例如 1/15 沒資料，用 1/8 填)
            df[col] = df[col].fillna(df[col].shift(7))
            
            # 情況 B: 找「7天後」的資料 (例如 1/1 沒資料，用 1/8 填；或 1/22 沒資料用 1/29)
            df[col] = df[col].fillna(df[col].shift(-7))
            
        # 4. 最後防線 (萬一整整兩個月都沒資料，避免程式當機)
        # 雖然依你的邏輯應該都補完了，但為了安全起見，剩下的填 0
        df[col] = df[col].fillna(0)

    # E. 處理星期特徵
    df['Day_of_Week'] = df.index.dayofweek
    df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

    return df

def initial_plots(time_series, num_lag):
    plt.figure(figsize=(12, 8)) # 把圖變大一點比較好看
    
    plt.subplot(3, 1, 1)
    plt.plot(time_series)
    plt.title('Original data across time')
    
    plt.subplot(3, 1, 2)
    plot_acf(time_series, lags=num_lag, ax=plt.gca())
    plt.title('Autocorrelation plot')
    
    plt.subplot(3, 1, 3)
    plot_pacf(time_series, lags=num_lag, ax=plt.gca())
    plt.title('Partial autocorrelation plot')
    plt.tight_layout()
    plt.show()

#Defining RMSE
def rmse(x,y):
    return sqrt(mean_squared_error(x,y))

def run_grid_search(trainingSeries, testingSeries, p_list, d_list, q_list, P_list, D_list, Q_list, s_list):
    
    # 1. 切分訓練集與測試集 (最後 30 天當作期末考驗證)
    train = trainingSeries
    test = testingSeries

    forecastLength = len(test)
    
    results = []
    total_combinations = len(p_list) * len(d_list) * len(q_list) * len(P_list) * len(D_list) * len(Q_list)
    current_iter = 0
    
    print(f"開始訓練... 總共有 {total_combinations} 種組合需要測試")

    # 2. 暴力迴圈測試所有組合
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
                                    # 建立並訓練模型 (先不加 exog，跑純時間序列)
                                    # enforce_stationarity=False 允許模型處理稍微不平穩的數據
                                    model = SARIMAX(train,
                                                    order=param,
                                                    seasonal_order=seasonal_param,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                                    
                                    results_model = model.fit(disp=False)
                                    
                                    # 預測測試集 (Out-of-sample forecast)
                                    pred = results_model.forecast(steps=forecastLength)
                                    
                                    # 計算分數
                                    rmse_score = rmse(test, pred)
                                    
                                    # 存檔
                                    results.append({
                                        'p': p, 'd': d, 'q': q,
                                        'P': P, 'D': D, 'Q': Q, 's': s,
                                        'AIC': results_model.aic,
                                        'RMSE': rmse_score
                                    })
                                    
                                    # 每 5 次印一次進度，讓你確認程式還在跑
                                    if current_iter % 5 == 0:
                                        print(f"[{current_iter}/{total_combinations}] RMSE: {rmse_score:.2f} | Order: {param} x {seasonal_param}")

                                except Exception as e:
                                    # 某些參數組合可能會導致數學運算錯誤 (如矩陣無法反轉)，直接跳過
                                    print(f"參數失敗 {param} x {seasonal_param}: {e}")
                                    continue
                                    
    # 3. 整理結果並排序
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by='RMSE') # 讓誤差最小的排前面
        
    return result_df

# main
trainingPath = './2024mrt'
testingPath = './2025mrt'

trainingRawData = prepare_mrt_data(trainingPath)
traingData = clean_num(trainingRawData)

testingRawData = prepare_mrt_data(testingPath)
testingData = clean_num(testingRawData)

trainingSeries = traingData['Total_Count']
testingSeries = testingData['Total_Count']

initial_plots(trainingSeries, 45)


# --- 設定參數範圍 (根據之前的圖表分析) ---

# 非季節性參數 (Trend)
p_list = [1, 2]    # PACF Lag 1 很高，試試看 AR(1) 或 AR(2)
d_list = [1]       # 確定要做一階差分來消除短期趨勢
q_list = [0, 1]    # ACF Lag 1 很高，試試看 MA(1)

# 季節性參數 (Seasonality)
P_list = [0, 1]    # 季節性 AR
D_list = [1]       # ⚠️ 關鍵：一定要做季節性差分 (消除每週循環)
Q_list = [0, 1]    # 季節性 MA
s_list = [7]       # ⚠️ 關鍵：週期為 7 天

# --- 執行搜索 (針對總運量 totalData) ---
# 注意：這可能需要跑 1~2 分鐘
best_models = run_grid_search(trainingSeries,testingSeries, 
                              p_list, d_list, q_list, 
                              P_list, D_list, Q_list, s_list)

# --- 查看冠軍模型 ---
print("\n=== 最佳模型排行榜 (Top 5) ===")
print(best_models.head())

# --- 4. 畫出最終結果 ---
# 取得最佳參數
top_model = best_models.iloc[0]
best_order = (int(top_model['p']), int(top_model['d']), int(top_model['q']))
best_seasonal = (int(top_model['P']), int(top_model['D']), int(top_model['Q']), int(top_model['s']))

# 重新訓練
final_model = SARIMAX(trainingSeries, 
                      order=best_order, 
                      seasonal_order=best_seasonal,
                      enforce_stationarity=False, 
                      enforce_invertibility=False)
results = final_model.fit(disp=False)

# 預測 2025
pred_steps = len(trainingSeries)
pred = results.get_forecast(steps=pred_steps)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# 繪圖
plt.figure(figsize=(15, 6))

# 畫 2024 的最後一段 (作為歷史參考)
plt.plot(trainingSeries.index[-60:], trainingSeries[-60:], label='Train (2024)', color='gray', alpha=0.5)

# 畫 2025 的真實數據
plt.plot(testingSeries.index, testingSeries, label='Test (2025 Real)', color='blue')

# 畫 2025 的預測數據
plt.plot(pred_mean.index, pred_mean, label='Prediction (2025)', color='red', linestyle='--')

plt.fill_between(pred_mean.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title(f'2024 Train -> 2025 Predict (RMSE: {top_model["RMSE"]:.0f})')
plt.legend()
plt.show()

'''
# --- 1. 設定冠軍參數 (從你的排行榜抄下來) ---
best_order = (2, 1, 1)
best_seasonal_order = (1, 1, 0, 7)

# --- 2. 重新訓練模型 (只用冠軍參數) ---
train = totalData[:-30]
test = totalData[-30:]

print("正在使用最佳參數重新訓練模型...")
model = SARIMAX(train, 
                order=best_order, 
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False, 
                enforce_invertibility=False)

results = model.fit(disp=False)

# --- 3. 預測最後 30 天 ---
pred = results.get_forecast(steps=30)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int() # 取得信賴區間 (陰影部分)

# --- 4. 畫圖 (這是報告要用的圖) ---
plt.figure(figsize=(15, 6))

# 畫出真實數據 (只畫最後 90 天比較看得清楚，不用畫整年)
plt.plot(totalData.index[-90:], totalData[-90:], label='Actual (History)', color='gray', alpha=0.5)
plt.plot(test.index, test, label='Actual (Ground Truth)', color='blue', linewidth=2)

# 畫出預測數據
plt.plot(pred_mean.index, pred_mean, label='Prediction (SARIMAX)', color='red', linestyle='--', linewidth=2)

# 畫出信賴區間 (陰影)
plt.fill_between(pred_mean.index, 
                 pred_ci.iloc[:, 0], 
                 pred_ci.iloc[:, 1], color='pink', alpha=0.3)

plt.title(f'SARIMAX Prediction vs Actual (RMSE: {23836.856827:.0f})', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()'''