import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

# --- 設定繪圖與警告 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows 適用 (Mac 請改 'Arial Unicode MS')
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# ==========================================
# 1. 資料讀取函式
# ==========================================
def prepare_mrt_data(path):
    files = sorted(glob.glob(os.path.join(path, "*.xlsx")))
    if not files:
        print(f"警告：在 {path} 找不到任何 .xlsx 檔案")
        return pd.DataFrame()

    df_list = []
    # 簡單判斷年份以決定每月天數 (僅做參考，實際以 clean_and_impute 處理)
    is_leap = '2024' in path or '2028' in path
    days_in_month = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for i, file in enumerate(files):
        if i >= len(days_in_month): break
        try:
            # 讀取 Excel (假設格式固定: 前4行Header, 取前5欄)
            df = pd.read_excel(file, header=4, nrows=days_in_month[i], usecols=[0, 1, 2, 3, 4])
            df.columns = ['Date', 'Day_of_Week', 'Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
            # 移除空日期行
            df = df.dropna(subset=['Date'])
            df_list.append(df)
        except Exception as e:
            print(f"讀取 {os.path.basename(file)} 失敗: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

# ==========================================
# 2. 資料清洗 (含颱風處理與缺失填補)
# ==========================================
def clean_and_impute(df):
    if df.empty: return df
    
    # 處理日期格式與索引
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # 防呆: 遇到"總計"變NaN，預防可能該月有某天忘記填寫日期，以至於多跑到總計那邊
    df = df.dropna(subset=['Date']) # 移除無效日期行
    df.set_index('Date', inplace=True) # 將日期設為索引
    
    # 設定頻率，移除重複
    df = df[~df.index.duplicated(keep='first')] 
    try:
        df = df.asfreq('D') #缺值補NaN
    except:
        pass

    cols = ['Red_Line_Count', 'Orange_Line_Count', 'Total_Count']
    
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # 防呆: 非數值轉NaN
        
        # 2. 補值邏輯：前後 7 天平均
        avg_neighbors = (df[col].shift(7) + df[col].shift(-7)) / 2
        df[col] = df[col].fillna(avg_neighbors)
        
        # 3. 連鎖填補 (處理連續缺失)
        for _ in range(3):
            df[col] = df[col].fillna(df[col].shift(7))
            df[col] = df[col].fillna(df[col].shift(-7))
            
        df[col] = df[col].fillna(0) # 最後防線

    df['Day_of_Week'] = df.index.dayofweek # 利用日期，將他轉為星期幾 (0=週一, 6=週日)
    return df

# ==========================================
# 3. 特徵工程 (加入演唱會人數 + 假日)
# ==========================================
def concert_features_enhanced(df):
    df['Concert_People'] = 0  # 預設為 0
    
    # --- A. 演唱會資料庫 (範例數據，請依實際情況擴充)，為連續數值 ---
    concert_map = {
        # === 一月 ===
        '2024-01-01': 11000, # 羅志祥(巨蛋1萬) + 夢時代跨年人流餘波/連假效應(估4萬)
        '2024-01-19': 13000, # OneRepublic (巨蛋)
        '2024-01-27': 18000, # Super Junior(巨蛋 1.3萬) + 理想混蛋(高流 0.5萬)
        '2024-01-28': 13000, # Super Junior(巨蛋 1.3萬)

        # === 二月 ===
        '2024-02-03': 56000, # Ed Sheeran (世運 - 大爆滿)
        '2024-02-04': 2000,  # VIXX (Live Warehouse - 小型場地，非世運)
        '2024-02-24': 10000, # 詹雅雯 (巨蛋)

        # === 三月 (人流高峰) ===
        '2024-03-22': 12000, # 櫻花祭 (夢時代)
        '2024-03-23': 45000+24000, # 五月天 (世運) + 櫻花祭(夢時代)
        '2024-03-24': 45000+24000, # 五月天 (世運) + 櫻花祭(夢時代)
        '2024-03-29': 45000, # 五月天 (世運)
        '2024-03-30': 75000, # ⚠️大魔王日：五月天(世運4.5萬) + 大港開唱(駁二3萬)
        '2024-03-31': 75000, # ⚠️大魔王日：五月天(世運4.5萬) + 大港開唱(駁二3萬)

        # === 四月 ===
        '2024-04-06': 5000,  # CNBLUE (高流)
        '2024-04-13': 40000, # Golden Wave (世運 - 拼盤演唱會)

        # === 五月 ===
        '2024-05-04': 4000,  # 玟星 (高流)
        '2024-05-11': 10000, # 韋禮安 (巨蛋)

        # === 六月 ===
        '2024-06-08': 5000,  # 麋先生 (高流)

        # === 七月 ===
        '2024-07-05': 25000, # 啤酒音樂節 (夢時代)
        '2024-07-06': 28000, # 啤酒音樂節 (夢時代)
        '2024-07-07': 25000, # 啤酒音樂節 (夢時代)
        '2024-07-13': 12000, # K-MEGA (巨蛋)
        '2024-07-19': 5000,  # 理想混蛋 (高流)

        # === 八月 ===
        '2024-08-10': 15000, # 宇宙人(巨蛋 1萬) + 怕胖團(高流 0.5萬)
        '2024-08-31': 5000,  # ECO LIVE (高流)

        # === 九月 (人流高峰) ===
        '2024-09-07': 63000, # ⚠️大魔王日：Bruno Mars(世運5萬) + Energy(巨蛋1.3萬)
        '2024-09-08': 63000, # ⚠️大魔王日：Bruno Mars(世運5萬) + Energy(巨蛋1.3萬)
        '2024-09-14': 10000, # 蔡健雅 (巨蛋)
        '2024-09-21': 45000, # ONE OK ROCK (世運)
        '2024-09-28': 10000, # 鄭中基 (巨蛋)
        '2024-09-29': 10000, # 鄭中基 (巨蛋)

        # === 十月 ===
        '2024-10-05': 11000, # 徐佳瑩 (巨蛋)

        # === 十一月 ===
        '2024-11-02': 45000, # Stray Kids (世運)
        '2024-11-16': 5000,  # Take That (高流)
        '2024-11-17': 10000, # LISA Fan MeetUp (巨蛋)
        '2024-11-23': 10000, # 鄭伊健 (巨蛋)
        '2024-11-30': 6000, # SCOOL (巨蛋)
        
        # === 十二月 ===
        '2024-12-01': 6000, # SCOOL (巨蛋)
        '2024-12-05': 13000, # Charlie Puth (巨蛋)
        '2024-12-28': 11000, # 羅志祥 (巨蛋)
        '2024-12-31': 11000,# 跨年夜：羅志祥(巨蛋) + 夢時代大跨年 (這天通常是捷運運量全年最高)

        # === 2025 (請務必填寫你已知的場次，模型才能預測未來!) ===
        # === 一月 ===
        '2025-01-25': 11000, # Super Junior-D&E (巨蛋)
        '2025-01-26': 11000, # Super Junior-D&E (巨蛋)

        # === 二月 ===
        '2025-02-14': 50000, # Maroon 5 (世運)
        '2025-02-15': 10000, # 民歌50 (巨蛋)

        # === 三月 ===
        '2025-03-01': 10000, # 麋先生 (巨蛋)
        '2025-03-15': 11000, # Kylie Minogue (巨蛋)
        '2025-03-28': 11000+12000, # 張學友 (巨蛋) + 櫻花祭(夢時代)
        '2025-03-29': 11000+24000+30000, # 張學友 (巨蛋) + 櫻花祭(夢時代) + 大港(駁二)
        '2025-03-30': 11000+24000+30000, # 張學友 (巨蛋) + 櫻花祭(夢時代) + 大港(駁二)

        # === 四月 ===
        '2025-04-19': 10000, # KKBOX(巨蛋)

        # === 五月 ===
        '2025-05-23': 11000, # 陳奕迅 (巨蛋)
        '2025-05-24': 11000, # 陳奕迅 (巨蛋)
        '2025-05-25': 11000, # 陳奕迅 (巨蛋)
        '2025-05-28': 5000, # Lauv(高流)
        '2025-05-29': 11000, # 陳奕迅 (巨蛋)
        '2025-05-30': 11000, # 陳奕迅 (巨蛋)
        '2025-05-31': 11000, # 陳奕迅 (巨蛋)

        # === 七月 ===
        '2025-07-04': 25000, # 啤酒音樂節(夢時代)
        '2025-07-05': 27000, # 啤酒音樂節(夢時代)
        '2025-07-11': 12000, # 江蕙 (巨蛋)
        '2025-07-12': 12000, # 江蕙 (巨蛋)
        '2025-07-15': 12000, # 江蕙 (巨蛋)
        '2025-07-18': 12000, # 江蕙 (巨蛋)
        '2025-07-19': 12000, # 江蕙 (巨蛋)
        '2025-07-22': 12000, # 江蕙 (巨蛋)
        '2025-07-25': 12000, # 江蕙 (巨蛋)
        '2025-07-26': 12000, # 江蕙 (巨蛋)

        # === 八月 ===
        '2025-08-02': 10000, # FNC BAND KINGDOM (巨蛋)
        '2025-08-03': 10000, # FNC BAND KINGDOM (巨蛋)
        '2025-08-09': 12000, # 蘇打綠 (巨蛋)
        '2025-08-10': 12000, # 蘇打綠 (巨蛋)
        '2025-08-16': 11000, # 孫燕姿 (巨蛋)
        '2025-08-17': 11000, # 孫燕姿 (巨蛋)
        '2025-08-23': 10000, # 蕭秉治 (巨蛋)
        '2025-08-30': 11000, # 八三夭 (巨蛋)

        # === 九月 ===
        '2025-09-06': 11000, # Energy (巨蛋)
        '2025-09-07': 11000, # Energy (巨蛋)
        '2025-09-13': 9000,  # 蔡琴 (巨蛋)

        # === 十月 ===
        '2025-10-18': 50000, # BLACKPINK (世運 - 預估滿場)
        '2025-10-19': 50000, # BLACKPINK (世運 - 預估滿場)
    }
    
    for date_str, people in concert_map.items():
        date = pd.to_datetime(date_str)
        if date in df.index:
            df.loc[date, 'Concert_People'] = people
    
    return df

def holiday_features_enhanced(df):
    df['Is_Holiday'] = 0      # 預設為 0

    # --- B. 國定假日 (包含連假)，為類別數值(0：無，1：有) ---
    holidays = [
        # 2024
        '2024-01-01', '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', 
        '2024-02-13', '2024-02-14', '2024-02-28', '2024-04-04', '2024-04-05', 
        '2024-06-10', '2024-09-17', '2024-10-10',
        # 2025
        '2025-01-01', '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28', 
        '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-28', '2025-04-03', 
        '2025-04-04', '2025-05-30', '2025-09-29', '2025-10-06', '2025-10-10', '2025-10-24',
    ]
    
    holiday_dt = pd.to_datetime(holidays)
    mask = df.index.isin(holiday_dt)
    df.loc[mask, 'Is_Holiday'] = 1

    return df

def typhoon_features_enhanced(df):
    df['Is_Typhoon'] = 0

    typhoon_dates = [
        # 2024
        '2024-07-24', '2024-07-25', '2024-07-26', # 凱米
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', # 山陀兒
        # 2025
        '2025-07-06', '2025-07-07', # 丹娜絲
        '2025-07-29', # 西南氣流
        '2025-08-13', # 楊柳
        '2025-11-12', # 鳳凰
    ]

    typhoon_dt = pd.to_datetime(typhoon_dates)
    mask = df.index.isin(typhoon_dt)
    df.loc[mask, 'Is_Typhoon'] = 1

    return df

# ==========================================
# 4. 自動尋找最佳參數函式 (Grid Search)
# ==========================================
def find_best_sarimax_params(y, exog, p_list, d_list, q_list, P_list, D_list, Q_list, s=7):
    # 產生所有參數組合
    pdq = list(itertools.product(p_list, d_list, q_list))
    seasonal_pdq = list(itertools.product(P_list, D_list, Q_list, [s]))
    
    best_aic = float("inf") # 初始化為無限大，AIC分數越低，表現越好
    best_order = None
    best_seasonal = None
    
    total_comb = len(pdq) * len(seasonal_pdq)
    print(f"開始網格搜索... (共 {total_comb} 種組合)")
    
    counter = 0
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            counter += 1
            try:
                mod = SARIMAX(y,    # endogenous，即主要分析資料，就是捷運流量
                              exog=exog, #exogenous即外生變數(演唱會、國定假日、颱風天)
                              order=param, # 傳入(p, d, q)給order參數，模型以當前參數組合訓練
                              seasonal_order=param_seasonal, # 傳入(P, D, Q, s)給seasonal_order參數
                              enforce_stationarity=False, # 關閉平穩性強制，讓模型強制算出一個AIC值，儘管當前組合很爛
                              enforce_invertibility=False) # 關閉可逆性強制，MA(q)可能會算超過邊界，但仍強制通過
                results = mod.fit(disp=False) # disp=False 關閉收斂訊息輸出
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
                    best_seasonal = param_seasonal
                
                # 每 10 次顯示一次進度
                if counter % 10 == 0:
                     print(f"進度 {counter}/{total_comb} | 目前最佳 AIC: {best_aic:.2f}")
            except:
                continue
                
    print(f"\n🎉 最佳參數組合找到: Order={best_order}, Seasonal={best_seasonal}, AIC={best_aic:.2f}")
    return best_order, best_seasonal

# ==========================================
# 5. 主程式執行
# ==========================================

# --- 1. 讀取與處理 ---
print("--- 步驟 1: 讀取與清洗資料 ---")
path_2024 = './dataset/高雄捷運113運量統計表' 
path_2025 = './dataset/高雄捷運114年運量統計表'

df_2024 = clean_and_impute(prepare_mrt_data(path_2024))
df_2025 = clean_and_impute(prepare_mrt_data(path_2025))

if df_2024.empty or df_2025.empty:
    print("錯誤：資料讀取失敗，請檢查路徑。")
    exit()

print("--- 步驟 2: 加入特徵 (活動+假日) ---")
df_2024 = concert_features_enhanced(df_2024)
df_2024 = holiday_features_enhanced(df_2024)
df_2024 = typhoon_features_enhanced(df_2024)
df_2025 = concert_features_enhanced(df_2025)
df_2025 = holiday_features_enhanced(df_2025)
df_2025 = typhoon_features_enhanced(df_2025)

# 準備訓練數據 (取 Log)
train_y_log = np.log(df_2024['Total_Count'])
train_exog = df_2024[['Concert_People', 'Is_Holiday', 'Is_Typhoon']]

# 準備測試數據 (全資料集，用於滾動預測)
full_df = pd.concat([df_2024, df_2025]) # 合併24、25年資料，因為要完整時間序列(如2024/12/31有演唱會，會影響2025/1/1的預測)
full_y_log = np.log(full_df['Total_Count']) 
full_exog = full_df[['Concert_People', 'Is_Holiday', 'Is_Typhoon']]

# --- 2. 尋找最佳參數 ---
print("--- 步驟 3: 自動尋找最佳參數 (Grid Search) ---")
# 設定搜索範圍 (d=0 因為已取Log趨勢平穩, s=7 週循環)
p_range = [1, 2]
d_range = [0, 1]
q_range = [0, 1]
P_range = [0, 1]
D_range = [1]
Q_range = [0, 1]

best_order, best_seasonal = find_best_sarimax_params(
    train_y_log, 
    train_exog,
    p_range, d_range, q_range, 
    P_range, D_range, Q_range, s=7
)

# --- 3. 訓練最終模型 ---
print(f"\n--- 步驟 4: 使用最佳參數 {best_order} x {best_seasonal} 訓練模型 ---")
model_train = SARIMAX(train_y_log, 
                      exog=train_exog,
                      order=best_order, 
                      seasonal_order=best_seasonal,
                      enforce_stationarity=False, 
                      enforce_invertibility=False)
results_train = model_train.fit(disp=False)
print(f"訓練完成。AIC: {results_train.aic:.2f}")

# --- 4. 滾動式預測 (Rolling Forecast) ---
print("\n--- 步驟 5: 執行 2025 滾動式預測 ---")

# 建立全模型架構
model_full = SARIMAX(full_y_log, 
                     exog=full_exog,
                     order=best_order, 
                     seasonal_order=best_seasonal,
                     enforce_stationarity=False, 
                     enforce_invertibility=False)

# 將 2024 訓練好的參數注入全模型
results_full = model_full.filter(results_train.params)

# 開始預測 2025 (使用 dynamic=False，即每次預測都基於前一天的真實數據)
split_date = df_2025.index[0]
pred_obj = results_full.get_prediction(start=split_date, dynamic=False) #start=split_date 指定從2025年1月1日開始預測

# 取出預測值 (Log) 並還原 (Exp)
pred_mean = np.exp(pred_obj.predicted_mean)
pred_ci = np.exp(pred_obj.conf_int()) # 信賴區間

# 計算 RMSE
rmse_total = sqrt(mean_squared_error(df_2025['Total_Count'], pred_mean)) # RMSE：代表預測誤差大小(猜錯多少人)
print(f"★ 2025 預測 RMSE: {rmse_total:.0f}")

# --- 5. 繪圖 ---
plt.figure(figsize=(15, 8))

# 畫歷史 (2024 Q4)
plt.plot(df_2024.index[-90:], df_2024['Total_Count'][-90:], label='歷史數據 (2024 Q4)', color='gray', alpha=0.5)

# 畫真實 2025
plt.plot(df_2025.index, df_2025['Total_Count'], label='真實運量 (2025)', color='blue', linewidth=1.5)

# 畫預測 2025
plt.plot(pred_mean.index, pred_mean, label='預測運量 (SARIMAX)', color='red', linestyle='--', linewidth=2)

# 標記演唱會
concert_dates = df_2025[df_2025['Concert_People'] >= 5000].index
for date in concert_dates:
    people = df_2025.loc[date, 'Concert_People']
    ymax_val = 0.3 if people > 40000 else 0.15
    plt.axvline(x=date, color='orange', linestyle=':', alpha=0.8, ymax=ymax_val)

# 畫信賴區間
plt.fill_between(pred_mean.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.2)

plt.title(f'高雄捷運運量預測 (自動網格搜索+Log優化)\n最佳參數: {best_order} x {best_seasonal} | RMSE: {rmse_total:.0f}', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()