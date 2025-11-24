import os
import glob

# 測試你的路徑
test_path = './dataset/高雄捷運114年運量統計表'

print(f"目前工作目錄: {os.getcwd()}")
print(f"嘗試讀取路徑: {test_path}")

# 檢查資料夾是否存在
if os.path.exists(test_path):
    print("✅ 資料夾存在！")
    # 檢查裡面有沒有 xlsx
    files = glob.glob(os.path.join(test_path, "*.xlsx"))
    print(f"📁 裡面有 {len(files)} 個 .xlsx 檔案")
else:
    print("❌ 找不到資料夾，請檢查路徑或資料夾名稱")