import os
import time

# 使用绝对路径
folder_path = r"C:\Users\zwq\Desktop\GT-PPO\GT-PPO\permissibleLS.py"

# 确保目录存在
if not os.path.exists(folder_path):
    print(f"错误: 目录不存在 -> {folder_path}")
else:
    # 设定新的修改时间（Unix 时间戳，秒）
    new_time = time.mktime((2024, 11, 3, 12, 0, 0, 0, 0, 0))

    # 修改所有文件的时间
    for root, _, files in os.walk(folder_path):
        os.utime(root, (new_time, new_time))  # 也修改子文件夹的时间
        for file in files:
            file_path = os.path.join(root, file)
            os.utime(file_path, (new_time, new_time))
            print(f"修改完成: {file_path}")

    print("所有文件和文件夹修改完成！")
