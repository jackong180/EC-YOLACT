import os
import json

txt_dir = r"C:\Users\16542\Desktop\yolact-pytorch-main\datasets\txt"
json_dir = r"C:\Users\16542\Desktop\yolact-pytorch-main\datasets\before"

os.makedirs(json_dir, exist_ok=True)

for file in os.listdir(txt_dir):
    if not file.endswith(".txt"):
        continue

    txt_path = os.path.join(txt_dir, file)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_path = os.path.join(
        json_dir,
        file.replace(".txt", ".json")
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

print("转换完成！")