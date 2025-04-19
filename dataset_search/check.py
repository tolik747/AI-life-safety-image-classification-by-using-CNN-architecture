import os

# Папка з усіма категоріями
base_dir = "./dataset/train"

# Мінімум бажаних зображень
min_required = 100

# Перевірка кожної папки
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if os.path.isdir(category_path):
        num_images = len([
            file for file in os.listdir(category_path)
            if file.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        status = "✅ Достатньо" if num_images >= min_required else "⚠️ Мало"
        print(f"{category}: {num_images} зображень → {status}")
