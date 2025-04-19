import os
import random
from PIL import Image, ImageEnhance, ImageOps

# Шляхи
source_dir = "./dataset/test"   # тут твої оригінальні зображення (по 100)
target_dir = "./dataset/train"  # сюди підуть аугментовані (до 1000)

# Скільки хочемо отримати у підсумку на категорію
target_count = 1000

# Аугментація: повороти, контраст, дзеркало, масштаб
def augment_image(img):
    aug_img = img.copy()

    if random.random() < 0.5:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        aug_img = aug_img.rotate(angle)

    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.8, 1.2))

    return aug_img

# По кожній категорії
for category in os.listdir(source_dir):
    src_path = os.path.join(source_dir, category)
    tgt_path = os.path.join(target_dir, category)
    os.makedirs(tgt_path, exist_ok=True)

    images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    original_count = len(images)
    saved_count = 0
    i = 0

    while saved_count < target_count:
        img_name = images[i % original_count]
        img_path = os.path.join(src_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            if saved_count < original_count:
                # Копіюємо оригінал
                img.save(os.path.join(tgt_path, f"original_{saved_count}.jpg"))
            else:
                # Аугментована копія
                aug = augment_image(img)
                aug.save(os.path.join(tgt_path, f"aug_{saved_count}.jpg"))
            saved_count += 1
        except Exception as e:
            print(f"⚠️ Пропущено {img_name}: {e}")

        i += 1

print("✅ Готово! У train/ по 1000 зображень у кожній категорії.")
