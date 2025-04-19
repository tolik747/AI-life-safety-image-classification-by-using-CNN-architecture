from icrawler.builtin import GoogleImageCrawler
import os

# Словник: категорія -> пошукова фраза
categories = {
    "under ": "urban construction site street photo"
}

base_dir = "./life_safety_dataset"
images_per_category = 700

for category, search_term in categories.items():
    category_dir = os.path.join(base_dir, category.replace(" ", "_"))
    os.makedirs(category_dir, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": category_dir})
    crawler.crawl(
        keyword=search_term,
        max_num=images_per_category,

        file_idx_offset=0
    )
