# 🧠 AI Life Safety Image Classification (Paper Replication)

This project is a **replication** of the IEEE research paper  
[**"AI Life Safety Image Classification by Using CNN Architecture" (2022)**](https://ieeexplore.ieee.org/document/9954724).

The goal is to implement the described CNN model from scratch using **PyTorch**,  
train it on a **custom dataset**, and **compare the results** to those in the original paper.

---

## 📌 Project Objective

To classify images showing **public safety hazards**, such as:
- Broken stairs
- Damaged sidewalks
- Icy roads
- Traffic accidents
- Construction zones
- Sinkholes
- Broken roads

---

## 🧰 Technologies

- `Python 3.10+`
- `PyTorch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `Pillow`

---

## 📊 Results Comparison

| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|----------------|
| Original (Paper)   | ~73%           | ~74%           |
| This Replication   | **86.9%**      | **99.5%**      |

---

## 📁 Project Structure

```text
life_safety_cnn/
├── dataset/
│   ├── train/
│   │   ├── broken_roads/
│   │   ├── traffic_accident/
│   │   └── ... (total 7 categories)
│   └── test/
│       ├── broken_roads/
│       ├── traffic_accident/
│       └── ...
├── model.py               # CNN architecture
├── plots/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
├── life_safety_model.pt   # Saved trained model
README.md              # Project documentation
```

  
---

## 💡 Notes

- The original dataset was unavailable, so a **custom dataset** was built using web scraping and image augmentation.
- The model architecture closely follows the original: 5× Conv2D layers with Dropout and MaxPooling.
- Loss, accuracy, and confusion matrix are visualized for performance evaluation.

---

## 📜 License

MIT — free to use and modify for educational or research purposes.
