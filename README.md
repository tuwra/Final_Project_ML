# Used Cars Price Prediction — ML Capstone Project

**Student:** Raeed Saud Alotaibi  
**Program:** ML, Deep Learning & NLP Applications — Tuwaiq Academy  
**Dataset:** [Used Cars — Craigslist (Kaggle)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

---

## Project Overview

This project builds a complete machine learning pipeline on a used cars dataset from Craigslist.
The goal is to predict whether a car is **cheap or expensive** based on features like year, mileage, manufacturer, fuel type, transmission, and condition.

The price column was converted into a binary label using the median as a threshold:
- `0` = cheap (price below median)
- `1` = expensive (price above median)

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/tuwra/Final_Project_ML.git
cd Final_Project_ML
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place `vehicles.csv` in `data/raw/`

4. Run the notebooks in order
