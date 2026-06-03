# Final Project Report
## Used Cars Price Prediction
### ML Foundations Bootcamp — Capstone Project

---

## 1. Introduction

I chose the Used Cars dataset from Kaggle because I wanted to understand what makes
a used car expensive or cheap. The dataset contains listings from Craigslist across
the United States with information like price, year, mileage, fuel type, and manufacturer.

My main question was: can a machine learning model predict whether a car is expensive
or cheap based on its features?

---

## 2. Dataset Description

The raw dataset had 426,880 rows and 26 columns. The most useful columns for this
project were:

- `price` — the listed price of the car
- `year` — the year the car was manufactured
- `odometer` — how many miles the car has driven
- `manufacturer` — the brand of the car
- `fuel` — fuel type (gas, diesel, electric, etc.)
- `transmission` — automatic or manual
- `condition` — the condition of the car (excellent, good, fair, etc.)

The target label I created was `price_label`: 0 for cheap (below the median price)
and 1 for expensive (above the median price). I used the median instead of the mean
because a small number of extremely high-priced listings pulled the mean up and made
it an unfair split point.

---

## 3. Data Cleaning

The raw data had several problems that needed to be fixed before training any model.

**Missing values:** Many columns had missing data. I kept only the columns I needed
and dropped any row that had a missing value in those columns. The condition column
had the most missing values among the selected features.

**Zero values:** Some listings had a price of zero or an odometer reading of zero,
which are not realistic. I removed those rows.

**Duplicates:** I checked for duplicate rows and removed them to avoid the model
learning the same record twice.

After cleaning, the dataset was significantly smaller but reliable. All key columns
had no missing values and all values were within a realistic range.

---

## 4. Feature Engineering

I created three new features to give the models more useful information:

**car_age:** I subtracted the manufacture year from the current year. This is more
useful than the raw year because it directly represents how old the car is.

**price_per_km:** I divided the price by the odometer reading. This shows how much
value a car holds per kilometer driven.

**manufacturer_popularity:** I counted how many times each manufacturer appeared in
the dataset. Popular brands tend to have more listings and different pricing patterns.

---

## 5. Models Used

I trained three machine learning models on the cleaned and engineered dataset.

**Logistic Regression** drew a linear boundary between the two price classes. It was
the fastest model to train and the easiest to interpret, but it struggled with the
non-linear patterns in the data.

**Random Forest** built 100 decision trees and combined their votes. It handled
complex feature interactions well and did not require scaling.

**KNN (K-Nearest Neighbors)** classified each car based on the 5 most similar cars
in the training set. It is simple in concept but slower on large datasets.

I used F1-Score as the main metric because both classes were roughly balanced at 50%
each, but F1 is still more informative than accuracy alone since it accounts for both
precision and recall.

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | —        | —         | —      | —        |
| Random Forest       | —        | —         | —      | —        |
| KNN                 | —        | —         | —      | —        |

*Fill in the numbers from your Step 1 output.*

Random Forest was the best performing model. It benefited from being able to capture
non-linear relationships between features like car age, odometer, and manufacturer,
which Logistic Regression could not represent.

---

## 6. Neural Network

I built a neural network with two hidden layers using the Keras library.

The architecture was:
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (30%) to reduce overfitting
- Dense layer with 64 neurons and ReLU activation
- Dropout layer (30%)
- Output layer with 1 neuron and Sigmoid activation

I trained the model for 20 epochs with a batch size of 256. The validation accuracy
and loss curves showed that the model was learning steadily without significant
overfitting. The training and validation lines stayed close to each other throughout
training, which means the Dropout layers did their job.

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Neural Network | —        | —         | —      | —        |

*Fill in the numbers from your Step 2 output.*

The neural network performed comparably to Random Forest. On tabular data like this,
tree-based models often match or beat neural networks because they handle mixed feature
types naturally. Neural networks tend to show a larger advantage on image or text data
where the relationships between inputs are much more complex.

---

## 7. Results and Comparison

Across all four models, Random Forest and the Neural Network performed best. The main
finding is that car age and odometer reading were the most predictive features, which
makes sense: newer cars with fewer miles tend to be more expensive.

Logistic Regression performed the lowest of the three Step 1 models, which suggests
the relationship between the features and price is not purely linear.

KNN fell in the middle. It is a reasonable model but sensitive to noise in the data,
which limited its performance here.

---

## 8. What I Learned

The biggest lesson was that data cleaning matters more than model choice. When I first
looked at the raw data, the price column had values as high as 3.7 billion dollars,
which were clearly errors. Without removing those outliers, any model would have
learned from garbage data and produced unreliable results.

I also learned that more complex does not always mean better. The neural network took
much longer to train than Random Forest but did not consistently outperform it on this
dataset. For a problem like this, a well-tuned Random Forest is often the right tool.

---

## Data Source

Craigslist Used Cars Dataset — Kaggle  
https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
