# CCFraud

CCFraud is a python project that uses a Linear Regression model from Scikit to identify credit card fraud.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Scikit, if you wish to run this code locally.

```bash
python -m pip install --upgrade pip
python -m pip install -U scikit-learn
```

## Usage
We first took a data set from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) that contains transactions made by credit cards in September 2013 by European cardholders. Our goal is to create a ML Model that is able to predict if a certain Credit Card transaction is a fraud or not. 

The first step we can take is to divide our credit_card_data into legit and fraud.

```python
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

```
The number of legitimate transactions is 284315, whilst the number of fraudulent transactions is 492. Therefore, the legit transactions should be sampled at n=492.


```python
legit_sample = legit.sample(n=492)

```

We then split the data into Features & Targets.
```python
X = new_dataset.drop(columns='Class', axis = 1)
Y = new_dataset['Class']
```

We further split the data using scikit's "train_test_split" function.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state =2)
```

The Logistic Regression can now be fitted and we achieve a 0.95 accuracy on Training data and a 0.92 accuracy on Test data.
```python
model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

```