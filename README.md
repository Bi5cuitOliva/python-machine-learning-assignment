**Python Machine Learning Assignment**  
Student Name: Dennis Njoroge 1988  
Project Title: Supervised and Unsupervised Machine Learning with Python

### Overview
This repo has two simple Python scripts that show the difference between supervised and unsupervised learning. Both use scikit-learn and each one has a Matplotlib graph as a bonus.

### Part A: Student Exam Score Prediction
The supervised script (supervised_ml.py) uses Linear Regression to predict an exam score based on how many hours a student studies.

It works with a small dataset of 10 examples (study hours vs exam scores). The code splits the data into the input feature (X) and the target (y), trains the model with fit() then asks the user for a number of study hours and uses predict() to give an estimated score. The result is shown to two decimal places.

The bonus graph shows the original data points, the regression line, and where the user’s prediction sits on that line.

### Part B: Customer Segmentation Using K-Means
The unsupervised script (unsupervised_ml.py) groups 10 customers using two features: monthly spending (in Kenyan shillings) and how many times they shop in a month.

Because the spending numbers are much bigger than the visit counts the features are scaled with StandardScaler before training. The K-Means model is set to 3 clusters with random_state=42 and n_init=10. After running it prints which cluster each customer belongs to and how many customers are in each cluster.

The bonus scatter plot colours the customers by their cluster.

### Requirements
- Python 3.10 or newer (recommended)
- pip
- The packages listed in requirements.txt

### Installation
Open a terminal in the project folder and run:

bash
python -m venv .venv


Activate the virtual environment:

**Windows PowerShell:**
powershell
.\.venv\Scripts\Activate


Then install the dependencies:

bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


### Running the programs
##Supervised learning:##
bash
python supervised_ml.py

Enter a number of study hours when prompted. It will print the predicted score and open the graph.

##Unsupervised learning##:
bash
python unsupervised_ml.py

It prints the cluster assignments and totals then opens the scatter plot.

Close the graph window when you’re done. Run the two scripts separately.

### Theory Questions

1. What is supervised machine learning?
Supervised learning is when the model is trained on labelled data. Every training example has both the input and the correct output so the model can learn the relationship between them and then make predictions on new data.

2. What is unsupervised machine learning?
Unsupervised learning works with data that has no labels. The model just looks for patterns, similarities, or natural groupings in the data on its own.

3. Explain the difference between supervised and unsupervised learning.
Supervised learning needs labelled examples while unsupervised learning doesn’t — it has to find structure by itself. Predicting an exam score from study hours is supervised. Grouping customers by spending behaviour is unsupervised.

4. Why is Linear Regression considered supervised learning?
Because it’s trained with both inputs and known target values. In this assignment the inputs are study hours and the targets are the actual exam scores. The model learns the relationship and can then predict a score for a new number of hours.

5. Why is K-Means considered unsupervised learning? 
K-Means doesn’t get any correct cluster labels during training. It just groups the data points based on how similar their features are. Here it groups customers using spending and shopping frequency, and assigns the cluster numbers itself.

6. What is the difference between fit() and predict()?
fit() is the training step — the model learns from the data you give it. predict() is the step where you ask the already-trained model to make estimates on new data. For clustering, scikit-learn also has fit_predict() which does both in one go.

7. Give two real-world applications of supervised learning and two of unsupervised learning.  

Supervised:
1. Predicting house prices from features like location, size and number of rooms (using past sales with known prices).
2. Detecting spam emails using messages that have already been labelled as spam or not spam.

Unsupervised:
1. Customer segmentation for marketing based on spending patterns and how often people shop.
2. Grouping news articles or documents by topic when you don’t have topic labels.
