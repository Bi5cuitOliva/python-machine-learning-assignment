
"""
Part A: Supervised Machine Learning
Project: Student Exam Score Prediction
Algorithm: Linear Regression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    # Step 1: Prepare the dataset.
    hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    exam_scores = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

    # Step 2: Separate input features (X) and target values (y).
    X = np.array(hours_studied).reshape(-1, 1)
    y = np.array(exam_scores)

    # Step 3: Create the Linear Regression model.
    model = LinearRegression()

    # Step 4: Train the model using the dataset.
    model.fit(X, y)

    # Step 5: Ask the user to enter the number of study hours.
    try:
        user_hours = float(input("Enter hours studied: "))

        if not 0 <= user_hours <= 24:
            print("Please enter a number of hours between 0 and 24.")
            return

    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    # Step 6: Predict the examination score.
    predicted_score = model.predict(
        np.array([[user_hours]])
    )[0]

    # Step 7: Display the prediction to two decimal places.
    print(f"Predicted exam score: {predicted_score:.2f}")

    # Bonus: Create a graph showing the relationship
    # between study hours and examination scores.
    line_hours = np.linspace(
        min(hours_studied),
        max(hours_studied),
        100
    )

    line_scores = model.predict(line_hours.reshape(-1, 1))

    plt.figure(figsize=(8, 5))

    plt.scatter(
        hours_studied,
        exam_scores,
        label="Actual exam scores"
    )

    plt.plot(
        line_hours,
        line_scores,
        label="Linear Regression line"
    )

    plt.scatter(
        [user_hours],
        [predicted_score],
        marker="x",
        s=100,
        label="Your prediction"
    )

    plt.title("Study Hours vs Examination Scores")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
