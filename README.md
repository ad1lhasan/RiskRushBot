RiskRushBot
Welcome to RiskRushBot, an AI-powered Credit Risk Advisor with an integrated chatbot, designed to assess creditworthiness with speed and precision. Built with Flask, scikit-learn, and a vibrant, animated frontend, this project combines exploratory data analysis (EDA), logistic regression, and an interactive chatbot to deliver insightful credit risk predictions.
Table of Contents

Project Overview
Features
Tech Stack
Installation
Usage
Folder Structure
Model Details
Contributing
License

Project Overview
RiskRushBot is a web application that predicts credit risk using machine learning and provides an engaging user experience through a chatbot interface. It leverages EDA to uncover data insights and logistic regression for accurate predictions, wrapped in a modern, animated design inspired by futuristic aesthetics.
Features

Credit Risk Prediction: Uses logistic regression to evaluate creditworthiness based on user inputs.
Interactive Chatbot: Suggests loan-related questions and provides detailed, unique responses.
Exploratory Data Analysis: Visualizes data distributions, correlations, and outliers for deeper insights.
Professional Design: Centered form layout with hover effects, Roboto font, and smooth animations.
Error Handling: Robust backend to manage invalid inputs and ensure a seamless experience.

Tech Stack

Backend: Flask, Python
Machine Learning: scikit-learn (Logistic Regression), pandas, numpy
Frontend: HTML, CSS, JavaScript
Styling: Custom CSS with gradients, shadows, and animations
Data Analysis: Matplotlib, seaborn (for EDA)

Installation

Clone the Repository:
git clone https://github.com/ad1lhasan/RiskRushBot.git
cd RiskRushBot


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install flask pandas numpy scikit-learn matplotlib seaborn


Ensure Model File:

Place your trained model.pkl (RandomForest model) in the project root.
If you don’t have one, train it using the provided train_model.py (see Model Details).



Usage

Run the Flask App:
python app.py


Access the App:

Open your browser and go to http://127.0.0.1:5000.
Fill out the credit risk form to get a prediction.
Interact with the chatbot for loan-related insights.


View Results:

Predictions are displayed on a dedicated result page.
Chatbot responses appear in a sleek, animated sidebar.



Folder Structure
RiskRushBot/
├── model.pkl              # Trained logistic regression model
├── app.py                 # Flask backend
├── train_model.py         # Script to train the model
├── templates/
│   ├── index.html         # Main form page
│   ├── result.html        # Prediction result page
├── static/
│   ├── css/
│   │   ├── style.css      # Custom styles
│   ├── js/
│   │   ├── enhancements.js # Animation and chatbot scripts
├── README.md              # This file

Model Details

Algorithm: RandomForest
Features: Numerical and categorical inputs (customize based on your dataset, e.g., Income, Credit_Score).
Training:
Run train_model.py to generate model.pkl using your dataset.
Example dataset: Kaggle’s loan data (merge on 'ID' column, as discussed).
Includes EDA for data cleaning, visualization, and outlier removal.



To train a new model:
python train_model.py

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Built with 💡 and 🚀 by [ad1lhasan]. Star this repo if you find it useful!
