from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import os
import re

# Load model assets for loan prediction
try:
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    model = model_data['model']
    feature_names = model_data['feature_names']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
except Exception as e:
    print(f"Error loading model assets: {e}")
    raise

# Initialize Flask app
app = Flask(__name__)

# SHAP explainer for loan model
try:
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"Error initializing SHAP explainer: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Map categorical credit score to numerical value
        credit_score_mapping = {
            'Good': 750,  # Mid-high value for 670–850
            'Fair': 625,  # Midpoint for 580–669
            'Bad': 450    # Mid-low value for 300–579
        }
        if 'credit_score' in data:
            data['credit_score'] = credit_score_mapping.get(data['credit_score'], 450)  # Default to 'Bad' if invalid
        else:
            return render_template('result.html', prediction=-1, probability=0, error="Credit Score is missing")

        # Convert numerical inputs to appropriate types
        for field in ['income', 'loan_amount', 'loan_term']:
            if field in data:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    return render_template('result.html', prediction=-1, probability=0, error=f"Invalid value for {field}: must be a number")

        df = pd.DataFrame([data])

        for col in label_encoders:
            if col in df.columns:  # Fixed typo: df.columnsYOU -> df.columns
                le = label_encoders[col]
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    df[col] = le.transform([le.classes_[0]])

        for col in feature_names:
            if col not in df.columns:
                if col == 'DAYS_BIRTH':
                    df[col] = -14600  # ~40 years old
                elif col == 'DAYS_EMPLOYED':
                    df[col] = -730  # ~2 years employed
                elif col == 'MONTHS_BALANCE':
                    df[col] = -60  # 5 years of account history
                else:
                    df[col] = 0

        df = df.reindex(columns=feature_names, fill_value=0)
        df_scaled = scaler.transform(df)

        prediction = int(model.predict(df_scaled)[0])
        proba = float(model.predict_proba(df_scaled)[0][1])

        shap_values = explainer.shap_values(df_scaled)

        return render_template('result.html', prediction=prediction, probability=round(proba * 100, 2))
    except Exception as e:
        print(f"Error in predict route: {e}")
        return render_template('result.html', prediction=-1, probability=0, error=str(e))

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.get_json()
        
        # Map categorical credit score to numerical value
        credit_score_mapping = {
            'Good': 750,
            'Fair': 625,
            'Bad': 450
        }
        if 'credit_score' in data:
            data['credit_score'] = credit_score_mapping.get(data['credit_score'], 450)  # Default to 'Bad' if invalid
        else:
            return jsonify({"lime_explanation": [], "error": "Credit Score is missing"}), 400

        # Convert numerical inputs to appropriate types
        for field in ['income', 'loan_amount', 'loan_term']:
            if field in data:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    return jsonify({"lime_explanation": [], "error": f"Invalid value for {field}: must be a number"}), 400

        df = pd.DataFrame([data])

        for col in label_encoders:
            if col in df.columns:
                le = label_encoders[col]
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    df[col] = le.transform([le.classes_[0]])

        for col in feature_names:
            if col not in df.columns:
                if col == 'DAYS_BIRTH':
                    df[col] = -14600  # ~40 years old
                elif col == 'DAYS_EMPLOYED':
                    df[col] = -730  # ~2 years employed
                elif col == 'MONTHS_BALANCE':
                    df[col] = -60  # 5 years of account history
                else:
                    df[col] = 0

        df = df.reindex(columns=feature_names, fill_value=0)
        df_scaled = scaler.transform(df)

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(df_scaled),
            feature_names=feature_names,
            class_names=['Not Approved', 'Approved'],
            mode='classification'
        )
        lime_exp = lime_explainer.explain_instance(df_scaled[0], model.predict_proba)
        explanation = lime_exp.as_list()

        return jsonify({"lime_explanation": explanation})
    except Exception as e:
        print(f"Error in explain route: {e}")
        return jsonify({"lime_explanation": [], "error": str(e)}), 500

# Endpoint for initial greeting with loan-related suggested questions
@app.route('/chatbot/init', methods=['GET'])
def chatbot_init():
    greeting = (
        "Hi,<br>How can I assist you today?<br>"
        "Here are some questions you might want to ask:<br>"
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'What factors affect loan approval?\')">What factors affect loan approval?</span>'
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'How does my income impact my loan?\')">How does my income impact my loan?</span>'
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'Does my credit score matter for loan approval?\')">Does my credit score matter for loan approval?</span>'
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'How does the loan term affect my application?\')">How does the loan term affect my application?</span>'
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'What role does employment history play in loan approval?\')">What role does employment history play in loan approval?</span>'
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'Does the loan purpose influence approval chances?\')">Does the loan purpose influence approval chances?</span>'
        '<span class="suggested-question" onclick="askSuggestedQuestion(\'Who are you?\')">Who are you?</span>'
    )
    return jsonify({"response": greeting})

# Chatbot endpoint to handle user messages
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user input from the request
    user_input = request.get_json().get("message", "").strip().lower()

    # Validate input
    if not user_input:
        return jsonify({"response": "Please provide a message to proceed."}), 400

    # Define responses for greetings
    greetings = ["hi", "hello", "hey", "greetings"]
    if any(greeting in user_input for greeting in greetings):
        return jsonify({"response": "Hello! How can I assist you today?"})

    # Define responses for farewells
    farewells = ["bye", "goodbye", "see you", "take care"]
    if any(farewell in user_input for farewell in farewells):
        return jsonify({"response": "Goodbye! If you have more questions, feel free to come back!"})

    # Define responses for loan-related queries
    if "loan" in user_input and "factor" in user_input or "approval" in user_input and "factor" in user_input:
        response = (
            "Several factors affect loan approval. Based on our model, the most important ones are: "
            "1. Your account balance history (MONTHS_BALANCE) - a longer, stable history helps. "
            "2. Your age (DAYS_BIRTH) - older applicants might have more financial stability. "
            "3. Employment duration (DAYS_EMPLOYED) - steady employment is a good sign. "
            "4. Total income (AMT_INCOME_TOTAL) - higher income improves your chances. "
            "Would you like to know more about any specific factor?"
        )
        return jsonify({"response": response})

    if "income" in user_input:
        return jsonify({"response": "Your total income (AMT_INCOME_TOTAL) is a key factor in loan approval. Higher income generally increases your chances of approval because it shows you can repay the loan. For example, if your income is above the average for your region, lenders may view you as a lower risk. Do you have a specific question about income requirements?"})

    if "employment" in user_input or "job" in user_input or "employment history" in user_input:
        return jsonify({"response": "Your employment history (DAYS_EMPLOYED) is crucial for loan approval. Lenders look for steady, long-term employment—ideally, at least 1-2 years with the same employer—as it indicates financial stability. Frequent job changes might raise concerns about your ability to repay. How long have you been with your current employer?"})

    if "age" in user_input:
        return jsonify({"response": "Your age (DAYS_BIRTH) plays a role in loan approval. Generally, older applicants might be seen as more financially stable due to longer credit histories, but younger applicants can still qualify if other factors like income and employment are strong. Lenders often prefer applicants between 25 and 55 years old. Would you like to know more?"})

    if "credit score" in user_input:
        return jsonify({"response": "Yes, your credit score is a significant factor in loan approval, even though it’s not the top feature in our model. A higher credit score (e.g., above 700) signals to lenders that you’re reliable with repayments. If your score is lower, you might still qualify by improving other factors like income or employment history. Do you know your current credit score?"})

    if "loan term" in user_input:
        return jsonify({"response": "The loan term (duration of the loan) can impact your application. Shorter terms often mean higher monthly payments but lower total interest, which might appeal to lenders as it reduces their risk. Longer terms lower monthly payments but increase total interest, potentially raising concerns about repayment capacity. What loan term are you considering?"})

    if "loan purpose" in user_input:
        return jsonify({"response": "The purpose of your loan can influence approval chances. For example, loans for homes or education are often seen as lower risk because they’re tied to tangible assets or future earning potential. Loans for business or personal use might be scrutinized more closely, depending on your financial profile. What’s the purpose of your loan?"})

    # Define responses for casual conversation
    if "how are you" in user_input:
        return jsonify({"response": "I'm doing great, thanks for asking! How about you?"})

    if "thank you" in user_input or "thanks" in user_input:
        return jsonify({"response": "You're welcome! If you have more questions, I'm here to help."})

    if "what is your name" in user_input or "who are you" in user_input:
        return jsonify({"response": "I'm the Loan Approval Chatbot, here to help with your loan-related questions or just chat casually. What's on your mind?"})

    # General knowledge responses
    if "capital of france" in user_input:
        return jsonify({"response": "The capital of France is Paris. Do you have any other questions, or would you like to talk about loans?"})

    if "weather" in user_input:
        return jsonify({"response": "I can't check the weather for you, but I can help with loan-related questions! Would you like to know about loan approval factors?"})

    if "time" in user_input:
        return jsonify({"response": "I can't tell the current time, but I can assist with loan queries or other topics. What would you like to talk about?"})

    # Default response for unrecognized queries
    return jsonify({"response": "I'm not sure I understand, but I'm here to help! Could you rephrase your question, or would you like to talk about loan approval?"})

if __name__ == '__main__':
    app.run(debug=True)