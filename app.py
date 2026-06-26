import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(page_title="Demat Account Prediction", layout="wide")
st.title("📊 Demat Account Opening Predictor")

# 2. Load Data
@st.cache_data
def load_data():
    # Replace with your actual path if different
    df = pd.read_csv("new data.csv")
    # Fill missing values in Previous_Investments with 'None'
    df['Previous_Investments'] = df['Previous_Investments'].fillna('None')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find 'new data.csv'. Please make sure it's in the same directory.")
    st.stop()

# 3. Train a quick baseline model behind the scenes
@st.cache_resource
def train_baseline_model(data):
    X = data.drop(columns=['Demat_Account_Open'])
    y = data['Demat_Account_Open']
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, encoders, X.columns

model, encoders, feature_cols = train_baseline_model(df)

# 4. Tabs for Navigation
tab1, tab2 = st.tabs(["📈 Dataset Overview", "🔮 Make Predictions"])

with tab1:
    st.subheader("Explore the Customer Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Conversion Rate", f"{(df['Demat_Account_Open'].mean() * 100):.1f}%")

with tab2:
    st.subheader("Input Customer Details to Predict Demat Conversion")
    
    # Form layout for user inputs
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 35)
            gender = st.selectbox("Gender", df['Gender'].unique())
            occupation = st.selectbox("Occupation", df['Occupation'].unique())
            education = st.selectbox("Education", df['Education'].unique())
            
        with c2:
            annual_income = st.number_input("Annual Income", min_value=0, value=500000, step=25000)
            savings_amount = st.number_input("Savings Amount", min_value=0, value=150000, step=10000)
            credit_score = st.slider("Credit Score", int(df['Credit_Score'].min()), int(df['Credit_Score'].max()), 700)
            city_tier = st.selectbox("City Tier", df['City_Tier'].unique())
            
        with c3:
            investment_exp = st.slider("Investment Experience (Years)", 0, int(df['Investment_Experience'].max()), 2)
            existing_bank = st.selectbox("Existing Bank Account?", df['Existing_Bank_Account'].unique())
            digital_usage = st.selectbox("Digital Banking Usage", df['Digital_Banking_Usage'].unique())
            prev_investments = st.selectbox("Previous Investments", df['Previous_Investments'].unique())
            
        submit_btn = st.form_submit_button("Predict Conversion")
        
    if submit_btn:
        # Create a DataFrame from user input
        input_data = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Occupation': occupation, 
            'Annual_Income': annual_income, 'Education': education, 'City_Tier': city_tier,
            'Existing_Bank_Account': existing_bank, 'Credit_Score': credit_score, 
            'Investment_Experience': investment_exp, 'Savings_Amount': savings_amount, 
            'Digital_Banking_Usage': digital_usage, 'Previous_Investments': prev_investments
        }])
        
        # Process inputs using trained encoders
        for col in encoders:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except ValueError:
                # Fallback if an unseen label appears
                input_data[col] = 0

        # Predict
        prediction = model.predict(input_data[feature_cols])[0]
        probability = model.predict_proba(input_data[feature_cols])[0][1]
        
        st.write("---")
        if prediction == 1:
            st.success(f"🎉 **Result:** High Likelihood to Open a Demat Account! (Probability: {probability:.2%})")
        else:
            st.warning(f"⚠️ **Result:** Low Likelihood to Open a Demat Account. (Probability: {probability:.2%})")