import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# Streamlit page configuration and title
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("👥 Employee Attrition Predictor")

#Load and Preprocess Data
df = pd.read_csv(r"C:\Users\priya\Downloads\Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
selected_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']
X = df[selected_features]
y = df['Attrition']

#Handle Categorical Variables
X = pd.get_dummies(X, drop_first=True)

#Split Data and Train Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Evaluate Model

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)


# Display Metrics

st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**ROC AUC Score:** {auc_score:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
  

#Confusion Matrix
fig_cm, ax_cm = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
st.pyplot(fig_cm)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend()
st.pyplot(fig_roc)
 
 #Create Input Form
with st.form(key='user_input_form'):
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    distance_from_home = st.number_input('Distance from Home (miles)', min_value=0, max_value=30, value=5)
    monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
    job_satisfaction = st.selectbox('Job Satisfaction', options=[1, 2, 3, 4], index=0)
    years_at_company = st.number_input('Years at Company', min_value=0, max_value=40, value=5)
    submit_button = st.form_submit_button(label='Predict')
# Prepare User Input

if submit_button:
    user_data = pd.DataFrame({
        'Age': [age],
        'DistanceFromHome': [distance_from_home],
        'MonthlyIncome': [monthly_income],
        'JobSatisfaction': [job_satisfaction],
        'YearsAtCompany': [years_at_company]
    })

    # Apply the same preprocessing as the training data
    user_data_encoded = pd.get_dummies(user_data, drop_first=True)
    user_data_encoded = user_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    user_prediction = model.predict(user_data_encoded)
    prediction_proba = model.predict_proba(user_data_encoded)[:, 1]

    st.write(f"Prediction: {'Attrition' if user_prediction[0] == 1 else 'No Attrition'}")
    st.write(f"Prediction Probability: {prediction_proba[0]:.2f}")