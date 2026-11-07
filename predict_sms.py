from pathlib import Path
import joblib



project_dir = Path(__file__).resolve().parent
model_data = joblib.load(project_dir /"sms_model.joblib")


vectorizer = model_data["vectorizer"]
model = model_data["model"]
threshold = model_data["threshold"]

message = "Congratulations! You have won a free prize."

vectorized_message = vectorizer.transform([message])
result = model.predict_proba(vectorized_message)
spam_index = list(model.classes_).index("spam")
spam_score = result[0, spam_index]

if spam_score > threshold:
    print("这条是垃圾短信")
else:
    print("这条是正常短信")
