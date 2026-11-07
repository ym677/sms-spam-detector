from pathlib import Path

import joblib
from flask import Flask, render_template, request


app = Flask(__name__)

# 找到当前项目文件夹
project_dir = Path(__file__).resolve().parent

# 加载之前训练并保存的模型数据
model_data = joblib.load(project_dir / "sms_model.joblib")

vectorizer = model_data["vectorizer"]
model = model_data["model"]
threshold = model_data["threshold"]


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    # 接收网页发送的 JSON 数据
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return {"message": "请求格式不正确"}, 400

    # 获取短信内容，去掉首尾空格
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return {"message": "请输入短信内容"}, 400

    message = message.strip()

    # 将短信转换成模型能处理的数字特征
    vectorized_message = vectorizer.transform([message])

    # 计算两个类别的预测概率
    probabilities = model.predict_proba(vectorized_message)

    # 找到 spam 对应的列
    spam_index = list(model.classes_).index("spam")
    spam_score = probabilities[0, spam_index]

    # 按保存的门槛判断
    if spam_score > threshold:
        prediction = "spam"
    else:
        prediction = "ham"

    # 将判断结果返回给网页
    return {"prediction": prediction}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)