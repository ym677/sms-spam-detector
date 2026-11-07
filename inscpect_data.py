from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import joblib

vectorizer = TfidfVectorizer()
model = MultinomialNB()

project_dir = Path(__file__).resolve().parent
data_path = project_dir /"data"/"sms+spam+collection"/"SMSSpamCollection"

ham_cnt = 0
spam_cnt = 0
x = []#存放对应短信内容
y = []#存放对应标签


with open(data_path,"r",encoding="utf-8") as file:
    lines = file.readlines()
    for i in lines:
        parts = i.rstrip("\n").split("\t")
        label = parts[0]
        y.append(label)
        message = parts[1]
        x.append(message)
        if label == "ham":
            ham_cnt += 1
        elif label == "spam":
            spam_cnt += 1
        else:
            print(label)


X_trainval, X_test, y_trainval, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_val_tfidf)
print(model.score(X_val_tfidf, y_val))

# print(ham_cnt,spam_cnt,"总数为:",ham_cnt + spam_cnt)
#
# print(x[0])
# print(len(X_test),len(y_test))
# print(len(X_train),len(y_train))
# print(len(X_val),len(y_val))

# print(X_train_tfidf.shape)
# print(X_val_tfidf.shape)

matrix = confusion_matrix(y_val, y_pred,labels=["ham","spam"])
report = classification_report(y_val,y_pred,labels=["ham","spam"])

print(y_pred[:10])
print(y_val[:10])
print(matrix)
# print(report)

probabilities = model.predict_proba(X_val_tfidf)
# print("类别顺序：",model.classes_)
# print("前5条短信的预测概论：",probabilities[:5])

spam_scores = probabilities[:,1]
y_pred_02 = np.where(spam_scores>0.2,"spam","ham")

matrix = confusion_matrix(y_val, y_pred_02,labels=["ham","spam"])
print(matrix)

X_test_tfidf = vectorizer.transform(X_test)
test_probabilities = model.predict_proba(X_test_tfidf)
spam_index = list(model.classes_).index("spam")
test_probabilities = test_probabilities[:,spam_index]

y_test_pred = np.where(test_probabilities>0.2,"spam","ham")

test_matrix = confusion_matrix(y_test,y_test_pred,labels=["ham","spam"])

test_report = classification_report(y_test,y_test_pred,labels=["ham","spam"])

print("测试集混淆矩阵：")
print(test_matrix)
print("测试集分类报告")
print(test_report)

model_data = {
    "vectorizer": vectorizer,
    "model": model,
    "threshold":0.2
}

model_path = project_dir /"sms_model.joblib"
joblib.dump(model_data,model_path)