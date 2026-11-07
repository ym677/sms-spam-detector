# 英文短信垃圾信息识别

一个机器学习学习项目：输入英文短信，网页返回正常短信或垃圾短信的预测结果。

## 使用

1. 使用 Python 3.12 创建项目环境，并在该环境中安装 `requirements.txt` 中的依赖。
2. 确认项目根目录中有 `sms_model.joblib`，然后运行 `app.py`。
3. 打开 `http://127.0.0.1:5001`，输入英文短信并点击“开始识别”。

模型文件保存了文字转换器、分类模型和判断门槛。序列化模型对 scikit-learn 版本敏感，依赖清单记录了生成模型时使用的版本。

## 实现与评估

- 使用 TF-IDF 将短信转换为数字特征，使用多项式朴素贝叶斯分类。
- 在验证集上选取 0.2 的垃圾短信判断门槛；保留测试集进行评估。
- 本次测试集包含 1,115 条短信：垃圾短信识别出 125 条、漏掉 24 条，正常短信误判 1 条。
- 发送者名字暂不参与预测。该项目为学习练习，主要支持英文短信，不能保证识别新型垃圾短信或诈骗信息。

## 数据来源

项目使用 UCI Machine Learning Repository 的 [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection)，由 Tiago Almeida 和 José Hidalgo 提供；数据集页面标明使用 CC BY 4.0 许可。数据及原始说明位于 `data/`。

## 文件说明

- `app.py`：Flask 页面和预测接口。
- `templates/`、`static/`：网页与交互代码。
- `inscpect_data.py`：数据读取、模型训练和评估练习。
- `predict_sms.py`：加载模型并预测单条短信的练习。
- `sms_model.joblib`：训练后的文字转换器、模型和判断门槛。
