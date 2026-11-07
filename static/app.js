// 获取页面上的按钮、输入框和结果显示位置
const button = document.getElementById("predict-button");
const messageInput = document.getElementById("message");
const result = document.getElementById("result");

// 用户点击按钮时执行
button.addEventListener("click", function () {
    const message = messageInput.value.trim();

    if (message === "") {
        result.textContent = "请输入短信内容";
    } else {
        result.textContent = "正在识别……";

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        })
        .then(function (response) {
            if (!response.ok) {
                throw new Error("请求失败，状态码：" + response.status);
            }

            return response.json();
        })
        .then(function (data) {
            if (data.prediction === "spam") {
                result.textContent = "垃圾短信";
            } else if (data.prediction === "ham") {
                result.textContent = "正常短信";
            } else {
                result.textContent = "无法识别返回结果";
            }
        })
        .catch(function (error) {
            console.log("请求出错：", error);
            result.textContent = "识别失败，请重试";
        });
    }
});