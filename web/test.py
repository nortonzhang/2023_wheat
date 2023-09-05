from flask import Flask, render_template, request, redirect, url_for, session
import os
import pymysql
import torch
from sqlalchemy.dialects import mysql
from torchvision import transforms
from PIL import Image
import os.path as osp
import shutil
import torch.nn as nn
from torchutils import get_torch_transforms
from train import SELFMODEL
import numpy as np
from torchvision.transforms import ToTensor, Normalize
from werkzeug.utils import secure_filename
from flask import send_from_directory
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import mysql.connector



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './path/save'
app.secret_key = '20191008238'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:123456@localhost/wheat'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = '20191008238'

model_path = "E:\\04-design\\2023_wheat\\web\\resnet50d_10epochs_accuracy0.96023_weights.pth"  # todo  模型路径
classes_names = ['CrownAndRootRot', 'HealthyWheat', 'LeafRust', 'PowderyMildew', 'WheatLooseSmut', 'WheatAphids',
                 'WheatCystNematode', 'WheatRedSpider', 'WheatScab', 'WheatSharpEyespot', 'WheatStalkRot',
                 'WheatTake-all']  # todo 类名
# model_path = "E:/04-design/checkpoints/efficientnet_b3a_pretrained_224/efficientnet_b3a_8epochs_accuracy0.98925_weights.pth"  # todo  模型路径
img_size = 224  # todo 图片大小
model_name = "resnet50d"  # todo 模型名称
num_classes = len(classes_names)  # todo 类别数目

# 数据库连接配置
db = pymysql.connect(
    host="localhost",
    user="root",
    password="123456",
    database="wheat"
)

# MySQL数据库配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'wheat'
}


# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 连接到MySQL数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 执行查询用户语句
        query = "SELECT * FROM user WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        user = cursor.fetchone()

        # 验证用户
        if user:
            # 将用户ID存储在会话中
            session['user_id'] = user[0]
            session['username'] = user[1]

            # 插入登录日志
            user_id = user[0]
            operation = 'login'
            insert_log(user_id, operation)

            # 重定向到首页或其他需要登录的页面
            return redirect('/')
        else:
            # 用户名或密码错误
            error = 'Invalid username or password'
            return render_template('login.html', error=error)

        # 关闭数据库连接
        cursor.close()
        conn.close()



    # GET请求返回登录页面
    return render_template('login.html')

# 登录路由
@app.route('/login')
def home():
    # 检查用户是否登录
    if 'user_id' in session:
        user_id = session['user_id']
        username = session['username']

        # 在此处可以根据需要进行其他操作

        return render_template('index.html', username=username)

    # 用户未登录，重定向到登录页面
    return redirect('/login')

# 登出路由
@app.route('/logout', methods=['POST'])
def logout():
    # 获取当前登录用户的信息
    user_id = session.get('user_id')

    # 插入登出日志
    operation = 'logout'
    insert_log(user_id, operation)
    # 清除会话中的用户数据
    session.clear()

    # 重定向到登录页面
    return render_template('index.html', username="")


@app.route('/')
def index():
    # 获取当前登录用户的信息
    username = session.get('username')

    # 对用户名进行空值处理
    if username is None:
        username = ""

    # 将用户名传递给模板
    return render_template('index.html', username=username)

def predict_single(model_path, image_path):
    # 加载网络
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片并应用预处理
    img = Image.open(image_path).convert("RGB")
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    # 进行推理
    output = model(img)
    label_id = torch.argmax(output).item()
    return label_id


# 图像预处理函数
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Validate if all fields are provided
        if not username or not password or not email:
            error = '所有需求均为必填项'
            return render_template('register.html', error=error)

        # Connect to MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        try:
            # Generate a new user ID
            query = "SELECT MAX(user_id) FROM user"
            cursor.execute(query)
            max_user_id = cursor.fetchone()[0]
            new_user_id = max_user_id + 1 if max_user_id else 1

            # Check if username is already taken
            query = "SELECT * FROM user WHERE username = %s"
            cursor.execute(query, (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                error = '该用户名已存在'
                return render_template('register.html', error=error)

            # Insert new user into database
            query = "INSERT INTO user (user_id, username, password, email) VALUES (%s, %s, %s, %s)"
            cursor.execute(query, (new_user_id, username, password, email))
            conn.commit()

            # Close database connection
            cursor.close()
            conn.close()

            # Insert log into log table
            insert_log(new_user_id, 'register')

            # Redirect to login page
            return redirect(url_for('login'))

        except Exception as e:
            # Handle database errors
            error = 'An error occurred while registering: {}'.format(str(e))
            return render_template('register.html', error=error)

    # If the request method is GET, render the register page
    return render_template('register.html', error=None)

@app.route('/wheat/CrownAndRootRot')
def CrownAndRootRot():
    return render_template('/wheat/CrownAndRootRot.html')
@app.route('/wheat/LeafRust')
def LeafRust():
    return render_template('/wheat/LeafRust.html')
@app.route('/wheat/PowderyMildew')
def PowderyMildew():
    return render_template('/wheat/PowderyMildew.html')
@app.route('/wheat/WheatAphids')
def WheatAphids():
    return render_template('/wheat/WheatAphids.html')
@app.route('/wheat/WheatCystNematode')
def WheatCystNematode():
    return render_template('/wheat/WheatCystNematode.html')
@app.route('/wheat/WheatLooseSmut')
def WheatLooseSmut():
    return render_template('/wheat/WheatLooseSmut.html')
@app.route('/wheat/WheatRedSpider')
def WheatRedSpider():
    return render_template('/wheat/WheatRedSpider.html')
@app.route('/wheat/WheatScab')
def WheatScab():
    return render_template('/wheat/WheatScab.html')
@app.route('/wheat/WheatSharpEyespot')
def WheatSharpEyespot():
    return render_template('/wheat/WheatSharpEyespot.html')
@app.route('/wheat/WheatStalkRot')
def WheatStalkRot():
    return render_template('/wheat/WheatStalkRot.html')
@app.route('/wheat/WheatTake')
def WheatTake():
    return render_template('/wheat/WheatTake.html')
@app.route('/mail')
def mail():
    return render_template('/mail.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['image']

        # 检查是否有文件上传
        if file:
            # 生成安全的文件名并保存文件
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 将图片信息插入数据库
            try:
                    # 获取当前登录用户的信息
                    user_id = session.get('user_id')

                    # 获取识别结果标签
                    labels = ['CrownAndRootRot',
                              'HealthyWheat',
                              'LeafRust',
                              'PowderyMildew',
                              'WheatLooseSmut',
                              'WheatAphids',
                              'WheatCystNematode',
                              'WheatRedSpider',
                              'WheatScab',
                              'WheatSharpEyespot',
                              'WheatStalkRot',
                              'WheatTake-all']  # 替换为实际标签

                    # 标签名称和ID的映射关系
                    label_mapping = {
                        'CrownAndRootRot': 0,
                        'HealthyWheat': 1,
                        'LeafRust': 2,
                        'PowderyMildew': 3,
                        'WheatLooseSmut': 4,
                        'WheatAphids': 5,
                        'WheatCystNematode': 6,
                        'WheatRedSpider': 7,
                        'WheatScab': 8,
                        'WheatSharpEyespot': 9,
                        'WheatStalkRot': 10,
                        'WheatTake-all': 11
                    }

                    predicted_label = labels[predict_single(model_path=model_path, image_path=file_path)]
                    # 获取预测标签ID
                    label_id = label_mapping.get(predicted_label)

                    # 获取当前时间
                    upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with db.cursor() as cursor:
                        sql = "INSERT INTO Image (filename, filepath, upload_time, label_id, user_id) VALUES (%s, %s, %s, %s, %s)"
                        cursor.execute(sql, (
                            filename, os.path.join(app.config['UPLOAD_FOLDER'], filename), upload_time, label_id, user_id))
                        db.commit()

                        # 插入上传日志
                        operation = 'upload'
                        insert_log(user_id, operation)

                    return render_template('result.html', image_path=file_path, predicted_label=predicted_label,
                                           image_name=filename)

            except Exception as e:
                print(e)
                db.rollback()

        else:
            return '未选择文件！'
    else:
        return redirect(url_for('index'))


    return '上传失败！'

def insert_log(user_id, operation):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        # Get the maximum log_id from the log table
        cursor.execute("SELECT MAX(log_id) FROM log")
        result = cursor.fetchone()
        max_log_id = result[0] if result[0] else 0

        # Generate the new log_id by incrementing the maximum log_id
        new_log_id = max_log_id + 1

        # Get current timestamp
        timestamp = datetime.now()

        # Insert log into database
        query = "INSERT INTO log (log_id, log_time, user_id, operation) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (new_log_id, timestamp, user_id, operation))
        conn.commit()

    except Exception as e:
        # Handle database errors
        print('An error occurred while inserting log:', str(e))

    finally:
        # Close database connection
        cursor.close()
        conn.close()

@app.route('/path/save/<path:filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False)
