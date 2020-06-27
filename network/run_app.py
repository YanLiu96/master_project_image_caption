import os
from flask import Flask
from flask import render_template, request, jsonify
from flask_cors import CORS
from utils import caption_img
from werkzeug.utils import secure_filename
from datetime import timedelta
# whether cache the uploaded image

IS_CACHE_IMG=True
image_format=set(['png', 'jpg', 'JPG', 'PNG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in image_format

def init_models(img_pth):
    caption_img(img_pth)

app = Flask(__name__)
CORS(app)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
    message=''
    try:
        if request.method=='POST':
            print('1')
            print(request.files)
            if 'images' not in request.files:
                message='Please uploading your image'
                print(message)
                return render_template('index.html', messge=message)
            img_files=request.files.getlist('images')
            # # check the format of image
            for file in img_files:
                if not (file and allowed_file(file.filename)):
                    message='The image format is wrong, please upload png or jpg format'
                    return render_template('index.html', message=message)
                else:
                    base_path= os.path.dirname(__file__)
                    if IS_CACHE_IMG:
                        uploaded_img_name = secure_filename(file.filename)
                    else:
                        uploaded_img_name='test.jpg'
                    uploaded_img_path= os.path.join(base_path, 'static/images',uploaded_img_name)
                    file.save(uploaded_img_path)
                    print('end')
    except:
        #上传的文件是其他文件改成jpg格式会报错
        message = 'something error,please make sure the uploaded file is in JPG format!'
        return render_template('index.html', message=message)
    print('5')
    return render_template('index.html', message=message)



if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=8081,debug=True)