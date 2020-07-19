import os,shutil
import torch
import json
import warnings
import subprocess
warnings.filterwarnings("ignore")
import matplotlib
from flask import Flask
from flask import render_template, request, jsonify, Response
import json
from flask_cors import CORS
from utils import caption_image_beam_search, visualize_att
from test.discrible import caption_img
from werkzeug.utils import secure_filename
from datetime import timedelta


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# whether cache the uploaded image

#IS_CACHE_IMG=True
image_format=set(['png', 'jpg', 'JPG', 'PNG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in image_format

def models_init():
    # Load model
    checkpoint = torch.load('saved_data/trained_models/best_checkpoint_trained_models.pth.tar', map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    # Load word map (word2ix)
    with open('saved_data/word2index/word2idx.json', 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    return decoder,encoder,word_map,rev_word_map

app = Flask(__name__)
CORS(app)

# # 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
decoder,encoder, word_map, rev_word_map = models_init()
caption_result=[]
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/demo', methods=['GET'])
def demo():
    return render_template('demo.html')

@app.route('/test', methods=['POST','GET'])
def upload():
    message=''
    try:
        if request.method=='POST':
            caption_result.clear()
            # delete all the privous uploaded files
            folder= os.path.join(os.path.dirname(__file__),'static/uploaded_img')
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            # delete all the privous test results 
            folder= os.path.join(os.path.dirname(__file__),'static/result_img')
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            if 'images' not in request.files:
                message='Please uploading your image'
                print(message)
                return render_template('index.html', messge=message)
            img_files=request.files.getlist('images')
            # check the format of image
            i =0
            for file in img_files:
                if not (file and allowed_file(file.filename)):
                    message='The image format is wrong, please upload png or jpg format'
                    return render_template('upload.html', message=message)
                else:
                    base_path= os.path.dirname(__file__)
                    #if IS_CACHE_IMG:
                    uploaded_img_name = secure_filename(file.filename)
                    uploaded_img_path= os.path.join(base_path, 'static/uploaded_img',uploaded_img_name)
                    file.save(uploaded_img_path)
                    seq, alphas = caption_image_beam_search(encoder, decoder, uploaded_img_path, word_map, 5)
                    alphas = torch.FloatTensor(alphas)
                    captions = [rev_word_map[ind] for ind in seq]
                    caption_result.append(captions)
                    #print(captions)
                    i=i+1
                    result_name='result_'+str(i)+'.png'
                    visualize_att(result_name, uploaded_img_path, seq, alphas, rev_word_map, True)
            print('You have uploaded the images')
            return jsonify({'status': 1})
        else:
            return render_template('upload.html')
    except:
        #上传的文件是其他文件改成jpg格式会报错
        message = 'something error,please make sure the uploaded file is in JPG format!'
        return render_template('upload.html', message=message)

@app.route('/showresult', methods=['GET'])
def showresult():
    img_src=[]
    sentence_list=[]
    folder= os.path.join(os.path.dirname(__file__),'static/result_img')
    result_list=sorted(os.listdir(folder))
    for filename in result_list:
        print(filename)
        file_path = os.path.join('../static/result_img', filename)
        img_src.append(file_path)
    for cap in caption_result:
        sentence=''
        for word in cap[1:-1]:
            sentence= sentence + ' '+word
        sentence_list.append(sentence)
    print(img_src,sentence_list)

    return render_template('showresult.html',results=zip(img_src, sentence_list))

# @app.route('/train')
# def train():
#     def begin():
#         proc = subprocess.Popen(
#             ['python main.py --control \'train\''],             #call something with a lot of output so we can see it
#             shell=True,
#             stdout=subprocess.PIPE,
#             universal_newlines=True
#         )
#         for line in iter(proc.stdout.readline,''):
#             yield line.rstrip() + '<br/>\n'
#     return Response(begin(), mimetype='text/html')  # text/html is required for most browsers to show th$

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=8081,debug=True)