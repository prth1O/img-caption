from flask import Flask,render_template,redirect,url_for,request
import os
from werkzeug.utils import secure_filename

from tensorflow.keras.models import Model
from pickle import load
from tensorflow.keras.models import load_model
from model_xep.caption_generator import generate_desc,cleanup_summary,extract_features
from tensorflow.keras.applications.xception import Xception


app = Flask(__name__)



# load the tokenizer
#tokenizer = load(open('/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 32
# load the model
#model = load_model('./model_9.h5')
# load and prepare the photograph
#photo = extract_features('example.jpg')
# generate description
tokenizer = load(open("./tokenizer.p","rb"))
model = load_model('./model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
#description = generate_desc(model, tokenizer, photo, max_length)
#description = cleanup_summary(description)
#print(description)
def prediction(img):
    photo = extract_features(img,xception_model)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    description = cleanup_summary(description)

    return description

@app.route('/',methods=['GET',])
def hello_world():
    return  render_template('Index.html')

@app.route('/Caption_prediction', methods=['GET', 'POST'])
def  Captioning():
    if request.method == 'POST':
        f = request.files['img']

        basepath ='./static/img/'
        file_path = os.path.join(basepath, f.filename)
        f.save(file_path)


        img=file_path
        desc=prediction(file_path)
        print(desc)




        #f.save(os.path.join(app.config['UPLOAD_FOLDER']))#, secure_filename(f.filename)))
        return render_template('pred.html' ,pred=desc,img=img)

        #print(description)
        #return redirect(url_for('.prediction',img=f))


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('hello_world'))





#return None



if __name__ == '__main__':
    app.run()
