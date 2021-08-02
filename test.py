from tensorflow.keras.models import Model
from pickle import load
from tensorflow.keras.models import load_model
from model_xep.caption_generator import generate_desc,cleanup_summary,extract_features
from tensorflow.keras.applications.xception import Xception
import argparse

tokenizer = load(open("./tokenizer.p","rb"))
model = load_model('./model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
print('model load succsesful')
max_length=32

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
#description = generate_desc(model, tokenizer, photo, max_length)
#description = cleanup_summary(description)
#print(description)
def prediction(img):
    photo = extract_features(img,xception_model)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    description = cleanup_summary(description)

    return description
tex=prediction(img_path)
print(tex)
