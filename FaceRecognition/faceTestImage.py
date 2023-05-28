from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import joblib
import sys
import numpy as np

# check the number of arguments passed
if len(sys.argv) != 3:
    print("Usage: python3 faceTest.py <model> <image>")
    sys.exit(1)

# model and image passed as command line arguments
model_filename = sys.argv[1]
image_filename = sys.argv[2]

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filename, model):
    # extract face
    face = extract_face(filename)
    # resize the image to the expected dimensions
    face = np.resize(face, (224, 224, 3))
    # convert into an array of samples
    samples = np.expand_dims(face, axis=0).astype('float64')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # perform prediction
    yhat = model.predict(samples)
    return yhat


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# get embeddings for the provided image
embedding = get_embeddings(image_filename, model)

# load the model from disk
loaded_model = joblib.load(model_filename)

# predict whether the person in the image is the same as in the training data
prediction = loaded_model.predict(embedding)

# Predict probabilities
probs = loaded_model.predict_proba(embedding)
print(probs)


print(prediction)

if prediction == 1:
    print("TRUE")
else:
    print("FALSE")
