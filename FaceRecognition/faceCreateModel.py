from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os
import numpy as np
from matplotlib import pyplot
import joblib
import sys

# check the number of arguments passed
if len(sys.argv) != 4:
    print("Usage: python3 faceTest.py <right_folder> <false_folder> <name_of_model>")
    sys.exit(1)

# directories passed as command line arguments
directory_right = sys.argv[1]
directory_false = sys.argv[2]
model_filename = sys.argv[3]

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
def get_embeddings(filenames, model):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# perform prediction
	yhat = model.predict(samples)
	return yhat

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# automatically detect face files in the directories
filenames_right = [os.path.join(directory_right, f) for f in os.listdir(directory_right) if f.endswith(('.jpg', '.png'))]
filenames_false = [os.path.join(directory_false, f) for f in os.listdir(directory_false) if f.endswith(('.jpg', '.png'))]

# get embeddings file filenames
embeddings_right = get_embeddings(filenames_right, model)
embeddings_false = get_embeddings(filenames_false, model)

# prepare labels: 1 for 'right', 0 for 'false'
labels_right = np.ones(embeddings_right.shape[0])
labels_false = np.zeros(embeddings_false.shape[0])

# concatenate embeddings and labels
X = np.concatenate((embeddings_right, embeddings_false))
y = np.concatenate((labels_right, labels_false))

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train a logistic regression classifier
clf = LogisticRegression(random_state=42).fit(X_train, y_train)

# save the model to disk
filename = f'{model_filename}.sav'
joblib.dump(clf, filename)
