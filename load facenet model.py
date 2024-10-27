from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model
import numpy as np

# Load the FaceNet model (or use VGGFace)
face_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_face(image, box):
    x, y, width, height = box
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, (224, 224))
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face, version=2)  # VGGFace version 2 preprocessing
    return face

def get_embedding(model, face_pixels):
    # Get the face embedding using the pre-trained model
    embedding = model.predict(face_pixels)
    return embedding