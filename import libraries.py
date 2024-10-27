import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# Initialize the MTCNN detector
detector = MTCNN()

def detect_faces(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    
    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    
    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
detect_faces('sample_image.jpg')