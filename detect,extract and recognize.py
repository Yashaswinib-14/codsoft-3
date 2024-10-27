def process_image(image_path, known_embeddings, threshold=0.5):
    # Read and detect faces in the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    
    for face in faces:
        box = face['box']
        face_pixels = extract_face(image_rgb, box)
        embedding = get_embedding(face_model, face_pixels)
        
        # Recognize face
        recognized, similarity = recognize_face(known_embeddings, embedding, threshold)
        x, y, width, height = box
        
        if recognized:
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, f"Matched ({similarity:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage with dummy known embeddings
known_embeddings = [np.random.rand(1, 2048) for _ in range(5)]  # Replace with actual known embeddings
process_image('sample_image.jpg', known_embeddings, threshold=0.5)