from sklearn.metrics.pairwise import cosine_similarity

def recognize_face(known_embeddings, test_embedding, threshold=0.5):
    # Calculate cosine similarity between known and test embeddings
    similarities = cosine_similarity(known_embeddings, test_embedding)
    max_similarity = np.max(similarities)
    
    if max_similarity > threshold:
        return True, max_similarity
    else:
        return False, max_similarity