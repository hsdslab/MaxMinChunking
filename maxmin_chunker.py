import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def process_sentences(sentences, embeddings, fixed_threshold=0.6, c=0.9, init_constant=1.5):
    """
    Process sentences into paragraphs based on semantic similarity.

    Args:
    - sentences (list of str): List of sentences to process.
    - embeddings (np.array): Sentence embeddings of shape (n_sentences, embedding_dim).
    - fixed_threshold (float): Fixed similarity threshold for joining sentences.
    - c (float): Coefficient for adjusting the similarity threshold.
    - init_constant (float): Initial constant for similarity comparison when cluster size is 1.

    Returns:
    - list of list of str: List of paragraphs, where each paragraph is a list of sentences.
    """
    
    def sigmoid(x):
        """Sigmoid function for adjusting threshold based on cluster size."""
        return 1 / (1 + np.exp(-x))

    paragraphs = []
    current_paragraph = [sentences[0]]
    cluster_start, cluster_end = 0, 1
    pairwise_min = -float('inf')

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]

            # Adjust threshold based on cluster size and similarity
            adjusted_threshold = pairwise_min * c * sigmoid((cluster_end - cluster_start) - 1)
            new_sentence_similarity = np.max(new_sentence_similarities)
            
            # Use the minimum of the minimum similarities and the pairwise_min
            pairwise_min = min(np.min(new_sentence_similarities), pairwise_min)
        else:
            adjusted_threshold = 0
            # Use an initial constant when there's only one sentence in the cluster
            pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
            new_sentence_similarity = init_constant * pairwise_min

        # Decide whether to add the sentence to the current paragraph or start a new one
        if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float('inf')

    # Append the last paragraph
    paragraphs.append(current_paragraph)
    return paragraphs
