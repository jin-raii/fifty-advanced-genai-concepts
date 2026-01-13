# Part of Chapter 01: Attention Basics
# accompanying handbook: handbook/01_introduction_to_attention.md

import numpy as np

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def dot_product_attention(Q, K, V):
    """
    Calculate dot product attention.
    
    Args:
        Q: Query matrix
        K: Key matrix
        V: Value matrix
        
    Returns:
        context_vectors: The weighted sum of V
        attention_weights: The attention weights
    """
    # 1. Calculate attention scores (Q dot K^T)
    # We transpose K to align dimensions for dot product
    # If d_k is the dimension of keys, dividing by sqrt(d_k) (scaled dot product) 
    # is often done, but here we implement the basic version.
    scores = np.dot(Q, K.T)
    
    # 2. Apply softmax to get attention weights
    weights = softmax(scores)
    
    # 3. Multiply weights by V to get context vectors
    output = np.dot(weights, V)
    
    return output, weights

# Example usage
if __name__ == "__main__":
    # Example Dimensions
    # Batch size = 1 (processed as single matrices here)
    # Sequence length = 3
    # Embedding info = 4
    
    np.random.seed(42)
    
    # Random input matrices
    Q = np.random.rand(3, 4)
    K = np.random.rand(3, 4)
    V = np.random.rand(3, 4)
    
    print("Query Matrix:\n", Q)
    print("\nKey Matrix:\n", K)
    print("\nValue Matrix:\n", V)
    
    context, attn_weights = dot_product_attention(Q, K, V)
    
    print("\nAttention Weights:\n", attn_weights)
    print("\nContext Vectors:\n", context)
