# Part of Chapter 02: Self-Attention Mechanics
# accompanying handbook: handbook/02_self_attention_mechanics.md

# self-attention allows position of sequence to attent to all the other positions in a sequence. 
# there are 4 main steps in self-attention mechanism:
# 1. Create Q, K, V matrices by multiplying input embeddings with weight matrices
# 2. Calculate attention scores by taking dot product of Q and K matrices
# 3. Apply softmax to attention scores to get attention weights
# 4. Multiply attention weights with V matrix to get the context vectors

# simple implementation of self-attention mechanism in python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    # Step 1: Create Q, K, V matrices
    Q = np.dot(X, W_Q)  # Query matrix
    K = np.dot(X, W_K)  # Key matrix
    V = np.dot(X, W_V)  # Value matrix

    # Step 2: Calculate attention scores
    attention_scores = np.dot(Q, K.T)  # Dot product of Q and K

    # Step 3: Apply softmax to get attention weights
    attention_weights = softmax(attention_scores)

    # Step 4: Multiply attention weights with V to get context vectors
    context_vectors = np.dot(attention_weights, V)

    return context_vectors, attention_weights

# example usage
if __name__ == "__main__":
    # Input sequence (4 tokens, embedding size 3)
    X = np.array([[1, 0, 1], # Token 1
                  [0, 2, 0], # Token 2
                  [1, 1, 0], # Token 3
                  [0, 0, 1]] # Token 4
                )

    # Weight matrices (embedding size 3, projection size 3)
    # these weights are usually learned during training of the model 
    # nn.Linear layers in pytorch or dense layers in tensorflow are used to create these weight matrices
    # bias is false because we don't need bias in self-attention mechanism
    W_Q = np.array([[0.2, 0.8, 0.5], 
                    [0.5, 0.1, 0.3], 
                    [0.6, 0.4, 0.9]])

    W_K = np.array([[0.9, 0.3, 0.4], # 
                    [0.2, 0.7, 0.6],
                    [0.5, 0.5, 0.5]])

    W_V = np.array([[0.1, 0.4, 0.7],
                    [0.3, 0.8, 0.2],
                    [0.6, 0.2, 0.9]])   

    W_V = np.array([[0.1, 0.4, 0.7],
                    [0.3, 0.8, 0.2],
                    [0.6, 0.2, 0.9]])

    context_vectors, attention_weights = self_attention(X, W_Q, W_K, W_V)

    print("Context Vectors:\n", context_vectors)
    print("Attention Weights:\n", attention_weights)


# expected output:
'''
Context Vectors:
 [[0.56531134 1.13811989 0.89934018]
 [0.57563696 0.93148965 1.07263733]
 [0.56785393 1.06532249 0.94617092]
 [0.56997773 1.01773361 0.97444593]]
Attention Weights:
 [[0.26672347 0.37473249 0.30680501 0.05173903]
 [0.38218845 0.18978917 0.31290944 0.11511295]
 [0.29156318 0.31584662 0.30651195 0.08607825]
 [0.3063185  0.27995403 0.30327058 0.1104569 ]]
 
 '''

# explanation of output:
# The "Context Vectors" output represents the new representations of each token in the sequence after
# applying the self-attention mechanism. Each vector is a weighted sum of the value vectors (V),
# where the weights are determined by the attention mechanism.
# The "Attention Weights" output shows how much focus each token in the sequence places on every other token.
# Each row corresponds to a token, and the values in that row indicate the importance of each token
# in relation to the token represented by that row. Higher values indicate greater attention paid to that token.
# Note: The actual numerical values may vary slightly due to floating-point precision.
