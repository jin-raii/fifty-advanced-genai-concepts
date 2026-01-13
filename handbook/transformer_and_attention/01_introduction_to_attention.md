# 01. Introduction to Attention

Attention mechanisms have revolutionized Natural Language Processing (NLP) and Deep Learning. Before attention, models like RNNs and LSTMs struggled with long-range dependencies because they had to compress the entire input sequence into a fixed-size vector.

## The Problem with Fixed-Size Vectors
In Sequence-to-Sequence (Seq2Seq) models (e.g., for translation), the encoder processes the input sentence and produces a "context vector". If the sentence is very long, this single vector becomes a bottleneck, losing information about the beginning of the sentence by the time it reaches the end.

## The Attention Solution
Attention mechanisms allow the model to "focus" on relevant parts of the input sequence *dynamically* at each step of the output generation. Instead of a single static context vector, the model creates a new context vector for every output token, computed as a weighted sum of all input states.

### Key Concepts
- **Query (Q)**: What we are currently looking for.
- **Key (K)**: What the input offers.
- **Value (V)**: The actual content we will extract.

In the next chapter [02_self_attention_mechanics.md](./02_self_attention_mechanics.md) , we will dive into **Self-Attention**, the core component of the Transformer architecture.
 