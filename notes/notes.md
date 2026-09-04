# Notes

## Chapter 1

Data preprocessing → pre-training → fine-tuning

Pre-training - training on a large unlabeled dataset by next-word prediction
→ the label is just the next word already sitting in the text, so nobody has to label anything
    → this is what "self-supervised" means

Fine-tuning - further training on a smaller labeled dataset
→ instruction fine-tuning, where the labels are prompt and response pairs
→ classification fine-tuning, where the labels are classes like spam / not spam

Encoder - turns text into vectors
Decoder - turns vectors into text
→ the original transformer had both
    → GPT is decoder only, BERT is encoder only
→ BERT predicts masked words anywhere in the sentence, GPT only ever predicts the next one

## Chapter 2

Word embeddings - mapping words into continuous vectors

Tokenizing - splitting text into individual tokens
→ each token has its own ID
→ special tokens
    → `<|unk|>` for words not in the vocab
    → `<|endoftext|>` to separate documents

Byte pair encoding - builds subword tokens by repeatedly merging the most common adjacent pair
→ eg. `d` + `e` → `de`, which then covers define, depend, made
→ the vocab is a merge table, `{byte sequence: rank}`
    → the rank is both the token ID and the order the merge was learned
    → low rank means learned early, which means common
    → GPT-2 vocab is $256 + 50{,}000 + 1 = 50{,}257$, being single bytes + merges + `<|endoftext|>`
→ encoding starts from the raw bytes and keeps merging the adjacent pair with the lowest rank
    → it stops when no adjacent pair is left in the table
    → it only ever merges neighbours, so it is not choosing between different spellings of a word
    → eg. `unbelievable` → `un` + `bel` + `iev` + `able`
        → but `" unbelievable"` with the leading space is one single token, ID 24479
        → the space-prefixed form was common enough to get merged whole
→ no `<|unk|>` is needed
    → worst case it falls back to single bytes, so any string can be encoded

Sliding window - training pairs made by sliding a window along the token stream
→ the target is the input shifted along by one
→ `max_length` is the window size, `stride` is how far the window moves each step

Token embeddings - a lookup table from token ID to vector

Encoding word positions - add a position embedding on top of the token embedding

$$\text{input}_i = E_{\text{tok}}[\text{id}_i] + E_{\text{pos}}[i]$$

→ the position table is a second lookup table
    → keyed on the slot index $0 \dots L-1$, not on the token ID
→ the same word in two different positions ends up with two different vectors
→ without it attention has no sense of order at all, since it is only ever a weighted sum

## Chapter 3

Attention - lets the decoder reach all of the encoder's hidden states, not only the final one

Self-attention - relates positions inside a single sequence to each other
→ each word picks up context from the rest of the sentence
→ attention scores - dot product of the query word against every word, $\omega_{ij} = x_i \cdot x_j$
    → the dot product measures how similar two vectors are
→ attention weights - the scores put through softmax, $\alpha_{ij} = \text{softmax}_j(\omega_{ij})$
    → all positive and summing to 1, so the context vector comes out as a weighted average
→ context vector - the weighted sum of every word vector, $z_i = \sum_j \alpha_{ij}\, x_j$
    → each word's vector now carries the context around it

### Scaled dot-product attention

Self-attention with trainable weights. Every word is multiplied by three learned matrices, which gives it three different jobs to do:

→ query, $q = xW_q$ - what am I looking for?
→ key, $k = xW_k$ - what am I, in case someone is looking for me?
→ value, $v = xW_v$ - what I hand over if someone picks me

Think of searching a library:
→ the query is what you type into the search box
→ the key is the title printed on the spine
→ the value is the text inside the book
→ you match what you typed against the titles, never against the full contents
    → then you read the contents of whatever matched
    → so the thing you compare against and the thing you walk away with are deliberately different

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

→ $q \cdot k$ decides how much, $v$ decides what
→ $W_q, W_k, W_v$ are learned and then frozen after training
    → the attention weights themselves are recomputed fresh for every input sequence
→ divide the scores by $\sqrt{d_k}$ before the softmax
    → $\text{Var}(q \cdot k)$ grows with $d_k$, so wide models produce huge scores
    → huge scores push softmax towards one-hot, which flattens the gradient and stalls learning
→ the three matrices also fix two things the plain version cannot do
    → without them, comparing A to B gives the exact same number as comparing B to A
        → but "red" needs "car" far more than "car" needs "red", and one number cannot say both
    → without them, a word compared to itself scores $x \cdot x = \lVert x \rVert^2$, its own maximum
        → so it mostly listens to itself, which tells it nothing new
        → $W_q$ and $W_k$ send the same word to two different places, so $q_i \cdot k_i$ is no longer a self dot product

### Causal attention

Masks the words after the current one, so the model cannot see the answer it is being asked to predict

→ two ways to do it, same result
    → set the upper triangle to $-\infty$ and then softmax
    → or softmax first, zero the upper triangle, then renormalise each row
    → the $-\infty$ version is the one worth keeping
→ masking leaves a lower triangular matrix
    → row $i$ is the context for the prefix ending at word $i$
    → so one forward pass over $n$ words gives $n$ training examples
→ dropout - randomly zeroes some attention weights during training
    → stops the model leaning on specific positions
    → survivors are scaled by $\frac{1}{1-p}$ so the totals stay comparable
    → training only, switched off at inference
    → applied to the attention weights after softmax, so after the causal mask
    → prevents overfitting 

### Multi-head attention

Several attention heads running in parallel, each free to learn a different kind of relation

→ the outputs are concatenated and passed through one final output projection
→ in practice there is a single $W_q, W_k, W_v$ of width `d_out`, split across the heads
    → head $i$ owns its own slice of the columns
    → same maths, far fewer matmuls
→ `d_out` must divide evenly by `num_heads`
    → and the $d_k$ inside the $\sqrt{\,}$ is the per-head width, not the full width


