The **BPE tokenizer** in your code follows these steps:

### **1. Preprocessing (`preprocess_text`)**
- Converts text to lowercase.
- Removes punctuation.
- Normalizes some Roman Urdu spelling variations.

### **2. Building Vocabulary (`get_vocab_from_corpus`)**
- Converts words into sequences of characters with an end-of-word marker (`</w>`).
- Counts the frequency of each unique word.

### **3. Finding Most Frequent Character Pairs (`get_stats`)**
- Identifies which character pairs appear most frequently in the vocabulary.

### **4. Merging the Most Frequent Pair (`merge_vocab`)**
- Merges the most frequent character pair into a new token.
- Repeats until the vocabulary reaches the desired size.

### **5. Training the Tokenizer (`train`)**
- Initializes character-level tokens and their IDs.
- Iteratively applies merges to form multi-character tokens.
- Stores the merge rules for later tokenization.

### **6. Encoding (`encode_word`, `encode`)**
- Splits words into characters.
- Applies learned merges to generate subword tokens.
- Maps subword tokens to unique IDs.

### **7. Decoding (`decode`)**
- Converts token IDs back into text.

### **8. Evaluation (`evaluate`)**
- Encodes a test set and calculates the proportion of `<UNK>` tokens.

<!-- --- -->
<br>

## **Example Walkthrough**

Let's take a small example to understand how the tokenizer works.

### **Input Corpus**
Suppose we have a training corpus with just **two sentences**:
```plaintext
mujhay acha laga
mujhy acha laga
```

---

### **Step 1: Preprocessing (`preprocess_text`)**
The text is converted to lowercase, punctuation is removed, and Roman Urdu spellings are normalized.

**Before Normalization:**
```plaintext
mujhay acha laga
mujhy acha laga
```
**After Normalization:**
```plaintext
mujhe acha laga
mujhe acha laga
```

---

### **Step 2: Building Initial Vocabulary (`get_vocab_from_corpus`)**
Each word is converted into **characters with an end-of-word marker (`</w>`)**.

| Word  | Character Representation |
|--------|-----------------------------|
| mujhe | (m, u, j, h, e, `</w>`) |
| acha | (a, c, h, a, `</w>`) |
| laga | (l, a, g, a, `</w>`) |

The **initial vocabulary** is:
```python
{
    ('m', 'u', 'j', 'h', 'e', '</w>'): 2,
    ('a', 'c', 'h', 'a', '</w>'): 2,
    ('l', 'a', 'g', 'a', '</w>'): 2
}
```

---

### **Step 3: Finding Most Frequent Character Pairs (`get_stats`)**
The frequency of adjacent character pairs is computed:

| Character Pair | Frequency |
|--------------|-----------|
| ('m', 'u')  | 2 |
| ('u', 'j')  | 2 |
| ('j', 'h')  | 2 |
| ('h', 'e')  | 2 |
| ('e', '</w>') | 2 |
| ('a', 'c')  | 2 |
| ('c', 'h')  | 2 |
| ('h', 'a')  | 2 |
| ('a', '</w>') | 2 |
| ('l', 'a')  | 2 |
| ('a', 'g')  | 2 |
| ('g', 'a')  | 2 |

---

### **Step 4: Merging Pairs (`merge_vocab`)**
The **most frequent pair is merged first**. Let's assume we start by merging **("m", "u") → "mu"**.

Updated Vocabulary:
```python
{
    ('mu', 'j', 'h', 'e', '</w>'): 2,
    ('a', 'c', 'h', 'a', '</w>'): 2,
    ('l', 'a', 'g', 'a', '</w>'): 2
}
```

We **repeat this process** until we reach the desired vocabulary size.

Final learned **merges** (example sequence):
1. ("m", "u") → "mu"
2. ("mu", "j") → "muj"
3. ("muj", "h") → "mujh"
4. ("mujh", "e") → "mujhe"
5. ("a", "c") → "ac"
6. ("ac", "h") → "ach"
7. ("ach", "a") → "acha"
8. ("l", "a") → "la"
9. ("la", "g") → "lag"
10. ("lag", "a") → "laga"

Now, our vocabulary consists of:
```python
{
    ('mujhe', '</w>'): 2,
    ('acha', '</w>'): 2,
    ('laga', '</w>'): 2
}
```

---

### **Step 5: Encoding a New Sentence (`encode`)**
Suppose we encode:
```plaintext
"mujhe acha laga"
```

The word is **split** into subword tokens using the learned merges:
```
['mujhe</w>', 'acha</w>', 'laga</w>']
```

Each token is mapped to an **ID**:
```python
{
    'mujhe</w>': 1,
    'acha</w>': 2,
    'laga</w>': 3
}
```

**Final encoded output (IDs):**
```plaintext
[1, 2, 3]
```

---

### **Step 6: Decoding (`decode`)**
The IDs `[1, 2, 3]` are mapped back to their tokens:
```plaintext
"mujhe acha laga"
```
Since we used the `</w>` marker to identify word boundaries, **we recover the original sentence correctly**.

---

### **Step 7: Evaluating Performance (`evaluate`)**
The **evaluation function** encodes test sentences and counts the number of `<UNK>` tokens (unknown words).

If the test sentence **contains only known tokens**, the `<UNK>` count is **zero**. Otherwise, it increases.

Example **test set**:
```plaintext
mujhe acha bohot laga
```
- "bohot" is **not in the learned vocabulary**, so it maps to `<UNK>`.

Evaluation result:
```python
{
    'total_tokens': 4,
    'unk_tokens': 1,
    'unk_ratio': 0.25  # (1 unknown token out of 4)
}
```

<!-- --- -->
<br>

## **Final Thoughts**
1. **Preprocess text** (lowercasing, removing punctuation, normalizing).
2. **Build character-level vocabulary** from a corpus.
3. **Find and merge the most frequent character pairs** iteratively.
4. **Train and store merge rules**.
5. **Encode and decode text** based on the learned merges.
6. **Evaluate unknown token ratio** on test data.

This method is useful for handling **out-of-vocabulary (OOV) words**, compressing vocabulary, and improving generalization in NLP models.