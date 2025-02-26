# Name: Muhammad Faheem
# Email: i220485@nu.edu.pk

import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset



class BPETokenizer:

    def __init__(self, vocab_size=1000):
        """
        Initialize the tokenizer with a target vocabulary size.
        The token-to-id mapping reserves 0 for the <UNK> token.
        """
        self.vocab_size = vocab_size
        self.merges = []             # List of learned merge operations (order matters)
        self.token2id = {"<UNK>": 0}   # Mapping token -> unique id; 0 reserved for <UNK>
        self.id2token = {0: "<UNK>"}   # Reverse mapping id -> token
        self.current_id = 1          # Next id to assign for new tokens

    def preprocess_text(self, text):
        """
        Preprocess the input text by:
          - Converting to lowercase
          - Removing punctuation
          - Normalizing known variant spellings (Roman Urdu challenges)
        """
        # Lowercase and remove punctuation (keep alphanumerics and whitespace)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        
        # Normalizing spelling variants
        normalization_map = {
            "mei": "mai",
            "main": "mai",
            "kia": "kya",
            "kiya": "kya",
            "mujhay": "mujhe",
            "mujhy": "mujhe"
        }
        norm_tokens = [normalization_map.get(token, token) for token in tokens]

        return norm_tokens

    def get_vocab_from_corpus(self, corpus):
        """
        Build the initial vocabulary from the training corpus.
        Each word is represented as a tuple of its characters plus a trailing '</w>' marker.
        Also, count the frequency of each word.
        """
        vocab = {}
        
        for text in corpus:
            tokens = self.preprocess_text(text)
        
            for token in tokens:
                # Representing the word as a tuple of symbols, with an end-of-word marker.
                word = tuple(token) + ("</w>",)
                vocab[word] = vocab.get(word, 0) + 1
        
        return vocab

    def get_stats(self, vocab):
        """
        Count frequency of each adjacent pair (bigram) in the current vocabulary.
        """
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            # word is a tuple of symbols
        
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        Merge all occurrences of the given pair in the vocabulary.
        For each word (represented as a tuple), scan through and merge whenever the pair is found.
        """
        new_vocab = {}
        
        for word, freq in vocab.items():
            new_word = []
            i = 0
        
            while i < len(word):
                # If the pair is found, merge and skip the next symbol.
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
        
            new_vocab[tuple(new_word)] = new_vocab.get(tuple(new_word), 0) + freq
        
        return new_vocab

    def train(self, corpus):
        """
        Train the BPE model on the provided corpus.
        Steps:
          1. Build initial vocabulary (word frequencies) from the corpus.
          2. Initialize the token mapping for each unique character (and the </w> marker).
          3. Iteratively merge the most frequent adjacent pairs until reaching the target vocabulary size.
        """
        vocab = self.get_vocab_from_corpus(corpus)
        
        # Initialize mapping: assign ids for all initial symbols (characters and '</w>')
        initial_symbols = set()
        for word in vocab:
            
            for symbol in word:
                initial_symbols.add(symbol)
        for symbol in sorted(initial_symbols):
            
            if symbol not in self.token2id:
                self.token2id[symbol] = self.current_id
                self.id2token[self.current_id] = symbol
                self.current_id += 1
        
        # Determining how many merges to perform.
        num_merges = self.vocab_size - len(self.token2id)
        
        # Performing merges iteratively.
        for i in range(num_merges):
            stats = self.get_stats(vocab)
            
            if not stats:
                break
        
            # Selecting the most frequent pair to merge.
            most_freq = max(stats, key=stats.get)
            self.merges.append(most_freq)
        
            # Creating a new token by concatenating the pair.
            new_token = most_freq[0] + most_freq[1]
            
            if new_token not in self.token2id:
                self.token2id[new_token] = self.current_id
                self.id2token[self.current_id] = new_token
                self.current_id += 1
        
            # Merging the vocabulary with the selected pair.
            vocab = self.merge_vocab(most_freq, vocab)
        
        self.vocab = vocab  # Saving the final vocabulary

    def encode_word(self, word):
        """
        Encode a single word (string) into subword tokens using the learned merges.
        The word is first split into characters with a trailing '</w>' marker.
        Then, iteratively, the merge rules (in the order learned) are applied.
        """
        symbols = list(word) + ["</w>"]
        # Iteratively apply merges until no more merges are applicable.
        while True:
            
            # Creating list of adjacent symbol pairs.
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            merge_applied = False
            
            # Applying merge rules in the order they were learned.
            for merge in self.merges:
                
                if merge in pairs:
                    new_symbols = []
                    i = 0
                
                    while i < len(symbols):
                        
                        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == merge:
                            new_symbols.append(symbols[i] + symbols[i + 1])
                            i += 2
                            merge_applied = True
                        else:
                            new_symbols.append(symbols[i])
                            i += 1
                
                    symbols = new_symbols
                    # Restarting from the first merge rule after a merge is applied.
                    break
            
            if not merge_applied:
                break
        
        return symbols

    def encode(self, text):
        """
        Encode an input text (which may consist of several words) into a list of token IDs.
        For each word, the learned BPE merges are applied and then tokens are mapped to their IDs.
        If a token is not found in the mapping, the <UNK> token (ID 0) is used.
        """
        tokens = self.preprocess_text(text)
        encoded_tokens = []
        
        for token in tokens:
            sub_tokens = self.encode_word(token)
            # Mapping each sub-token to its id (using <UNK> if missing)
            
            for sub in sub_tokens:
                token_id = self.token2id.get(sub, self.token2id["<UNK>"])
                encoded_tokens.append(token_id)
        
        return encoded_tokens

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.
        This method reconstructs words by joining tokens and using the '</w>' marker to detect word boundaries.
        """
        tokens = [self.id2token.get(tid, "<UNK>") for tid in token_ids]
        words = []
        current_word = ""
        
        for token in tokens:
        
            # If token ends with '</w>', it signals end-of-word.
            if token.endswith("</w>"):
                current_word += token.replace("</w>", "")
                words.append(current_word)
                current_word = ""
            else:
                current_word += token
        
        if current_word:
            words.append(current_word)
        
        return " ".join(words)

    def evaluate(self, corpus):
        """
        Evaluate the tokenization performance on a test corpus.
        For each diary entry, we encode the text and count the occurrence of <UNK> tokens.
        Returns total token count, <UNK> count, and the ratio.
        """
        total_tokens = 0
        unk_tokens = 0
        
        for text in corpus:
            encoded = self.encode(text)
            total_tokens += len(encoded)
            unk_tokens += encoded.count(self.token2id["<UNK>"])
        
        return {
            "total_tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "unk_ratio": unk_tokens / total_tokens if total_tokens > 0 else 0
        }



if __name__ == "__main__":
    dataset_directory = r"22I-0485_BS-AI-B_NLP-Assignment1\data\dataset"

    print("Loading the corpus:")
    loaded_corpus = load_dataset(dataset_directory)

    train_corpus, test_corpus = train_test_split(loaded_corpus, train_size=0.7, random_state=42, shuffle=True)

    # # For, testing
    # print(f"\nTotal entries: {len(loaded_corpus)}")
    # print(f"Training set size: {len(train_corpus)}")
    # print(f"Testing set size: {len(test_corpus)}")

    # print(type(train_corpus))

    # print("\nSample training entry:", train_corpus[0])
    # print("Sample testing entry:", test_corpus[0])

    # Initializing and train the BPE tokenizer
    bpe_tokenizer = BPETokenizer(vocab_size=1000)

    print("\nTraining...")
    bpe_tokenizer.train(train_corpus)

    # Encoding a sample text
    sample_text = "aaj mei ne kaam kia aur musaam enjoy kiya"
    encoded = bpe_tokenizer.encode(sample_text)
    print("\nEncoded tokens:", encoded)

    # Decoding the token ids back to text
    decoded = bpe_tokenizer.decode(encoded)
    print("\nDecoded text:", decoded)

    # Evaluating the model on the test corpus
    print("\nTesting...")
    eval_results = bpe_tokenizer.evaluate(test_corpus)
    print("\nResults:", eval_results)
