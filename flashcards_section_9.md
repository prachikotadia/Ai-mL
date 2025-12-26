# Section 9: NLP Concepts & Tokenization

## 401. Natural Language Processing (NLP)
🟦 **What is NLP (Natural Language Processing)?**

🟩 **Definition**
NLP is the field of making computers understand and generate human language. It includes tasks like translation, search, and chatbots. NLP combines language rules with machine learning.

🟨 **How It Works / Example**
A spam filter reads email text and predicts spam or not spam. It learns patterns like common spam phrases. The same idea supports sentiment analysis and question answering.

🟪 **Quick Tip**
Computers reading text.

---

## 402. Tokenization
🟦 **What is tokenization in NLP?**

🟩 **Definition**
Tokenization splits text into smaller pieces called tokens. Models use tokens instead of raw text. Good tokenization helps models handle many languages and rare words.

🟨 **How It Works / Example**
The text "I can't go." might become tokens like ["I", "can", "'t", "go", "."]. The model then converts tokens to IDs. These IDs become the input sequence.

🟪 **Quick Tip**
Chopping up text.

---

## 403. Token
🟦 **What is a token in NLP models?**

🟩 **Definition**
A token is a unit of text a model processes, like a word or part of a word. Tokens are mapped to numbers for the model. Token choice affects speed, cost, and meaning representation.

🟨 **How It Works / Example**
"unhappiness" might be split into "un", "happi", "ness". This helps the model handle words it has not seen before. The model learns meanings from these parts across many examples.

🟪 **Quick Tip**
Data unit for AI.

---

## 404. Subword Tokenization
🟦 **What is subword tokenization and why is it used?**

🟩 **Definition**
Subword tokenization breaks words into common pieces instead of full words. It reduces unknown words and keeps vocabulary size manageable. It is widely used in modern LLMs.

🟨 **How It Works / Example**
A rare word like "electroencephalogram" can be split into smaller parts the model knows. The model can still process it without an "unknown" token. This improves coverage across languages and domains.

🟪 **Quick Tip**
Breaking words down.

---

## 405. Vocabulary
🟦 **What is a vocabulary in tokenizers?**

🟩 **Definition**
A vocabulary is the list of tokens a tokenizer can produce. Each token has an ID. Vocabulary size affects memory, speed, and how text is split.

🟨 **How It Works / Example**
If a vocabulary has 50,000 tokens, most common words are single tokens. If it's smaller, words split into more subwords. More tokens in a prompt can increase inference cost.

🟪 **Quick Tip**
The model's dictionary.

---

## 406. Unknown Token (UNK)
🟦 **What is an unknown token (UNK) and why does it matter?**

🟩 **Definition**
UNK is a special token used when the tokenizer cannot represent a piece of text. Too many UNKs reduce model understanding. Subword tokenizers reduce UNK usage a lot.

🟨 **How It Works / Example**
Old word-level tokenizers might map rare words to UNK. Then different rare words become the same token and meaning is lost. Subword tokenization avoids this by splitting rare words into known pieces.

🟪 **Quick Tip**
The "I don't know" label.

---

## 407. Byte Pair Encoding (BPE)
🟦 **What is Byte Pair Encoding (BPE)?**

🟩 **Definition**
BPE is a subword tokenization method that merges common character pairs to form tokens. It builds a vocabulary based on frequent patterns. Many LLM tokenizers are based on BPE-like ideas.

🟨 **How It Works / Example**
Start with characters as tokens, then repeatedly merge the most frequent pair like "t" + "h" -> "th". Over many merges, common word pieces become tokens. This creates efficient subword units.

🟪 **Quick Tip**
Merging frequent pairs.

---

## 408. WordPiece
🟦 **What is WordPiece tokenization?**

🟩 **Definition**
WordPiece is a subword method that chooses tokens to best improve likelihood on training data. It is used in models like BERT. It often marks continuation subwords with special symbols.

🟨 **How It Works / Example**
A word like "playing" might become "play" and "##ing". The "##" shows it is a continuation piece. The model combines these to understand the full word meaning.

🟪 **Quick Tip**
Continual splitting.

---

## 409. Unigram Tokenization
🟦 **What is unigram tokenization?**

🟩 **Definition**
Unigram tokenization learns a set of subword tokens and chooses the best split for each word using probabilities. It is used in some SentencePiece setups. It can represent multiple ways to split text.

🟨 **How It Works / Example**
A word may be split as "inter" + "national" or "intern" + "ational" depending on token probabilities. The tokenizer picks the most likely segmentation. This flexibility can help across languages.

🟪 **Quick Tip**
Probabilistic splitting.

---

## 410. SentencePiece
🟦 **What is SentencePiece and why is it popular?**

🟩 **Definition**
SentencePiece is a tokenizer tool that treats text as a stream of characters and learns subwords. It works well without requiring pre-splitting on spaces. It supports many languages, including those without spaces.

🟨 **How It Works / Example**
For Japanese or Chinese, words are not separated by spaces, which makes classic tokenizers harder. SentencePiece learns subword units directly from raw text. This makes multilingual modeling easier.

🟪 **Quick Tip**
Universal tokenizer.

---

## 411. Whitespace Tokenization
🟦 **What is whitespace tokenization?**

🟩 **Definition**
Whitespace tokenization splits text only by spaces. It is simple but weak for many languages and punctuation. It often creates too many unknown words.

🟨 **How It Works / Example**
"can't" stays as one token, and "hello!" stays as "hello!" including punctuation. Rare words become hard to handle. Modern NLP mostly uses subword tokenization instead.

🟪 **Quick Tip**
Split on spaces.

---

## 412. Stemming
🟦 **What is stemming in NLP?**

🟩 **Definition**
Stemming reduces words to a rough root form. It is a rule-based method and can produce non-real words. It helps match related word forms in search and classic NLP.

🟨 **How It Works / Example**
"connected," "connecting," and "connection" might become "connect". A search engine can match more results using stems. But stemming can sometimes harm meaning if done too aggressively.

🟪 **Quick Tip**
Chopping off endings.

---

## 413. Lemmatization
🟦 **What is lemmatization in NLP?**

🟩 **Definition**
Lemmatization reduces words to a dictionary form called a lemma. It uses language rules and usually produces real words. It is more accurate than stemming but slower.

🟨 **How It Works / Example**
"better" becomes "good" and "running" becomes "run". This helps grouping word forms with the same meaning. It is useful in classic text processing pipelines.

🟪 **Quick Tip**
Finding the dictionary root.

---

## 414. Stop Word
🟦 **What is a stop word in NLP?**

🟩 **Definition**
Stop words are very common words like "the," "is," and "and." In some tasks, removing them can reduce noise. In modern neural NLP, they are often kept because context matters.

🟨 **How It Works / Example**
In keyword search, removing stop words can improve matching. But in sentiment tasks, words like "not" are critical and should not be removed. You choose based on the task.

🟪 **Quick Tip**
Common filler words.

---

## 415. Bag-of-Words (BoW)
🟦 **What is a bag-of-words (BoW) representation?**

🟩 **Definition**
BoW represents text by counting word occurrences, ignoring word order. It creates a sparse vector over a vocabulary. It is simple but misses context and meaning.

🟨 **How It Works / Example**
"dog bites man" and "man bites dog" have the same BoW counts. A classifier using BoW cannot tell the difference in meaning. This is why order-aware models like transformers help.

🟪 **Quick Tip**
Word soup.

---

## 416. TF-IDF
🟦 **What is TF-IDF and why is it used?**

🟩 **Definition**
TF-IDF weights words based on how common they are in a document and how rare they are across documents. It highlights words that are important for a specific document. It is common in search and classic text classification.

🟨 **How It Works / Example**
In a news article, the word "the" appears often but gets low TF-IDF because it's common everywhere. A word like "earthquake" gets high TF-IDF if it's rare across the corpus. This helps retrieval and categorization.

🟪 **Quick Tip**
Rarity score.

---

## 417. N-gram
🟦 **What is an n-gram in NLP?**

🟩 **Definition**
An n-gram is a sequence of n tokens, like 2-grams (bigrams) or 3-grams (trigrams). N-grams capture some local word order. They are used in classic language models and features.

🟨 **How It Works / Example**
Bigrams in "New York City" are "New York" and "York City." These phrases can help detect topics or named entities. N-grams improve over BoW by adding local order.

🟪 **Quick Tip**
Word clusters.

---

## 418. Language Model
🟦 **What is a language model in NLP?**

🟩 **Definition**
A language model predicts the next token (or missing token) in text. It learns patterns of language from data. LLMs are large neural language models.

🟨 **How It Works / Example**
Given "I'm going to the," a language model predicts likely next words like "store" or "park." It assigns probabilities to many possible tokens. This is the foundation for text generation.

🟪 **Quick Tip**
Predicting text.

---

## 419. Perplexity
🟦 **What is perplexity in language modeling?**

🟩 **Definition**
Perplexity measures how well a language model predicts tokens. Lower perplexity means the model is less "surprised" by the test text. It is mainly used for base models, not instruction quality.

🟨 **How It Works / Example**
If the model assigns high probability to the true next tokens, perplexity drops. If it often assigns low probability, perplexity increases. Two models can have similar perplexity but different chat usefulness.

🟪 **Quick Tip**
Confusion score.

---

## 420. Sequence Labeling
🟦 **What is a sequence labeling task in NLP?**

🟩 **Definition**
Sequence labeling assigns a label to each token in a sequence. Examples include named entity recognition and part-of-speech tagging. It requires understanding both token meaning and context.

🟨 **How It Works / Example**
In "Apple released a phone," sequence labeling can tag "Apple" as an organization. The model outputs a label for each token. Transformers do this by producing contextual embeddings per token.

🟪 **Quick Tip**
Tagging every word.

---

## 421. Named Entity Recognition (NER)
🟦 **What is named entity recognition (NER)?**

🟩 **Definition**
NER finds and labels entities like people, places, and organizations in text. It is important for search, analytics, and information extraction. It is a common NLP interview topic.

🟨 **How It Works / Example**
In "Barack Obama visited Paris," NER tags "Barack Obama" as PERSON and "Paris" as LOCATION. A model learns patterns from labeled examples. The output can feed into knowledge graphs or indexing.

🟪 **Quick Tip**
Spotting nouns.

---

## 422. Part-of-Speech (POS) Tagging
🟦 **What is part-of-speech (POS) tagging?**

🟩 **Definition**
POS tagging assigns grammar labels like noun, verb, and adjective to each word. It helps understand sentence structure. It is used in classic NLP and some downstream pipelines.

🟨 **How It Works / Example**
In "She runs fast," "runs" is tagged as a verb and "fast" as an adverb. A model predicts tags using context. POS tags can help parsing and rule-based systems.

🟪 **Quick Tip**
Grammar labeling.

---

## 423. Dependency Parsing
🟦 **What is dependency parsing in NLP?**

🟩 **Definition**
Dependency parsing finds relationships between words, like which word is the subject or object. It builds a tree of connections. It helps deeper understanding of sentence structure.

🟨 **How It Works / Example**
In "The dog chased the cat," parsing links "dog" as subject of "chased" and "cat" as object. These links help extract who did what. This is useful for information extraction.

🟪 **Quick Tip**
Sentence structure tree.

---

## 424. Sentiment Analysis
🟦 **What is sentiment analysis?**

🟩 **Definition**
Sentiment analysis predicts whether text expresses positive, negative, or neutral feeling. It is widely used in reviews and social media monitoring. It can be done with classic models or transformers.

🟨 **How It Works / Example**
A model reads "This product is amazing" and predicts positive. It learns from labeled examples. Companies use it to track customer satisfaction trends.

🟪 **Quick Tip**
Mood detector.

---

## 425. Text Classification
🟦 **What is text classification in NLP?**

🟩 **Definition**
Text classification assigns one or more labels to a text piece, like topic or intent. It is used for spam detection, routing, and content moderation. It relies on good features or embeddings.

🟨 **How It Works / Example**
A support system classifies tickets as "billing," "technical," or "shipping." The model reads the ticket text and outputs a label. That label decides which team gets the ticket.

🟪 **Quick Tip**
Sorting text.

---

## 426. Multi-Label Classification
🟦 **What is multi-label classification in NLP?**

🟩 **Definition**
Multi-label classification allows a text to have multiple labels at once. This differs from multi-class where only one label is allowed. It is common in tagging systems.

🟨 **How It Works / Example**
A news article can be tagged as both "politics" and "economy." The model outputs independent probabilities for each label. You choose labels above a threshold.

🟪 **Quick Tip**
Many tags allowed.

---

## 427. Text Similarity
🟦 **What is text similarity in NLP?**

🟩 **Definition**
Text similarity measures how close two texts are in meaning. It is used in search, deduplication, and retrieval. It often uses embeddings and distance metrics.

🟨 **How It Works / Example**
You embed two sentences into vectors and compute cosine similarity. High similarity means they talk about the same idea. This helps find duplicate support tickets or similar documents.

🟪 **Quick Tip**
Measuring closeness.

---

## 428. Semantic Search
🟦 **What is semantic search?**

🟩 **Definition**
Semantic search finds results based on meaning, not just exact keywords. It uses embeddings to represent queries and documents. It often improves search quality for natural language questions.

🟨 **How It Works / Example**
A user searches "how to cancel my plan," and results include a page titled "terminate subscription." Keyword search might miss it, but embeddings match the meaning. A vector database returns the closest documents.

🟪 **Quick Tip**
Searching by meaning.

---

## 429. Embedding-Based Retriever
🟦 **What is an embedding-based retriever?**

🟩 **Definition**
An embedding-based retriever finds relevant texts by comparing vector embeddings. It supports semantic matching beyond exact words. It is a core part of RAG systems.

🟨 **How It Works / Example**
You embed the user question and all document chunks. Then you search for nearest neighbors in vector space. The top chunks are given to the LLM to answer.

🟪 **Quick Tip**
Vector search engine.

---

## 430. Special Token
🟦 **What is a tokenizer "special token"?**

🟩 **Definition**
A special token is a reserved token with a specific meaning, like padding or end-of-sequence. It helps control model behavior and formatting. Different models use different special tokens.

🟨 **How It Works / Example**
A model may use an EOS token to know when to stop generating. PAD tokens fill shorter sequences in a batch. Chat models may have special role tokens for system and user messages.

🟪 **Quick Tip**
Control signals.

---

## 431. Padding
🟦 **What is padding in NLP batching?**

🟩 **Definition**
Padding adds extra tokens to make sequences in a batch the same length. It helps run efficient matrix operations. Padding tokens should be masked so they do not affect attention.

🟨 **How It Works / Example**
If one sentence has 5 tokens and another has 8, you pad the first to 8 with PAD tokens. An attention mask prevents the model from attending to PAD. This keeps results correct while enabling batching.

🟪 **Quick Tip**
Filling the gaps.

---

## 432. Truncation
🟦 **What is truncation in tokenization?**

🟩 **Definition**
Truncation cuts off tokens when text is longer than a max length. It is used to fit model context limits. Truncation can remove important information if done blindly.

🟨 **How It Works / Example**
If a model limit is 512 tokens, longer documents are cut to 512. You might keep the start, the end, or a smart selection. For long documents, chunking is often better than truncation.

🟪 **Quick Tip**
Cutting to fit.

---

## 433. Max Sequence Length
🟦 **What is a "max sequence length" in NLP models?**

🟩 **Definition**
Max sequence length is the maximum number of tokens a model can process in one pass. It is limited by architecture and memory. It affects training and inference cost.

🟨 **How It Works / Example**
A BERT model might use 512 tokens. If your text is longer, you split it into multiple segments. You then combine results with pooling or a separate aggregation step.

🟪 **Quick Tip**
Input size limit.

---

## 434. Byte-Level Tokenization
🟦 **What is byte-level tokenization?**

🟩 **Definition**
Byte-level tokenization works directly on bytes rather than characters or words. It can represent any text without UNK tokens. It is common in some GPT-style tokenizers.

🟨 **How It Works / Example**
Even rare symbols or emojis can be represented as bytes. The tokenizer merges byte patterns into tokens. This makes coverage strong but can increase token count for some text.

🟪 **Quick Tip**
Bytes as atoms.

---

## 435. Unicode Normalization
🟦 **What is Unicode normalization and why does it matter in NLP?**

🟩 **Definition**
Unicode normalization makes text consistent when the same character can be written in different ways. It reduces weird tokenization differences. It is important for multilingual and user-generated text.

🟨 **How It Works / Example**
Some accented characters can be stored as one symbol or as a base letter plus an accent mark. Normalization converts them to a standard form. This helps tokenization and matching behave consistently.

🟪 **Quick Tip**
Standardizing symbols.

---

## 436. Text Normalization
🟦 **What is text normalization in NLP pipelines?**

🟩 **Definition**
Text normalization is cleaning and standardizing text before modeling. It can include lowercasing, removing extra spaces, and handling punctuation. The right normalization depends on the task.

🟨 **How It Works / Example**
For search, you might lowercase and remove repeated spaces. For sentiment, you might keep punctuation because "!!!" can matter. For LLMs, you often keep raw text and rely on tokenizer rules.

🟪 **Quick Tip**
Cleaning text.

---

## 437. Out-of-Vocabulary (OOV)
🟦 **What is out-of-vocabulary (OOV) and how do tokenizers handle it?**

🟩 **Definition**
OOV means a word is not in the tokenizer vocabulary as a single token. Subword tokenizers handle OOV by splitting into smaller known parts. This keeps coverage high.

🟨 **How It Works / Example**
A new product name might not exist as one token. The tokenizer breaks it into parts like "Mega" + "Phone" + "X." The model can still process it and learn usage from context.

🟪 **Quick Tip**
Unknown words.

---

## 438. Detokenization
🟦 **What is detokenization?**

🟩 **Definition**
Detokenization converts tokens back into readable text. It handles joining subwords, spaces, and punctuation correctly. Good detokenization is important for clean outputs.

🟨 **How It Works / Example**
If tokens are "play" and "##ing," detokenization merges them into "playing." If tokens include space markers, it restores spacing. This is how model outputs become human-readable strings.

🟪 **Quick Tip**
Reassembling text.

---

## 439. Language Pair
🟦 **What is a language pair in machine translation?**

🟩 **Definition**
A language pair is the source and target languages for translation, like English->Spanish. Translation models often behave differently across pairs. Tokenization and data quality matter a lot.

🟨 **How It Works / Example**
A model trained on English->French learns mappings and grammar for that pair. For low-resource pairs, quality may be worse due to less data. Multilingual models share vocabulary and parameters across many pairs.

🟪 **Quick Tip**
Source and target.

---

## 440. BLEU Score
🟦 **What is BLEU score in NLP evaluation?**

🟩 **Definition**
BLEU measures how close a generated translation is to a reference translation using n-gram overlap. It is common for machine translation. It does not fully capture meaning or fluency.

🟨 **How It Works / Example**
If the model uses similar word sequences as the reference, BLEU is higher. If it uses different wording with the same meaning, BLEU may still be low. That's why people also use human evaluation or newer metrics.

🟪 **Quick Tip**
N-gram overlap score.

---

## 441. ROUGE Score
🟦 **What is ROUGE score in NLP evaluation?**

🟩 **Definition**
ROUGE measures overlap between generated text and reference text, often used for summarization. It focuses on recall of n-grams. Like BLEU, it can miss meaning if wording differs.

🟨 **How It Works / Example**
A summary that shares many phrases with the reference gets a high ROUGE score. A good paraphrase may score lower. Teams often use ROUGE plus human judgment.

🟪 **Quick Tip**
Recall overlap score.

---

## 442. Exact Match (EM)
🟦 **What is exact match (EM) in question answering evaluation?**

🟩 **Definition**
Exact match checks if the predicted answer matches the reference answer exactly. It is strict and easy to compute. It can be too harsh for answers with valid paraphrases.

🟨 **How It Works / Example**
If the reference is "Paris" and the model outputs "Paris," EM is 1. If it outputs "The capital is Paris," EM might be 0. Some datasets also use F1 to allow partial matches.

🟪 **Quick Tip**
Strict equality.

---

## 443. F1 Score
🟦 **What is F1 score used for in NLP tasks?**

🟩 **Definition**
F1 combines precision and recall into one number. It is used for tasks like NER and QA span matching. It is helpful when both false positives and false negatives matter.

🟨 **How It Works / Example**
In NER, precision measures how many predicted entities are correct, and recall measures how many real entities were found. F1 balances them. If you miss many entities, recall drops and F1 drops too.

🟪 **Quick Tip**
Balanced accuracy metric.

---

## 444. Data Augmentation
🟦 **What is data augmentation in NLP?**

🟩 **Definition**
Data augmentation creates extra training examples from existing text. It helps reduce overfitting and improves robustness. It must be done carefully to avoid changing meaning.

🟨 **How It Works / Example**
You can paraphrase sentences or swap synonyms. For classification, labels stay the same if meaning stays the same. This helps the model handle more writing styles.

🟪 **Quick Tip**
Multiplying data.

---

## 445. Back-Translation
🟦 **What is back-translation in NLP?**

🟩 **Definition**
Back-translation creates new training data by translating text to another language and back. It generates paraphrases naturally. It is used in translation and other NLP tasks.

🟨 **How It Works / Example**
You take English text, translate it to French, then translate it back to English. The result is a paraphrase with the same meaning. This adds variety to training data.

🟪 **Quick Tip**
Translate there and back.

---

## 446. Vocabulary Mismatch
🟦 **What is a vocabulary mismatch problem in NLP?**

🟩 **Definition**
Vocabulary mismatch happens when user wording differs from document wording. Keyword methods may fail because they rely on exact matches. Embeddings reduce this issue by matching meaning.

🟨 **How It Works / Example**
A user asks "refund," but the document says "reimbursement." Keyword search may miss it. Embedding search can still match because meanings are close. This is why semantic search is useful.

🟪 **Quick Tip**
Different words, same meaning.

---

## 447. Word Sense Disambiguation (WSD)
🟦 **What is word sense disambiguation (WSD)?**

🟩 **Definition**
WSD is identifying which meaning of a word is used in context. Many words have multiple meanings. Correct disambiguation improves understanding.

🟨 **How It Works / Example**
"Bank" could mean a river bank or a financial bank. The words around it give context. Modern transformers handle this by creating different embeddings based on context.

🟪 **Quick Tip**
Which definition?

---

## 448. Coreference Resolution
🟦 **What is coreference resolution?**

🟩 **Definition**
Coreference resolution finds when different words refer to the same thing. It connects pronouns and nouns to their real references. This improves understanding of who or what is being discussed.

🟨 **How It Works / Example**
In "Alice said she would come," "she" refers to "Alice." A model or pipeline links these mentions. This helps tasks like summarization and information extraction.

🟪 **Quick Tip**
Linking pronouns.

---

## 449. Token Budget
🟦 **What is a "token budget" in LLM applications?**

🟩 **Definition**
Token budget is the maximum tokens you can spend on prompt plus output. It affects cost and latency. Managing token budget is important in production systems.

🟨 **How It Works / Example**
If your model allows 8,000 tokens total, a 7,500-token prompt leaves little room for output. Teams shorten prompts, summarize history, or retrieve only relevant chunks. This keeps responses complete and affordable.

🟪 **Quick Tip**
Spending tokens wisely.

---

## 450. Token-Level vs Word-Level
🟦 **What is "token-level vs word-level" modeling and why does it matter?**

🟩 **Definition**
Token-level modeling predicts tokens, which may be subwords, not full words. Word-level modeling predicts whole words. Token-level modeling handles rare words better and supports many languages more easily.

🟨 **How It Works / Example**
With token-level models, "unhappiness" can be predicted as smaller parts. A word-level model might treat it as unknown. Token-level modeling is why modern LLMs can handle new words and names better.

🟪 **Quick Tip**
Subwords vs whole words.
