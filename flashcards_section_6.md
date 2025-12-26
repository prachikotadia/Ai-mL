# Section 6: Transformers & Attention

## 251. Attention
🟦 **What is attention in deep learning?**

🟩 **Definition**
Attention is a method that helps a model focus on the most relevant parts of the input. It assigns weights to different tokens or positions. This helps the model handle long inputs better.

🟨 **How It Works / Example**
In a sentence, attention can make the word "it" focus on the earlier noun it refers to. The model gives higher weight to that noun's token. This improves understanding and prediction.

🟪 **Quick Tip**
Spotlight on what matters.

---

## 252. Self-Attention
🟦 **What is self-attention?**

🟩 **Definition**
Self-attention is attention where a sequence attends to itself. Each token looks at other tokens to gather context. This is the core mechanism inside transformers.

🟨 **How It Works / Example**
When processing "The dog chased the cat," the token "chased" can attend to "dog" and "cat." This helps the model understand who did what. The result is a better token representation.

🟪 **Quick Tip**
Context from within.

---

## 253. Transformer Model
🟦 **What is a transformer model?**

🟩 **Definition**
A transformer is a neural network built mainly from attention and feedforward layers. It processes tokens in parallel, unlike RNNs. Transformers power most modern NLP and LLM systems.

🟨 **How It Works / Example**
A transformer reads all tokens at once and computes attention between them. This captures relationships across the whole input. Then it produces outputs like next-token predictions or classifications.

🟪 **Quick Tip**
Parallel processing powerhouse.

---

## 254. Attention is All You Need
🟦 **What is the main idea behind "Attention is All You Need"?**

🟩 **Definition**
The main idea is that attention can replace recurrence and convolutions for sequence modeling. This makes training more parallel and often more effective. It led to the transformer architecture.

🟨 **How It Works / Example**
Instead of reading tokens one-by-one like an RNN, the model uses attention to connect any tokens directly. This speeds training on GPUs. It also helps learn long-range dependencies.

🟪 **Quick Tip**
Goodbye RNNs.

---

## 255. Query (Q)
🟦 **What is the query (Q) in attention?**

🟩 **Definition**
A query is a vector that represents what a token is "looking for." It is used to score relevance against keys. Queries are learned projections of token representations.

🟨 **How It Works / Example**
When token A wants context, its query compares with keys from all tokens. Higher similarity means more attention weight. The token then pulls information from the matching values.

🟪 **Quick Tip**
The search intent.

---

## 256. Key (K)
🟦 **What is the key (K) in attention?**

🟩 **Definition**
A key is a vector that represents what information a token "offers." Keys are compared to queries to decide attention weights. Keys are also learned projections.

🟨 **How It Works / Example**
Token B's key is matched with token A's query. If they match strongly, token A pays more attention to token B. This lets the model choose useful context dynamically.

🟪 **Quick Tip**
The ID tag.

---

## 257. Value (V)
🟦 **What is the value (V) in attention?**

🟩 **Definition**
A value is the information that gets combined based on attention weights. After computing attention scores, the model takes a weighted sum of values. Values are learned projections of tokens.

🟨 **How It Works / Example**
If token A attends mostly to token B and C, it will mix their value vectors strongly. That mixed vector becomes token A's updated representation. This is how attention "moves" information.

🟪 **Quick Tip**
The actual content.

---

## 258. Scaled Dot-Product Attention
🟦 **What is scaled dot-product attention?**

🟩 **Definition**
Scaled dot-product attention computes similarity between queries and keys using dot products. It divides by a scale factor to keep scores stable. Then it applies softmax to get weights.

🟨 **How It Works / Example**
For each token, compute Q·K for all tokens, divide by √d, then softmax. The softmax results are attention weights. These weights are used to combine the V vectors.

🟪 **Quick Tip**
Math of matching.

---

## 259. Scaling Factor
🟦 **Why do we scale attention scores by √d?**

🟩 **Definition**
Scaling prevents dot products from becoming too large when vector size d is big. Large scores can make softmax too peaky and hurt learning. Scaling keeps gradients more stable.

🟨 **How It Works / Example**
If d is large, Q·K can grow in magnitude. Softmax may then output near one-hot attention, making training unstable. Dividing by √d keeps scores in a healthier range.

🟪 **Quick Tip**
Preventing exploding scores.

---

## 260. Multi-Head Attention
🟦 **What is multi-head attention?**

🟩 **Definition**
Multi-head attention runs several attention "heads" in parallel. Each head can learn different types of relationships. The results are combined to form a richer representation.

🟨 **How It Works / Example**
One head may learn to link pronouns to nouns. Another may focus on nearby words for grammar. Combining heads gives the model multiple views of context at the same time.

🟪 **Quick Tip**
Many viewpoints at once.

---

## 261. Multi-Head Utility
🟦 **Why is multi-head attention useful?**

🟩 **Definition**
It lets the model capture different patterns at once. Different heads can specialize in different dependencies. This improves expressiveness without needing separate models.

🟨 **How It Works / Example**
In translation, one head may align subject words, another may align verbs, and another may handle phrase boundaries. Each head contributes useful context. The combined result improves translation quality.

🟪 **Quick Tip**
Diversity of thought.

---

## 262. Positional Encoding
🟦 **What is a positional encoding?**

🟩 **Definition**
Positional encoding tells a transformer the order of tokens. Attention alone does not know positions. Positional signals are added to token embeddings.

🟨 **How It Works / Example**
If you swap word order, meaning changes, so the model needs position info. Positional encodings add a unique pattern for each position. The transformer then learns patterns like "word at position 2 often follows position 1."

🟪 **Quick Tip**
Adding order to chaos.

---

## 263. Sinusoidal Encoding
🟦 **What is sinusoidal positional encoding?**

🟩 **Definition**
Sinusoidal positional encoding uses sine and cosine waves to encode positions. It does not require learning position vectors. It can generalize to longer lengths than seen in training in some cases.

🟨 **How It Works / Example**
Each position gets a set of sine/cosine values at different frequencies. The pattern is unique per position. The model adds this to token embeddings to inject order information.

🟪 **Quick Tip**
Waves of position.

---

## 264. Learned Positional Embedding
🟦 **What is learned positional embedding?**

🟩 **Definition**
Learned positional embeddings are trainable vectors for each position index. The model learns the best position representations during training. This is common in many transformer implementations.

🟨 **How It Works / Example**
Position 1 has an embedding vector, position 2 has another, and so on. These are learned like word embeddings. They are added to token embeddings before attention layers.

🟪 **Quick Tip**
Learning where things are.

---

## 265. Transformer Encoder
🟦 **What is the encoder in a transformer?**

🟩 **Definition**
The encoder reads the input sequence and builds contextual representations. It usually uses self-attention plus feedforward layers. It is common in tasks like classification and retrieval.

🟨 **How It Works / Example**
In translation, the encoder processes the source sentence and produces embeddings for each token. These embeddings capture meaning and context. The decoder then uses them to generate the target language.

🟪 **Quick Tip**
Understanding the input.

---

## 266. Transformer Decoder
🟦 **What is the decoder in a transformer?**

🟩 **Definition**
The decoder generates an output sequence, usually one token at a time. It uses self-attention plus cross-attention to the encoder outputs (in encoder-decoder models). It is used in tasks like translation and text generation.

🟨 **How It Works / Example**
In translation, the decoder predicts the next target word based on previously generated words. It also attends to encoder outputs to focus on relevant source words. This repeats until an end token is produced.

🟪 **Quick Tip**
Generating the output.

---

## 267. Cross-Attention
🟦 **What is cross-attention?**

🟩 **Definition**
Cross-attention lets a decoder attend to encoder outputs. Queries come from the decoder, and keys/values come from the encoder. This connects input and output sequences.

🟨 **How It Works / Example**
While generating a French word, the decoder uses cross-attention to look at the English words it should translate. It assigns high weight to the most relevant source tokens. This improves alignment and accuracy.

🟪 **Quick Tip**
Connecting input and output.

---

## 268. Masked Self-Attention
🟦 **What is masked self-attention?**

🟩 **Definition**
Masked self-attention prevents a token from seeing future tokens. It is used in autoregressive generation. This ensures the model predicts the next token using only past context.

🟨 **How It Works / Example**
When predicting token 5, the model can attend only to tokens 1–4. A triangular mask blocks attention to tokens 6 and beyond. This matches how text is generated step-by-step.

🟪 **Quick Tip**
No peeking ahead.

---

## 269. Autoregressive Transformer
🟦 **What is an autoregressive transformer?**

🟩 **Definition**
An autoregressive transformer predicts the next token based on previous tokens. It uses masked attention to avoid looking ahead. GPT-style models are autoregressive.

🟨 **How It Works / Example**
Given "I like pizza," the model predicts the next token, maybe "because." Then it appends that token and predicts again. This continues to produce a full text output.

🟪 **Quick Tip**
Predicting the future, one step at a time.

---

## 270. Bidirectional Transformer
🟦 **What is a bidirectional transformer?**

🟩 **Definition**
A bidirectional transformer can use both left and right context when building token representations. It is common for understanding tasks like classification. BERT-style models are bidirectional.

🟨 **How It Works / Example**
In "The bank is near the river," bidirectional context helps interpret "bank." The model can look at "river" on the right side. This improves understanding compared to only left context.

🟪 **Quick Tip**
Looking both ways.

---

## 271. Transformer Block
🟦 **What is a transformer block?**

🟩 **Definition**
A transformer block is a repeating unit in a transformer. It usually contains attention, a feedforward network, and normalization with residual connections. Stacking blocks makes the model deeper and more powerful.

🟨 **How It Works / Example**
A token passes through attention to mix context from other tokens. Then it passes through a small MLP to transform features. Residual connections help keep information flowing across many layers.

🟪 **Quick Tip**
The Lego brick of AI.

---

## 272. Feedforward Network (FFN)
🟦 **What is the feedforward network (FFN) in a transformer?**

🟩 **Definition**
The FFN is a small neural network applied to each token independently. It increases and then reduces dimensions to add more modeling power. It helps transform representations after attention mixing.

🟨 **How It Works / Example**
After attention combines context, the FFN refines the token representation. It can learn useful non-linear transformations. This helps the model represent complex language patterns.

🟪 **Quick Tip**
Processing individual tokens.

---

## 273. Residual Connection
🟦 **What is a residual connection in transformers?**

🟩 **Definition**
A residual connection adds a layer's input to its output. It helps gradients flow through deep networks. This makes training more stable and allows deeper models.

🟨 **How It Works / Example**
Instead of replacing a token embedding completely, a layer adds a "delta" update. If a layer is not helpful, it can learn a small change. This prevents deeper transformers from getting worse.

🟪 **Quick Tip**
Shortcut for gradients.

---

## 274. Layer Normalization
🟦 **What is layer normalization in transformers?**

🟩 **Definition**
Layer normalization stabilizes training by normalizing activations within each token representation. It works well even with small batch sizes. It is standard in transformer blocks.

🟨 **How It Works / Example**
For each token vector, layer norm centers and scales values. This keeps activations in a stable range across layers. It reduces training issues like exploding values.

🟪 **Quick Tip**
Keeping values stable.

---

## 275. Pre-Norm vs Post-Norm
🟦 **What is "pre-norm" vs "post-norm" in transformers?**

🟩 **Definition**
Pre-norm applies layer normalization before attention/FFN, while post-norm applies it after. Pre-norm often trains more stably for deep transformers. The choice affects gradient flow and convergence.

🟨 **How It Works / Example**
In pre-norm, you normalize input then apply attention, then add residual. This helps keep signals stable early in the block. Many large LLMs use pre-norm to reduce training instability.

🟪 **Quick Tip**
Order matters.

---

## 276. Attention Complexity
🟦 **What is attention complexity and why does it matter?**

🟩 **Definition**
Standard attention has time and memory cost that grows with the square of sequence length. This can become expensive for long documents. It is a major scaling limit for transformers.

🟨 **How It Works / Example**
If you double the sequence length, attention cost can become about 4×. For long inputs like 32k tokens, this is very heavy. That's why long-context methods and efficient attention are important.

🟪 **Quick Tip**
N-squared problem.

---

## 277. Causal Attention
🟦 **What is causal attention?**

🟩 **Definition**
Causal attention is attention with a mask that blocks future tokens. It is used for generation tasks. It ensures the model predicts like a left-to-right language model.

🟨 **How It Works / Example**
When generating a story, the model can only use earlier words to predict the next word. The causal mask enforces this rule. This prevents cheating and matches inference behavior.

🟪 **Quick Tip**
Strictly chronological.

---

## 278. Attention Head
🟦 **What is "attention head" in a transformer?**

🟩 **Definition**
An attention head is one parallel attention mechanism inside multi-head attention. Each head has its own Q, K, and V projections. Heads can learn different relationships in data.

🟨 **How It Works / Example**
One head might focus on local neighbors for grammar. Another might connect matching brackets or quotes far apart. Combining multiple heads gives richer context mixing.

🟪 **Quick Tip**
Single strand of thought.

---

## 279. Head Dimension
🟦 **What is "head dimension" in multi-head attention?**

🟩 **Definition**
Head dimension is the size of Q/K/V vectors inside each attention head. Total model dimension is split across heads. This affects capacity and compute cost.

🟨 **How It Works / Example**
If model dimension is 768 and you have 12 heads, each head might use 64 dimensions. Each head learns attention patterns using those 64-d vectors. More heads or larger head dim changes model behavior and cost.

🟪 **Quick Tip**
Size of the spotlight.

---

## 280. Attention Mask
🟦 **What is "attention mask" and how is it used?**

🟩 **Definition**
An attention mask controls which tokens can attend to which others. It can block padding tokens or future tokens. Masks help attention behave correctly for different tasks.

🟨 **How It Works / Example**
In a batch with different-length sentences, padding tokens are added. A padding mask prevents the model from attending to padding. In generation, a causal mask prevents looking at future tokens.

🟪 **Quick Tip**
Blinders for the model.

---

## 281. Key-Value Cache
🟦 **What is "key-value cache" in transformer inference?**

🟩 **Definition**
A key-value (KV) cache stores past attention keys and values during generation. It avoids recomputing them at every step. This makes autoregressive decoding much faster.

🟨 **How It Works / Example**
When generating token 200, you reuse keys/values from tokens 1–199 from the cache. You only compute the new token's K/V once. This reduces compute and latency in chatbots.

🟪 **Quick Tip**
Remembering the past.

---

## 282. Greedy Decoding
🟦 **What is greedy decoding in text generation?**

🟩 **Definition**
Greedy decoding always picks the highest-probability next token. It is fast and simple. But it can produce repetitive or less creative text.

🟨 **How It Works / Example**
At each step, the model outputs probabilities over tokens. Greedy decoding selects the top one every time. This often works for short factual outputs but can get stuck in loops.

🟪 **Quick Tip**
Taking the safest step.

---

## 283. Beam Search
🟦 **What is beam search decoding?**

🟩 **Definition**
Beam search keeps multiple top candidate sequences at each step. It tries to find a higher-probability overall sequence than greedy decoding. It is common in translation and summarization.

🟨 **How It Works / Example**
With beam size 5, you keep the 5 best partial sequences after each token. You expand each one and prune back to 5 again. This explores more options but costs more compute.

🟪 **Quick Tip**
Exploring multiple paths.

---

## 284. Temperature
🟦 **What is temperature in sampling-based decoding?**

🟩 **Definition**
Temperature controls how random text generation is. Higher temperature makes output more diverse, lower temperature makes it more deterministic. It reshapes the probability distribution before sampling.

🟨 **How It Works / Example**
If temperature is 0.2, the model strongly prefers top tokens. If temperature is 1.0, it samples more freely. For creative writing you use higher temperature; for factual answers you use lower.

🟪 **Quick Tip**
Creativity dial.

---

## 285. Top-k Sampling
🟦 **What is top-k sampling?**

🟩 **Definition**
Top-k sampling limits choices to the k most likely next tokens. Then it samples from only those tokens. This avoids very unlikely tokens that can hurt quality.

🟨 **How It Works / Example**
If k=50, the model sorts token probabilities and keeps the top 50. It renormalizes and samples from them. This balances diversity and safety compared to full sampling.

🟪 **Quick Tip**
Shortlisting the best.

---

## 286. Top-p (Nucleus) Sampling
🟦 **What is top-p (nucleus) sampling?**

🟩 **Definition**
Top-p sampling chooses the smallest set of tokens whose probabilities add up to p. Then it samples from that set. It adapts the number of options based on how confident the model is.

🟨 **How It Works / Example**
If the model is confident, only a few tokens reach p=0.9, so sampling is focused. If the model is uncertain, more tokens are included. This often produces better text than fixed top-k.

🟪 **Quick Tip**
Smart shortlisting.

---

## 287. Repetition Penalty
🟦 **What is repetition penalty in decoding?**

🟩 **Definition**
Repetition penalty reduces the chance of repeating the same tokens again and again. It helps avoid loops and boring outputs. It is a decoding-time trick, not a training change.

🟨 **How It Works / Example**
If the model starts repeating "very very very," the penalty lowers the probability of previously used tokens. This encourages new words. It often improves readability in long generations.

🟪 **Quick Tip**
Stopping the echo.

---

## 288. Length Normalization
🟦 **What is length normalization in beam search?**

🟩 **Definition**
Length normalization adjusts beam search scoring to avoid favoring short sequences too much. Without it, shorter outputs often win because probabilities multiply. It helps produce more complete answers.

🟨 **How It Works / Example**
Beam search sums log probabilities across tokens. Longer sequences naturally get lower total probability. Length normalization divides or adjusts the score so longer, better outputs can compete.

🟪 **Quick Tip**
Fair play for long sentences.

---

## 289. Encoder-Only Transformer
🟦 **What is an encoder-only transformer and how is it used?**

🟩 **Definition**
An encoder-only transformer builds contextual embeddings for the input. It is mainly used for understanding tasks like classification and retrieval. BERT is a popular encoder-only model.

🟨 **How It Works / Example**
You input a sentence and get a contextual representation for each token. You use a special pooled output to classify sentiment. Or you use token embeddings for search and retrieval.

🟪 **Quick Tip**
Understanding specialist.

---

## 290. Decoder-Only Transformer
🟦 **What is a decoder-only transformer and how is it used?**

🟩 **Definition**
A decoder-only transformer predicts the next token in a sequence. It uses causal masking for generation. GPT-style LLMs are decoder-only models.

🟨 **How It Works / Example**
Given a prompt, the model predicts the next word repeatedly. It uses KV cache to speed decoding. This is how chat assistants generate responses.

🟪 **Quick Tip**
Generation specialist.

---

## 291. Encoder-Decoder Transformer
🟦 **What is an encoder-decoder transformer and where is it used?**

🟩 **Definition**
An encoder-decoder transformer uses an encoder to read input and a decoder to generate output. It is common for translation and summarization. T5 is an example.

🟨 **How It Works / Example**
The encoder processes the source text into embeddings. The decoder generates the target text while attending to encoder outputs. This structure helps map one sequence to another.

🟪 **Quick Tip**
Translator specialist.

---

## 292. Context Window
🟦 **What is "context window" in transformers?**

🟩 **Definition**
Context window is the maximum number of tokens the model can attend to at once. It limits how much text the model can use as input. Larger context windows allow longer documents but cost more compute.

🟨 **How It Works / Example**
A model with 4k context can't fully process a 20k-token document in one pass. You must chunk or summarize. Long-context models can handle bigger inputs but use more memory.

🟪 **Quick Tip**
Memory span.

---

## 293. Long-Context Attention
🟦 **What is "long-context attention" and why is it needed?**

🟩 **Definition**
Long-context attention methods reduce cost or improve performance for very long sequences. Standard attention becomes too expensive at large lengths. Long-context approaches help models read long documents.

🟨 **How It Works / Example**
Some methods use sparse attention so each token attends to fewer tokens. Others use special kernels or memory tricks. This lets the model handle longer inputs like books or long chats.

🟪 **Quick Tip**
Reading the whole book.

---

## 294. FlashAttention
🟦 **What is FlashAttention?**

🟩 **Definition**
FlashAttention is an efficient way to compute attention with less memory and faster speed. It uses optimized GPU-friendly computation. It helps train and run transformers more efficiently.

🟨 **How It Works / Example**
Instead of storing large attention matrices in memory, it computes attention in blocks. This reduces memory reads and writes. As a result, training large transformers can be faster and fit in memory more easily.

🟪 **Quick Tip**
Attention on fast-forward.

---

## 295. Attention Dropout
🟦 **What is "attention dropout"?**

🟩 **Definition**
Attention dropout randomly drops some attention weights during training. It acts like regularization for attention. It helps reduce overfitting and improves robustness.

🟨 **How It Works / Example**
During training, some attention connections are zeroed out. The model can't rely on a single strong link every time. This encourages it to learn multiple useful patterns.

🟪 **Quick Tip**
Thickening the plot (robustness).

---

## 296. Attention Head Pruning
🟦 **What is "attention head pruning" and why do it?**

🟩 **Definition**
Attention head pruning removes attention heads that contribute little. It reduces model size and speed cost. It can help deployment on limited hardware.

🟨 **How It Works / Example**
You measure which heads have low importance or low impact on accuracy. Then you delete them and fine-tune the model. The result can be faster inference with minimal quality loss.

🟪 **Quick Tip**
Trimming the fat.

---

## 297. Token Mixing
🟦 **What is "token mixing" in transformers?**

🟩 **Definition**
Token mixing is how information moves between tokens. In transformers, attention is the token-mixing mechanism. It lets each token gather context from other tokens.

🟨 **How It Works / Example**
A token's representation becomes a weighted sum of other tokens' values. This mixes information across the sentence. Then the FFN refines each token independently.

🟪 **Quick Tip**
Sharing the knowledge.

---

## 298. Positional Bias
🟦 **What is "positional bias" in attention?**

🟩 **Definition**
Positional bias is a built-in preference based on token distances or positions. It helps attention focus on nearby tokens or certain patterns. It can improve performance and stability.

🟨 **How It Works / Example**
Some models add a bias so tokens pay more attention to nearby words. This helps with local grammar patterns. Other biases help the model handle long contexts more efficiently.

🟪 **Quick Tip**
Local preference.

---

## 299. RoPE (Rotary Positional Embedding)
🟦 **What is RoPE (Rotary Positional Embedding)?**

🟩 **Definition**
RoPE is a positional method that rotates query and key vectors based on position. It helps encode relative positions naturally. Many modern LLMs use RoPE for better long-context behavior.

🟨 **How It Works / Example**
Instead of adding a position vector, RoPE changes Q and K using a rotation that depends on token index. This makes attention scores depend on relative distance. It helps the model keep track of order and distance more smoothly.

🟪 **Quick Tip**
Rotation for location.

---

## 300. Attention Interpretability
🟦 **What is "attention interpretability" and why is it tricky?**

🟩 **Definition**
Attention interpretability is trying to explain model decisions using attention weights. It can be helpful but is not always a full explanation. Attention weights do not always equal true "importance."

🟨 **How It Works / Example**
You might see a token attends strongly to a certain word and assume that word caused the prediction. But other layers and heads may matter more. For better understanding, people combine attention views with ablations or gradient-based methods.

🟪 **Quick Tip**
Peeking into the black box.
