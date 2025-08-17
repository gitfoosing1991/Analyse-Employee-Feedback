# Learning Notes: 

**When using Direct GPT Analysis method, why is the batch size for Theme Discovery (step 1) set at 100, but Theme Assignment (step 2) set at 5?**

Theme assignment (step 2) requires a much longer prompt and output per entry than theme discovery (step 1).

In theme discovery, you typically send only the raw feedback entries and ask the model to extract themes. The output is short (just a list of themes). In theme assignment, you send each entry plus the full list of discovered themes, and ask the model to assign each entry to a main/sub-theme, plus possibly sentiment and other scores. The prompt and expected output per entry are much longer.

For theme assignment, the prompt grows with both the number of entries and the number/length of discovered themes. The output per entry is also longer (multiple fields per entry). This quickly pushes you over the model’s token limit, even with a small batch size.


--- 

**In the clustering method, why not do batching when embedding request (step 1) to save tokens?**

Batching embedding requests does not save tokens—the total token usage is the same whether you send entries one at a time or in batches. The main benefit of batching is improved speed and efficiency (fewer API calls, faster processing).

Token cost is calculated per input text, not per API call.
Batching helps reduce network overhead and can speed up processing, but does not reduce the total number of tokens used or billed.

So, batching embeddings helps with performance, not token savings.

---
**Can you explain Principal Component Analysis Variance?**

Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning. It transforms high-dimensional data into a lower-dimensional representation while retaining most of the original data's variance.

Imagine you have a big box of colored pencils—100 different colors. You want to organize them, but carrying all 100 is heavy and sometimes you don’t need every single shade.

PCA variance is like deciding how many pencils you need to keep so you can still draw almost any picture you want:

If you keep 99 pencils (99% variance), you have almost every color—your drawings will look very close to the original.
If you keep only 80 pencils (80% variance), you have fewer colors, but you can still draw most things, just with less detail or fewer shades.

In data analysis, PCA variance is about how much of the “color” (information) you keep when you simplify your data. Higher variance means you keep more detail; lower variance means you keep less, making things simpler but possibly missing some subtle differences.

So, adjusting PCA variance is like choosing how many colored pencils to carry: more pencils (higher variance) means more detail, fewer pencils (lower variance) means less detail but easier to manage. It doesn’t change how many pictures (clusters) you draw—just how detailed each picture can be.

---
**Why is Principal Component Analysis (PCA) used in the clustering methodology?**

PCA is used in the clustering methodology to make the clustering process more effective and efficient. Here’s a simple analogy and explanation:

Imagine you have a huge, messy closet with clothes scattered everywhere (your data has many features or “dimensions”). If you try to organize it all at once, it’s overwhelming and hard to see patterns.

PCA is like first sorting your clothes into a few big, meaningful piles—shirts, pants, jackets—so you can focus on the most important categories. This makes it much easier to organize (cluster) your closet, because you’re working with fewer, clearer groups.

In technical terms:

PCA reduces the number of features (dimensions) in your data, keeping only the most important information.
This helps clustering algorithms (like KMeans) find patterns more easily, avoids noise, and speeds up processing.
Without PCA, clustering on high-dimensional data can be slow, less accurate, and more likely to find random or meaningless groups.

So, PCA helps you “tidy up” your data before clustering, making the results more meaningful and the process more efficient.


---

**What is the breakdown of the token cost estimates for both analysis method?**

**Clustering Method (Assuming 1000 Responses):**
- **Embedding generation:**
  - 1 API call per response; 1000 responses = 1000 API calls.
  - Estimated token usage: ~20 tokens per call; 1000 API calls ≈ 20,000 tokens.

- **Sentiment analysis:**
  - ~10 API calls per main theme (one per sample, up to 10 samples per theme).
  - For example, 10 main themes × 10 samples = 100 API calls.
  - Estimated token usage: ~1,000 tokens per call; 100 API calls ≈ 100,000 tokens.

- **Theme naming:**
  - 1 API call per main theme.
  - Estimated token usage: ~1,000 tokens per call; 10 API calls ≈ 10,000 tokens.

- **Sub-theme naming:**
  - 1 API call per sub-theme.
  - Estimated token usage: ~1,000 tokens per call; e.g., 20 sub-themes ≈ 20,000 tokens.

- **Theme summary:**
  - 1 API call per main theme (using 10 responses each).
  - Estimated token usage: ~2,000 tokens per call; 10 API calls ≈ 20,000 tokens.

- **Estimated total:**
  - API calls: ~1000 (embedding) + ~100 (sentiment) + main/sub-theme naming and summary (varies by dataset).
  - Token usage: ~170,000 tokens (for 1000 responses, 10 main themes, 20 sub-themes; actual usage may vary).

**Direct GPT Method (Assuming 1000 Responses):**
- **Theme exploration:**
  - Responses are batched in sizes of 100 per batch and sent to ChatGPT for theme exploration.
  - 1 batch (up to 100 responses) = 1 API call; 10 batches (1000 responses) = 10 API calls.
  - Estimated token usage: ~6,000 tokens per API call; 10 API calls ≈ 60,000 tokens.

- **Sentiment & theme assignment:**
  - Responses are batched in sizes of 5 per batch and sent to ChatGPT for sentiment and theme assignment.
  - 1 batch (up to 5 responses) = 1 API call; 200 batches (1000 responses) = 200 API calls.
  - Estimated token usage: ~3,000 tokens per API call; 200 API calls ≈ 600,000 tokens.

- **Theme summary:**
  - 1 API call per main theme (using 10 responses each).
  - Depends on number of main themes (e.g., 10 themes × ~2,000 tokens = ~20,000 tokens).

- **Estimated total:** ~680,000 tokens, ~220 API calls (for 1000 responses, batch sizes as above).
