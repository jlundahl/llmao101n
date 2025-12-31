# LLM101n: Let's Build a Storyteller

You're about to do something remarkable. Not download a language model. Not fine-tune someone else's work. Not call an API and hope for the best. You're going to *build* one. From the ground up. Starting from nothing but Python, C, CUDA, and your own curiosity.

By the time you finish this course, you'll have a working AI storyteller, a web application where you can create, refine, and illustrate stories with a model you understand completely. Not because someone explained it to you in a blog post, but because you wrote every layer yourself. The tokenizer that turns text into numbers? You built it. The attention mechanism that lets words talk to each other across a paragraph? You implemented it. The optimizer that nudges millions of parameters toward coherent prose? You coded it from scratch.

This is not the fast path. If you want to ship something by Friday, go use the OpenAI API. This is the *deep* path, the one where magic dissolves into mechanics, and the mechanics turn out to be more beautiful than the magic ever was.

Here's what you'll gain: You'll understand how language models actually work, not as black boxes but as systems you could rebuild from memory. You'll know why attention scales with the square of sequence length, why BF16 training is more stable than FP16, why KV-caching makes generation fast, why RLHF makes models helpful. You'll have opinions about tokenization strategies and initialization schemes and learning rate schedules, opinions grounded in code you've written and experiments you've run.

More than that, you'll have *confidence*. When a new architecture paper drops, you'll read it and think "I could implement that." When someone claims transformers can't do X, you'll know enough to evaluate the claim. When you need to debug a training run at 3am, you'll have the mental model to figure out what's wrong.

The journey starts simple, embarrassingly simple. A bigram model that predicts the next character by counting. From there, we build: automatic differentiation, neural networks, attention, transformers, efficient training, fast inference, alignment, multimodality. Each chapter adds one concept, implemented from scratch, until suddenly you have GPT.

Let's begin.

## Table of Contents

1. [Bigram Language Model](#bigram-language-model)
   - [Lesson 1: Language Modeling Fundamentals](#lesson-1-language-modeling-fundamentals)
   - [Lesson 2: Character-level Bigram Model](#lesson-2-character-level-bigram-model)
   - [Lesson 3: Training Loop](#lesson-3-training-loop)
   - [Lesson 4: Sampling and Generation](#lesson-4-sampling-and-generation)
2. [Micrograd](#micrograd)
   - [Lesson 1: Automatic Differentiation](#lesson-1-automatic-differentiation)
   - [Lesson 2: Value Class Implementation](#lesson-2-value-class-implementation)
   - [Lesson 3: Forward Pass](#lesson-3-forward-pass)
   - [Lesson 4: Backward Pass](#lesson-4-backward-pass)
   - [Lesson 5: Training a Neural Network](#lesson-5-training-a-neural-network)
3. [N-gram Model](#n-gram-model)
   - [Lesson 1: Multi-layer Perceptrons](#lesson-1-multi-layer-perceptrons)
   - [Lesson 2: Embedding Layers](#lesson-2-embedding-layers)
   - [Lesson 3: Matrix Multiplication](#lesson-3-matrix-multiplication)
   - [Lesson 4: MLP with GELU Activation](#lesson-4-mlp-with-gelu-activation)
   - [Lesson 5: Batched Training](#lesson-5-batched-training)
4. [Attention](#attention)
   - [Lesson 1: Attention Mechanism](#lesson-1-attention-mechanism)
   - [Lesson 2: Query, Key, Value Projections](#lesson-2-query-key-value-projections)
   - [Lesson 3: Scaled Dot-Product Attention](#lesson-3-scaled-dot-product-attention)
   - [Lesson 4: Softmax Implementation](#lesson-4-softmax-implementation)
   - [Lesson 5: Positional Encodings](#lesson-5-positional-encodings)
5. [Transformer](#transformer)
   - [Lesson 1: Transformer Architecture](#lesson-1-transformer-architecture)
   - [Lesson 2: Transformer Block](#lesson-2-transformer-block)
   - [Lesson 3: Full GPT-2 Architecture](#lesson-3-full-gpt-2-architecture)
   - [Lesson 4: Weight Loading](#lesson-4-weight-loading)
   - [Lesson 5: Text Generation](#lesson-5-text-generation)
6. [Tokenization](#tokenization)
   - [Lesson 1: Byte Pair Encoding](#lesson-1-byte-pair-encoding)
   - [Lesson 2: Character-level Tokenization](#lesson-2-character-level-tokenization)
   - [Lesson 3: BPE Training Algorithm](#lesson-3-bpe-training-algorithm)
   - [Lesson 4: BPE Encoding and Decoding](#lesson-4-bpe-encoding-and-decoding)
   - [Lesson 5: Special Tokens](#lesson-5-special-tokens)
7. [Datasets](#datasets)
   - [Lesson 1: Data Loading and Generation](#lesson-1-data-loading-and-generation)
   - [Lesson 2: Efficient Data Loading](#lesson-2-efficient-data-loading)
   - [Lesson 3: Batching and Shuffling](#lesson-3-batching-and-shuffling)
   - [Lesson 4: Synthetic Data Generation](#lesson-4-synthetic-data-generation)
   - [Lesson 5: Data Quality](#lesson-5-data-quality)
8. [Optimization](#optimization)
   - [Lesson 1: Initialization and Optimization](#lesson-1-initialization-and-optimization)
   - [Lesson 2: Weight Initialization](#lesson-2-weight-initialization)
   - [Lesson 3: SGD with Momentum](#lesson-3-sgd-with-momentum)
   - [Lesson 4: Adam and AdamW](#lesson-4-adam-and-adamw)
   - [Lesson 5: Learning Rate Schedulers](#lesson-5-learning-rate-schedulers)
9. [Need for Speed I,  Device](#need-for-speed-i-device)
   - [Lesson 1: Hardware Understanding](#lesson-1-hardware-understanding)
   - [Lesson 2: CPU vs GPU Benchmarks](#lesson-2-cpu-vs-gpu-benchmarks)
   - [Lesson 3: CUDA Kernel Basics](#lesson-3-cuda-kernel-basics)
   - [Lesson 4: Matrix Multiplication on GPU](#lesson-4-matrix-multiplication-on-gpu)
   - [Lesson 5: Memory Transfer Patterns](#lesson-5-memory-transfer-patterns)
10. [Need for Speed II,  Precision](#need-for-speed-ii-precision)
   - [Lesson 1: Mixed Precision Training](#lesson-1-mixed-precision-training)
   - [Lesson 2: Floating Point Representations](#lesson-2-floating-point-representations)
   - [Lesson 3: Mixed Precision Training Loop](#lesson-3-mixed-precision-training-loop)
   - [Lesson 4: Loss Scaling](#lesson-4-loss-scaling)
   - [Lesson 5: Automatic Mixed Precision](#lesson-5-automatic-mixed-precision)
11. [Need for Speed III,  Distributed](#need-for-speed-iii-distributed)
   - [Lesson 1: Distributed Training](#lesson-1-distributed-training)
   - [Lesson 2: Data Parallel Training](#lesson-2-data-parallel-training)
   - [Lesson 3: Gradient Synchronization](#lesson-3-gradient-synchronization)
   - [Lesson 4: DDP Implementation](#lesson-4-ddp-implementation)
   - [Lesson 5: ZeRO Optimizer Sharding](#lesson-5-zero-optimizer-sharding)
12. [Inference I,  KV-Cache](#inference-i-kv-cache)
   - [Lesson 1: KV-Cache Optimization](#lesson-1-kv-cache-optimization)
   - [Lesson 2: KV-Cache Implementation](#lesson-2-kv-cache-implementation)
   - [Lesson 3: Incremental Generation Loop](#lesson-3-incremental-generation-loop)
   - [Lesson 4: Batched Generation](#lesson-4-batched-generation)
   - [Lesson 5: Cache Memory Management](#lesson-5-cache-memory-management)
13. [Inference II,  Quantization](#inference-ii-quantization)
   - [Lesson 1: Quantization Fundamentals](#lesson-1-quantization-fundamentals)
   - [Lesson 2: Post-Training Quantization](#lesson-2-post-training-quantization)
   - [Lesson 3: Quantization-Aware Training](#lesson-3-quantization-aware-training)
   - [Lesson 4: INT8 and INT4 Implementations](#lesson-4-int8-and-int4-implementations)
   - [Lesson 5: Calibration and Error Analysis](#lesson-5-calibration-and-error-analysis)
14. [Deployment](#deployment)
   - [Lesson 1: API and Web App](#lesson-1-api-and-web-app)
   - [Lesson 2: REST API](#lesson-2-rest-api)
   - [Lesson 3: Streaming Responses](#lesson-3-streaming-responses)
   - [Lesson 4: Web Frontend](#lesson-4-web-frontend)
   - [Lesson 5: Infrastructure Patterns](#lesson-5-infrastructure-patterns)
15. [Finetuning I,  SFT](#finetuning-i-sft)
   - [Lesson 1: Supervised Finetuning](#lesson-1-supervised-finetuning)
   - [Lesson 2: Chat Format Training](#lesson-2-chat-format-training)
   - [Lesson 3: SFT Training Loop](#lesson-3-sft-training-loop)
   - [Lesson 4: LoRA Implementation](#lesson-4-lora-implementation)
   - [Lesson 5: Parameter-Efficient Finetuning](#lesson-5-parameter-efficient-finetuning)
16. [Finetuning II,  RL](#finetuning-ii-rl)
   - [Lesson 1: Reinforcement Learning from Human Feedback](#lesson-1-reinforcement-learning-from-human-feedback)
   - [Lesson 2: Reward Model Training](#lesson-2-reward-model-training)
   - [Lesson 3: PPO Implementation](#lesson-3-ppo-implementation)
   - [Lesson 4: DPO](#lesson-4-dpo)
   - [Lesson 5: Alignment Evaluation](#lesson-5-alignment-evaluation)
17. [Multimodal](#multimodal)
   - [Lesson 1: Beyond Text](#lesson-1-beyond-text)
   - [Lesson 2: VQVAE Image Tokenizer](#lesson-2-vqvae-image-tokenizer)
   - [Lesson 3: Diffusion Model Basics](#lesson-3-diffusion-model-basics)
   - [Lesson 4: Text-to-Image Generation](#lesson-4-text-to-image-generation)
   - [Lesson 5: Multimodal Integration](#lesson-5-multimodal-integration)

---

# Bigram Language Model

## Lesson 1: Language Modeling Fundamentals

You're going to build something that generates text. Not by following rules someone wrote down, not by filling in templates, but by learning patterns from examples. Before we write a single line of code, we need to understand what we're actually trying to do.

Here's the question that drives everything: given some text, what comes next?

That's it. That's language modeling. Everything else, from the bigram model we'll build this week to GPT-4, is just increasingly sophisticated ways of answering that question.

### Why This Matters

Open ChatGPT or Claude or Gemini. Type something. Watch the response appear word by word. What's happening under the hood?

The model is predicting the next token. Then it takes that prediction, adds it to what came before, and predicts the next token again. And again. And again. This process, sometimes called "dreaming" by researchers, continues until the model decides it's done or hits some limit.

The bigram language model predicts the next token based solely on the current token, serving as a simple baseline for language modeling. It's almost embarrassingly simple. Given the letter 'a', what letter comes next? Given the word 'the', what word comes next? The model doesn't understand grammar. It doesn't understand meaning. It just counts patterns.

And yet this simple idea scales. The same fundamental operation, predict the next token, is what powers systems that write poetry, debug code, and carry on conversations. The difference is in how much context they use and how sophisticated their pattern matching becomes.

### How Text Becomes Numbers

Computers don't understand letters. They understand numbers. So the first thing we need is a way to convert text into something a computer can process.

Under the hood, text is tokenized into discrete chunks called tokens that form a one-dimensional sequence. For our bigram model, each character will be its own token. The letter 'a' might become the number 0, 'b' becomes 1, and so on. Models like GPT-4 use a vocabulary of roughly 200,000 tokens, but they're doing the same basic thing: turning text into a sequence of integers.

Let's say we have the text "cat chased mouse". We split this into characters and assign each unique character a number. Now we have a sequence of integers. The model's job is to look at one integer and predict what integer comes next.

Here's where probability comes in. The model doesn't just predict one answer. It predicts a probability distribution over all possible next tokens. After the letter 'c', maybe 'a' has probability 0.3, 'h' has probability 0.2, 'o' has probability 0.15, and so on. All these probabilities sum to 1.

Why probabilities instead of just picking one answer? Because language is genuinely uncertain. After "The cat", you might write "sat" or "ran" or "meowed" or a thousand other things. A good language model captures this uncertainty.

### The Training Loop

So how does the model learn these probabilities?

We show it examples. Lots of examples. We take text like "cat chased mouse" and create training pairs: given 'c', the answer is 'a'. Given 'a', the answer is 't'. Given 't', the answer is ' ' (space). Given ' ', the answer is 'c'. And so on.

The model starts with random guesses. It might think that after 'c', the letter 'z' is most likely. Obviously wrong. So we need a way to measure how wrong it is.

This is where cross-entropy loss comes in. Cross-entropy measures the gap between the model's predicted probabilities and reality. If the model says 'a' has probability 0.01 when the true next character is 'a', the loss is high. If the model says 'a' has probability 0.95, the loss is low.

The training loop works like this:
1. Show the model a character
2. Get its probability distribution over next characters
3. Check what the actual next character was
4. Compute the cross-entropy loss
5. Adjust the model's parameters to reduce this loss
6. Repeat millions of times

After enough repetitions, the model has learned the statistical patterns in your training data. It knows that 'q' is almost always followed by 'u'. It knows that spaces often follow periods. It knows that in your dataset, certain character combinations appear more frequently than others.

### Generating Text

Once trained, how do we get the model to generate new text?

We give it a starting character. It produces probabilities for the next character. We sample from those probabilities, picking a character randomly but weighted by likelihood. Then we feed that character back in and repeat.

This is where temperature becomes important. Temperature controls how "creative" or "random" the sampling is. A low temperature makes the model stick to high-probability choices. A high temperature makes it more willing to pick unlikely options.

Temperature 0 means always pick the most likely next character. This produces repetitive, boring text. Temperature 1 means sample according to the true probabilities. Higher temperatures make rare choices more likely, producing more surprising but potentially nonsensical output.

### Where Learners Get Stuck

The first confusion is thinking the model "knows" anything. It doesn't. LLMs are probabilistic and may hallucinate. Their responses reflect statistical patterns in training data rather than guaranteed facts. A bigram model that's seen "cat chased mouse" many times will learn that 'c' is often followed by 'a' and 'h'. It has no concept of cats or chasing.

The second confusion is about what "training" means. You're not programming rules. You're adjusting numbers, lots of numbers, so that the model's predictions get closer to reality. The model discovers patterns you never explicitly told it about.

The third confusion is thinking generation is deterministic. It's not. The same prompt can produce different outputs because of the random sampling step. This is a feature, not a bug.

### The Context Window

A conversation with an LLM is represented as a growing token sequence called the context window, where user and model take turns appending tokens. Starting a new chat resets this window.

For a bigram model, the context window is exactly one character. The model sees only the current character and predicts the next. This is extremely limiting. Consider "cat chased mouse" versus "mouse chased cat". A bigram model treats these identically character by character. It can't understand that the meaning is completely different.

The context window is a limited resource. Longer conversations increase computational cost and slow response times. For our simple bigram model, this isn't an issue. But as we build more sophisticated models, managing context becomes critical.

### Success Indicators

You understand this lesson if you can answer:
- What is a language model trying to predict?
- Why do we use probabilities instead of single predictions?
- What does cross-entropy loss measure?
- What happens when you increase temperature during sampling?
- Why can't a bigram model understand that "cat chased mouse" and "mouse chased cat" mean different things?

In the next lesson, we'll actually build this. You'll write the code that converts characters to numbers, counts bigram frequencies, computes probabilities, and generates new text. The theory becomes real when you can run it.

## Lesson 2: Character-level Bigram Model

Let's build something that generates text. Right now. Before we understand everything about it.

Open a new Python file. We're going to construct a bigram language model from scratch, and by the end of this lesson, you'll have code that produces new character sequences based on patterns it learned from data.

### Why Bigrams?

A bigram model predicts the next character based solely on the immediately preceding character. That's it. No memory of what came before. No understanding of words or grammar. Just: "Given that I just saw the letter 'h', what letter probably comes next?"

This sounds almost uselessly simple. And yet this constraint forces us to confront the essential machinery of language modeling. The loss function we use here, the training loop structure, the sampling procedure, all of these carry forward to GPT and beyond. We start simple so we can see the shape of the problem before complexity obscures it.

### Building the Count Matrix

First, we need data. Let's use a list of names. You can grab any text file with names, one per line.

```python
words = open('names.txt', 'r').read().splitlines()
```

Now here's the key insight: we need to count how often each character pair appears. If 'th' appears 5000 times and 'tx' appears 3 times, we want our model to strongly prefer 'h' after 't'.

We also need special tokens to mark where sequences begin and end. These boundary markers let the model learn which characters typically start names and which typically end them.

```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  # special token for start/end
itos = {i:s for s,i in stoi.items()}
```

The dot serves as both our start token and end token. When we see '.', we know a name is beginning or ending.

Now we count bigrams:

```python
import torch

N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

What did we just build? A 27x27 matrix where row i, column j contains the count of how many times character j followed character i in our dataset. Row 0 tells us which characters start names. Column 0 tells us which characters end names.

### From Counts to Probabilities

Raw counts aren't useful for sampling. We need probabilities. If 'a' is followed by 'n' 2847 times and by 'b' 541 times, we want to know the relative likelihood.

We create a row-stochastic matrix, where each row sums to 1:

```python
P = (N + 1).float()  # add 1 for smoothing
P = P / P.sum(dim=1, keepdim=True)
```

Why add 1 to every count? This is called smoothing. Without it, if some character pair never appeared in training, its probability would be zero, and we could never generate that pair. Adding 1 gives every possible transition at least a tiny probability.

Each row of P now represents the conditional probability distribution of the next character given the current character. Row 5 might be the distribution P(next | current='e').

### Sampling New Names

Here's where it gets fun. We can generate new names by repeatedly sampling from these probability distributions:

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    out = []
    ix = 0  # start with '.'
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:  # hit end token
            break
        out.append(itos[ix])
    print(''.join(out))
```

The function torch.multinomial draws indices according to given probabilities. If P[ix] says 'a' has probability 0.3 and 'b' has probability 0.1, then multinomial will return the index for 'a' roughly three times as often as 'b'.

Run this. You'll get output like:

```
junide
janasah
p
cfay.
```

Some look plausible. Some look terrible. That's the bigram model. It captures local patterns but has no concept of longer structure. It doesn't know that names rarely start with 'x' unless it learned that from the data. It doesn't know that 'qu' usually comes together.

### What Just Happened?

Let's be precise about what we built:

1. We collected bigram statistics by counting consecutive character pairs
2. We normalized counts into probability distributions
3. We sampled from those distributions to generate new sequences

The model learned something real. It learned that 'e' often ends names, that 'th' is common, that 'q' is rare. But it learned nothing about three-character patterns, nothing about the rhythm of syllables, nothing about meaning.

### Common Mistakes

**Forgetting boundary tokens.** Without the special start/end token, the model can't learn which characters begin or end sequences. You'll generate text that starts mid-word and never stops.

**Not normalizing rows.** If you forget to divide by row sums, you're sampling from counts, not probabilities. The model will still run but produce garbage.

**Integer division.** In Python 3, this is less of an issue, but make sure your probability matrix is float, not int. Integer division truncates, destroying your distributions.

**Seed confusion.** If you want reproducible results for debugging, set a generator seed. If you want variety, don't.

### Success Indicators

You know you've got it working when:

- Your probability matrix rows each sum to 1.0 (within floating point tolerance)
- Generated sequences start with plausible first characters
- Generated sequences actually terminate (hit the end token)
- The output looks vaguely like your training data, capturing at least some local patterns

### The Bigger Picture

This bigram model is the simplest possible language model. ChatGPT, Gemini, Claude, all of these are language models too. They predict the next token based on context. The difference is that they use much more context, much more sophisticated architectures, and vastly more parameters.

But the core loop is identical: look at what came before, predict what comes next, sample from the prediction.

In the next lesson, we'll measure how good our model is using cross-entropy loss. This will let us compare models quantitatively and set up the machinery for training neural network language models.

For now, play with your bigram model. Try different training data. Look at the probability matrix. Which character transitions are most likely? Which are impossible? The patterns you see are the patterns your model learned.

## Lesson 3: Training Loop

You have a bigram model. It has weights. Those weights are random garbage right now. The model predicts the next character about as well as a coin flip predicts the weather.

Time to fix that.

The training loop is where learning actually happens. It's the same fundamental structure whether you're training a character-level bigram model or GPT-4. The scale changes. The architecture changes. This loop stays the same.

Let's build it.

### The Loop in Code

```python
for step in range(num_steps):
    # 1. Forward pass: make predictions
    logits = model(inputs)
    
    # 2. Loss Calculation: how wrong were we?
    loss = cross_entropy(logits, targets)
    
    # 3. Gradient Computation: which direction to move?
    loss.backward()
    
    # 4. Parameter Updates: nudge the weights
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        p.grad = None  # reset for next iteration
```

That's it. Four steps, repeated thousands of times. Forward pass, Loss Calculation, Gradient Computation, Parameter Updates. Everything else is optimization and bookkeeping.

Let's understand each piece.

### Step 1: Forward Pass

The forward pass runs your input through the model to get predictions. For our bigram model, this means: given character 'h', what's the probability distribution over the next character?

```python
# inputs is a batch of character indices, shape (batch_size,)
# Each index points to a row in our weight matrix
logits = W[inputs]  # shape (batch_size, vocab_size)
```

The logits are raw scores. Higher score means the model thinks that character is more likely to come next. These aren't probabilities yet. They could be negative, they could be huge. We'll convert them to probabilities when we compute the loss.

### Step 2: Loss Calculation

How wrong is the model? We need a single number that captures this. That number is the loss.

For language modeling, we use cross-entropy loss. Here's the intuition: if the model assigns high probability to the correct next character, the loss is low. If it assigns low probability to the correct character, the loss is high.

```python
# Convert logits to probabilities
probs = softmax(logits)  # each row sums to 1

# For each example, look up the probability assigned to the correct answer
correct_probs = probs[range(batch_size), targets]

# Cross-entropy: negative log of correct probability
loss = -torch.log(correct_probs).mean()
```

Why negative log? Because we want:
- Probability 1.0 (perfect) → loss 0
- Probability 0.5 → loss 0.69
- Probability 0.01 (terrible) → loss 4.6

The log turns multiplication into addition, which matters for numerical stability when you're multiplying many small probabilities. The negative flips it so lower loss is better.

This is the same loss function used in GPT-4. Same math. Same meaning. The model that predicts next tokens well has low cross-entropy loss.

### Step 3: Gradient Computation

Here's where the magic happens. We need to know: for each weight in the model, if I nudge it up a tiny bit, does the loss go up or down?

The gradient of the loss with respect to a weight indicates how much the loss would change if the weight were nudged slightly. Positive gradient means increasing the weight increases the loss. Negative gradient means increasing the weight decreases the loss.

Backpropagation is an algorithm that efficiently computes gradients of a loss function with respect to neural network weights by applying the chain rule recursively through a computational graph. You don't compute each gradient separately. You compute them all in one backward sweep through the computation.

```python
loss.backward()
```

One line. But behind it: the chain rule applied recursively through every operation. For a multiplication operation L = d * f, the gradient of L with respect to d is f, and the gradient of L with respect to f is d. These local rules compose through the entire network.

Neural networks are mathematical expressions that take inputs (data and weights) and produce outputs (predictions or loss), and backpropagation works on any such expression, not just neural networks. Any differentiable computation can be trained this way.

### Step 4: Parameter Updates

Now we have gradients. Time to use them.

```python
for p in model.parameters():
    p.data -= learning_rate * p.grad
    p.grad = None
```

We move each weight in the opposite direction of its gradient. If increasing a weight would increase the loss, we decrease that weight. If increasing a weight would decrease the loss, we increase that weight.

The learning rate controls how big a step we take. Too big and you overshoot. Too small and training takes forever. Finding the right learning rate is an art we'll cover in the optimization chapter.

Setting the gradient to None after the update is crucial. Otherwise gradients accumulate across steps, which is usually not what you want.

### Loop Continuation

Why do we loop? Because one update isn't enough. The gradients tell you which direction is downhill from where you currently stand. But after you take a step, you're in a new place. The gradients change. You need to recompute them and take another step.

Iterative Refinement. Each step makes the model a little better. Thousands of steps make it a lot better.

### The Essential Trade

Here's something important that the source materials emphasize: there's a fundamental trade-off in how we train these models. We sacrifice Theoretical Guarantees for Empirical Stability and Scalability.

We can't prove the training loop will find the best possible weights. We can't prove it will converge at all. What we have is empirical evidence that it works remarkably well in practice, especially as models get larger.

This is The Transformer era philosophy. We don't fully understand why these methods work as well as they do. But they do work, and they scale.

### Common Mistakes

**Forgetting to zero gradients.** If you don't reset gradients after each update, they accumulate. Your model will behave erratically.

**Learning rate too high.** The loss will spike or oscillate wildly. Start small, like 0.01 or 0.001.

**Not shuffling data.** If you always train on examples in the same order, the model can develop weird biases. Shuffle your training data.

**Training for too few steps.** The model needs time to learn. If your loss is still dropping, keep training.

**Training for too many steps.** Eventually the model memorizes the training data instead of learning patterns. We'll cover this when we discuss validation sets.

### Success Indicators

How do you know training is working?

1. **Loss goes down over time.** This is the primary signal. Plot your loss curve. It should trend downward, especially early in training.

2. **Generated samples improve.** Sample from your model periodically. Early samples will be garbage. Later samples should look more like real text.

3. **Loss stabilizes.** Eventually the loss stops dropping. The model has learned what it can from the data.

For a character-level bigram model on names, you should see the loss drop from around 3.5 (random) to around 2.4 within a few thousand steps. The model learns that 'q' is almost always followed by 'u', that names often end with 'a' or 'n', that certain character combinations are common.

### What You've Built

You now have a complete training loop. The same structure scales up. More parameters, more data, more compute, but the same four steps: forward pass, Loss Calculation, Gradient Computation, Parameter Updates.

In the next lesson, we'll add the final piece: sampling from your trained model to generate new text. Your model has learned a probability distribution over character sequences. Time to dream with it.

## Lesson 4: Sampling and Generation

You've built a bigram model. You've trained it. Now comes the fun part: making it dream.

That's not a metaphor. When you sample from a language model, you're asking it to generate sequences that look like its training data. The model learned patterns from thousands of names, compressed those patterns into a probability matrix, and now you're going to let it hallucinate new names that never existed but feel like they could have.

Let's write the code first, then understand what we're doing.

```python
def sample_name(P, itos, stoi):
    name = []
    ix = stoi['.']  # start token
    
    while True:
        # get probability distribution for next character
        probs = P[ix]
        # sample from distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # check for end token
        if ix == stoi['.']:
            break
        name.append(itos[ix])
    
    return ''.join(name)

# generate 10 names
for _ in range(10):
    print(sample_name(P, itos, stoi))
```

Run this. You'll get output like:

```
ander
h
andra
alex
etetet
ananan
christopher
ph
```

Some of these look reasonable. Some are garbage. That's exactly what we expect from a bigram model. Let's understand why.

### The Sampling Loop

The process of sampling from a character-level model to generate names is fundamentally the same as the inference process described for large language models, just applied at a different granularity. Both processes operate on the principle of next-token prediction and iterative generation.

Here's what happens step by step:

1. Start with the special token `.` representing the beginning of a name
2. Look up row `ix` in our probability matrix P to get probabilities for the next character
3. Sample a character according to those probabilities using `torch.multinomial`
4. If we sampled the end token `.`, stop
5. Otherwise, append the character and repeat from step 2

This is Iterative Generation. Each predicted token is appended to the sequence and fed back into the model to predict the next token. For our bigram model, "fed back in" just means we use the new character's row in P. For GPT-4, it means running the entire transformer forward again with the extended context.

Same Core Mechanism. Different scale.

### Understanding torch.multinomial

The key function here is `torch.multinomial`. It draws indices according to given probabilities. If you pass it `[0.5, 0.3, 0.2]`, it returns index 0 about half the time, index 1 about 30% of the time, and index 2 about 20% of the time.

```python
# example
probs = torch.tensor([0.5, 0.3, 0.2])
samples = torch.multinomial(probs, num_samples=1000, replacement=True)
print(torch.bincount(samples))  # roughly [500, 300, 200]
```

This is what makes generation interesting. We don't always pick the most likely next character. We sample from the distribution. This introduces variability, which is why you get different names each time you run the generator.

### Why Sampling Instead of Argmax?

You might wonder: why not just always pick the most probable character? That's called greedy decoding. Try it:

```python
def sample_name_greedy(P, itos, stoi):
    name = []
    ix = stoi['.']
    
    while True:
        probs = P[ix]
        ix = torch.argmax(probs).item()  # always pick highest probability
        if ix == stoi['.']:
            break
        name.append(itos[ix])
        if len(name) > 20:  # safety limit
            break
    
    return ''.join(name)
```

Run this multiple times. You'll get the same name every time. Boring. And often it gets stuck in loops because the greedy path leads to repetitive patterns.

Sampling Strategies let us balance between exploitation (picking likely characters) and exploration (allowing variety). Both bigram models and large language models can use similar sampling techniques: greedy, temperature-based, top-k, top-p.

### Temperature

Temperature is a way to sharpen or flatten the probability distribution before sampling. Here's how it works:

```python
def sample_name_with_temperature(P, itos, stoi, temperature=1.0):
    name = []
    ix = stoi['.']
    
    while True:
        probs = P[ix]
        # apply temperature
        if temperature != 1.0:
            logits = torch.log(probs + 1e-10)  # convert to log space
            logits = logits / temperature
            probs = torch.softmax(logits, dim=0)
        
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == stoi['.']:
            break
        name.append(itos[ix])
    
    return ''.join(name)
```

What does temperature do?

- **Temperature = 1.0**: Use the learned probabilities as-is
- **Temperature < 1.0**: Sharpen the distribution. High-probability characters become even more likely. Output becomes more "conservative" and predictable.
- **Temperature > 1.0**: Flatten the distribution. Low-probability characters get boosted. Output becomes more "creative" and chaotic.
- **Temperature → 0**: Approaches greedy decoding
- **Temperature → ∞**: Approaches uniform random sampling

Try it:

```python
print("Low temperature (0.5):")
for _ in range(5):
    print(sample_name_with_temperature(P, itos, stoi, temperature=0.5))

print("\nHigh temperature (2.0):")
for _ in range(5):
    print(sample_name_with_temperature(P, itos, stoi, temperature=2.0))
```

Low temperature gives you safer, more common-looking names. High temperature gives you weird stuff like "xqvz" because even unlikely characters get sampled.

### Local Computation

Here's something important: sampling is cheap. Training required iterating over the entire dataset many times, computing losses, computing gradients, updating weights. Sampling just requires running the trained model forward. No gradients. No updates. Just matrix lookups and random sampling.

This is why inference (generation) is so much cheaper than training. You're not learning anything new. You're just using what was already learned.

### The Model is Dreaming from Training Distribution

When you generate names, you're watching the model dream. It's producing sequences that follow the patterns and distributions learned from training names. Just as LLMs generate documents that resemble their training data, character-level models generate names that follow the character-pair frequencies they observed.

The bigram model learned that 'a' often follows start-of-name, that 'n' often follows 'a', that 'd' often follows 'n'. So it generates "and." and similar patterns. It's not intelligent. It's pattern matching, compressed into a 27x27 matrix.

### Common Mistakes

**Forgetting the end condition**: Without checking for the end token, your loop runs forever. Always have a termination condition.

**Not handling edge cases**: What if your probability row sums to zero? Add small epsilon values or handle this explicitly.

**Confusing indices and characters**: Keep `itos` and `stoi` straight. One maps integers to strings, the other maps strings to integers.

### Success Indicators

You know you've got sampling working when:

1. Running the generator multiple times produces different outputs
2. Lower temperature produces more "normal" looking names
3. Higher temperature produces stranger outputs
4. The generated names feel vaguely like they could be real names, even if many are nonsense

### What We Built

Over these four lessons, you built a complete language model from scratch:

1. **Lesson 1**: Loaded data and understood bigram statistics
2. **Lesson 2**: Built a probability matrix representing P(next | current)
3. **Lesson 3**: Trained with cross-entropy loss
4. **Lesson 4**: Sampled to generate new sequences

This is the same loop that GPT uses. Different architecture, different scale, same fundamental process: learn probability distributions over next tokens, then sample from those distributions to generate.

The bigram model is terrible at generating realistic names because it only looks at one character of context. In Chapter 3, we'll extend to n-grams, using multiple previous characters. In Chapter 4, we'll introduce attention, letting tokens communicate across arbitrary distances.

But the core idea you learned here, Iterative Generation from learned probability distributions, that stays the same all the way up to the largest models.

## Practice

Build a character-level bigram model from scratch, implement a training loop that updates weights based on prediction errors, and create a sampling function that generates text character by character.

### Setup

You'll need Python with NumPy. No deep learning frameworks, we're building from first principles.

Start with a small text file (a few paragraphs of any text you like). Create these components:

1. **Character vocabulary**: Map each unique character to an integer and back
2. **Count matrix**: A 2D array where `counts[i][j]` tracks how often character `j` follows character `i`
3. **Probability matrix**: Normalize each row of counts to get `P(next | current)`
4. **Sampling function**: Given a starting character, repeatedly sample from the probability distribution to generate new text

### Success Criteria

- Your model learns from the training text (probabilities reflect actual patterns)
- Generated text shows character-level patterns from your training data (common letter pairs appear frequently)
- You can generate 100+ characters without crashing
- You understand why the output looks like gibberish but *structured* gibberish

### Common Pitfalls

**Division by zero**: Some characters might never appear at the start of your text. Add smoothing (add 1 to all counts before normalizing) or handle the edge case.

**Forgetting to normalize**: Each row of your probability matrix must sum to 1. Use `row / row.sum()`.

**Confusing indices**: Keep your mappings straight. If `char_to_idx['a'] = 5`, then `idx_to_char[5]` must equal `'a'`.

**Sampling wrong**: Use `np.random.choice` with your probability distribution, not `argmax`. Argmax gives you the most likely character every time, boring and repetitive.

### Stretch Goals

- Compare output trained on Shakespeare vs. Python code vs. song lyrics
- Compute the loss (negative log likelihood) on held-out text
- Visualize your probability matrix as a heatmap

## Assessment

By the end of this module, you should be able to:

- **Explain language modeling** as the task of predicting the next token given previous context, and articulate why this simple formulation underlies all modern text generation systems
- **Work with probability distributions** over vocabularies, understanding how a model outputs probabilities for every possible next token and why these probabilities must sum to one
- **Calculate and interpret cross-entropy loss** as the measure of how wrong your model's predictions are, and explain why minimizing this loss improves generation quality
- **Implement sampling from a distribution** to generate text, understanding why we sample rather than always picking the highest-probability token
- **Apply temperature scaling** to control the randomness of generation, predicting how temperature values above and below 1.0 will affect the output distribution

Success in this module means you can take a trained bigram model, feed it a character, and explain exactly what happens at each step: how the input becomes a number, how that number produces a probability distribution, how loss measures prediction quality during training, and how sampling with temperature produces the next character during generation. You should be able to look at a probability distribution, adjust the temperature, and correctly predict whether the output will become more random or more deterministic. Most importantly, you should see that this simple model, despite its limitations, contains the same fundamental operations that power systems a million times larger.

---

# Micrograd

## Lesson 1: Automatic Differentiation

You are about to learn the single most important algorithm in modern machine learning. Not attention. Not transformers. The algorithm that makes all of them possible: automatic differentiation.

Here is the whole idea: neural networks learn by adjusting their parameters to make better predictions. But how do you know which direction to adjust them? You need to compute gradients. Derivatives. The slope that tells you "if I nudge this number up a tiny bit, does my error go up or down?"

For a network with ten parameters, you could compute these gradients by hand. For a network with a billion parameters, you cannot. Automatic differentiation is more scalable than manual counting for learning from data because it leverages the chain rule to compute gradients efficiently and exactly for arbitrarily complex computational graphs, without requiring manual derivation or approximation.

Let me show you what this means with actual code.

### Why This Matters

Think about what training actually is. You have parameters, which are numerical values in a massive function that collectively implement a statistical model of language patterns. Training is the process of finding these values. But finding them requires knowing which direction to move each one.

Consider a simple function:

```python
def f(x):
    return x * x + 3 * x
```

If x = 2, then f(2) = 4 + 6 = 10. Now, what is the derivative at x = 2? You probably remember from calculus: f'(x) = 2x + 3, so f'(2) = 7.

But what if your function looks like this:

```python
def messy_function(a, b, c, d, e):
    x = a * b + c
    y = x * x
    z = y + d * e
    w = sin(z) * cos(x)
    return w * w + log(abs(z) + 1)
```

Now compute the partial derivative with respect to each of a, b, c, d, e. By hand. Go ahead, I will wait.

This is what neural networks look like. Millions of operations chained together. Manual counting, symbolically deriving gradients for each operation, does not scale beyond trivial models due to the exponential growth in complexity.

### How Automatic Differentiation Works

The Key Components are surprisingly simple. We need to track two things:

1. The value of each computation
2. How that value depends on its inputs

Let us build this from scratch. Here is a minimal Value class:

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
```

Look at what happens with multiplication. When we compute `a * b`, the derivative of the output with respect to `a` is `b`, and the derivative with respect to `b` is `a`. We store this relationship in `_backward`. Later, when we know how much the final loss depends on the output (`out.grad`), we can compute how much it depends on the inputs using the chain rule.

This is the Implementation Strategy: build a computational graph forward, then walk it backward to accumulate gradients.

### The Chain Rule in Action

The chain rule says: if y = f(g(x)), then dy/dx = dy/dg * dg/dx.

In code, this becomes: each node's gradient equals its local gradient times the gradient flowing back from downstream.

```python
a = Value(2.0)
b = Value(3.0)
c = a * b      # c = 6.0
d = c + a      # d = 8.0
```

Now suppose d is our loss and we want gradients. We set d.grad = 1.0 (the derivative of d with respect to itself is 1). Then we walk backward:

1. d = c + a, so c.grad += 1.0 and a.grad += 1.0
2. c = a * b, so a.grad += b.data * c.grad = 3.0 * 1.0 = 3.0, and b.grad += a.data * c.grad = 2.0 * 1.0 = 2.0

Final gradients: a.grad = 4.0 (it contributed twice), b.grad = 2.0.

Why It Works: we are just applying the chain rule systematically, operation by operation, in reverse order. No symbolic manipulation. No approximation. Exact gradients.

### Forward Mode vs Reverse Mode

There are actually two ways to do automatic differentiation.

Forward mode: compute derivatives alongside values, propagating from inputs to outputs. Good when you have few inputs and many outputs.

Reverse mode: compute values first, then propagate gradients from outputs to inputs. Good when you have many inputs and few outputs.

Neural networks have millions of inputs (parameters) and one output (loss). Reverse mode wins. This is why backpropagation exists. It is just reverse mode automatic differentiation applied to neural networks.

### Common Mistakes

**Mistake 1: Forgetting to zero gradients.** Notice we use `+=` in the backward functions. Gradients accumulate. If you run backward twice without zeroing, you get wrong answers. Every training loop needs `param.grad = 0` before backward.

**Mistake 2: Breaking the graph.** If you do `x.data = x.data + 1` instead of `x = x + 1`, you modify the value without creating a new node. The gradient path is broken. Always use operations that create new Value objects.

**Mistake 3: Thinking this is numerical differentiation.** Numerical differentiation computes (f(x+h) - f(x))/h for tiny h. It is slow (one forward pass per parameter) and approximate. Automatic differentiation is exact and computes all gradients in one backward pass.

### Context: Where This Fits

Remember from the Core Architecture of this course: LLMs are created through lossy compression of internet data. The compression happens during training. Training requires gradients. Gradients require automatic differentiation.

Just to give you a sense, it would only require about 500 lines of C with no other dependencies to implement the neural network architecture, and that uses basically the parameters to run the model. But training that model, finding those parameters in the first place, requires running automatic differentiation billions of times.

The circuits inside a neural network, all those weights and connections, are found by gradient descent. Gradient descent needs gradients. Now you know how to compute them.

### Success Indicators

You understand this lesson if you can:

1. Explain why manual differentiation does not scale
2. Trace through a simple computation graph and compute gradients by hand
3. Explain why we use reverse mode instead of forward mode for neural networks
4. Identify when gradient accumulation will cause bugs

In the next lesson, we will implement the full Value class with more operations, including the nonlinearities that give neural networks their power. Then we will build actual neurons and train them.

The magic of deep learning, the thing that makes billions of parameters learnable, is this simple application of calculus. The chain rule, applied systematically, at scale. That is automatic differentiation. That is how neural networks learn.

## Lesson 2: Value Class Implementation

Let's build something. Open your editor and type this:

```python
class Value:
    def __init__(self, data):
        self.data = data
```

That's a Value. It holds a number. Not very impressive yet. But here's the thing: this tiny class is going to become the foundation of everything. By the end of this lesson, you'll have a Value that remembers where it came from, knows who its parents are, and can trace its own ancestry back through an entire computation.

### Why We Need More Than Numbers

Think about what happens when you multiply two numbers:

```python
a = 2.0
b = 3.0
c = a * b  # c is 6.0
```

Python gives you 6.0. That's it. The number 6.0 has no memory. It doesn't know it came from multiplying 2 and 3. It's just 6.

But for backpropagation to work, we need memory. We need to know that c came from a and b through multiplication. We need a computational graph.

This is what the source materials mean when they say: "each value in a computational graph tracks its data, gradient, operation that produced it, and child nodes, enabling gradient computation via backpropagation."

Let's build that.

### The Full Value Class

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
```

Four attributes. Let's understand each one:

**self.data** - The actual number. The 6.0.

**self.grad** - The gradient. Starts at zero. This will store "how much does the final loss change if I nudge this value a little?" We initialize it to zero because before we do any backpropagation, we don't know the gradient yet.

**self._prev** - The children. The values that created this one. For c = a * b, the children of c are {a, b}. We use a set because order doesn't matter and we want fast lookup.

**self._op** - The operation. A string like '*' or '+' that tells us how the children combined to make this value.

The underscore prefix on _prev and _op is a Python convention meaning "this is internal, don't touch it from outside." Users of our Value class shouldn't need to mess with these directly.

### Making Values Talk to Each Other

A Value that just holds data is useless. We need Values that can combine:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

Now watch what happens:

```python
a = Value(2.0)
b = Value(3.0)
c = a + b  # calls a.__add__(b)
d = a * b  # calls a.__mul__(b)
```

When we compute c = a + b, Python calls our __add__ method. That method creates a NEW Value with:
- data = 5.0 (the sum)
- _prev = {a, b} (the parents)
- _op = '+' (how we got here)

The value c now remembers its entire history. It knows it came from adding a and b.

### Building a Computational Graph

Let's trace through a more complex example:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b      # d = -6.0, children = {a, b}, op = '*'
e = d + c      # e = 4.0, children = {d, c}, op = '+'
```

What does the graph look like?

```
a (2.0) ----\
             *---> d (-6.0) ----\
b (-3.0) ---/                    +---> e (4.0)
                                /
c (10.0) ----------------------/
```

Every Value knows its immediate parents. And those parents know their parents. So from e, we can trace back through the entire computation: e came from d and c, d came from a and b.

This is the computational graph. It's a directed acyclic graph (DAG) where nodes are Values and edges represent "was used to compute."

### A Debugging Trick

Add a repr method so you can actually see what's happening:

```python
def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
```

Now when you print a Value, you see both its data and its gradient:

```python
a = Value(2.0)
print(a)  # Value(data=2.0, grad=0.0)
```

This is invaluable for Debugging Difficulty. When backpropagation goes wrong, and it will go wrong, you need to see what's happening inside your Values.

### Common Mistakes

**Mistake 1: Forgetting to wrap raw numbers**

```python
a = Value(2.0)
b = a * 3  # CRASH! 3 doesn't have a.data attribute
```

Our __mul__ assumes other is a Value. It tries to access other.data, but 3 is just an int. We'll fix this later, but for now, always wrap your numbers.

**Mistake 2: Mutating data instead of creating new Values**

```python
# WRONG
a.data = a.data + 5

# RIGHT
a = a + Value(5.0)
```

The whole point is to track the computation graph. If you mutate data directly, you lose the history.

**Mistake 3: Confusing the graph direction**

_prev stores the INPUTS to an operation, not the outputs. When we compute c = a + b, the _prev of c is {a, b}. The children are the things that came BEFORE, not after. This feels backwards at first but makes sense for backpropagation: we start at the loss and walk backwards through _prev to reach the weights.

### Why This Matters for Neural Networks

Remember from the source materials: "Neural networks are mathematical expressions that take inputs (data and weights) and produce outputs (predictions or loss), and backpropagation works on any such expression, not just neural networks."

A neural network is just a big pile of additions and multiplications (and a few other operations). If every addition and multiplication creates a Value that remembers its parents, then the entire neural network becomes one giant computational graph.

The loss at the end is a single Value. That Value's _prev points to the Values that created it. Those point to their parents. All the way back to the input data and the weights.

And once we have that graph, we can walk backwards through it, computing gradients as we go. That's backpropagation. That's how neural networks learn.

### Success Indicators

You understand this lesson if you can:

1. Create Values and combine them with + and *
2. Trace through a computation and draw the graph by hand
3. Explain what _prev and _op store and why
4. Predict what the _prev set will contain for any expression

Try this exercise: what are the _prev sets for each Value in this computation?

```python
x = Value(3.0)
y = Value(4.0)
z = x * x + y * y
```

If you can answer that, you're ready for the next lesson, where we'll add the _backward function that actually computes gradients.

The Value class is complete enough to build graphs. Next, we teach it to differentiate.

## Lesson 3: Forward Pass

You have a Value class that can wrap numbers and track what created them. Now we make it think.

The forward pass is where computation actually happens. You feed inputs through a series of operations, and outputs emerge at the other end. In neural networks, this is where predictions get made. In our micrograd engine, this is where we build the computational graph that backpropagation will later traverse.

Let me show you what I mean.

### Building Your First Computation Graph

Open your Python environment. We are going to build something and watch it grow.

```python
# Start with two input values
a = Value(2.0, label='a')
b = Value(-3.0, label='b')

# Perform operations
c = a * b
c.label = 'c'

d = Value(10.0, label='d')
e = c + d
e.label = 'e'

f = Value(-2.0, label='f')
L = e * f
L.label = 'L'
```

What just happened? We created a chain of operations. Each Value object knows its data, but more importantly, each one knows its children and what operation produced it. When we computed `c = a * b`, the resulting Value `c` stored references to `a` and `b` as its children, and recorded that multiplication was the operation.

This is the computational graph. It is a directed acyclic graph where nodes are values and edges represent dependencies. The forward pass builds this graph, node by node, operation by operation.

Print out `L.data` and you should get:

```python
print(L.data)  # -8.0
```

Walk through it manually. `a * b = 2.0 * -3.0 = -6.0`. Then `c + d = -6.0 + 10.0 = 4.0`. Then `e * f = 4.0 * -2.0 = -8.0`. The math checks out.

But the magic is not in the math. The magic is in the structure we built while doing the math.

### Why Structure Matters

Here is the thing about neural networks. The expert puts it well: the neural network architecture and the forward pass of that network, everything is algorithmically understood and open. We can give the network some inputs and we can measure the outputs. We can basically measure their behavior.

The forward pass is how we measure behavior. You feed in data, you get predictions. But to learn, to improve those predictions, you need to know how each piece contributed to the final answer. That is what the computational graph gives you.

Think of it like a recipe. If your cake tastes bad, knowing the final taste does not help you fix it. You need to know which ingredient caused the problem. The computational graph is your ingredient list with quantities and mixing instructions.

### Extending Operations

Our Value class needs more operations to be useful. Let us add them one at a time.

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

Now add subtraction and division:

```python
    def __sub__(self, other):
        return self + (other * Value(-1.0))
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')
        return out
```

Notice something clever here. Subtraction is implemented as addition with a negated value. Division is multiplication by an inverse. This means we only need to implement backward passes for addition, multiplication, and power. The others come for free.

### The Exponential and Its Special Property

For neural networks, we need the exponential function. It has a beautiful property that makes backpropagation simple.

```python
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        return out
```

The derivative of the exponential function e^x is e^x. So in backpropagation through exp, the local derivative is the same as the forward pass output. This is not just mathematically elegant. It means during the backward pass, we already have the derivative computed. We just reuse the output value.

This matters for softmax, which neural networks use constantly. Softmax involves exponentials, and this property keeps the backward pass clean.

### A Neuron's Forward Pass

Let us build an actual neuron. A neuron takes inputs, multiplies each by a weight, sums them up, adds a bias, and applies an activation function.

```python
# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias
b = Value(6.8813735870195432, label='b')

# forward pass
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'

# activation (tanh)
o = n.tanh(); o.label = 'o'
```

We need to implement tanh:

```python
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        return out
```

The forward pass computes `o`, the neuron's output. But it also builds the entire graph from inputs to output. Every Value knows what created it. Every operation is recorded.

### Common Mistakes

When initializing neural network parameters, avoid setting biases exactly to zero. This can mask gradient calculation errors by simplifying expressions. If your bias is zero and something goes wrong in backpropagation, the bug might hide because zero times anything is zero.

Use small random values instead. The bias value I used above, 6.8813735870195432, looks arbitrary. It is. That is the point. Arbitrary values expose bugs that clean values hide.

Another mistake: forgetting that each operation creates a new Value. If you write `a = a + b`, you are not modifying the original `a`. You are creating a new Value and rebinding the name `a` to it. The old `a` still exists in the graph. This is actually what you want, but it confuses people who expect in-place modification.

### Success Indicators

You know your forward pass is working when:

1. You can trace any output back to its inputs through the `_prev` attribute
2. Each Value knows its operation via `_op`
3. The numerical results match hand calculations
4. You can visualize the graph and it makes sense

Try this: starting from your output `L` or `o`, recursively walk through `_prev` and print each node. You should see the entire computation history.

The forward pass is complete when you arrive at a single number, the loss, that expresses how well this neural network works with the current setting of parameters. That number is the starting point for backpropagation.

Next lesson, we go backwards.

## Lesson 4: Backward Pass

You have a computation graph. You have a forward pass that flows from inputs to outputs. Now comes the part that makes neural networks learn: flowing backwards.

The backward pass is where we answer the question that matters. For every single weight in our network, we need to know: if I nudge this weight a tiny bit, how does the loss change? That answer is the gradient, and computing it efficiently is what backpropagation does.

### Why We Go Backwards

Here is the thing that took researchers decades to figure out. You could compute gradients by brute force. Nudge each weight by a tiny epsilon, run the forward pass again, see how the output changed. But if you have a million weights, that is a million forward passes. Neural networks would be completely impractical.

The backward pass computes all gradients in a single sweep. One forward pass, one backward pass, and you have the gradient for every weight in the network. This is not magic. This is the chain rule, applied systematically.

### The Chain Rule in Action

Remember from the last lesson: every Value object knows what operation created it and what its inputs were. The backward pass exploits this. We start at the output (the loss) and work backwards through the graph, at each step asking: how does this node's output affect the final loss?

The chain rule says: the gradient of a node with respect to the loss equals the local derivative of the operation times the upstream gradient (the gradient flowing in from above).

Let me show you what this means concretely. Say we have `c = a + b`, and we know that `dc/dloss = 0.5` (the upstream gradient). What are `da/dloss` and `db/dloss`?

For addition, the local derivative with respect to each input is 1. So:
- `da/dloss = 1.0 * 0.5 = 0.5`
- `db/dloss = 1.0 * 0.5 = 0.5`

The gradient flows through unchanged. Addition distributes the gradient equally.

Now multiplication. Say `c = a * b` with `a = 2.0`, `b = 3.0`, and upstream gradient 0.5.
- `da/dloss = b * 0.5 = 3.0 * 0.5 = 1.5`
- `db/dloss = a * 0.5 = 2.0 * 0.5 = 1.0`

For multiplication, the local derivative with respect to one input is the other input. This is why multiplication mixes gradients in interesting ways.

### Implementing the Backward Pass

We need to add a `_backward` function to each operation. This function takes the upstream gradient and computes the gradients for the inputs.

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
```

Notice something critical: we use `+=` not `=`. If a tensor is used multiple times in the forward pass, the gradients from each usage must be summed during the backward pass. This is one of the most common bugs when implementing backprop manually. If a value contributes to the loss through two different paths, both contributions matter.

### The Topological Sort

We cannot just call `_backward()` on nodes in any order. We need to process nodes from outputs to inputs, ensuring that when we process a node, all nodes that depend on it have already been processed. This is a topological sort.

```python
def backward(self):
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

We start by setting the gradient of the output to 1.0. The loss with respect to itself has gradient 1. Then we walk backwards through the sorted nodes, calling each `_backward()` function.

### More Operations

Let us add tanh, which we need for our neurons:

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')
    
    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
```

You see here when we implemented tanh, the forward pass of this operation to calculate the tanh was actually a fairly complicated mathematical expression, but because it is a clustered mathematical expression, when we did the backward pass we did not individually backward through the x and the two times and the exp. We used the known derivative of tanh, which is `1 - tanh(x)^2`. This is much simpler and more efficient.

For elementwise operations like log(x), the local derivative is 1/x, and the chain rule multiplies this by the upstream gradient. The derivative of the exponential function e^x is e^x, so in backpropagation through exp, the local derivative is the same as the forward pass output.

### Common Mistakes

When initializing neural network parameters, avoid setting biases exactly to zero, as this can mask gradient calculation errors by simplifying expressions. If your bias is zero and you have a bug in how you handle it, the bug might not show up because zero times anything is zero.

The second common mistake: forgetting to accumulate gradients. Use `+=`, always.

Third: getting the topological order wrong. If you process a node before processing all nodes that depend on it, you will have incomplete gradients.

### Verifying Your Implementation

Using PyTorch's autograd for gradient checking is a reliable way to verify the correctness of manually implemented backward passes. Build the same computation in PyTorch, call `.backward()`, and compare the gradients. They should match to many decimal places.

```python
import torch

# Your implementation
a = Value(2.0)
b = Value(3.0)
c = a * b
c.backward()

# PyTorch verification
a_pt = torch.tensor(2.0, requires_grad=True)
b_pt = torch.tensor(3.0, requires_grad=True)
c_pt = a_pt * b_pt
c_pt.backward()

print(f"Your grad: {a.grad}, PyTorch grad: {a_pt.grad.item()}")
```

### Why This Matters

Manually implementing the backward pass is a valuable exercise that improves debugging skills and deepens understanding of how gradients flow. About ten years ago, manually writing backward passes was standard practice in deep learning, whereas today it is mostly done only for educational purposes.

But understanding it is not optional. Backpropagation is a leaky abstraction. You cannot just stack arbitrary differentiable functions, backpropagate, and expect everything to work automatically without understanding the internals. When things go wrong, and they will, you need to understand what the gradients are doing.

### Success Indicators

You know you have understood this lesson when:
- You can trace gradient flow through any computation graph by hand
- You understand why gradients must be accumulated, not replaced
- You can implement the backward pass for new operations given their mathematical derivatives
- Your gradients match PyTorch's to numerical precision

The whole idea behind this approach is to manually implement the backward pass as an educational exercise to deepen understanding of how gradients flow, rather than relying solely on automatic differentiation as a black box.

Next lesson, we put everything together and train an actual neural network with the autograd engine we have built.

## Lesson 5: Training a Neural Network

You have built something remarkable. Over the last four lessons, you constructed a `Value` class that tracks computations, implemented the forward pass to compute outputs, and built the backward pass to compute gradients via the chain rule. Now we put it all together.

This is the moment where the magic happens. We are going to train a tiny neural network entirely with your autograd engine. No PyTorch. No TensorFlow. Just the code you wrote yourself.

### Why This Matters

Remember from the course overview: "parameters or weights are not like entries in a database, but are numerical values in a massive function that collectively implement a statistical model of language patterns, and that training is the process of finding these values."

That sentence contains the whole idea. Training is finding values. Your backward pass tells you how to nudge each value to make the output better. Now we do the nudging.

### Building a Neuron

A neuron computes a weighted sum of inputs, adds a bias, and applies a nonlinearity. Here it is in your micrograd:

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
```

Look at what is happening. Each weight `wi` and bias `b` is a `Value` object. When you compute `wi * xi`, your `__mul__` method creates a new `Value` that remembers its parents. When you sum them up, your `__add__` chains more operations. When you call `tanh()`, you add the final nonlinearity to the graph.

The entire computation is being recorded automatically.

### Building a Layer

A layer is just multiple neurons looking at the same inputs:

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

### Building the MLP

Stack layers together. This is your multi-layer perceptron:

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Create a small network:

```python
n = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4, 1 output
```

This network has 3 inputs, two hidden layers with 4 neurons each, and 1 output. Count the parameters: first layer has 4 neurons each with 3 weights and 1 bias, so 16. Second layer has 4 neurons each with 4 weights and 1 bias, so 20. Output layer has 1 neuron with 4 weights and 1 bias, so 5. Total: 41 parameters.

Tiny by modern standards. GPT-3 has 175 billion. But the gestalt is identical.

### The Training Loop

Here is where everything comes together. We have:
- A dataset (inputs and desired outputs)
- A model (your MLP)
- A loss function (how wrong are we?)
- Gradients (which direction to move each parameter)

```python
# Simple dataset: learn XOR-like function
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs
```

Now the training loop:

```python
for k in range(100):
    
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Backward pass
    for p in n.parameters():
        p.grad = 0.0  # zero gradients first!
    loss.backward()
    
    # Update
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(f"step {k} loss {loss.data}")
```

Run this. Watch the loss decrease. That is learning.

### How It Works

The forward pass builds a massive computation graph. Every operation on every `Value` creates nodes and edges. When you compute the loss, you have a single `Value` at the top that depends on all 41 parameters through chains of operations.

The backward pass walks this graph in reverse. Starting from the loss, it computes how much each parameter contributed to the error. This is the chain rule, applied automatically, through the `_backward` functions you implemented.

The update step is gradient descent. Each parameter moves a small amount in the direction that reduces the loss. The learning rate (0.05 here) controls how big a step you take.

### Common Mistakes

**Forgetting to zero gradients.** Gradients accumulate by default. If you do not reset them to zero before each backward pass, they pile up from previous iterations. Your training will behave erratically.

**Learning rate too high.** The loss will explode or oscillate wildly. Start small (0.01 or 0.001) and increase if training is too slow.

**Learning rate too low.** Training will work but take forever. You will see the loss decrease in tiny increments.

**Not enough iterations.** Neural networks need many passes over the data. If loss is still decreasing, keep going.

### Success Indicators

After 100 iterations, your loss should be very small (under 0.01). Check your predictions:

```python
for x, y in zip(xs, ys):
    print(f"target: {y}, predicted: {n(x).data:.4f}")
```

The predictions should be close to the targets. Not perfect, but close.

### The Bigger Picture

You just trained a neural network using an autograd engine you built yourself. The process you followed is exactly what happens inside PyTorch, JAX, and every other deep learning framework. The only difference is scale and optimization.

Working with scalar values instead of tensors is pedagogically useful because it allows for a clear, step-by-step understanding of the fundamental mechanics of automatic differentiation and backpropagation without the abstraction and complexity of tensor operations. You can manually trace gradient flow through each elementary operation, seeing exactly how the chain rule is applied at each node.

Production systems use tensors for efficiency through parallelized operations and hardware acceleration. But the underlying mathematics is what you just implemented.

In the next chapter, we build n-gram models with proper matrix multiplication and batching. The concepts you learned here, computational graphs, gradients, backpropagation, training loops, carry forward unchanged. You understand them now not as vocabulary but as code you wrote.

That understanding is permanent. When you use `loss.backward()` in PyTorch, you will know exactly what is happening under the hood. When something breaks, you will have intuition about where to look. When you read papers about improved training procedures or improved architectures, you will understand what they are improving.

This is the deep path. It takes longer, but you arrive somewhere different.

## Practice

Build a Value class that tracks computation graphs, implement forward and backward passes, and train a tiny neural network entirely with your autograd engine.

### Setup

Create a new file called `micrograd.py`. You'll build everything from scratch, no PyTorch, no NumPy for the core implementation.

### Part 1: The Value Class

Implement a `Value` class with these operations:
- Addition (`__add__`, `__radd__`)
- Multiplication (`__mul__`, `__rmul__`)
- Power (`__pow__`)
- Negation (`__neg__`)
- Subtraction (`__sub__`)
- Division (`__truediv__`)
- ReLU activation (`relu`)
- Tanh activation (`tanh`)

Each operation must:
1. Compute the forward pass (the actual value)
2. Store a `_backward` function that computes gradients when called

### Part 2: Backward Pass

Implement a `backward()` method on `Value` that:
1. Builds a topological ordering of all nodes in the graph
2. Sets `self.grad = 1.0` (the gradient of a value with respect to itself)
3. Calls `_backward()` on each node in reverse topological order

### Part 3: A Tiny Neural Network

Build these classes using only your `Value` objects:
- `Neuron`: weights, bias, and an activation function
- `Layer`: a list of neurons
- `MLP`: a multi-layer perceptron (stack of layers)

Then train your network on this toy dataset:

```python
# XOR-like problem
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # targets
```

### Success Criteria

1. **Gradient check passes**: For a simple expression like `a * b + c`, verify your gradients match numerical gradients (compute `(f(x+h) - f(x-h)) / 2h` for small h ≈ 1e-5)

2. **Backward propagates correctly**: After calling `backward()` on a loss value, all upstream nodes should have non-zero gradients

3. **Network trains**: Your MLP's loss should decrease over ~100 iterations of gradient descent

4. **Final loss < 0.01**: The network should fit the tiny dataset nearly perfectly

### Common Pitfalls

**Gradients accumulate, not replace.** Use `+=` in your backward functions:
```python
self.grad += out.grad  # correct
self.grad = out.grad   # wrong, breaks when a value is used twice
```

**Forgetting to zero gradients.** Before each backward pass during training, reset all gradients to zero. Otherwise they accumulate across iterations.

**Topological sort must handle DAGs.** A value can be used multiple times. Your sort needs to visit each node only once, after all its children.

**Integer division in Python.** Make sure `__truediv__` handles the case where `other` is a plain Python number.

### Stretch Goals

- Visualize your computation graph using graphviz
- Add more operations: `exp`, `log`
- Implement a `zero_grad()` method that walks the graph and resets all gradients
- Compare your gradients to PyTorch's on the same computation

## Assessment

By the end of this module, you should be able to:

- **Explain computational graphs**: Describe how mathematical expressions can be represented as directed acyclic graphs where nodes are operations and edges carry values, and why this representation is essential for automatic differentiation.

- **Apply the chain rule**: Given a composition of functions, correctly compute derivatives by multiplying local gradients along paths from output to input, understanding why this decomposition makes complex derivatives tractable.

- **Compute gradients manually and verify them**: For small computational graphs, calculate gradients by hand using the chain rule, then verify your answers match what automatic differentiation produces.

- **Distinguish forward mode from reverse mode autodiff**: Explain why forward mode computes one input's influence on all outputs while reverse mode computes all inputs' influence on one output, and articulate why reverse mode is preferred for neural networks where outputs (loss) are few and inputs (parameters) are many.

- **Implement backpropagation**: Write code that traverses a computational graph in reverse topological order, accumulating gradients from outputs back to inputs, handling the case where a node contributes to multiple downstream computations.

Success in this module means you can take any small mathematical expression, draw its computational graph, and trace through backpropagation step by step, predicting exactly what gradient each node will receive before running any code. When you implement your own Value class and it produces the same gradients as PyTorch's autograd, you will know the magic has become mechanics. More importantly, when you encounter a bug in gradient computation later in this course, you will have the mental model to diagnose it: you will think in terms of graph structure, local derivatives, and gradient flow, not in terms of mysterious framework errors.

---

# N-gram Model

## Lesson 1: Multi-layer Perceptrons

You have a bigram model. It predicts the next character by looking at exactly one character. That is embarrassingly limited. The word "the" appears after thousands of different single letters, so your model has no idea whether "t" should be followed by "h" or "o" or "a" because it cannot see what came before the "t".

We need depth. We need to look at more context. We need layers.

This is where multi-layer perceptrons come in. An MLP is just layers of neurons stacked on top of each other, where each layer transforms its input through a weighted sum followed by a nonlinearity. That is the entire idea. Weighted sums and squishing functions, repeated.

Let us build one.

### Why Depth Matters

Here is the problem with a single layer. Suppose you want to learn that "th" predicts "e" but "to" predicts "o". With one layer looking at one character, you cannot distinguish these cases. The model sees "t" and has to make a single prediction that works for all contexts where "t" appears.

But with two characters of context and a hidden layer, something interesting happens. The first layer can learn features like "this is a 't' followed by 'h'" versus "this is a 't' followed by 'o'". The second layer can then map these features to different predictions.

Depth lets you build hierarchical representations. The first layer detects simple patterns. The second layer combines those patterns into more complex ones. This is how you go from "I see the letter 't'" to "I see the beginning of the word 'the'".

Transformer neural networks process token sequences through mathematical expressions involving embedding, attention blocks, and multi-layer perceptron blocks. The MLP blocks are doing exactly this kind of hierarchical feature extraction. You are learning the building block that scales all the way up to GPT-4.

### The Architecture

An MLP has three types of components:

**Input layer**: Your raw data. For us, this will be embeddings of the previous N characters.

**Hidden layers**: Where the learning happens. Each hidden layer computes:
```
output = activation(input @ weights + bias)
```

**Output layer**: Produces your final prediction. For language modeling, this is a probability distribution over the next token.

The `@` symbol is matrix multiplication. The `activation` is a nonlinear function. The `weights` and `bias` are learned parameters. That is the entire recipe.

Let me show you what this looks like in code:

```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # First layer: input to hidden
        self.W1 = torch.randn(input_size, hidden_size) * 0.01
        self.b1 = torch.zeros(hidden_size)
        
        # Second layer: hidden to output
        self.W2 = torch.randn(hidden_size, output_size) * 0.01
        self.b2 = torch.zeros(output_size)
    
    def forward(self, x):
        # First layer with activation
        h = x @ self.W1 + self.b1
        h = gelu(h)
        
        # Output layer
        out = h @ self.W2 + self.b2
        return out
```

That is a complete two-layer MLP. The forward pass flows from input through hidden layer to output. Each transition involves matrix multiplication, bias addition, and (except for the output) a nonlinearity.

### Matrix Multiplication: The Core Operation

Matrix multiplication is the fundamental operation of deep learning. When you compute `x @ W`, you are doing something specific: each row of the output is a weighted combination of the columns of W, with weights determined by the corresponding row of x.

Think of it this way. Your input x is a batch of vectors. Your weight matrix W contains a set of learned directions. The multiplication asks: "How much does each input align with each learned direction?"

This is why GPUs are so important. Matrix multiplication is embarrassingly parallel. Every element of the output can be computed independently. Modern GPUs can do trillions of these operations per second.

When you see a model described as having billions of parameters, those parameters are almost entirely in weight matrices. The computation is almost entirely matrix multiplication. Everything else is just bookkeeping.

### Activation Functions: GELU vs ReLU

Why do we need the nonlinearity at all? Because without it, stacking layers does nothing. If you compute `(x @ W1) @ W2`, that is mathematically identical to `x @ (W1 @ W2)`. You could collapse all your layers into one. The nonlinearity breaks this equivalence and lets depth actually matter.

The classic activation is ReLU: `max(0, x)`. It is simple. It works. But it has a problem called "dying ReLU" where neurons that output negative values get stuck at zero forever.

GELU (Gaussian Error Linear Unit) is smoother. Instead of a hard cutoff at zero, it gradually transitions. The formula is approximately:
```
gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Do not memorize that. The intuition is: GELU lets small negative values through with small probability, which helps gradients flow during training. Modern transformers use GELU almost universally.

### Where Learners Get Stuck

**Dimension mismatches**: The output size of layer N must equal the input size of layer N+1. If your hidden layer outputs 128 dimensions, the next layer's weight matrix must have 128 rows. Draw out the shapes before you code.

**Forgetting the nonlinearity**: If you write `h = x @ W1 + b1` and then immediately `out = h @ W2 + b2`, you have a linear model. You need the activation between layers.

**Initialization scale**: If you initialize weights too large, activations explode. Too small, gradients vanish. The `* 0.01` in the code above is crude but functional. We will cover proper initialization later.

**Confusing batch dimension**: When you have a batch of 32 examples, each with 64 features, your input shape is (32, 64). The matrix multiplication broadcasts correctly, but you need to track which dimension is which.

### How This Connects to Transformers

The Transformer architecture, from the paper "Attention Is All You Need", uses MLPs inside each block. After the attention mechanism lets tokens talk to each other, an MLP processes each token's representation independently. This is sometimes called a position-wise feed-forward network.

The pattern is: attention (tokens communicate) then MLP (tokens think). Attention then MLP. Repeated many times. The MLP you just learned is half of each transformer block.

Neural network parameters are randomly initialized at the start of training and adjusted through optimization to match statistical patterns in the training data. The MLP learns what patterns matter. The attention learns which tokens should influence which. Together, they build representations powerful enough to predict the next token in arbitrary text.

### Success Indicators

You understand this lesson if you can:

1. Explain why a single linear layer cannot learn XOR but two layers can
2. Write the forward pass of a 3-layer MLP from memory
3. Identify the shapes of all weight matrices given input and output dimensions
4. Explain why removing activation functions collapses depth to width

Next lesson, we build embeddings, the mechanism that turns discrete tokens into continuous vectors that MLPs can actually process.

## Lesson 2: Embedding Layers

You have a problem. Your neural network speaks one language: continuous numbers. Your data speaks another: discrete characters. The letter 'a' is not a number. It is not 0.7 or -3.2 or any floating point value. It is just 'a'. A symbol. A discrete thing.

How do you bridge this gap?

The answer is embedding layers. And understanding them deeply will change how you think about neural networks forever.

### The Lookup Table That Learns

Let us start with code. Here is the simplest possible embedding layer:

```python
import numpy as np

# 27 characters: a-z plus space
vocab_size = 27
# Each character becomes a vector of 10 numbers
embedding_dim = 10

# Initialize random embedding matrix
E = np.random.randn(vocab_size, embedding_dim) * 0.01

# Look up the embedding for character index 5
char_idx = 5
embedding_vector = E[char_idx]  # Shape: (10,)
```

That is it. An embedding layer is a matrix where each row corresponds to one token in your vocabulary. When you want the embedding for token 5, you grab row 5. Pure lookup.

But here is what makes this powerful: the values in that matrix are learnable parameters. They start random. Then gradient descent adjusts them. Over thousands of training steps, tokens that behave similarly in language get pushed toward similar regions of the embedding space.

The quote from our source materials captures this precisely: "Embedding layers in neural networks convert discrete tokens (words or characters) into continuous vector representations that can capture semantic relationships through learned proximity in vector space."

Proximity in vector space. That phrase deserves your attention. After training, the embedding for 'a' might be close to the embedding for 'e' because both are vowels and appear in similar contexts. The embedding for 'q' might be close to 'u' because they almost always appear together in English. These relationships emerge from the data. You do not program them. The network discovers them.

### Why Not One-Hot Encoding?

You might be thinking: why not just use one-hot vectors? Represent 'a' as [1,0,0,0,.], 'b' as [0,1,0,0,.], and so on.

You could. Early models did. But one-hot encoding has a fatal flaw: every token is equally distant from every other token. The vector for 'a' is exactly as far from 'b' as it is from 'z'. There is no structure. No relationships. No semantic meaning in the geometry.

Embeddings fix this. With a 10-dimensional embedding space, you have 10 degrees of freedom per token. The network can use those dimensions to encode whatever structure helps it predict the next character. Maybe dimension 3 encodes vowel-ness. Maybe dimension 7 encodes frequency. You do not decide. The gradients decide.

This connects to a deeper principle from the source materials: "The parameters of a neural network are a lossy compression of the training data, similar to a zip file but with information loss, capturing the gestalt rather than exact copies." Your embedding matrix is compressing the statistical relationships between characters into a dense numerical format.

### The Shape of Things

Pay attention to shapes. They will save you hours of debugging.

```python
# Vocabulary size: V (how many unique tokens)
# Embedding dimension: D (how many numbers per token)
# Batch size: B (how many examples at once)
# Sequence length: T (how many tokens per example)

E = np.random.randn(V, D)  # Shape: (V, D)

# Single token lookup
token_idx = 5
single_embedding = E[token_idx]  # Shape: (D,)

# Batch of tokens
token_indices = np.array([5, 2, 8, 1])  # Shape: (4,)
batch_embeddings = E[token_indices]  # Shape: (4, D)

# Full batch with sequences
batch_indices = np.array([
    [5, 2, 8],
    [1, 0, 3],
])  # Shape: (B, T) = (2, 3)
full_embeddings = E[batch_indices]  # Shape: (B, T, D) = (2, 3, D)
```

The embedding layer takes integer indices and outputs floating point vectors. Input shape: (B, T). Output shape: (B, T, D). This output then feeds into your neural network layers, which expect continuous inputs.

### Dimensionality Alignment

How big should your embeddings be? This is a design choice with real tradeoffs.

Small embeddings (say, 8 or 16 dimensions) are fast and memory efficient but limit how much information each token can carry. Large embeddings (256, 512, or more) can encode richer representations but require more parameters and computation.

For character-level models with small vocabularies, embedding dimensions of 16 to 64 work well. For word-level models or subword tokenizers with vocabularies of 50,000 or more, you might use 768 or 1024 dimensions.

The key constraint is Dimensionality Alignment: your embedding dimension must match what your subsequent layers expect as input. If your MLP's first layer expects 64-dimensional inputs, your embeddings must be 64-dimensional.

### Context Window: Concatenating Embeddings

In our n-gram model, we predict the next character from several previous characters. If we use a context of 3 characters, we need to combine 3 embeddings into one input vector.

The simplest approach: concatenation.

```python
context_length = 3
embedding_dim = 16

# Three characters in context
context_indices = np.array([5, 2, 8])  # 'f', 'c', 'i' maybe
context_embeddings = E[context_indices]  # Shape: (3, 16)

# Flatten to single vector
mlp_input = context_embeddings.flatten()  # Shape: (48,)
```

Your MLP now receives a 48-dimensional input: the concatenation of three 16-dimensional embeddings. The first 16 dimensions encode "what was the first character," the next 16 encode "what was the second character," and so on.

This is where Sequence Length matters. Longer contexts mean larger input vectors, which mean more parameters in your first MLP layer. The computational cost grows with context length. This is one reason why attention mechanisms, which we will cover later, became so important. They handle long contexts more efficiently.

### Common Mistakes

**Forgetting to include embeddings in your optimizer.** The embedding matrix contains learnable parameters. If you do not pass it to your optimizer, it will never update, and your model will be stuck with random embeddings forever.

**Using embeddings that are too small.** If your embedding dimension is smaller than the information content you need to encode, your model will struggle. Start with at least vocab_size / 4 as a rough heuristic.

**Confusing token indices with embeddings.** Token index 5 is an integer. The embedding for token 5 is a vector of floats. These are different objects with different shapes and different purposes.

### Success Indicators

You know your embeddings are working when:

1. Loss decreases during training. If embeddings are broken, the model cannot learn.
2. Similar characters cluster in embedding space. After training, compute distances between embeddings. Vowels should be closer to vowels than to consonants.
3. The model generates plausible character sequences, not uniform random noise.

### What Comes Next

You now have a way to convert discrete characters into continuous vectors. These vectors feed into matrix multiplications, which we cover next. The embedding layer is your interface between the symbolic world of language and the numerical world of neural networks.

Remember: this exact same principle scales up. GPT models use embeddings. Llama uses embeddings. The vocabulary is larger (50,000 tokens instead of 27 characters), the dimensions are bigger (4096 instead of 16), but the core idea is identical. A learnable lookup table that converts discrete symbols into continuous vectors.

That is the power of understanding fundamentals. The same pattern appears at every scale.

## Lesson 3: Matrix Multiplication

You've got embeddings now. Each character maps to a vector of numbers. But a lookup table doesn't learn anything. It just stores. To actually learn patterns in language, we need to transform those vectors, combine them, extract features from them.

The operation that does all of this is matrix multiplication.

This isn't just one tool among many. Matrix multiplication is THE fundamental mathematical primitive being scaled up across language modeling, autograd, and neural network training. When people talk about GPUs being essential for deep learning, they're really talking about hardware optimized for matrix multiplication. When transformer models get bigger, they're adding more matrix multiplications. When we train with gradient descent, we're computing matrix multiplications in the forward pass and more matrix multiplications in the backward pass.

Let's build it from scratch so you understand exactly what's happening.

### Starting with Vectors

Before matrices, let's multiply vectors. Say you have two vectors of the same length:

```python
a = [2, 3, 1]
b = [4, 1, 5]
```

The dot product multiplies corresponding elements and sums them:

```python
def dot(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

result = dot(a, b)  # 2*4 + 3*1 + 1*5 = 8 + 3 + 5 = 16
```

That's it. Multiply element-wise, sum everything up, get a single number.

Why does this matter? The dot product measures similarity. If two vectors point in the same direction, their dot product is large. If they're perpendicular, it's zero. If they point opposite ways, it's negative. This geometric intuition will matter enormously when we get to attention in Chapter 04.

### From Dots to Matrices

A matrix is just a grid of numbers. Matrix multiplication takes two grids and produces a third. Here's the rule: to get the value at row i, column j of the output, take the dot product of row i from the first matrix with column j from the second matrix.

```python
def matmul(A, B):
    # A is shape (m, n)
    # B is shape (n, p)
    # Output is shape (m, p)
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # Initialize output with zeros
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            # Dot product of row i of A with column j of B
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C
```

Notice the constraint: the inner dimensions must match. If A is (m, n), B must be (n, something). The n's have to agree. This is where shape errors come from, and you'll see a lot of them.

### Why This is the Core Operation

Think about what a neural network layer does. You have a batch of inputs, each a vector. You want to transform each vector into a new vector. The transformation should be learnable.

Matrix multiplication does exactly this. If your input is a matrix X where each row is one example, and W is a weight matrix, then X @ W gives you transformed outputs. Each row of the output is a weighted combination of the features in that row of the input, with the weights coming from W.

The Attention Weight Matrices in transformers? Matrix multiplication. The Forward Network Matrices in the MLP blocks? Matrix multiplication. The Embedding Weight Matrices that turn token IDs into vectors? Also matrix multiplication (though often implemented as a lookup for efficiency).

Every layer in a transformer is fundamentally doing: take some input, multiply by some learned weights, maybe add a bias, apply a nonlinearity. The multiply step is matmul.

### Concrete Matrix Operations in Our Model

Let's trace through what happens in our n-gram model. Say we're looking at 3 characters of context, each embedded as a 16-dimensional vector. We concatenate them into a single vector of length 48.

Our first hidden layer has 128 neurons. The weight matrix W1 is shape (48, 128). The bias b1 is shape (128,).

```python
# x is our concatenated input, shape (batch_size, 48)
# W1 is shape (48, 128)
# After matmul, we get shape (batch_size, 128)
hidden = matmul(x, W1)  # Each example becomes 128 numbers
hidden = [h + b for h, b in zip(hidden, b1)]  # Add bias
hidden = gelu(hidden)  # Apply nonlinearity
```

The second layer takes those 128 features and produces scores for each possible next character. If we have 65 characters in our vocabulary, W2 is shape (128, 65).

```python
# hidden is shape (batch_size, 128)
# W2 is shape (128, 65)
logits = matmul(hidden, W2)  # shape (batch_size, 65)
```

Those logits become probabilities through softmax. The whole thing is just matmuls with nonlinearities sandwiched between.

### Hardware Utilization and Why GPUs Matter

Here's something that might seem like a detail but changes everything: matrix multiplication is embarrassingly parallel. Look at our implementation again. Each element C[i][j] can be computed independently. Nothing depends on any other element.

GPUs have thousands of small cores designed to do exactly this kind of parallel computation. A CPU might compute one dot product at a time. A GPU computes thousands simultaneously. This is why Hardware Utilization matters so much in deep learning. A well-optimized matmul uses all those cores. A poorly written one leaves most of them idle.

When you hear about Hardware Precision and mixed-precision training, it's about doing these matrix multiplications in lower precision (16-bit instead of 32-bit floats) to double throughput. The numbers are a bit less accurate, but you can do twice as many operations per second.

### Common Mistakes

**Shape mismatches.** You'll see errors like "cannot multiply matrices of shape (64, 48) and (64, 128)." The inner dimensions don't match. Check your shapes obsessively.

**Forgetting the batch dimension.** When you have a batch of examples, your input isn't a vector, it's a matrix where each row is one example. Your weight matrix stays the same, but now you're doing many transformations at once.

**Confusing matmul with element-wise multiplication.** Element-wise multiplication (Hadamard product) multiplies corresponding elements: same shapes in, same shape out. Matmul combines dimensions differently. They're completely different operations.

### Success Indicators

You understand matrix multiplication when you can:

1. Predict the output shape from two input shapes without running code
2. Explain why the inner dimensions must match
3. Trace through a two-layer network and state the shape at each step
4. Explain why GPUs accelerate matmul but not arbitrary Python code

The Conceptual Journey here is seeing that all the complexity of neural networks reduces to this one operation repeated many times with different learned weights. The Input Representation goes in, gets transformed by matmul after matmul, and predictions come out. Everything else is just choosing which matmuls to do and how to arrange them.

Next lesson, we'll add the nonlinearities that go between these matmuls. Without them, stacking layers would be pointless. With them, depth creates power.

## Lesson 4: MLP with GELU Activation

You have embeddings. You have matrix multiplication. Now we build the thing that actually learns.

The Layer Perceptron is where neural networks get their power. Not from any single layer, but from stacking them. Depth creates representational capacity. A single layer can only learn linear relationships. Two layers can approximate any continuous function. This is not theory. This is the architecture that scales from our tiny character model all the way up to the 70 billion parameters in Llama 2 70B.

Let us build one.

### The Core Architecture

Here is what we are making:

```python
class MLP:
    def __init__(self, n_embd, hidden_dim):
        self.W1 = randn(n_embd, hidden_dim) * 0.02
        self.b1 = zeros(hidden_dim)
        self.W2 = randn(hidden_dim, n_embd) * 0.02
        self.b2 = zeros(n_embd)
    
    def forward(self, x):
        # x shape: (batch, seq_len, n_embd)
        h = x @ self.W1 + self.b1      # project up
        h = gelu(h)                     # nonlinearity
        out = h @ self.W2 + self.b2    # project back down
        return out
```

That is the whole thing. Two matrix multiplications with a nonlinearity sandwiched between them. The Forward Network Matrices take your embedding dimension, expand it to a larger hidden dimension, apply GELU, then project back down.

Run this on some random data:

```python
import numpy as np

def randn(*shape):
    return np.random.randn(*shape)

def zeros(*shape):
    return np.zeros(shape)

# We'll define gelu in a moment
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

n_embd = 64
hidden_dim = 256  # typically 4x the embedding dimension
batch_size = 4
seq_len = 8

mlp = MLP(n_embd, hidden_dim)
x = randn(batch_size, seq_len, n_embd)
out = mlp.forward(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
```

Input and output have the same shape. The MLP transforms each position independently. No information flows between positions here. That is what attention does. The MLP just processes each position through the same learned transformation.

### Why GELU and Not ReLU

You might know ReLU: `max(0, x)`. Simple. Fast. Worked great for years.

The Gaussian Error Linear Unit is different:

```python
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

Plot them both:

```python
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, gelu(x), label='GELU')
plt.legend()
plt.title('Activation Functions')
plt.grid(True)

plt.subplot(1, 2, 2)
# Derivatives
relu_grad = (x > 0).astype(float)
gelu_grad = np.gradient(gelu(x), x)
plt.plot(x, relu_grad, label='ReLU gradient')
plt.plot(x, gelu_grad, label='GELU gradient')
plt.legend()
plt.title('Gradients')
plt.grid(True)
plt.show()
```

ReLU has a hard cutoff at zero. Everything negative becomes exactly zero. This creates Sparse Activation Patterns, which sounds good but causes problems. Dead neurons. Harsh gradients. The gradient is either 1 or 0, nothing in between.

GELU is smooth. Small negative values get small negative outputs instead of hard zeros. The gradient flows more naturally. Empirical Performance on language models is consistently better. This is not about Mathematical Fidelity to some theoretical ideal. This is about what works when you train on billions of tokens.

Besides Leaky ReLU and the Exponential Linear Unit, GELU has become the standard for transformers. GPT-2, GPT-3, Llama, all use GELU or close variants.

### The Kaiming Initialization

Notice the `* 0.02` when we create weights:

```python
self.W1 = randn(n_embd, hidden_dim) * 0.02
```

This matters more than you might think.

Neural networks for language models expect inputs in a reasonable range. If your weights are too large, activations explode. Too small, gradients vanish. The Kaiming initialization scales weights based on layer dimensions to keep activations stable.

The proper formula:

```python
def kaiming_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return randn(fan_in, fan_out) * std
```

For our MLP:

```python
class MLP:
    def __init__(self, n_embd, hidden_dim):
        self.W1 = kaiming_init(n_embd, hidden_dim)
        self.b1 = zeros(hidden_dim)
        self.W2 = kaiming_init(hidden_dim, n_embd) * 0.02  # extra scaling for residual
        self.b2 = zeros(n_embd)
```

The extra `* 0.02` on W2 is specific to residual networks. We will cover why when we build the full transformer. For now, just know that initialization determines whether your network trains at all.

### Computational Efficiency

The hidden dimension is typically 4x the embedding dimension. Why?

```python
n_embd = 768       # GPT-2 small
hidden_dim = 3072  # 4 * 768
```

This expansion ratio comes from Empirical Performance studies. Smaller ratios underfit. Larger ratios cost too much compute for minimal gain. The 4x ratio hits a sweet spot.

Count the parameters:

```python
def count_params(n_embd, hidden_dim):
    w1_params = n_embd * hidden_dim + hidden_dim  # W1 + b1
    w2_params = hidden_dim * n_embd + n_embd      # W2 + b2
    return w1_params + w2_params

params = count_params(768, 3072)
print(f"MLP parameters: {params:,}")  # 4,722,432 per layer
```

Nearly 5 million parameters per MLP layer. GPT-2 small has 12 layers. That is 56 million parameters just in MLPs, roughly half the total model.

### Batch Normalization vs Layer Norm

You might have heard of Batch Normalization. We do not use it here.

Batch Normalization normalizes across the batch dimension. This creates problems during inference when batch size is 1. It also creates dependencies between samples in a batch, which breaks the autoregressive property we need for language modeling.

Layer normalization normalizes across the feature dimension:

```python
def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)
```

This will wrap our MLP in the full transformer. For now, understand that normalization keeps Extreme Values under control during training.

### Where Learners Get Stuck

**Problem 1: Shape mismatches.** The Embedding Weight Matrices output `(batch, seq_len, n_embd)`. Your MLP must preserve this. Check shapes at every step.

**Problem 2: Forgetting biases.** Biases are small but matter. A network without biases has less expressive power.

**Problem 3: Wrong GELU formula.** The exact approximation matters for Numerical Precision. Use the tanh version shown above, not the erf version, for Computational Efficiency.

### Success Indicators

You have built this correctly if:

1. Output shape equals input shape
2. Random inputs produce outputs with mean near 0 and std near 1 (with proper initialization)
3. Gradients flow through both layers (we will verify this when we add backprop)

### Putting It Together

Here is the complete, tested implementation:

```python
class MLP:
    def __init__(self, n_embd, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = 4 * n_embd
        
        self.W1 = np.random.randn(n_embd, hidden_dim) * np.sqrt(2.0 / n_embd)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_embd) * 0.02
        self.b2 = np.zeros(n_embd)
    
    def forward(self, x):
        self.x = x  # cache for backward
        self.h_pre = x @ self.W1 + self.b1
        self.h = gelu(self.h_pre)
        out = self.h @ self.W2 + self.b2
        return out
```

We cache intermediate values for the backward pass. That comes next lesson.

The Key Components are in place: embedding lookup, matrix multiplication, and now the MLP. The Output Projection Matrix in W2 brings us back to embedding dimension. The Attention Weight Matrices are what we build in Chapter 4.

This MLP, simple as it looks, is the same architecture used in every transformer. The same one processing your queries to ChatGPT. The same one in Llama 2 70B. Scale changes. The Implementation Strategy does not.

# Lesson 5: Batched Training

You have an MLP. You have GELU activations. You have embeddings. Now we need to make this thing actually learn, and learn efficiently. The secret is batching, and it changes everything about how neural networks train in practice.

## Why Batching Matters

Let's start with the naive approach. You could train your model one example at a time:

```python
for i in range(len(training_data)):
    x, y = training_data[i]
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    update_weights()
```

This works. It's also painfully slow. Every single example triggers a weight update. Every single example requires its own forward pass, backward pass, gradient computation. Your GPU sits mostly idle because you're feeding it one tiny computation at a time.

Here's what batching looks like instead:

```python
batch_size = 32

for i in range(0, len(training_data), batch_size):
    batch = training_data[i:i+batch_size]
    xs = stack([x for x, y in batch])  # Shape: (32, context_length)
    ys = stack([y for x, y in batch])  # Shape: (32,)
    
    logits = model(xs)  # Shape: (32, vocab_size)
    loss = cross_entropy(logits, ys)
    loss.backward()
    update_weights()
```

Same total computation, but now we process 32 examples simultaneously. The GPU loves this. Matrix multiplication is embarrassingly parallel, and modern GPUs have thousands of cores waiting to be used. Feed them a batch and they all light up at once.

## Building the Data Pipeline

Let's construct batches for our n-gram model. Remember, we're predicting the next character from a context window:

```python
def create_dataset(text, context_length):
    """Convert text into (context, target) pairs."""
    # Convert characters to integers
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    data = [stoi[ch] for ch in text]
    
    X, Y = [], []
    for i in range(len(data) - context_length):
        context = data[i:i+context_length]
        target = data[i+context_length]
        X.append(context)
        Y.append(target)
    
    return torch.tensor(X), torch.tensor(Y)

X, Y = create_dataset(text, context_length=3)
# X shape: (num_examples, 3)
# Y shape: (num_examples,)
```

Now the batch generator:

```python
def get_batch(X, Y, batch_size):
    """Sample a random batch."""
    ix = torch.randint(0, len(X), (batch_size,))
    return X[ix], Y[ix]

xb, yb = get_batch(X, Y, batch_size=32)
# xb shape: (32, 3)
# yb shape: (32,)
```

That random sampling is important. We don't iterate through the data in order. We sample randomly. This is Stochastic Gradient Descent in action. The randomness prevents the model from memorizing sequences and helps it generalize.

## The Training Loop

Here's the complete training loop with batching:

```python
# Hyperparameters
batch_size = 32
learning_rate = 0.01
max_steps = 10000

# Model (from previous lessons)
model = MLP(vocab_size=27, embed_dim=10, hidden_dim=100, context_length=3)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for step in range(max_steps):
    # Get batch
    xb, yb = get_batch(X, Y, batch_size)
    
    # Forward pass
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Logging
    if step % 1000 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

Let me walk through what happens each iteration:

1. `get_batch` samples 32 random examples
2. Forward pass computes predictions for all 32 simultaneously
3. `cross_entropy` averages the loss across the batch
4. `zero_grad` clears old gradients (crucial, they accumulate otherwise)
5. `backward` computes gradients for all parameters
6. `step` updates weights using those gradients

The loss you see is an estimate based on 32 examples, not the true loss over all data. That's fine. The estimate is noisy but unbiased, and the noise actually helps Training Stability by preventing the model from getting stuck in sharp local minima.

## Understanding Batch Dimensions

Here's where people get confused. Let's trace the shapes through the model:

```python
# Input batch
xb = get_batch(.)  # Shape: (batch_size, context_length) = (32, 3)

# Embedding lookup
emb = self.embedding(xb)  # Shape: (32, 3, embed_dim) = (32, 3, 10)

# Flatten for MLP
emb_flat = emb.view(32, -1)  # Shape: (32, 30)

# First linear layer
h = self.fc1(emb_flat)  # Shape: (32, hidden_dim) = (32, 100)

# GELU activation
h = F.gelu(h)  # Shape: (32, 100)

# Output layer
logits = self.fc2(h)  # Shape: (32, vocab_size) = (32, 27)
```

The batch dimension flows through unchanged. Every matrix multiplication, every activation function, operates independently on each example in the batch. This is why batching is so efficient. The operations are identical, just applied to more data simultaneously.

## Sequence Length Multiplies Your Batch

Here's something subtle that the source materials highlight. In training batches for language modeling, each position in a sequence provides both an input (context) and target (next token) example, effectively multiplying the batch size by Sequence Length.

For our n-gram model with context length 3, each training example is one prediction. But later, with transformers, a single sequence of length 512 gives you 512 training examples. A batch of 32 sequences becomes 32 × 512 = 16,384 predictions per step. This is why transformers train so efficiently despite their size.

## Common Mistakes

**Forgetting zero_grad**: Gradients accumulate by default. If you don't zero them, your updates become garbage after the first step. Always call `optimizer.zero_grad()` before `backward()`.

**Batch size too large**: Bigger isn't always better. Very large batches can hurt generalization. Start with 32 or 64, increase if training is stable.

**Batch size too small**: Below 8 or so, the gradient estimates become too noisy. Training becomes unstable.

**Not shuffling data**: If you iterate through data in order, the model sees correlated examples in each batch. Random sampling breaks this correlation.

## Success Indicators

How do you know training is working?

1. **Loss decreases**: Should drop quickly at first, then slow down
2. **Loss is noisy but trending down**: Some variation is normal and healthy
3. **Generated text improves**: Sample periodically and check quality
4. **No NaN or Inf**: If you see these, something is broken (usually learning rate too high)

Try this: train with batch size 1 versus batch size 32. The batch-32 version should train roughly 10-20x faster in wall clock time while achieving similar final loss.

## The Bigger Picture

Batching seems like a simple optimization trick, but it's fundamental to how modern deep learning works. Model inference is computationally cheap and can be done locally, but training is extremely expensive and requires massive computational resources. Batching is what makes training feasible at all.

The models that power modern AI, everything from GPT to the systems described in the source materials, all rely on batched training across massive datasets. The Llama 2 70B model with its 70 billion parameters was trained on batches flowing through clusters of GPUs for weeks. The principles are identical to what you just implemented. The scale is different.

You now have all the pieces for Part I: a language model that predicts next characters, automatic differentiation to compute gradients, an MLP architecture with embeddings and GELU activations, and batched training to make it all efficient. In Part II, we add attention, and everything changes.

## Practice

Build embedding layers for character lookup, implement matrix multiplication, create a multi-layer perceptron with GELU activation, and train with batched gradient descent.

### Setup

Start with your bigram model from Chapter 01. You'll extend it to use multiple characters of context.

```python
import torch

# Load your text data
text = open('input.txt', 'r').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Context length: how many characters to look at
context_length = 3
```

### Your Tasks

**1. Build the embedding layer**

Create a lookup table that converts character indices to dense vectors:

```python
embedding_dim = 16
# Your embedding matrix: vocab_size x embedding_dim
# Given a character index, return its embedding vector
```

**2. Implement the forward pass**

For a context of N characters:
- Look up embeddings for each character
- Concatenate them into one vector (size: N × embedding_dim)
- Pass through hidden layer: `h = gelu(x @ W1 + b1)`
- Pass through output layer: `logits = h @ W2 + b2`

**3. Implement GELU activation**

```python
def gelu(x):
    # GELU(x) = x * Φ(x), where Φ is the cumulative distribution function
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    pass
```

**4. Train with batched gradient descent**

- Sample random batches of (context, target) pairs
- Compute cross-entropy loss
- Backpropagate and update weights

### Success Criteria

- Loss drops below 2.5 within 10,000 steps
- Generated text shows recognizable patterns beyond single characters (you should see common 2-3 letter combinations)
- Model produces different predictions for "th" versus "to" contexts

### Common Pitfalls

**Embedding concatenation shape**: If context_length is 3 and embedding_dim is 16, your flattened input to the MLP should be size 48, not 16.

**Weight initialization scale**: Initialize weights with small values (`* 0.01`). Large initial weights cause exploding gradients.

**Forgetting the bias terms**: Every linear layer needs both weights and biases.

**GELU versus ReLU**: GELU is smoother and works better for language models. Don't substitute ReLU just because it's simpler.

**Batch dimension confusion**: Your forward pass should handle batches. If `x` has shape `(batch_size, input_dim)`, then `x @ W` should work without reshaping.

### Stretch Goals

- Try different context lengths (2, 3, 5, 8) and compare loss curves
- Add a second hidden layer and observe the effect
- Visualize what the embeddings learn, do similar characters cluster together?

## Assessment

By the end of this module, you should be able to:

- **Explain what embeddings are and why they matter**: You understand that embeddings are learned vector representations that capture relationships between discrete tokens, and you can articulate why looking up a dense vector is more powerful than using one-hot encodings.

- **Perform matrix multiplication by hand and in code**: You can trace through what happens when a batch of embedded inputs multiplies against a weight matrix, understanding both the shapes involved and the semantic meaning of the operation.

- **Describe activation functions and their purpose**: You know why we need nonlinearities between layers, can explain the difference between ReLU and GELU, and understand what happens to a network that has no activation functions at all.

- **Build an MLP architecture from scratch**: Given a context size, embedding dimension, hidden size, and vocabulary size, you can construct a multi-layer perceptron that takes character indices as input and produces next-character probabilities as output.

- **Implement batching correctly**: You understand why we process multiple examples simultaneously, can explain how batch dimensions flow through matrix operations, and can debug shape mismatches when they occur.

Success in this module means you can take a blank Python file and build a working n-gram language model without looking at reference code. You should be able to draw the architecture on paper, annotate every tensor shape at every layer, and explain to someone else why each component exists. When you see an MLP block inside a transformer diagram, you should think "I know exactly what that does, I built one." The math should feel like machinery you operate, not magic you invoke.

---

# Attention

## Lesson 1: Attention Mechanism

You have built an n-gram model. It works. It predicts the next character by looking at a fixed window of previous characters. And that fixed window is exactly the problem we need to solve now.

Think about this sentence: "The cat that chased the mouse across the kitchen floor was hungry." When you get to "was," you need to know it refers to "cat," not "mouse" or "floor." That relationship spans eight words. A fixed window either misses it entirely or wastes computation on every position equally.

Attention solves this. Instead of treating all previous positions the same, attention lets each token ask: "Which previous tokens matter for predicting what comes next?"

### The Core Intuition

Let me give you a concrete example before we touch any math.

Imagine you have the sequence: "France -> Paris, Japan -> ?"

You need to predict "Tokyo." How do you know? Because you recognize a pattern. France maps to Paris. Japan should map to its capital. You are looking at "France -> Paris" and asking "what is the mapping correspondence here?" The answer is "capital of." Then you apply that same relationship to Japan.

This is attention. The query "Japan -> ?" looks back at previous tokens, finds the relevant pattern (France -> Paris), extracts the relationship (capital of), and uses it to generate the answer.

Now let's build this from scratch.

### Queries, Keys, and Values

Every token in your sequence gets three vectors: a Query, a Key, and a Value. Think of it like a library system:

- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "Here is my actual content."

When token 5 wants to gather information from previous tokens, it broadcasts its Query. Every previous token (1 through 4) has a Key. We compute compatibility scores between the Query and each Key. High score means "this token has what I'm looking for." Low score means "not relevant."

Then we use those scores to compute a weighted sum of all the Values. Tokens with high compatibility contribute more. Tokens with low compatibility contribute less.

Here is the concrete matrix operations version:

```python
# Assume we have embeddings X of shape (sequence_length, embedding_dim)
# We learn three weight matrices: W_Q, W_K, W_V

Q = X @ W_Q  # Queries: what each token is looking for
K = X @ W_K  # Keys: what each token offers
V = X @ W_V  # Values: the actual content to retrieve

# Compute attention weights
scores = Q @ K.T  # How well does each query match each key?
weights = softmax(scores)  # Normalize to probabilities
output = weights @ V  # Weighted sum of values
```

That's it. That's attention. Everything else is details.

### The Softmax Function

The softmax turns raw compatibility scores into attention weights that sum to one. If token 3 has a high score with your query and tokens 1, 2, 4 have low scores, softmax will give most of the weight to token 3.

```python
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()
```

But this naive implementation has a problem. If any value in x is large (say, 100), then exp(100) overflows. Your computer cannot represent that number. You get infinity, then NaN, then garbage.

The fix is simple and essential:

```python
def softmax(x):
    x_shifted = x - x.max()  # Subtract maximum value
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()
```

Subtracting the maximum does not change the output (the ratios stay the same) but keeps all exponents in a reasonable range. This numerical stability trick shows up everywhere in deep learning. Remember it.

### Scaling the Dot Product

There is another numerical issue. When your embedding dimension is large, the dot products between queries and keys can get very large. Large values push softmax into regions where it is nearly one-hot, meaning almost all attention goes to a single token. Gradients vanish. Learning stops.

The fix from The Transformer paper "Attention Is All You Need": divide by the square root of the key dimension.

```python
scores = (Q @ K.T) / np.sqrt(d_k)
```

This keeps the variance of the scores reasonable regardless of dimension. Scaled dot-product attention is now standard.

### Causal Masking: No Cheating Allowed

Here is a critical constraint for language modeling. When predicting token 5, you cannot look at tokens 6, 7, 8. That would be cheating. You would be using the answer to predict the answer.

We enforce this with a mask structure. Before softmax, we set all "future" positions to negative infinity:

```python
# Create causal mask
mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # Upper triangle
scores = scores - mask * 1e9  # Future positions become -infinity
weights = softmax(scores)  # -infinity -> 0 probability
```

This creates a pattern of gradual unblocking. Position 1 can only attend to itself. Position 2 can attend to positions 1 and 2. Position 3 can attend to 1, 2, and 3. And so on. The mask structure ensures autoregressive generation works correctly.

During training, this lets you process an entire sequence in parallel while still respecting the causal constraint. During inference, you generate one token at a time, and the mask naturally prevents looking ahead.

### Why This Matters

The abstract view is that attention implements a kind of Sparse Distributed Memory. Instead of storing information at fixed addresses, you store it in a distributed way and retrieve it by content. The Query describes what you want. The Keys and Values together form an associative memory that returns relevant content.

This is fundamentally different from the fixed-window approach of n-gram models. Attention can, in principle, look back across the entire context. A token at position 1000 can attend to position 1 if that is where the relevant information lives. The model learns which positions matter for which predictions.

The phase change from n-gram to attention is not just quantitative. It enables qualitatively different capabilities. Long-range dependencies. In-context learning. The ability to recognize patterns like "France -> Paris, Japan -> ?" and apply them on the fly.

### Common Mistakes

1. **Forgetting the scale factor.** Your model trains but learns nothing useful. Check that you are dividing by sqrt(d_k).

2. **Wrong mask shape.** Off-by-one errors in masking cause subtle bugs. Token 5 attending to token 6 during training corrupts your loss.

3. **Numerical instability in softmax.** If you see NaN in your attention weights, add the max-subtraction trick.

4. **Confusing Q, K, V roles.** Remember: Query asks, Key advertises, Value delivers. The Query comes from the token doing the attending. Keys and Values come from tokens being attended to.

### Success Indicators

You know you understand attention when:

- You can explain why we need three separate projections (Q, K, V) instead of just comparing embeddings directly
- You can draw the mask structure for a sequence of length 5
- You can trace through the computation for a single query attending to three keys
- You understand why scaling by sqrt(d_k) prevents gradient problems

Next lesson, we will implement this in code and see attention weights visualized. You will watch your model learn which tokens to attend to for different predictions.

## Lesson 2: Query, Key, Value Projections

In the previous lesson, you built an intuition for the Attention Mechanism as a way for tokens to communicate across arbitrary distances. You saw how tokens can "look at" other tokens and gather information from them. But we left something vague: how does a token decide what to look for? And how does it decide what information to share?

That's what Query, Key, Value projections solve. Let's build them.

### The Problem We're Solving

Think about what happens when you read the sentence "The cat sat on the mat because it was tired."

When you hit the word "it," your brain does something remarkable. It asks a question: "What does 'it' refer to?" Then it scans backward through the sentence, checking each word to see if it's a good match. "mat"? No, mats don't get tired. "cat"? Yes, cats get tired.

Your brain just performed attention. The word "it" had a query ("what noun am I referring to?"), and each previous word had a key ("here's what kind of thing I am"). The match between query and key determined where attention flowed.

But here's the thing: the word "cat" contains more information than just "I'm a noun that can get tired." It also carries semantic content that "it" needs to absorb. That's the value.

Three roles. Three projections.

### Building the Projections

Let's make this concrete. To remind you very briefly, we have these tokens lined up in a sequence and each token at this stage of the attention emits three vectors: the query, key, and value.

Here's the code:

```python
import torch
import torch.nn as nn

class QKVProjection(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        
        # Three separate linear projections
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_head)
        K = self.W_k(x)  # (batch, seq_len, d_head)
        V = self.W_v(x)  # (batch, seq_len, d_head)
        return Q, K, V
```

That's it. Three matrix multiplications. Each token's embedding gets projected into three different spaces.

Let's trace through what happens with a concrete example:

```python
d_model = 64   # embedding dimension
d_head = 32    # head dimension
seq_len = 4    # "The cat sat."

# Random embeddings for our 4 tokens
x = torch.randn(1, seq_len, d_model)

qkv = QKVProjection(d_model, d_head)
Q, K, V = qkv(x)

print(f"Input shape: {x.shape}")      # (1, 4, 64)
print(f"Query shape: {Q.shape}")      # (1, 4, 32)
print(f"Key shape: {K.shape}")        # (1, 4, 32)
print(f"Value shape: {V.shape}")      # (1, 4, 32)
```

Each of the 4 tokens now has three vectors of dimension 32. The Input Representation has been transformed into three specialized representations.

### What Each Projection Does

**Query (Q):** "What am I looking for?"

The query vector encodes what information this token wants to gather from other tokens. When token 3 ("sat") computes its query, it's essentially asking "which tokens have information relevant to me?"

**Key (K):** "What do I have to offer?"

The key vector encodes what information this token has available for others. It's like a label or a tag. When computing Attention Scores, queries get compared against keys to determine Compatibility Scores.

**Value (V):** "Here's my actual content."

The value vector contains the information that will actually be transmitted. Once attention decides that token 1 ("cat") is relevant to token 3 ("sat"), it's the value vector from "cat" that gets sent over.

The split is crucial. Keys and queries determine *where* attention flows. Values determine *what* information flows. This separation lets the network learn different things for "how to match" versus "what to transmit."

### The Mapping Correspondence

Here's an analogy that might help. Think of the France -> Paris, Japan -> ? pattern. If you're trying to complete this analogy:

- Your query encodes "I'm looking for a capital city relationship"
- Each country's key encodes "I have a capital city relationship with my value"
- Each country's value encodes the actual capital name

The query-key match tells you which countries are relevant. The value gives you the answer.

### Why Learned Projections?

You might wonder: why not just use the embeddings directly? Why transform them?

The raw embedding of a word like "cat" contains everything about that word: its part of speech, its semantic meaning, its typical contexts, everything. But for any given attention operation, we only care about specific aspects.

When "it" is looking for its referent, it cares about "animate noun that can get tired." When a verb is looking for its subject, it cares about "noun that can perform actions." Same word, different queries.

The projections let the network extract the relevant slice of information for each role. The weights W_q, W_k, and W_v are Learned Biases (in the sense of learned parameters, not bias terms) that the network discovers during training.

### Computing Attention from Q, K, V

Once we have the projections, the Concrete Matrix Operations for attention are straightforward:

```python
def attention(Q, K, V):
    d_k = Q.shape[-1]
    
    # Attention Scores: how well does each query match each key?
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Attention Weights: normalize with softmax
    weights = torch.softmax(scores, dim=-1)
    
    # Output Computation: weighted sum of values
    output = torch.matmul(weights, V)
    
    return output, weights
```

If you think about this not in terms of the matrix multiplies but in terms of comparing each query to each key: in normal self-attention, we have one query for each key, so every point compares to every other point.

The operations that get applied are: number one, the queries and the keys interact to give us the attention; then the softmax here normalizes the attention so it sums to one; then we use those Attention Weights to compute a weighted average of values.

### Common Mistakes

**Confusing keys and values.** Keys are for matching. Values are for content. If you mix them up, your attention will learn to match on the wrong features.

**Forgetting the scaling factor.** That division by sqrt(d_k) is critical. Without it, dot products can get very large, pushing softmax into regions where gradients vanish.

**Using bias terms carelessly.** Notice we set `bias=False` in our linear layers. Some implementations use biases, some don't. The original transformer used them, but many modern architectures (like Llama) skip them. Be consistent.

### Success Indicators

You understand this lesson if you can:

1. Explain why we need three separate projections instead of just one
2. Describe what role each of Q, K, V plays in the attention computation
3. Implement the projection layers from scratch without looking at the code
4. Trace through the shapes at each step of the computation

### What's Next

We've built the projections that create Q, K, and V. We've sketched how they combine to compute attention. But we glossed over something important: that softmax operation has numerical stability issues that will bite you in practice. And we haven't addressed how attention knows word order, since nothing we've built so far distinguishes position 1 from position 7.

Next lesson: we'll implement softmax properly and add positional encodings.

## Lesson 3: Scaled Dot-Product Attention

You have queries, keys, and values. Now we make them talk to each other.

In the last lesson, you built the projections that transform your input embeddings into Q, K, and V matrices. These are the raw ingredients. This lesson is about the recipe: how do we combine them to let every token decide which other tokens matter?

Let's build it first, then understand why it works.

### The Core Operation

Here is scaled dot-product attention in four lines:

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = softmax(scores, dim=-1)
    output = weights @ V
    return output
```

That's it. Four lines contain the mechanism that made transformers possible. Let's trace through what each line does.

### Step 1: Computing Compatibility Scores

```python
scores = Q @ K.transpose(-2, -1)
```

This matrix multiplication computes how much each query "matches" each key. If you have a sequence of 10 tokens, Q is shape `(10, d_k)` and K transposed is shape `(d_k, 10)`. The result is a `(10, 10)` matrix where entry `[i, j]` tells you how compatible token i's query is with token j's key.

Think about what this means. Token 5 broadcasts a query: "I'm looking for context about verbs." Tokens 0 through 4 each have keys that describe what information they offer. The dot product measures alignment between what token 5 wants and what each previous token provides.

Why dot product specifically? The expert explains this directly: "And we took the dot-product attention formulation largely because it allowed us to do attention as a matrix multiplication." And further: "dot-product attention could be expressed as a multiplication, and they're already kernels for being able to do matrix multiplication very effectively on the GPU."

This is a theme you'll see throughout deep learning. Hardware Utilization drives architectural choices. Matrix multiplications are fast on GPUs because decades of optimization have gone into them. The dot product isn't necessarily the best similarity function mathematically, but it's the one we can compute efficiently at scale.

### Step 2: The Scaling Factor

```python
scores = scores / (d_k ** 0.5)
```

Why divide by the square root of the key dimension? This is about Numerical Stability.

Here's the problem. When d_k is large (say, 64 or 128), dot products between random vectors tend to have large variance. Some scores end up huge, others tiny. When you feed these into softmax, the huge scores dominate completely. You get attention weights like `[0.99, 0.003, 0.002, 0.001,.]` where one token gets almost all the weight.

The expert puts it this way: "if the dot product actually becomes too big, and you can solve it under certain assumptions of mean and variance in the representations, you can" show that dividing by sqrt(d_k) keeps the variance roughly at 1 regardless of dimension.

Without scaling, your gradients vanish. The softmax saturates, producing near-one-hot outputs, and the gradient signal dies. Scaling keeps the Attention Scores in a reasonable range where softmax can produce meaningful distributions.

### Step 3: Softmax for Attention Weights

```python
weights = softmax(scores, dim=-1)
```

Softmax converts raw Compatibility Scores into Attention Weights that sum to 1. Each row of the weights matrix is a probability distribution over which tokens to attend to.

But here's where you'll get burned if you're not careful. Naive softmax has a numerical stability problem:

```python
# DON'T do this
def naive_softmax(x):
    return exp(x) / sum(exp(x))
```

If any value in x is large (say, 100), `exp(100)` overflows to infinity. Your attention weights become NaN.

The fix is simple but essential:

```python
# DO this
def stable_softmax(x):
    x_max = max(x)
    exp_x = exp(x - x_max)
    return exp_x / sum(exp_x)
```

Subtracting the maximum value before exponentiating doesn't change the result mathematically (you're multiplying numerator and denominator by the same constant), but it prevents overflow. The largest exponent is now `exp(0) = 1`.

This is the kind of detail that separates working code from code that mysteriously produces NaN after a few training steps. Hardware Precision matters.

### Step 4: Feature Weight Application

```python
output = weights @ V
```

The final step is straightforward. We take a weighted average of the value vectors, where the weights come from our attention distribution.

If token 5 assigned weights `[0.1, 0.05, 0.3, 0.15, 0.4]` to tokens 0-4, then token 5's output is:

```
output[5] = 0.1*V[0] + 0.05*V[1] + 0.3*V[2] + 0.15*V[3] + 0.4*V[4]
```

Each token's output is a mixture of information from all tokens it attended to, blended according to relevance.

### The Abstract View

Let's step back and see the full picture.

The Mapping Correspondence between the Concrete Matrix Operations and the Abstract View is:

1. **Q @ K.T**: "Who should I pay attention to?" Each query asks which keys are relevant.
2. **/ sqrt(d_k)**: Keep scores in a stable range for gradient flow.
3. **softmax**: Convert scores to a proper probability distribution.
4. **weights @ V**: Gather information according to those probabilities.

This is Input Representation flowing through Forward Processing to produce Output Computation. Every token starts with its own embedding, consults all other tokens through attention, and emerges with a context-aware representation.

### Common Mistakes

**Forgetting the scaling factor.** Your model might still train, but slowly and poorly. The gradients will be weak because softmax is saturated.

**Applying softmax on the wrong dimension.** You want each query to have its own distribution over keys. That's `dim=-1` (the last dimension, which corresponds to keys after the transpose).

**Not handling numerical stability.** Works fine on small toy examples, explodes on real data. Always use the max-subtraction trick.

**Confusing scores and weights.** Scores are raw dot products (can be any real number). Weights are post-softmax probabilities (between 0 and 1, sum to 1 per row).

### Success Indicators

You know you've got it when:

- Your attention weights sum to 1.0 along each row (within floating point tolerance)
- No NaN values appear during training, even after thousands of steps
- Attention patterns show structure (not uniform, not one-hot) when you visualize them
- The output shape matches your input shape: `(batch, seq_len, d_model)`

### Looking Ahead

We've built the core attention mechanism. But there's something missing. Right now, attention treats position 0 and position 100 identically. The model has no idea that some tokens come before others. In the next lesson, we add positional encodings so attention knows word order, and we implement causal masking so tokens can only attend to the past.

For now, make sure you can trace through the four lines of scaled dot-product attention and explain what each operation accomplishes. This is the beating heart of the transformer.

## Lesson 4: Softmax Implementation

You have attention scores sitting in a matrix. Raw numbers. Some positive, some negative, some huge, some tiny. Before you can use these to weight your values, you need to turn them into probability distributions. Each row needs to sum to one. Each element needs to be between zero and one.

That is what softmax does. And getting it right, with all the numerical stability tricks, is more important than you might think.

Let us build it.

### The Naive Implementation

Here is softmax in its purest mathematical form:

```python
def softmax_naive(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

Three lines. Exponentiate everything, then divide by the sum. Each row becomes a probability distribution.

Try it:

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 1.0, 1.0]])
print(softmax_naive(x))
```

You get something like:

```
tensor([[0.0900, 0.2447, 0.6652],
        [0.3333, 0.3333, 0.3333]])
```

The first row has most weight on the largest value (3.0). The second row, where all values are equal, gives uniform probabilities. So if all the logits are one, then because of the normalization inside the softmax, this will actually come out okay.

This works. Until it does not.

### Why Naive Softmax Breaks

Try this:

```python
x = torch.tensor([[1000.0, 1001.0, 1002.0]])
print(softmax_naive(x))
```

You get `nan`. Not a number. Your computation exploded.

Here is what happened: `exp(1000)` is approximately 10^434. That number does not fit in a 32-bit float. It overflows to infinity. Infinity divided by infinity gives you nan. Your gradients become garbage. Your training collapses.

Now, there are 17,000 logits here in a real model, so on top of there we have the softmax layer. Some of those logits will get large during training. If you do not handle this, you run into numerical issues if some of the logits take on too large values because we end up exponentiating them.

This is where Numerical Precision Safeguards become essential.

### The Stable Implementation

The fix exploits a mathematical property of softmax. For any constant c:

```
softmax(x) = softmax(x - c)
```

Subtracting the same value from every element in a row does not change the output. The exponentials scale, but the ratio stays identical.

So we pick c to be the maximum value in each row:

```python
def softmax_stable(x):
    # Find max in each row
    x_max = x.max(dim=-1, keepdim=True).values
    # Subtract max for stability
    x_shifted = x - x_max
    # Now exponentiate safely
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

What I am doing is I am finding the maximum in each row and I am subtracting it for the purposes of numerical stability. Now the largest exponent you compute is `exp(0) = 1`. Everything else is smaller. No overflow.

```python
x = torch.tensor([[1000.0, 1001.0, 1002.0]])
print(softmax_stable(x))
```

Now you get:

```
tensor([[0.0900, 0.2447, 0.6652]])
```

Same answer as before, but it actually computes.

### What Softmax Does to Attention

In the context of attention, softmax makes the distribution "peakier." The softmax operation makes attention distributions peakier by emphasizing large similarity values and suppressing smaller ones through exponential weighting.

Consider two attention score vectors:

```python
scores_flat = torch.tensor([[1.0, 1.1, 1.2, 1.3]])
scores_varied = torch.tensor([[0.0, 0.0, 0.0, 4.0]])

print(softmax_stable(scores_flat))   # [0.22, 0.24, 0.26, 0.29]
print(softmax_stable(scores_varied)) # [0.02, 0.02, 0.02, 0.95]
```

When scores are similar, attention spreads across tokens. When one score dominates, attention concentrates. This is how the model learns to focus on relevant context.

There is a deep connection here worth knowing. The softmax operation in Transformer attention can be closely approximated by the intersection of hyperspheres in a high-dimensional vector space, as used in Sparse Distributed Memory. This geometric interpretation helps explain why attention works so well for retrieval and association tasks.

### The Backward Pass

If you are implementing this manually, as we encouraged in the Micrograd chapter, you need the gradient. Manually implementing the backward pass for a neural network is a valuable exercise that improves debugging skills and deepens understanding of how gradients flow.

For softmax, the backward pass is subtle. The derivative of the exponential function e^x is e^x, so in backpropagation through exp, the local derivative is the same as the forward pass output. But softmax involves division too, and all outputs depend on all inputs through the normalization.

The full Jacobian is:

```
d(softmax_i) / d(x_j) = softmax_i * (delta_ij - softmax_j)
```

Where delta_ij is 1 when i equals j, 0 otherwise.

In code:

```python
def softmax_backward(grad_output, softmax_output):
    # grad_output: upstream gradient, same shape as softmax_output
    # For each row, compute: s * (grad - sum(grad * s))
    s = softmax_output
    grad_sum = (grad_output * s).sum(dim=-1, keepdim=True)
    return s * (grad_output - grad_sum)
```

Using PyTorch's autograd for gradient checking is a reliable way to verify the correctness of manually implemented backward passes:

```python
x = torch.randn(2, 5, requires_grad=True)
y = softmax_stable(x)
loss = y.sum()
loss.backward()

# Compare x.grad with your manual implementation
```

### Common Mistakes

**Forgetting keepdim.** When you compute the max or sum along a dimension, you need `keepdim=True` to broadcast correctly. Without it, shapes mismatch and PyTorch either errors or silently broadcasts wrong.

**Wrong dimension.** Softmax should operate on the last dimension in attention, where each row represents one query's scores across all keys. Using `dim=0` instead of `dim=-1` normalizes across the wrong axis.

**Ignoring the mask.** In causal attention, you set future positions to negative infinity before softmax. If you apply the mask after softmax, you break the probability distribution. The mask must come before.

```python
# Correct: mask before softmax
scores = scores.masked_fill(mask == 0, float('-inf'))
attn_weights = softmax_stable(scores)
```

**Not testing edge cases.** What happens when an entire row is masked? All values become negative infinity, exp gives zeros, and you divide zero by zero. Handle this case explicitly if your architecture allows it.

### Success Indicators

You know your softmax implementation is correct when:

1. Each row sums to exactly 1.0 (within floating point tolerance)
2. All values are between 0 and 1
3. Large input values (1000+) produce valid outputs, not nan or inf
4. Gradients match PyTorch's autograd to at least 5 decimal places
5. Uniform inputs produce uniform outputs

Run this test:

```python
def test_softmax():
    # Test 1: Rows sum to 1
    x = torch.randn(10, 50)
    s = softmax_stable(x)
    assert torch.allclose(s.sum(dim=-1), torch.ones(10))
    
    # Test 2: Numerical stability
    x_large = torch.tensor([[1000., 1001., 1002.]])
    s_large = softmax_stable(x_large)
    assert not torch.isnan(s_large).any()
    
    # Test 3: Gradient check
    x = torch.randn(3, 4, requires_grad=True)
    torch.autograd.gradcheck(softmax_stable, x)
    
    print("All tests passed")

test_softmax()
```

### Looking Ahead

You now have the two core operations of attention: scaled dot-product and softmax. In the next lesson, we add positional encodings. Because right now, your attention mechanism has no idea about word order. "The cat sat on the mat" and "mat the on sat cat the" look identical to it. We need to fix that.

## Lesson 5: Positional Encodings

Here is a puzzle for you. Run this code:

```python
import torch
import torch.nn.functional as F

# Three sentences with identical words, different order
sentences = [
    "dog bites man",
    "man bites dog", 
    "man dog bites"
]

# Pretend embeddings (same vector for same word)
vocab = {"dog": torch.tensor([1.0, 0.0]), 
         "bites": torch.tensor([0.0, 1.0]), 
         "man": torch.tensor([0.5, 0.5])}

def get_embeddings(sentence):
    return torch.stack([vocab[w] for w in sentence.split()])

# Compute attention scores (simplified, no projections)
def simple_attention(embeddings):
    scores = embeddings @ embeddings.T
    weights = F.softmax(scores, dim=-1)
    output = weights @ embeddings
    return output.sum(dim=0)  # Aggregate for comparison

for s in sentences:
    emb = get_embeddings(s)
    result = simple_attention(emb)
    print(f"{s}: {result}")
```

Run it. What do you notice?

All three sentences produce the exact same output. "dog bites man" and "man bites dog" are treated identically by attention. This is a news headline versus a bizarre incident, and our attention mechanism cannot tell them apart.

This is the problem. Self-attention is permutation-invariant, requiring the explicit addition of positional information to model sequence order. The attention mechanism we built in the previous lessons computes similarity between tokens based purely on their content. It has no idea which token came first, second, or third. To attention, your sentence is a bag of words floating in space.

### Why Position Matters

Think about what attention actually computes. For each query, it asks: "Which keys are similar to me?" Similarity is computed through dot products between vectors. Nothing in that computation depends on where tokens sit in the sequence.

But language is deeply positional. "The cat chased mouse" is incomplete. "The cat chased the mouse" makes sense. "The mouse chased cat" reverses the meaning entirely. Word order carries semantic information that pure content similarity cannot capture.

The original transformer paper, "Attention Is All You Need," solved this elegantly. In the original transformer, the positional embeddings are actually initialized and fixed, if I remember correctly, to sinusoids and cosines of different frequencies, and that's the positional coding, and it's fixed. Let me show you what that means.

### Sinusoidal Positional Encodings

The idea is simple: add a unique vector to each position. Position 0 gets one vector, position 1 gets another, and so on. The trick is choosing vectors that have useful mathematical properties.

```python
import numpy as np
import torch

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.
    
    max_len: maximum sequence length
    d_model: embedding dimension
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    # Create frequency bands
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-np.log(10000.0) / d_model)
    )
    
    # Even dimensions get sine, odd dimensions get cosine
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Generate encodings
pe = sinusoidal_positional_encoding(max_len=100, d_model=64)
print(f"Shape: {pe.shape}")
print(f"Position 0: {pe[0,:8]}")
print(f"Position 1: {pe[1,:8]}")
```

Each dimension oscillates at a different frequency. Low dimensions change slowly across positions. High dimensions change rapidly. This creates a unique fingerprint for each position, like a binary encoding but continuous.

Why sines and cosines? They have a beautiful property: the encoding at position p+k can be expressed as a linear function of the encoding at position p. This means the model can potentially learn to attend to relative positions, not just absolute ones.

### Adding Position to Your Model

Integration is straightforward. You add the positional encoding to your token embeddings before feeding them into attention:

```python
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        
        # Create positional encodings (fixed, not learned)
        pe = sinusoidal_positional_encoding(max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        seq_len = x.size(1)
        
        # Get token embeddings
        tok_emb = self.token_embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encodings
        pos_emb = self.pe[:seq_len,:]  # (seq_len, d_model)
        
        return tok_emb + pos_emb
```

Now test our earlier puzzle with positions:

```python
# With positional encodings added
pe = sinusoidal_positional_encoding(10, 2)

def get_embeddings_with_position(sentence):
    words = sentence.split()
    embs = torch.stack([vocab[w] for w in words])
    positions = pe[:len(words),:]
    return embs + positions

for s in sentences:
    emb = get_embeddings_with_position(s)
    result = simple_attention(emb)
    print(f"{s}: {result}")
```

Now the outputs differ. "dog bites man" and "man bites dog" are no longer identical to attention.

### Learned vs Fixed Encodings

The original transformer used fixed sinusoidal encodings. But there is another approach: let the model learn positional embeddings from scratch.

```python
class LearnedPositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        return tok_emb + pos_emb
```

Both approaches work. Learned embeddings are more flexible but cannot extrapolate beyond the maximum length seen during training. Sinusoidal encodings can theoretically handle any length, though model performance may degrade.

### Relative Position Encodings

Modern architectures often use relative position encodings, like rotary embeddings. These are superior to absolute encodings for modeling temporal relationships and enable better extrapolation to longer sequences.

The intuition: instead of asking "what position is this token at?", ask "how far apart are these two tokens?" Relative encodings bake this distance information directly into the attention computation.

We will not implement rotary embeddings here, but understand that they represent the current best practice. The key insight remains the same: attention needs position information injected somehow.

### Common Mistakes

**Forgetting to add positions entirely.** Your model will train. It will even produce somewhat coherent text. But it will struggle with anything order-dependent. Always check that positions are being added.

**Adding positions after attention instead of before.** The positional information needs to be present when computing Q, K, V projections. Add it to the input embeddings.

**Using positions beyond max_len with learned embeddings.** This crashes or produces garbage. Either use sinusoidal encodings or ensure your sequences never exceed the trained length.

### Success Indicators

You understand positional encodings when you can:

1. Explain why "cat chased mouse" and "mouse chased cat" would be identical to vanilla attention
2. Implement sinusoidal encodings from the formula
3. Choose between learned and fixed encodings for your use case
4. Recognize when position information is missing from a model's behavior

The attention mechanism is now complete. You have queries, keys, and values. You have scaled dot-product attention. You have numerically stable softmax. You have causal masking. And now you have positional encodings so your model knows that word order matters.

In the next chapter, we assemble these pieces into the full transformer architecture. The individual components you have built will stack into something far more powerful than their sum.

## Practice

Build Query, Key, Value projections, implement scaled dot-product attention, create a numerically stable softmax, and add positional encodings.

### Setup

Create a new file called `attention.py`. You'll need only NumPy for this exercise:

```python
import numpy as np
np.random.seed(42)
```

Start with these dimensions:
- Sequence length: 8 tokens
- Embedding dimension: 64
- Create random input embeddings: `X = np.random.randn(8, 64)`

### What to Build

**Part 1: Projections**
Initialize weight matrices W_Q, W_K, W_V (all 64x64). Project your input X into queries, keys, and values.

**Part 2: Numerically Stable Softmax**
The naive `exp(x) / sum(exp(x))` explodes when x contains large values. Fix this. Hint: subtracting the maximum value from x before exponentiating doesn't change the result but prevents overflow.

**Part 3: Scaled Dot-Product Attention**
Compute `Q @ K.T`, but scale by `1/sqrt(d_k)` where d_k is the key dimension. Without scaling, dot products grow large with dimension, pushing softmax into regions where gradients vanish.

**Part 4: Positional Encodings**
Attention is permutation-invariant, it doesn't know token order. Add sinusoidal positional encodings to your input embeddings before projection. Use `sin` for even dimensions, `cos` for odd dimensions, with frequencies that decrease along the embedding dimension.

### Success Criteria

- Your softmax output sums to 1.0 along the last axis (within floating point tolerance)
- `softmax(np.array([1000, 1001, 1002]))` returns valid probabilities, not NaN
- Attention weights matrix has shape (8, 8)
- Each row of attention weights sums to 1.0
- Swapping two rows of X (without positional encoding) and swapping the same two rows of the output gives identical results. With positional encoding, this symmetry breaks.

### Common Pitfalls

**Softmax axis confusion**: Make sure you're normalizing across keys (the dimension you're attending over), not across queries.

**Forgetting the scale**: Unscaled attention still "works" but trains poorly. The scale factor is `sqrt(64) = 8` for your dimensions.

**Positional encoding formula errors**: Position 0 should not be all zeros. The encoding for position `pos` at dimension `i` uses frequency `1 / (10000^(2i/d_model))`.

**Shape mismatches**: Q @ K.T should give (seq_len, seq_len). If you get (64, 64), your matrices are transposed wrong.

### Stretch Goal

Implement causal masking: token 5 should only attend to tokens 1-5, not 6-8. Set future positions to negative infinity before softmax.

## Assessment

By the end of this module, you should be able to:

- Explain the core intuition behind self-attention: why it exists, what problem it solves, and how it differs from fixed-window approaches
- Implement the Query, Key, and Value transformations from scratch and explain what each represents
- Compute attention scores using the dot product between queries and keys, and explain why this measures "compatibility"
- Apply softmax to convert raw scores into normalized attention weights, including handling numerical stability
- Implement positional encoding and explain why transformers need it (hint: attention alone has no notion of order)
- Build causal masking to prevent tokens from attending to future positions, and explain why this is essential for autoregressive language modeling

Success means you can take a sequence of embeddings and walk through the entire attention computation by hand, computing Q, K, V projections, calculating the attention matrix, applying the mask, running softmax, and producing the weighted output. You should be able to draw the attention pattern for a simple sequence and explain which tokens are attending to which, and why. When someone asks "why do we divide by the square root of the dimension?" you should have an answer that isn't just "the paper said so." You're not just implementing attention, you're understanding it deeply enough that you could reinvent it if you had to.

---

# Transformer

## Lesson 1: Transformer Architecture

You just spent an entire chapter building attention from scratch. You understand how tokens look at each other, how queries meet keys, how values get weighted and summed. That was the hard part.

Now we stack it.

The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need," is presented as a radical simplification: it takes the attention mechanism and makes it the central, standalone operation, discarding the RNN structures entirely. Before this paper, most models used LSTMs and LSTMs with attention mechanisms. The attention was a helper, a side channel. The Transformer said: what if attention is everything?

### Why This Architecture Exists

Let me give you the history, because it matters.

Before Transformers, models like RNNs, LSTMs, and GRUs were used but struggled with long sequences and encoding context effectively. The sequence-to-sequence architecture from 2014 used encoder-decoder LSTMs for machine translation but suffered from an "encoder bottleneck" where the entire input sentence was compressed into a single vector. Imagine trying to translate a 50-word sentence, but you have to squeeze all the meaning through a single fixed-size vector before you can start producing output. Information gets crushed.

The 2015 paper "Neural Machine Translation by Jointly Learning to Align and Translate" introduced attention to alleviate the encoder bottleneck by allowing the model to soft-search relevant parts of the source sentence. This was traditional attention, where you have two separate sequences and each token in the source sequence gets associated with a soft alignment to elements in the target sequence.

Then came the key insight. Self-attention eliminates separate source and target sequences by making them the same, allowing each element in a sequence to relate to other elements within that same sequence. Self-attention learns relationships within a sequence, such as how an adjective like "blue" refers to a noun like "ball" in natural language.

The Transformer architecture introduced two key innovations: multi-head attention and self-attention, combined with fast autoregressive decoding. You already built multi-head attention. You already understand self-attention. Now we need to understand how to stack these pieces into something that actually works.

### The Transformer Block

Here is what a single transformer block looks like, in pseudocode:

```python
def transformer_block(x):
    # Attention with residual connection
    attn_out = multi_head_attention(layer_norm(x))
    x = x + attn_out
    
    # MLP with residual connection  
    mlp_out = mlp(layer_norm(x))
    x = x + mlp_out
    
    return x
```

That is it. Seriously. A transformer block is attention, then MLP, with normalization and residual connections wrapping each piece.

Let me break down why each component exists.

**The MLP**: Attention lets tokens talk to each other. But after they talk, they need to think. The MLP is where individual token representations get transformed, where features get combined and recombined. You built MLPs in Chapter 3. Same thing here, just applied position-wise to each token independently.

**Layer Normalization**: Neural networks get unstable when activations grow too large or shrink too small. LayerNorm keeps things in a reasonable range by normalizing across the feature dimension. Without it, training deep networks becomes a nightmare of exploding and vanishing values.

**Residual Connections**: This is the `x = x + something` pattern. Instead of replacing x with the output, we add the output to x. Why? Because it gives gradients a highway to flow backward through the network. During backpropagation, the gradient can flow directly through the addition, skipping potentially problematic layers. This is what lets us stack dozens of blocks without the gradient signal dying.

### Stacking Deep

A single transformer block is not very powerful. GPT-2 small uses 12 blocks. GPT-2 medium uses 24. GPT-3 uses 96. You stack these blocks one after another, and each block refines the representations a little more.

The input flows like this:

1. Tokens get embedded into vectors
2. Position information gets added
3. The vectors pass through block 1, then block 2, then block 3.
4. After the final block, a linear layer projects to vocabulary size
5. Softmax gives you probabilities over next tokens

Each block has the same structure, but different learned weights. Early blocks might learn syntactic patterns. Later blocks might learn semantic relationships. The representations get progressively more abstract and useful for prediction.

Here is something that might surprise you: Transformers can achieve strong performance with near-shallow architectures when properly designed and trained, challenging the assumption that deep networks are always necessary for complex tasks. Near-shallow transformer architectures may offer computational efficiency advantages while maintaining competitive performance on benchmark tasks. Depth helps, but it is not magic. Careful attention to transformer component design and training methodology can enable effective learning with fewer layers than traditional deep models.

### The Full Picture

When people say "GPT architecture" or "decoder-only transformer," they mean:

1. Token embeddings plus position embeddings
2. A stack of transformer blocks (attention + MLP, with norms and residuals)
3. A final layer norm
4. A linear projection to vocabulary logits

Just to give you a sense, it would only require about 500 lines of C with no other dependencies to implement the neural network architecture, and that uses basically the parameters to run the model. The architecture is not complicated. The scale is what makes it powerful.

For example, if you are using ChatGPT or something like that, the model architecture was never released. But with open models like Llama, the weights and the architecture and a paper were all released by Meta, so anyone can work with this model very easily by themselves. We will be building GPT-2, which is fully open and well-documented.

### Where People Get Stuck

**Mistake 1**: Forgetting the residual connections. If you just do `x = attention(x)` instead of `x = x + attention(x)`, your gradients will die in deep networks. The residual is not optional.

**Mistake 2**: Putting layer norm in the wrong place. There are two conventions: "pre-norm" (normalize before attention/MLP) and "post-norm" (normalize after). Modern architectures use pre-norm because it trains more stably. The pseudocode above uses pre-norm.

**Mistake 3**: Confusing the attention output dimension with the MLP hidden dimension. Attention outputs the same dimension as its input. The MLP has a hidden layer that is typically 4x larger than the model dimension, then projects back down.

**Mistake 4**: Thinking each block is independent. The blocks share the same residual stream. Each block reads from and writes to this shared representation. Information accumulates across blocks.

### How You Know It Is Working

When you have implemented a transformer correctly:

- You can load pretrained weights and the model produces coherent text immediately
- The residual connections mean early layers still get gradients even in deep networks
- Layer norms keep activation magnitudes stable across all blocks
- The output logits are in a reasonable range for softmax

In the next lesson, we will build the transformer block in code. You will implement layer normalization, wire up the residual connections, and see exactly how attention and MLP combine. Then we will stack blocks and load GPT-2 weights to prove it works.

The attention mechanism excels at understanding context, such as predicting "French" in "I grew up in France. I speak fluent ___" by connecting distant words in a sequence. By the end of this chapter, you will have built the full system that makes this possible.

## Lesson 2: Transformer Block

You have attention. You have MLPs. Now we stack them together into something that actually works.

The Transformer block is where these pieces combine into the fundamental repeating unit of every modern language model. GPT-2, GPT-3, Llama, all of them: they are just Transformer blocks stacked on top of each other, dozens or hundreds deep. Understanding this single block means understanding the architecture that powers everything.

Let us build one.

### The Structure

A Transformer block has four components arranged in a specific order:

1. Layer normalization
2. Multi-head self-attention
3. Another layer normalization
4. A feed-forward network (MLP)

But here is the critical part that makes deep networks actually trainable: residual connections wrap around both the attention and the MLP. The information kind of flows through into the multi-layer perceptron block and so on, but it also has a direct path that bypasses each sub-layer entirely.

Here is what this looks like in code:

```python
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x
```

That is it. That is the entire Transformer block. The simplicity is deceptive because scale changes everything. When you stack 12 of these blocks with 768-dimensional representations, you get GPT-2 small. Stack 96 blocks with 12288 dimensions and you get GPT-3.

### Why Residual Connections Matter

Residual connections, originally popularized by ResNet for computer vision, are adopted in Transformers through skip connections that bypass one or more layers. The operation looks like this:

```
output = x + Sublayer(x)
```

Why does adding the input back to the output matter so much? Three reasons:

First, stable gradient flow. When you backpropagate through 96 layers, gradients can vanish or explode. The residual path provides a highway for gradients to flow directly to earlier layers without passing through dozens of nonlinearities. This is what makes training deep networks possible at all.

Second, the network learns to refine rather than replace. Each block does not need to reconstruct the entire representation from scratch. It just needs to compute what to add or subtract. This makes optimization dramatically easier.

Third, at initialization, when weights are small and random, each block contributes almost nothing. The network starts as approximately an identity function and gradually learns to do something useful. This is superior training dynamics compared to forcing every layer to immediately produce meaningful transformations.

### Layer Normalization

Before each sub-layer, we normalize the activations. Layer normalization computes the mean and variance across the feature dimension for each position independently:

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = Parameter(ones(d_model))
        self.beta = Parameter(zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

The learnable parameters gamma and beta allow the network to undo the normalization if that turns out to be useful. The epsilon prevents division by zero.

There is an important architectural choice here: Pre-LN versus Post-LN. The original "Attention Is All You Need" paper from 2017 placed normalization after the residual connection. Modern architectures place it before. Pre-LN provides more stable training, especially for very deep networks. The code I showed above uses Pre-LN.

### The MLP Block

Here kind of like the attention block of this Transformer handles communication between positions. The MLP block processes each position independently, giving the network a chance to think about what it learned from attention.

```python
class MLP:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        return x
```

The hidden dimension d_ff is typically 4 times d_model. So if your model dimension is 768, the MLP expands to 3072, applies the nonlinearity, then projects back to 768. This expansion gives the network more capacity to compute complex functions.

Why GELU instead of ReLU? We covered this in the previous chapter, but briefly: GELU provides smoothing that helps with optimization, and empirically it works better for language models.

### Putting It All Together

Each Transformer block takes a sequence of vectors and returns a sequence of vectors with the same shape. This architectural modularity means you can stack as many blocks as you want. The relationship map between tokens gets refined at each layer, with early layers learning simpler patterns and later layers learning more abstract relationships.

Here is a complete block with all the pieces:

```python
class TransformerBlock:
    def __init__(self, config):
        d_model = config.d_model
        n_heads = config.n_heads
        d_ff = config.d_ff
        
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    
    def forward(self, x, mask=None):
        # Attention sublayer with residual
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        
        # MLP sublayer with residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        
        return x
```

The mask parameter handles causal masking for autoregressive generation. Each position can only attend to previous positions, which we enforce by setting future attention weights to negative infinity before the softmax.

### Common Mistakes

The most frequent error is getting the order wrong. Normalization comes before the sublayer in modern architectures, and the residual adds the original input, not the normalized input. This matters:

```python
# Correct (Pre-LN)
x = x + self.attn(self.ln1(x))

# Wrong (normalizing the residual)
x = self.ln1(x + self.attn(x))
```

Another mistake is forgetting that attention and MLP each get their own normalization layer. Using a single shared LayerNorm breaks the architecture.

Finally, watch your dimensions. The attention output and MLP output must match the input dimension exactly, or the residual addition will fail. This is why the MLP projects back to d_model at the end.

### Success Indicators

You know your Transformer block is working when:

1. A single block with random weights produces output with the same shape as input
2. Gradients flow through without NaN or infinity values
3. Stacking multiple blocks does not cause activation magnitudes to explode
4. The loss decreases when you train, even slowly

Test each component in isolation before combining them. Verify that your LayerNorm actually normalizes (mean near zero, variance near one). Verify that your residual connections preserve the input when sublayer weights are zero.

### What We Built

You now have the fundamental building block of modern language models. The Transformer block combines attention for communication between positions with MLPs for computation at each position, wrapped in normalization and residual connections for efficient knowledge representation and stable training.

In the next lesson, we stack these blocks into the full GPT-2 architecture and load real pretrained weights. The jump from one block to a complete model is smaller than you might expect. Most of the complexity lives right here, in this single repeating unit.

## Lesson 3: Full GPT-2 Architecture

You have a transformer block. Now we stack twelve of them together, add some bookkeeping, and suddenly you have GPT-2. The 124 million parameter version. The one that caused a stir when it was released because OpenAI initially refused to publish the weights.

Here is the thing that will surprise you: the architecture itself is about 500 lines of code. The magic is not in the code. The magic is in the parameters. Those 124 million numbers, trained on internet text, compressed into a gestalt that can write poetry and answer questions and pretend to be a pirate. But the neural network architecture? Everything is algorithmically understood and open.

Let us build it.

### The Shape of GPT-2

The GPT-2 124M model has 12 transformer layers, 768 embedding dimensions, a vocabulary size of 50,257 tokens, and a maximum sequence length of 1,024 tokens. Write those numbers down. They define everything.

```python
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
```

Why these specific numbers? Scaling laws. GPT-2 is a series of models with different sizes, ranging from 124 million to 1.5 billion parameters, where larger models generally perform better on downstream tasks like translation and summarization. The 124M version is the smallest. It is also the one you can train yourself in about an hour for roughly $10 on cloud compute. When it was originally released five years ago, this was much more computationally intensive.

### Decoder-Only: What We Removed

The original Transformer from the "Attention Is All You Need" paper had an encoder and a decoder with cross-attention between them. GPT-2 uses a decoder-only transformer architecture, removing the encoder and cross-attention entirely.

Why? Because language modeling is simpler than translation. In translation, you need to look at the entire source sentence (encoder) while generating the target sentence (decoder). In language modeling, you just predict the next token given all previous tokens. One direction. One stack of blocks.

GPT-2 also modifies layer normalization placement to use pre-normalization. Layer norm before attention and MLP blocks, not after. This creates clean residual pathways where layer norms are placed before attention and MLP blocks, not inside the residual stream. Why does this matter? Because it allows gradients to flow unchanged from top to bottom, improving training stability.

### The Full Architecture

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

Two embedding tables. One for tokens (`wte`), one for positions (`wpe`). In GPT-2, positional embeddings are learned parameters initialized randomly and trained from scratch, unlike the original transformer which used fixed sinusoidal encodings. Interestingly, they often develop sinusoidal-like patterns during optimization anyway. The model discovers what the original authors hand-designed.

Then twelve transformer blocks (`h`). Each block does the same thing: attention, then MLP, with residual connections and layer norms. Transformers interleave two phases: communication through multi-headed attention and computation through the multi-layer perceptron. This is the repeated map-reduce pattern. Attention acts as a communication and reduce operation where tokens exchange information, while the MLP acts as an individual map operation where each token processes information independently.

Finally, a layer norm (`ln_f`) and a linear projection to vocabulary size (`lm_head`).

### The Forward Pass

```python
def forward(self, idx):
    B, T = idx.shape
    assert T <= self.config.block_size
    
    # Token and position embeddings
    pos = torch.arange(0, T, device=idx.device)
    tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
    pos_emb = self.transformer.wpe(pos)      # (T, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)
    
    # Transformer blocks
    for block in self.transformer.h:
        x = block(x)
    
    # Final layer norm and projection
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)                 # (B, T, vocab_size)
    return logits
```

That is it. Embed tokens, add positions, pass through twelve blocks, normalize, project to vocabulary. The Transformer architecture has proven remarkably resilient since 2017, with most modern variants like GPT using essentially the same core architecture.

### Loading Pretrained Weights

Now comes the practical part. You want to load the actual GPT-2 weights from Hugging Face Transformers into your implementation.

```python
@classmethod
def from_pretrained(cls, model_type):
    from transformers import GPT2LMHeadModel
    
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    
    config = GPTConfig()
    model = GPT(config)
    sd = model.state_dict()
    
    # Copy weights, transposing where necessary
    for k in sd_hf.keys():
        # Handle transposition for Conv1D vs Linear
        if k.endswith('.attn.c_attn.weight') or k.endswith('.attn.c_proj.weight'):
            sd[k].copy_(sd_hf[k].t())
        elif k.endswith('.mlp.c_fc.weight') or k.endswith('.mlp.c_proj.weight'):
            sd[k].copy_(sd_hf[k].t())
        else:
            sd[k].copy_(sd_hf[k])
    
    return model
```

When loading pre-trained GPT-2 weights from Hugging Face Transformers into a custom PyTorch implementation, some weight tensors may need transposition to match PyTorch's expected layout due to differences from the original TensorFlow implementation. This catches many people. The original GPT-2 was written in TensorFlow, and the weight layouts are different.

### Generating Text

```python
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        # Crop to block_size if necessary
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        with torch.no_grad():
            logits = self(idx_cond)
        
        # Take logits at last position, apply temperature
        logits = logits[:, -1,:] / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    
    return idx
```

During text generation with GPT-2, using `torch.no_grad()` context manager prevents PyTorch from caching intermediate tensors for backward passes, saving memory and computation when only inference is needed. Without this, you will run out of memory quickly.

### Common Mistakes

**Forgetting the transpose.** Your model loads without errors but generates garbage. The weights are in the wrong shape.

**Wrong layer norm placement.** Post-normalization versus pre-normalization. GPT-2 uses pre-norm. If you put layer norm after the attention and MLP instead of before, training becomes unstable.

**Position embedding overflow.** If you try to generate more than 1024 tokens, your position embedding lookup will fail. You need to either crop the context or handle this gracefully.

**Missing the final layer norm.** There is a layer norm after all the transformer blocks but before the final projection. Easy to forget. Your model will work but perform worse.

### Success Indicators

You know you have it right when:

1. Your model loads pretrained weights and generates coherent English text
2. The parameter count matches: 124,439,808 for GPT-2 small
3. Given the same prompt and random seed, your output matches the Hugging Face implementation

This is what we call the Transformer neural network architecture. You have now built it from scratch. The architecture is about 500 lines of code. The parameters file that makes it useful is 500 megabytes. The lossy compression of internet data into those parameters is where the real magic lives.

Next lesson, we load the weights and make it talk.

## Lesson 4: Weight Loading

You have built a GPT-2 architecture from scratch. Every layer, every parameter, every connection. But right now your model is filled with random numbers. It knows nothing about language, nothing about the world, nothing about anything.

In this lesson, we fix that. We take the weights that OpenAI trained on massive amounts of text and load them into your architecture. When we are done, your from-scratch implementation will generate coherent English text. Same weights, same architecture, same outputs. The only difference is that you wrote the code yourself.

This is the moment where theory becomes reality.

### Why Weight Loading Matters

The GPT-2 124M model has 12 transformer layers, 768 embedding dimensions, a vocabulary size of 50,257 tokens, and a maximum sequence length of 1,024 tokens. Training this model from scratch would take hours and cost real money. But we do not need to train it. OpenAI already did that work. We just need to copy their homework.

Weight loading is how you take a pretrained checkpoint and put those learned values into your own implementation. This is not cheating. This is how the entire field works. You build the architecture, someone else provides the weights, and suddenly you have a working model.

The catch is that your architecture must match theirs exactly. Every dimension, every layer order, every parameter name. If anything is off, the weights will not fit, or worse, they will fit but produce garbage.

### The Hugging Face Connection

We will load weights from Hugging Face Transformers, which hosts pretrained checkpoints for thousands of models including GPT-2. The process looks like this:

```python
from transformers import GPT2LMHeadModel

# Load the pretrained model from Hugging Face
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Get the state dict (all the weights as a dictionary)
hf_state_dict = hf_model.state_dict()
```

Now you have a dictionary mapping parameter names to tensors. Print out the keys and you will see things like:

```
transformer.wte.weight
transformer.wpe.weight
transformer.h.0.ln_1.weight
transformer.h.0.attn.c_attn.weight
transformer.h.0.mlp.c_fc.weight.
```

Each key corresponds to a specific part of the model. `wte` is the token embedding. `wpe` is the positional embedding. `h.0` is the first transformer block. `ln_1` is the first layer norm. And so on for all 12 layers.

### The Transposition Problem

Here is where people get stuck. When loading pre-trained GPT-2 weights from Hugging Face Transformers into a custom PyTorch implementation, some weight tensors may need transposition to match PyTorch's expected layout due to differences from the original TensorFlow implementation.

The original GPT-2 was written in TensorFlow. TensorFlow and PyTorch have different conventions for how they store weight matrices. Specifically, the linear layer weights are transposed relative to each other.

In PyTorch, a linear layer with input size `in_features` and output size `out_features` stores its weight as a tensor of shape `(out_features, in_features)`. TensorFlow uses `(in_features, out_features)`. The Hugging Face checkpoint preserves the TensorFlow convention.

This means you need to transpose certain weights:

```python
def load_weights(our_model, hf_state_dict):
    # These keys need transposition
    transpose_keys = [
        'attn.c_attn.weight',
        'attn.c_proj.weight', 
        'mlp.c_fc.weight',
        'mlp.c_proj.weight'
    ]
    
    our_state_dict = our_model.state_dict()
    
    for name, param in hf_state_dict.items():
        # Check if this weight needs transposition
        needs_transpose = any(k in name for k in transpose_keys)
        
        if needs_transpose:
            param = param.t()  # Transpose the tensor
            
        # Map HF naming to your naming convention
        our_name = convert_name(name)  # You implement this
        our_state_dict[our_name] = param
    
    our_model.load_state_dict(our_state_dict)
```

If you skip the transposition, your model will run without errors. The shapes might even match. But the outputs will be complete nonsense because the weights are oriented wrong. This is a silent failure. The code works, the model generates text, but the text is garbage.

### Name Mapping

The other challenge is matching parameter names. Hugging Face uses one naming convention. You probably used a different one when building your architecture. You need a mapping function.

For example, Hugging Face might call something `transformer.h.0.ln_1.weight` while your code calls it `blocks.0.ln1.weight`. Same tensor, different name. You need to handle every such mapping.

The safest approach is to print both sets of keys side by side and manually verify the correspondence:

```python
print("Hugging Face keys:")
for k in sorted(hf_state_dict.keys()):
    print(f"  {k}: {hf_state_dict[k].shape}")

print("\nOur keys:")
for k in sorted(our_model.state_dict().keys()):
    print(f"  {k}: {our_model.state_dict()[k].shape}")
```

Check that every shape matches. If you have a weight of shape `(768, 2304)` in Hugging Face and `(2304, 768)` in your model, that is the transposition issue. If you have `(768, 768)` versus `(768, 512)`, your architecture is wrong somewhere.

### Verification Through Generation

Once weights are loaded, verify by generating text. During text generation with GPT-2, using `torch.no_grad()` context manager prevents PyTorch from caching intermediate tensors for backward passes, saving memory and computation when only inference is needed.

```python
@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=50):
    tokens = prompt_tokens.clone()
    
    for _ in range(max_new_tokens):
        # Get logits for the last position
        logits = model(tokens)[:, -1,:]
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append and continue
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokens
```

Encode a prompt like "Hello, I am a language model" and generate. If you get coherent English that continues the thought, your weight loading worked. If you get random characters or repeated tokens, something is wrong with the loading.

Compare your outputs to the Hugging Face model directly:

```python
# Generate with Hugging Face model
hf_output = hf_model.generate(prompt_tokens, max_length=50)

# Generate with your model
our_output = generate(our_model, prompt_tokens, max_new_tokens=50)

# They should match (with same random seed)
```

### Common Mistakes

**Forgetting to transpose**: The model runs, but outputs are garbage. Always check those four weight types.

**Wrong layer order**: If your transformer blocks are numbered differently, weights go to wrong layers. Verify block-by-block.

**Missing parameters**: If your model has parameters that the checkpoint lacks, or vice versa, `load_state_dict` will complain. Use `strict=False` only after understanding what is missing.

**Device mismatch**: Loading weights to CPU then moving to GPU is fine. Loading directly to wrong device causes errors.

### Success Indicators

You know weight loading succeeded when:

1. `load_state_dict` completes without errors
2. Generated text is coherent English
3. Your outputs match Hugging Face outputs exactly given the same random seed
4. The model responds sensibly to various prompts

This is the payoff. You built the architecture. You loaded the weights. You have a working GPT-2. Every layer, every attention head, every feedforward network, you understand because you wrote it.

In the next lesson, we generate text properly, exploring sampling strategies and what makes generation actually work well.

## Lesson 5: Text Generation

You have built a GPT-2 from scratch. You have loaded 124 million parameters from a pretrained checkpoint. Now comes the moment where all of this becomes real: you are going to make your transformer dream.

Text generation is where the rubber meets the road. Everything we have built in this module exists for this purpose. The attention layers, the MLPs, the residual connections, the layer normalization, the careful weight loading from Hugging Face. All of it culminates in a simple loop that predicts one word at a time.

Let me show you what this looks like in practice.

### The Generation Loop

Here is the core of text generation:

```python
import torch

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    """
    Generate text autoregressively.
    
    idx: (B, T) tensor of starting token indices
    max_new_tokens: number of tokens to generate
    """
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to max sequence length
            idx_cond = idx[:, -1024:]
            
            # Forward pass to get logits
            logits = model(idx_cond)
            
            # Take logits at the last position
            logits = logits[:, -1,:] / temperature
            
            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
    
    return idx
```

That `torch.no_grad()` context manager is doing something important. During text generation, we only need the forward pass. We are not training, so we do not need PyTorch to cache intermediate tensors for backward passes. This saves memory and computation when only inference is needed.

### Why This Works

The neural network predicts what comes next. So, for example, in this case, this neural network might predict that in this context of four words, the next word will probably be "mat" with, say, 97% probability. The model outputs a probability distribution over all 50,257 tokens in the vocabulary. We sample from that distribution, append the result, and repeat.

This is the dreaming process. The model hallucinates text one token at a time, feeding each prediction back into itself as context for the next prediction. What's happening here is this text verbatim is not found in a training set document, but this information, if you actually look it up, is actually roughly correct with respect to this fish. The model has compressed patterns from its training data into its weights, and generation is the process of decompressing those patterns into coherent text.

The temporal feedback mechanism that keeps later words consistent with earlier ones is simply the growing context window. Each new token sees all previous tokens through attention. The model conditions on everything it has generated so far.

### Temperature and Sampling

Temperature controls randomness. A temperature of 1.0 uses the raw probabilities. Lower temperatures make the distribution sharper, concentrating probability mass on the most likely tokens. Higher temperatures flatten the distribution, making unlikely tokens more probable.

```python
# Temperature = 0.1: Very deterministic, picks highest probability tokens
# Temperature = 1.0: Uses learned distribution as-is
# Temperature = 2.0: More random, creative, potentially incoherent
```

Top-k sampling with k=50 is the default setting in Hugging Face's text generation pipeline. It restricts sampling to only the k most likely tokens, zeroing out everything else. This prevents the model from occasionally sampling extremely unlikely tokens that would derail coherence.

These are control mechanisms that prevent the dreaming process from diverging into completely incoherent outputs. Without them, the model might occasionally sample a bizarre token that throws off all subsequent generation.

### Running Your Model

Let us generate some text:

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode a prompt
prompt = "The quick brown fox"
tokens = tokenizer.encode(prompt, return_tensors='pt')

# Generate
output = generate(model, tokens, max_new_tokens=50, temperature=0.8, top_k=40)

# Decode back to text
text = tokenizer.decode(output[0])
print(text)
```

The GPT-2 124M model has 12 transformer layers, 768 embedding dimensions, a vocabulary size of 50,257 tokens, and a maximum sequence length of 1,024 tokens. That 1024 limit is why we crop the context in our generation loop. If the generated sequence exceeds this length, we can only condition on the most recent 1024 tokens.

A 70B would be running about 10 times slower, but I wanted to give you an idea of sort of just the text generation and what that looks like. The 124M model generates quickly enough to feel interactive on a modern GPU.

### Common Mistakes

**Forgetting `model.eval()`**: This switches off dropout and other training-specific behaviors. Your generations will be noisier without it.

**Not using `torch.no_grad()`**: Your code will still work, but it will be slower and use more memory because PyTorch is building a computation graph you will never use.

**Context overflow**: If you generate more than 1024 tokens total, you need to handle the sliding window. The model cannot attend to tokens beyond its maximum sequence length.

**Greedy decoding**: Always taking the highest probability token (temperature approaching 0) produces repetitive, boring text. Language models need some randomness to generate diverse outputs.

**Ignoring Hugging Face quirks**: Hugging Face's implementation of text generation has multiple quirks and implementation details that affect its behavior. If your outputs differ from theirs, check for subtle differences in how attention masks or position IDs are handled.

### What You Have Built

You now have a complete text generation system built from scratch. The decoder transformer architecture you implemented removes the encoder and cross-attention from the original Attention Is All You Need transformer. The pre-normalization pattern places layer norm before attention and MLP blocks, creating clean residual pathways where gradients flow unchanged from top to bottom.

The attention mechanism acts as a communication operation where tokens exchange information. The MLP acts as an individual map operation where each token processes information independently. This repeated map-reduce pattern, stacked 12 layers deep, gives the model its representational power.

GPT-2 is a series of models with different sizes, ranging from 124 million to 1.5 billion parameters, where larger models generally perform better on downstream tasks like translation and summarization due to scaling laws. You have built the smallest version, but the architecture is identical across all sizes. Only the dimensions change.

### Success Indicators

You know you have succeeded when:

1. Your model generates grammatically coherent English text
2. The generated text stays on topic for at least a few sentences
3. Temperature changes produce noticeably different output styles
4. Your outputs roughly match Hugging Face's generate function given the same prompt and settings

Reproducing the GPT-2 124M model today can be done in about an hour for roughly $10 on cloud compute, compared to when it was originally released five years ago when it was more computationally intensive. You have not just used this model. You have built every piece of it yourself, from the attention computation to the weight loading to the generation loop.

This is the foundation for everything that follows. ChatGPT, developed by OpenAI and deployed in 2022, was the first widely accessible text interface for interacting with a large language model, sparking viral adoption. That interface is just a more sophisticated version of what you have built here: a transformer that predicts the next word, running in a loop, generating text one token at a time.

The transformer performance appears to be largely modality-independent, meaning the same architecture can achieve excellent results across different data types when scaled appropriately. The architecture you have built for text will reappear when we work with images in the multimodal chapter. The bitter lesson of AI research is that general methods that leverage computation win in the long run. The transformer is that general method.

You have built a storyteller. Now you understand exactly how it works.

## Practice

Build a transformer block with attention, MLP, LayerNorm, and residual connections; implement full GPT-2 architecture; load pretrained weights; and generate text.

### Setup

You'll need your attention implementation from Chapter 04. If you don't have it working, go back, everything here builds on it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# You'll also need tiktoken for GPT-2's tokenizer
# pip install tiktoken
```

Download GPT-2 weights from Hugging Face. We'll use the 124M parameter version (GPT-2 "small"):

```python
from transformers import GPT2LMHeadModel
pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
```

You'll extract weights from this and load them into your own architecture.

### Part 1: The Transformer Block

Build a single transformer block with:
- LayerNorm (pre-norm style, as in GPT-2)
- Multi-head causal self-attention (from your Chapter 04 code)
- Residual connection around attention
- Another LayerNorm
- MLP with GELU activation and 4x hidden dimension expansion
- Residual connection around MLP

Test it: random input of shape `(batch=2, seq_len=16, d_model=768)` should produce output of the same shape.

### Part 2: Full GPT-2 Architecture

Stack your blocks into the full architecture:
- Token embedding (vocab_size=50257, d_model=768)
- Positional embedding (max_seq_len=1024, d_model=768)
- 12 transformer blocks
- Final LayerNorm
- Output projection to vocabulary (tied with token embeddings)

GPT-2 small config: `n_layer=12, n_head=12, d_model=768`

### Part 3: Load Pretrained Weights

Map the Hugging Face weight names to your architecture. Print both sets of parameter names and match them up. Common mappings:
- `wte` → token embeddings
- `wpe` → positional embeddings  
- `h.0`, `h.1`, ... → transformer blocks
- `ln_f` → final layer norm

After loading, verify: your model's output logits should match the pretrained model's logits exactly (within floating point tolerance) for the same input.

### Part 4: Generate Text

Implement autoregressive generation:
1. Tokenize a prompt using tiktoken's GPT-2 encoder
2. Forward pass to get logits
3. Sample or argmax from the last token's logits
4. Append to sequence, repeat

Generate a few completions. They should be coherent English, you're running the real GPT-2.

### Success Criteria

- [ ] Single transformer block preserves tensor shape
- [ ] Forward pass through full model runs without error
- [ ] Your logits match pretrained logits (use `torch.allclose` with `atol=1e-4`)
- [ ] Generated text is coherent and matches what you'd get from the Hugging Face model

### Common Pitfalls

**Wrong normalization placement**: GPT-2 uses pre-norm (LayerNorm before attention/MLP), not post-norm. This matters for weight loading.

**Forgetting weight tying**: The output projection matrix should be the same tensor as the token embedding matrix, transposed. Not a copy, the same tensor.

**GELU variant**: GPT-2 uses the approximate GELU (`torch.nn.functional.gelu` with `approximate='tanh'`). The exact version gives slightly different results.

**Attention mask off-by-one**: Your causal mask should allow position i to attend to positions 0 through i (inclusive). Verify this.

**Positional embedding indexing**: Positions are absolute indices 0, 1, 2, ... not relative. Make sure you're slicing correctly for sequences shorter than 1024.

### Stretch Goals

Once the basics work:
- Implement temperature and top-k/top-p sampling
- Load GPT-2 medium (345M) or large (774M) and verify those work too
- Measure tokens per second on your hardware
- Profile where time is spent (attention vs MLP vs other)

## Assessment

By the end of this module, you should be able to:

- **Explain why residual connections exist** and what would happen to gradients in a deep network without them
- **Implement layer normalization from scratch** and articulate why it's applied before (not after) attention and MLP in modern transformers
- **Build a complete transformer block** that combines multi-head attention, MLP, normalization, and residual connections in the correct order
- **Stack multiple transformer blocks into a GPT architecture** with proper embedding layers, positional encoding, and output projection
- **Trace a token's journey through the entire architecture**, from input embedding through every layer to final logits

Success means you can draw the full GPT architecture on a whiteboard without looking anything up, not as memorized boxes and arrows, but as a sequence of operations you understand deeply enough to implement. When someone asks "why is there a residual connection here?" you don't say "because the paper did it." You say "because without it, gradients in layer 47 would be essentially zero, and the model couldn't learn." When you see `x = x + attn_out`, you understand this as the architectural decision that makes deep transformers trainable. You should be able to take the attention mechanism you built in the previous chapter and wrap it in the scaffolding that turns a single operation into a complete language model.

---

# Tokenization

## Lesson 1: Byte Pair Encoding

You have text. Neural networks need numbers. That gap is where everything breaks.

Here is a sentence: "The cat sat on the mat."

Here is what your neural network actually sees: `[464, 3466, 3290, 319, 262, 2603, 13]`

That translation, text to integers and back again, is tokenization. And if you get it wrong, nothing else matters. Your attention mechanism could be perfect. Your training data could be flawless. But if your tokenization is broken, your model will be broken in ways that are incredibly hard to debug.

So we are going to build it ourselves.

### Why Tokenization Exists

Neural networks for language models expect a one-dimensional sequence of symbols from a finite vocabulary. That is the constraint. You cannot feed raw text into a transformer. You need to convert characters, words, or chunks of text into integers that index into an embedding table.

The question is: what should those chunks be?

**Option 1: Characters.** The word "happiness" becomes `['h', 'a', 'p', 'p', 'i', 'n', 'e', 's', 's']`. Your vocabulary is tiny, maybe 256 symbols if you use raw bytes. But your sequences get very long. A 1000-word document might become 5000 tokens. Long sequences are expensive. Attention is quadratic in sequence length.

**Option 2: Words.** The word "happiness" becomes `['happiness']`. Your sequences are short. But your vocabulary explodes. English has hundreds of thousands of words. What about "unhappiness"? What about "unhappinesses"? What about misspellings? What about other languages? You cannot have a separate token for every possible word.

**Option 3: Something in between.** This is where Byte Pair Encoding comes in.

### The Core Idea

Byte pair encoding creates tokens by grouping common consecutive byte sequences into new symbols, allowing for shorter sequences with larger vocabularies. It is a compression algorithm that learns which chunks of text appear together frequently and merges them.

Think of it like this. You start with individual characters. Then you look at your training data and ask: which pair of adjacent characters appears most often? Maybe it is "th". You create a new token for "th" and replace every occurrence. Now you ask again: which pair appears most often? Maybe it is "the" (the new "th" token followed by "e"). You merge again.

You keep doing this until you hit your target vocabulary size. GPT-4 uses a vocabulary of 100,277 possible symbols. That number is a choice. More tokens means shorter sequences but a larger embedding table. Fewer tokens means longer sequences but a smaller vocabulary to learn.

### The Algorithm

Let us walk through byte pair encoding step by step. Say your training text is:

```
aaabdaaabac
```

**Step 0: Start with bytes.**

Your initial tokens are just the individual characters: `['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']`

**Step 1: Count pairs.**

Look at every adjacent pair:
- "aa" appears 4 times
- "ab" appears 2 times  
- "bd" appears 1 time
- "da" appears 1 time
- "ba" appears 1 time
- "ac" appears 1 time

The most frequent pair is "aa".

**Step 2: Merge.**

During byte pair encoding, when a pair is merged, all occurrences of that pair in the data are replaced with the newly created token. Let us call the new token "Z".

Before: `a a a b d a a a b a c`
After:  `Z a b d Z a b a c`

In byte pair encoding, the merge index starts at 0 and corresponds to the first new token ID at 256. So if your original bytes are 0-255, your first merged token is 256.

**Step 3: Repeat.**

Now count pairs again on the new sequence:
- "Za" appears 2 times
- "ab" appears 2 times
- "bd" appears 1 time
- "da" appears 1 time (wait, this is now "dZ")
- etc.

Pick the most frequent, merge, repeat. You do this for however many merges you want, which determines your final vocabulary size.

### Why This Works

The beauty of byte pair encoding is that it automatically discovers meaningful units. Run it on English text and it will learn that "ing" is common, that "tion" is common, that "the" is common. Run it on code and it will learn that "def " is common, that "return" is common.

Common words become single tokens. Rare words get split into pieces. The word "unhappiness" might become `["un", "happiness"]` or `["un", "happi", "ness"]` depending on what the algorithm learned from the training corpus.

This gives you the best of both worlds. Common patterns get compressed into single tokens, keeping sequences short. Rare patterns can still be represented by combining pieces, keeping vocabulary finite.

### The Tradeoff

Tokenization converts text into tokens by trading off between vocabulary size and sequence length. A larger vocabulary means more parameters in your embedding table and output projection. A smaller vocabulary means longer sequences, which means more computation in attention.

There is no perfect answer. Different models make different choices. The key is understanding that this choice exists and affects everything downstream.

### Where People Get Stuck

**Mistake 1: Thinking tokens are words.** They are not. A token might be a word, part of a word, a punctuation mark, or even a space followed by a word. When you see token IDs, you cannot assume anything about word boundaries.

**Mistake 2: Forgetting about the training corpus.** The BPE merges are learned from data. If your training corpus has lots of Python code, you will get good tokens for Python. If it has no Spanish, Spanish text will be tokenized inefficiently into many small pieces. Language filtering decisions affect model performance.

**Mistake 3: Ignoring special tokens.** Real tokenizers have special tokens for things like beginning of sequence, end of sequence, padding. We will cover these later, but know they exist.

### How You Know It Is Working

When you have implemented BPE correctly:

1. Encoding then decoding returns the original text exactly
2. Common words in your training data become single or few tokens
3. Rare words get split into sensible subword pieces
4. Your vocabulary size matches your target

### What Comes Next

In this module, we will implement all of this from scratch. Character-level tokenization first as a baseline. Then the full BPE training algorithm with frequency counting and incremental updates. Then encoding and decoding functions. Then special tokens.

By the end, you will have a tokenizer that can convert any text into a sequence of integers and back. That sequence is what your transformer will actually process. Everything in the model, the attention, the embeddings, the predictions, operates on these token IDs.

Tokenization is complex and full of hidden pitfalls. But once you understand it, you understand the interface between human language and neural computation. That interface is where a lot of the weird behaviors of language models come from.

Let us build it.

## Lesson 2: Character-level Tokenization

Let's build the simplest tokenizer that could possibly work.

Before we get into Byte Pair Encoding and all the sophisticated machinery modern LLMs use, we need a baseline. Something so simple you can hold the entire thing in your head. Something that definitely works, even if it's not optimal. This is character-level tokenization, and it's going to teach us exactly what problem we're trying to solve.

### Why Start Here?

Here's the thing about building systems: you want to start with a Naive Tokenizer that you completely understand. Character-level tokenization is that baseline. Every character becomes its own token. The letter 'a' is token 0. The letter 'b' is token 1. A space is token 32. You get the idea.

This approach has one massive advantage: Information Preservation is perfect. You can encode any text and decode it back to exactly what you started with. No information lost. No weird edge cases. If you can type it, you can tokenize it.

Let's build it.

```python
class CharTokenizer:
    def __init__(self, text):
        # Get all unique characters from the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings both directions
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        # Turn string into list of integers
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        # Turn list of integers back into string
        return ''.join([self.idx_to_char[i] for i in indices])
```

That's it. About 15 lines of Python. Let's see it work:

```python
text = "hello world"
tokenizer = CharTokenizer(text)

print(f"Vocabulary size: {tokenizer.vocab_size}")
# Vocabulary size: 8

encoded = tokenizer.encode("hello")
print(f"Encoded: {encoded}")
# Encoded: [1, 0, 3, 3, 4]

decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
# Decoded: hello
```

The vocabulary here is just the unique characters: space, d, e, h, l, o, r, w. Eight tokens total. Simple.

### The Problem: Sequence Length Explosion

Now here's where things get interesting. Let's think about what happens when we feed this to a transformer.

Remember from earlier chapters that transformers have a Context Window Limitation. They can only process sequences up to a certain length. GPT-2 could handle 1024 tokens. GPT-4 can handle much more, but there's always a limit.

With character-level tokenization, the word "understanding" is 13 tokens. The sentence "The quick brown fox jumps over the lazy dog" is 43 tokens including spaces. A typical paragraph might be 500 characters, which means 500 tokens.

Now think about this: attention in transformers scales quadratically with sequence length. Double your sequence length, quadruple your compute. This is the Computational Overhead problem. Character-level tokenization creates very long sequences, which means:

1. You hit your context window limit faster
2. Training takes longer because sequences are longer
3. Inference is slower for the same reason

There's another problem too. When you tokenize at the character level, your model has to learn everything from scratch at a very low level. It has to learn that 't', 'h', 'e' often appear together. It has to learn that 'i', 'n', 'g' is a common ending. It has to learn what words are before it can learn what they mean.

This is the fundamental tension: character-level gives you perfect Information Preservation and handles any text you throw at it, but it forces the model to do a lot of work that we could potentially shortcut.

### How Character Tokenization Actually Works

Let's dig deeper into the mechanics. The key insight is that we're building a bidirectional mapping between characters and integers, because neural networks need numbers as Model Input.

```python
def build_vocab(text):
    """Build vocabulary from text corpus."""
    chars = sorted(list(set(text)))
    return {ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)}

# With a larger corpus
corpus = """The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs."""

char_to_idx, idx_to_char = build_vocab(corpus)
print(f"Vocab: {char_to_idx}")
```

Notice we're sorting the characters. This isn't strictly necessary, but it makes the vocabulary deterministic. Same input text, same vocabulary every time.

### Special Token Handling

Real tokenizers need to handle special cases. What happens when you try to encode a character that wasn't in your training data?

```python
class RobustCharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        
        # Reserve index 0 for unknown characters
        self.unk_token = '<UNK>'
        self.char_to_idx = {self.unk_token: 0}
        self.idx_to_char = {0: self.unk_token}
        
        for i, ch in enumerate(chars, start=1):
            self.char_to_idx[ch] = i
            self.idx_to_char[i] = ch
        
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])
```

Now unknown characters map to the `<UNK>` token instead of crashing. This is Special Token Handling in its simplest form. Production tokenizers have many special tokens: beginning of sequence, end of sequence, padding, and more.

### Unicode and UTF-8

Here's where things get tricky. When I said "character," what did I mean exactly?

```python
text = "hello 世界 🌍"
print(f"Length: {len(text)}")  # 11 characters
print(f"Bytes: {len(text.encode('utf-8'))}")  # 18 bytes
```

Python strings are Unicode, so Chinese characters and emoji count as single characters. But under the hood, they take multiple bytes in UTF-8 encoding. A Sophisticated Tokenizer needs to handle this properly.

For now, our character tokenizer works with Python's Unicode characters directly. This means our vocabulary could get very large if we include all possible Unicode characters. The full Unicode standard has over 140,000 characters. That's a big vocabulary.

### When Character-Level Makes Sense

Despite the Optimization Challenges, character-level tokenization isn't useless. It's actually great for:

1. Learning and debugging. You can see exactly what's happening.
2. Languages with small alphabets where Sequence Length isn't a huge problem.
3. Tasks where character-level patterns matter, like spelling correction.
4. As a baseline to measure how much a Sophisticated Tokenizer actually helps.

### Common Mistakes

**Mistake 1:** Building vocabulary on test data. Your vocabulary should come from your training corpus only. If you include test data, you're leaking information.

**Mistake 2:** Forgetting about case sensitivity. 'A' and 'a' are different characters. Decide whether you want case-sensitive tokenization and be consistent.

**Mistake 3:** Not handling unknown characters. Your model will see characters in production that weren't in training. Plan for this.

### Success Indicators

You understand character-level tokenization when you can:

- Build a tokenizer from scratch given any text corpus
- Explain why `encode(decode(tokens)) == tokens` should always be true
- Articulate the tradeoff between vocabulary size and sequence length
- Predict approximately how many tokens a sentence will produce

### Looking Ahead

Character-level tokenization is our Naive Tokenizer baseline. It works, it's simple, and it preserves all information. But the Sequence Length Explosion problem is real. We need something that gives us shorter sequences without losing the ability to handle any text.

That something is Byte Pair Encoding. Instead of treating each character as a token, BPE learns to merge frequent character pairs into single tokens. The word "the" becomes one token instead of three. Common patterns get compressed. Rare patterns still work because we can fall back to smaller units.

This is the general pattern in machine learning: start simple, understand the baseline, then add complexity where it actually helps. Character tokenization taught us the problem. BPE will teach us one solution.

## Lesson 3: BPE Training Algorithm

In the last lesson, we built a character-level tokenizer. It works. It's simple. And it produces sequences that are way too long for any practical transformer. If you tried to feed a novel into a model using character tokens, you'd blow past your context window before finishing the first chapter.

So we need something smarter. We need tokens that represent chunks of text, not individual characters. But how do we decide what those chunks should be?

This is where Byte Pair Encoding comes in. The BPE algorithm is elegantly simple: start with individual bytes, find the pair that appears most frequently, merge them into a new token, and repeat. That's it. That's the whole algorithm.

Let's build it.

### Starting Point: Raw Bytes

Remember from last lesson that we can convert any text into UTF-8 bytes:

```python
text = "the cat sat on the mat"
tokens = list(text.encode("utf-8"))
print(tokens)
# [116, 104, 101, 32, 99, 97, 116, 32, 115, 97, 116, 32, 111, 110, 32, 116, 104, 101, 32, 109, 97, 116]
```

We start with a vocabulary of 256 byte tokens. Every possible byte value from 0 to 255 is already a valid token. This is our foundation.

The problem? This sequence is 22 tokens long for a simple sentence. Raw UTF-8 bytes produce excessively long sequences that would exceed the finite context window of the transformer. We need to compress.

### Step 1: Counting Pairs

The first step in BPE training is frequency counting. We need to know which pairs of adjacent tokens appear most often:

```python
def get_pair_counts(tokens):
    counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

Let's run this on our example:

```python
counts = get_pair_counts(tokens)
print(sorted(counts.items(), key=lambda x: -x[1])[:5])
# [((32, 116), 2), ((116, 104), 2), ((104, 101), 2), ((97, 116), 2),.]
```

The pair `(32, 116)` appears twice. That's a space followed by 't'. The pair `(116, 104)` also appears twice, that's 't' followed by 'h'. We're seeing "the" show up multiple times.

### Step 2: Merging the Most Frequent Pair

Now we take the most frequent pair and merge it into a new token. The new token gets the next available ID, which is 256 (since we started with 0-255):

```python
def merge(tokens, pair, new_id):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens
```

If the most frequent pair is `(116, 104)` (which is 't' followed by 'h'), we merge it:

```python
top_pair = max(counts, key=counts.get)
tokens = merge(tokens, top_pair, 256)
print(len(tokens))  # Now shorter!
```

Each merge in BPE creates a mapping from a pair of tokens (children) to a new token ID. We need to save this mapping:

```python
merges = {}
merges[top_pair] = 256
```

### Step 3: Repeat Until Done

The full BPE training algorithm just repeats this process. You pick a target vocabulary size, say 500 tokens, and keep merging until you get there:

```python
def train_bpe(text, vocab_size):
    tokens = list(text.encode("utf-8"))
    merges = {}
    
    num_merges = vocab_size - 256
    
    for i in range(num_merges):
        counts = get_pair_counts(tokens)
        if not counts:
            break
        
        top_pair = max(counts, key=counts.get)
        new_id = 256 + i
        
        tokens = merge(tokens, top_pair, new_id)
        merges[top_pair] = new_id
        
        print(f"Merge {i+1}: {top_pair} -> {new_id}")
    
    return merges
```

Run this on a larger text and watch the merges happen. Early merges tend to be common character pairs like 'th', 'he', 'in'. Later merges combine those into larger chunks like 'the', 'ing', 'tion'.

### What the Merges Look Like

After training on English text, you'll see a pattern. The algorithm builds a forest of merges rather than a single tree. Some merge chains go deep:

- Bytes 't' and 'h' merge into token 256
- Token 256 and byte 'e' merge into token 312
- Token 312 and byte ' ' merge into token 487

Now "the " is a single token. Other chains stay shallow. Rare character combinations never get merged at all.

### The Vocabulary Size Tradeoff

The vocabulary size of a tokenizer is a hyperparameter. There's a sweet spot where text is dense enough for efficient attention but the embedding table and softmax output remain manageable.

Too small (like 256 raw bytes): sequences are too long, you run out of context.

Too large (like 1 million tokens): your embedding table becomes enormous, training slows down, and rare tokens don't get enough examples to learn good representations.

GPT-2 used a vocabulary of roughly 50,257 tokens. GPT-4 uses roughly 100,000 tokens (cl100k_base), which allows the same text to be represented with fewer tokens, effectively increasing context. More tokens per vocabulary means fewer tokens per sequence.

### Efficiency Optimizations

The naive implementation above is slow. Really slow. On a large corpus, you're recounting every pair after every merge. Here's what production implementations do differently:

**Efficient Data Structures**: Instead of a list, use a data structure that supports fast pair counting and updates. Some implementations use linked lists so merges don't require rebuilding the whole sequence.

**Incremental Updates**: When you merge a pair, you don't need to recount everything. You only need to update counts for pairs that touch the merged positions.

**Batch Processing**: Process the corpus in chunks and aggregate statistics.

For learning purposes, the naive version is fine. For training a real tokenizer on gigabytes of text, you'll want the optimized version. Libraries like Hugging Face provide fast implementations.

### What Can Go Wrong

A few things trip people up here:

**Forgetting to save the merge order**: The order of merges matters. When you encode new text, you need to apply merges in the same order they were learned. Save the merges dictionary carefully.

**Training on too little data**: If you train BPE on a tiny corpus, you'll get weird merges that don't generalize. The algorithm learns whatever patterns exist in your training data.

**Confusing training with encoding**: Training builds the vocabulary. Encoding applies it to new text. They're separate operations. We'll cover encoding in the next lesson.

### Success Indicators

You know your BPE training is working when:

1. Sequence lengths decrease steadily with each merge
2. Early merges are common letter pairs ('th', 'he', 'er', 'in')
3. Later merges form recognizable word pieces ('ing', 'tion', 'the')
4. Your final vocabulary size matches your target

Try training on different texts. Train on Python code and you'll see different patterns, things like 'def ', 'self.', 'return'. This is why improvements in tokenizer design can significantly improve model performance on specific tasks without changing the core neural architecture. GPT-4's tokenizer handles whitespace in Python better than GPT-2's did.

### Looking Ahead

We've built the training algorithm. We can take any corpus and learn a vocabulary of subword tokens. But we haven't actually used this vocabulary to encode new text yet. That's next lesson: taking the merges dictionary and using it to convert arbitrary strings into token sequences.

The encoding process has its own subtleties. Tokenization can be case-sensitive and context-dependent. The same word can be tokenized differently based on capitalization, preceding spaces, or position in a sentence. We'll see exactly why when we implement the encoder.

## Lesson 4: BPE Encoding and Decoding

You have trained a BPE tokenizer. You have a vocabulary of merged tokens and the merge rules that created them. Now comes the part that actually matters: using those rules to convert text into numbers and back again.

This is where the rubber meets the road. Encoding transforms arbitrary text into a sequence of token IDs that become Model Input. Decoding reverses the process, turning those IDs back into readable text. Get this wrong and your model sees garbage. Get it right and you have a working tokenizer.

Let us build both operations from scratch.

### The Encoding Problem

Here is what encoding needs to accomplish. You have a string like "lower" and you need to produce a sequence of integers. Those integers index into your vocabulary. The model only understands integers.

The naive approach would be: look up each character, done. But that throws away everything BPE training gave us. The whole point of Byte Pair Encoding was to discover common patterns and merge them into single tokens. If your training found that "er" appears constantly and merged it into token 256, then encoding "lower" should produce something like [108, 111, 119, 256] rather than [108, 111, 119, 101, 114].

So encoding must apply merges. But which merges, and in what order?

### The Greedy Encoding Algorithm

The answer comes from your merge list. During training, you recorded merges in order: first you merged the most frequent pair, then the next most frequent, and so on. That order matters.

Here is the algorithm:

```python
def encode(text, merges):
    # Start with raw bytes
    tokens = list(text.encode("utf-8"))
    
    while len(tokens) >= 2:
        # Find all adjacent pairs
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        
        # Find the pair with lowest merge index (earliest merge)
        pair_to_merge = None
        lowest_index = float('inf')
        for pair in pairs:
            if pair in merges:
                idx = merges[pair]
                if idx < lowest_index:
                    lowest_index = idx
                    pair_to_merge = pair
        
        # If no mergeable pair found, we are done
        if pair_to_merge is None:
            break
        
        # Merge all occurrences of this pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
                new_tokens.append(merges[pair_to_merge])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return tokens
```

Walk through this carefully. You start with raw UTF-8 bytes. Then you repeatedly find the pair that was merged earliest during training and merge all occurrences of it. You keep going until no more merges apply.

Why earliest? Because merge order encodes frequency. The first merges captured the most common patterns. Applying them first ensures consistent tokenization. If you applied merges in random order, the same text could tokenize differently each time.

### The Decoding Problem

Decoding is simpler. You have a list of token IDs and you need to produce text. Your vocabulary maps IDs to byte sequences. Just concatenate and decode.

```python
def decode(ids, vocab):
    # vocab maps id -> bytes
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
```

The `errors="replace"` handles edge cases where byte sequences do not form valid UTF-8. This happens rarely but you need to handle it.

One subtlety: your vocabulary must store the actual bytes each token represents, not just the merge rules. During training, when you merge pair (65, 66) into token 256, you should record that token 256 represents the bytes [65, 66]. For the base vocabulary, token 65 represents byte 65, and so on.

### Efficient Data Structures Matter

The naive encoding algorithm above works but scales poorly. For each merge iteration, you scan the entire token list to find pairs, then scan again to apply the merge. For long texts with many merges, this becomes slow.

Production tokenizers use Efficient Data Structures to speed this up. Common optimizations include:

1. Priority queues to track which pair to merge next
2. Linked lists to make merging O(1) instead of O(n)
3. Caching pair counts with Incremental Updates

The Hugging Face tokenizers library implements these Efficiency Optimizations in Rust for speed. But understanding the basic algorithm matters more than optimizing it. You can always swap in a fast implementation later.

### Handling Special Tokens

Real tokenizers need special tokens that do not appear in normal text. Common ones include:

- `<|endoftext|>` to mark document boundaries
- `<|pad|>` for Batch Processing when sequences have different lengths
- `<|startoftext|>` for some architectures

These get added to your vocabulary with reserved IDs. During encoding, you typically handle them separately from the BPE algorithm. You might split on special token patterns first, encode the regular text pieces with BPE, then insert the special token IDs at the right positions.

```python
def encode_with_special(text, merges, special_tokens):
    # special_tokens is dict like {"<|endoftext|>": 50256}
    
    # Split text on special tokens, keeping them
    import re
    pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
    parts = re.split(pattern, text)
    
    result = []
    for part in parts:
        if part in special_tokens:
            result.append(special_tokens[part])
        elif part:
            result.extend(encode(part, merges))
    
    return result
```

### Where Learners Get Stuck

The most common mistake is applying merges in the wrong order. If your training produced merges [(101, 114) -> 256, (256, 115) -> 257], then encoding "ers" must first merge "er" into 256, then merge that 256 with "s" into 257. If you somehow tried the second merge first, it would not match because 256 does not exist yet in your token list.

Another pitfall: forgetting that the base vocabulary is bytes, not characters. The text "café" encodes to bytes [99, 97, 102, 195, 169] in UTF-8. That last character "é" becomes two bytes. Your BPE operates on these bytes, not on the Unicode characters.

Third mistake: inconsistent vocabulary building. Your vocab must be complete. Every token ID the encoder can produce must have a corresponding entry in the decoder's vocabulary. If token 300 can appear in encoded output, the decoder must know what bytes token 300 represents.

### Verifying Your Implementation

Test roundtrip consistency:

```python
text = "Hello, world! This is a test of BPE encoding."
encoded = encode(text, merges)
decoded = decode(encoded, vocab)
assert decoded == text
```

Test that Sequence Length decreases with merges:

```python
text = "aaaaaaaaaa"  # 10 a's
tokens = encode(text, merges)
# Should be shorter than 10 if 'aa' was merged
print(f"Original length: {len(text.encode('utf-8'))}, Encoded length: {len(tokens)}")
```

Test special tokens:

```python
text = "Hello<|endoftext|>World"
tokens = encode_with_special(text, merges, {"<|endoftext|>": 50256})
# Should contain 50256 in the middle
assert 50256 in tokens
```

### The Connection to Model Input

When you feed tokens to a transformer, each token ID indexes into an embedding matrix. The model never sees raw text. It sees a sequence of integers, each representing a chunk of text that BPE decided was worth treating as a unit.

This is why tokenization matters so much. The Byte Pair Encoding algorithm compresses common patterns into single tokens, keeping Sequence Length manageable. Without this compression, you would hit Sequence Length Explosion on any substantial text, and Standard Transformers have quadratic attention cost in sequence length.

You now have a complete tokenizer: training to discover merges, encoding to apply them, decoding to reverse them. The next lesson covers the edge cases and practical considerations that separate toy implementations from production systems.

## Lesson 5: Special Tokens

You have built a working BPE tokenizer. It can encode any text into tokens and decode those tokens back into text. But if you try to use this tokenizer with an actual language model, you will immediately hit a problem: how does the model know where a document starts? Where it ends? Where one conversation turn stops and another begins?

This is what special tokens solve. And understanding them is essential because, as the source materials note, training efficiency can be significantly improved by adding special tokens to compress common patterns. The GPT family of models uses special tokens extensively, and getting them wrong will break your model in subtle, frustrating ways.

Let us add special tokens to our tokenizer.

### The Problem Special Tokens Solve

Open your BPE implementation and try this thought experiment. You want to train a language model on multiple documents. You concatenate them together:

```python
training_data = document1 + document2 + document3
```

Now your model sees one continuous stream of text. When it reaches the end of document1, it will try to predict the first token of document2 as if they were connected. But they are not connected. Document1 might be a news article about politics. Document2 might be Python code. The model will learn spurious patterns at these boundaries.

You need a way to say "this is where one document ends and another begins." That is what special tokens do.

### Implementing Special Tokens

Special tokens are tokens that do not come from the BPE merging process. They are tokens we define explicitly and reserve specific IDs for. Here is how to add them to your tokenizer:

```python
class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        
        # Reserve IDs for special tokens FIRST
        self.special_tokens = {
            "<|endoftext|>": 0,
            "<|padding|>": 1,
            "<|startoftext|>": 2,
        }
        
        # Build base vocabulary starting AFTER special tokens
        self.base_vocab_start = len(self.special_tokens)
        for i in range(256):
            self.vocab[i + self.base_vocab_start] = bytes([i])
```

The key insight: special tokens get the lowest IDs, and everything else shifts up. This matters because when you decode, you need to handle special tokens differently from regular tokens.

### The Essential Special Tokens

Different models use different special tokens, but there are patterns. GPT-2 and GPT-3 use `<|endoftext|>` as their primary special token. It appears at the end of every document during training. When the model generates this token during inference, it signals "I am done generating."

Here are the special tokens you will encounter most often:

**End of text token**: Marks document boundaries. During training, you insert this between concatenated documents. During inference, generating this token means the model wants to stop.

**Padding token**: When you batch sequences of different lengths together, shorter sequences need padding. The padding token fills these gaps. The model learns to ignore it.

**Beginning of sequence token**: Some architectures want an explicit start marker. This gives the model a consistent starting point.

**Separator token**: For tasks involving pairs of texts (like question and answer), this separates them.

### Encoding with Special Tokens

Your encode function needs to handle special tokens before applying BPE:

```python
def encode(self, text, add_special_tokens=True):
    # First, check for special tokens in the text
    if add_special_tokens:
        text = "<|startoftext|>" + text + "<|endoftext|>"
    
    # Split text around special tokens
    pattern = '(' + '|'.join(re.escape(t) for t in self.special_tokens) + ')'
    parts = re.split(pattern, text)
    
    tokens = []
    for part in parts:
        if part in self.special_tokens:
            # Special token: use its reserved ID directly
            tokens.append(self.special_tokens[part])
        else:
            # Regular text: apply BPE encoding
            tokens.extend(self._encode_chunk(part))
    
    return tokens
```

The critical detail: special tokens are never broken apart by BPE. The string `<|endoftext|>` must always become a single token, not a sequence of tokens for `<`, `|`, `e`, and so on.

### Decoding with Special Tokens

Decoding needs the reverse mapping:

```python
def decode(self, tokens, skip_special_tokens=False):
    # Build reverse mapping for special tokens
    id_to_special = {v: k for k, v in self.special_tokens.items()}
    
    result = []
    for token_id in tokens:
        if token_id in id_to_special:
            if not skip_special_tokens:
                result.append(id_to_special[token_id])
        else:
            result.append(self.vocab[token_id].decode('utf-8', errors='replace'))
    
    return ''.join(result)
```

The `skip_special_tokens` parameter matters for user-facing output. When showing generated text to users, you typically want to hide the special tokens. But for debugging, you want to see them.

### Special Tokens for Code

The source materials specifically mention that training efficiency on code can be significantly improved by adding special tokens to compress runs of spaces. Code has patterns that natural language does not: indentation, for instance.

Consider Python code:

```python
def foo():
    if True:
        return 1
```

Those leading spaces are common in code but inefficiently tokenized by standard BPE trained on natural language. You can add special tokens like:

```python
self.special_tokens["<|indent4|>"] = 3  # represents 4 spaces
self.special_tokens["<|indent8|>"] = 4  # represents 8 spaces
self.special_tokens["<|newline|>"] = 5  # explicit newline
```

This is a form of domain-specific optimization. The model learns that `<|indent4|>` means "four spaces of indentation" and can reason about code structure more efficiently.

### Common Mistakes

**Mistake 1: Not reserving IDs before training BPE.** If you train BPE first and then try to add special tokens, you will have ID collisions. Special token IDs must be reserved before any other vocabulary is built.

**Mistake 2: Letting BPE split special tokens.** If your special token is `<|endoftext|>` but BPE sees the characters `<`, `|`, `e`... it will merge them according to its learned rules. You must handle special tokens before BPE encoding.

**Mistake 3: Forgetting special tokens during training data preparation.** Your training loop must insert `<|endoftext|>` between documents. If you forget, the model learns wrong patterns at document boundaries.

**Mistake 4: Mismatched special tokens between training and inference.** If you train with `<|endoftext|>` but your inference code looks for `</s>`, nothing will work. Special token handling must be consistent.

### Success Indicators

You know your special token implementation is correct when:

1. `encode("<|endoftext|>")` returns a single token ID, not multiple
2. `decode([0])` returns `"<|endoftext|>"` (assuming 0 is your end token ID)
3. `encode("hello<|endoftext|>world")` returns three logical parts: tokens for "hello", the special token ID, tokens for "world"
4. Round-trip works: `decode(encode(text)) == text` for text containing special tokens

### Looking Forward

Special tokens become even more important when you reach the finetuning chapters. Supervised fine-tuning (SFT) uses special tokens to mark conversation turns. Reinforcement learning from human feedback uses special tokens to structure prompts and responses.

The model scale matters here too. As the source materials note, model scale is a critical factor for the emergence and improvement of zero-shot and few-shot capabilities. Larger models learn to use special tokens more effectively, understanding that `<|endoftext|>` is not just another token but a structural signal.

You now have a complete tokenizer. It handles Unicode through UTF-8, compresses text through BPE, and manages special tokens for document structure. This is the same fundamental approach used by GPT-2, GPT-3, and their descendants.

In the next module, we will use this tokenizer to prepare actual training data.

## Practice

Build character-level tokenization baseline, implement BPE training algorithm, create BPE encoding and decoding, and handle special tokens.

### Setup

Create a new file called `tokenizer.py`. You'll also need a text file for training, use any plain text file with at least a few thousand characters. A chapter from a book or a collection of Wikipedia articles works well.

```python
# Start with this skeleton
class Tokenizer:
    def __init__(self):
        self.vocab = {}  # token_id -> bytes
        self.merges = {}  # (token_id, token_id) -> new_token_id
    
    def train(self, text: str, vocab_size: int):
        """Learn BPE merges from training text."""
        pass
    
    def encode(self, text: str) -> list[int]:
        """Convert text to token ids."""
        pass
    
    def decode(self, ids: list[int]) -> str:
        """Convert token ids back to text."""
        pass
```

### Part 1: Character-Level Baseline

Before implementing BPE, build a simple character-level tokenizer:

1. Create a vocabulary mapping each unique byte (0-255) to itself
2. Implement `encode`: convert string to UTF-8 bytes, return as list of integers
3. Implement `decode`: convert list of integers back to bytes, decode as UTF-8

Test it: `decode(encode(text)) == text` must be true for any valid UTF-8 string.

### Part 2: BPE Training

Implement the `train` method:

1. Convert training text to a list of byte tokens
2. Count all adjacent pairs
3. Find the most frequent pair
4. Create a new token ID for this pair (start at 256)
5. Replace all occurrences of the pair with the new token
6. Record the merge rule
7. Repeat until you reach `vocab_size`

### Part 3: BPE Encoding

Update `encode` to use your learned merges:

1. Start with raw bytes
2. Apply merges in the order they were learned
3. Return final token sequence

### Part 4: BPE Decoding

Update `decode` to reverse the process:

1. For each token, look up its bytes (recursively expanding merged tokens)
2. Concatenate all bytes
3. Decode as UTF-8

### Part 5: Special Tokens

Add support for special tokens like `<|endoftext|>`:

1. Reserve token IDs for special tokens before training
2. Modify encode to handle special token strings
3. Modify decode to emit special token strings

### Success Criteria

- [ ] `decode(encode(text)) == text` for any input text
- [ ] Training on 100KB of text with vocab_size=500 completes in under 30 seconds
- [ ] Encoded sequences are shorter than raw bytes (compression is working)
- [ ] Special tokens encode to single token IDs and decode correctly

### Common Pitfalls

**Merge order matters.** When encoding new text, you must apply merges in the same order they were learned during training. Applying them in a different order produces different tokenizations.

**UTF-8 edge cases.** Some byte sequences are invalid UTF-8. Your decoder should handle this gracefully, consider using `errors='replace'` when decoding.

**Counting pairs after merging.** When you replace a pair with a new token, the pair counts change. You need to recount, not just decrement.

**Off-by-one in vocabulary size.** Remember you start with 256 byte tokens. If you want vocab_size=1000, you need 744 merges.

### Stretch Goals

- Implement regex-based pre-tokenization (split on whitespace and punctuation before BPE)
- Add a `save` and `load` method to persist your trained tokenizer
- Compare compression ratios across different vocabulary sizes
- Visualize what tokens your BPE learns (you'll see common words and subwords emerge)

## Assessment

By the end of this module, you should be able to:

- **Explain why tokenization exists** and articulate the tradeoffs between character-level, word-level, and subword tokenization approaches
- **Implement Byte Pair Encoding from scratch**, including the iterative merge process that builds a vocabulary from raw text
- **Define and manipulate vocabulary mappings**, converting between text, tokens, and integer IDs in both directions
- **Describe how subword units solve the open vocabulary problem**, handling rare words, misspellings, and morphological variations without exploding vocabulary size
- **Explain Unicode and UTF-8 encoding**, understanding why text is ultimately bytes, how variable-width encoding works, and why tokenizers typically operate on UTF-8 byte sequences rather than Unicode code points

Success means you can take any piece of text, trace exactly how it becomes a sequence of integers, and explain every decision along the way. You should be able to look at a tokenized sequence and predict where the boundaries fell and why. When someone asks "why did the model struggle with this word?" you should immediately think about how it was tokenized. You understand that tokenization is not preprocessing, it is a fundamental part of what the model learns and how it sees language.

---

# Datasets

## Lesson 1: Data Loading and Generation

You have a transformer. You have an optimizer. You have a loss function. Now you need something to feed it.

This is where most tutorials wave their hands and say "download this dataset" and move on. We are not going to do that. Understanding where your data comes from, how it gets to your model, and what makes it good or bad is the difference between a model that works and a model that generates garbage.

Let us start with the simplest possible thing.

### The Shape of the Problem

Your model expects batches of token sequences. That is it. A tensor of shape `[batch_size, sequence_length]` containing integer token IDs. Everything else in this chapter is about getting data into that shape efficiently.

```python
# This is what your model wants
batch = torch.tensor([
    [104, 2891, 553, 12, 9920, 442],  # sequence 1
    [7721, 33, 1002, 856, 44, 221],   # sequence 2
    [901, 4455, 23, 778, 6632, 19],   # sequence 3
])
# Shape: [3, 6] - three sequences, six tokens each
```

The autoregressive loss function, which predicts the next token in a sequence, is the foundational training mechanism that enabled the development of modern language models. Your data loader exists to serve this function. Every design decision flows from this.

### Loading Text Data

Here is the naive approach:

```python
def load_dataset_naive(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens)
```

This works for small files. It fails catastrophically for anything resembling real training data. Common Crawl alone is hundreds of terabytes. You cannot load that into memory.

The solution is memory mapping. Instead of loading the entire file, you create a view into the file on disk and only load the parts you need:

```python
import numpy as np
import os

class MemmapDataset:
    def __init__(self, filepath, sequence_length):
        # Memory-map the file - does not load into RAM
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Only loads this specific chunk from disk
        chunk = self.data[idx:idx + self.sequence_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
```

The `x` is your input sequence. The `y` is your target sequence, shifted by one position. The model learns to predict each next token from the tokens before it.

### Batching and Shuffling

Random access matters. If you train on sequences in order, your model sees the same patterns in the same order every epoch. This creates weird correlations and slows learning.

```python
from torch.utils.data import DataLoader

dataset = MemmapDataset('training_data.bin', sequence_length=1024)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel loading
    pin_memory=True  # Faster GPU transfer
)

for batch_x, batch_y in loader:
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
    # Train.
```

The `num_workers` parameter spawns separate processes to load data while your GPU is computing. Without this, your GPU sits idle waiting for data. With it, your next batch is ready before your GPU finishes the current one.

### Synthetic Data Generation

Here is where things get interesting.

You can generate training data from existing language models. Self-instruct data generation involves using a stronger language model to expand a small set of human prompts into a large, diverse dataset of question-answer pairs without human annotation.

The Alpaca model demonstrated that fine-tuning on synthetic data generated by a stronger model (like OpenAI's API models) can produce capable instruction-following models. This was a watershed moment. Suddenly you did not need thousands of human annotators.

The basic pattern:

```python
def generate_synthetic_examples(seed_prompts, generator_model):
    """
    Expand a small set of human-written prompts into
    a larger dataset using a capable model.
    """
    synthetic_data = []
    
    for seed in seed_prompts:
        # Ask the model to generate variations
        prompt = f"""Given this example task:
{seed}

Generate 5 similar but different tasks with answers:"""
        
        response = generator_model.generate(prompt)
        examples = parse_examples(response)
        synthetic_data.extend(examples)
    
    return synthetic_data
```

Synthetic data generation from stronger models has become the dominant approach for creating instruction-tuning datasets, while human-written data remains rare but valuable. Quality synthetic data generated this way can bootstrap surprisingly capable models.

For Phi, Microsoft Research showed that carefully curated synthetic data can match or exceed models trained on much larger web scrapes. The key word is "carefully." Bad synthetic data produces bad models.

### Data Quality: What Makes Data Good

Not all data is equal. Here are the things that matter:

**Diversity**: Your model learns what it sees. If 90% of your data is news articles, your model will sound like a news anchor. ShareGPT provided open language model builders with prompts similar to what users asked ChatGPT, significantly improving data diversity and accelerating progress in fine-tuning.

**Correctness**: Garbage in, garbage out. A model trained on wrong answers learns to give wrong answers. Verification methods matter here. You need ways to check that your data is actually correct.

**Format consistency**: If your instruction data has ten different formats, your model has to learn ten different things. Pick one format and stick with it.

```python
def filter_quality(examples):
    """Basic quality filtering."""
    filtered = []
    for ex in examples:
        # Length checks
        if len(ex['input']) < 10 or len(ex['output']) < 10:
            continue
        if len(ex['output']) > 10000:  # Suspiciously long
            continue
        
        # Basic content checks
        if ex['output'].count('I cannot') > 3:  # Refusal spam
            continue
        if detect_language(ex['output']) != 'en':  # Wrong language
            continue
            
        filtered.append(ex)
    return filtered
```

The OpenAssistant project demonstrated that large-scale human data collection (10k+ annotated trees, 1000+ volunteers) is crucial for alignment and remains extensively used today. Human annotation is expensive but produces signal that synthetic data struggles to match.

### Common Mistakes

**Loading everything into memory**: Use memory mapping. Always.

**Not shuffling**: Your model will overfit to sequence order.

**Ignoring data quality**: More data is not always better. Filtered data beats raw data.

**Single-threaded loading**: Your GPU is expensive. Do not let it sit idle waiting for data.

**Forgetting the +1**: Your target sequence needs to be shifted by one token. The model predicts position i+1 from positions 0 through i.

### Success Indicators

You know your data pipeline is working when:

1. GPU utilization stays above 90% during training
2. Your loss decreases smoothly without sudden jumps
3. Generated samples show diversity matching your training data
4. You can restart training from any checkpoint without data ordering issues

### What Comes Next

You have the mechanics of getting data into your model. But we have not talked about where the data comes from in the first place, or how to curate it for specific tasks like storytelling. That is the next lesson.

The transition from human-annotated datasets to synthetic/self-instruct data enabled the rapid proliferation of open fine-tuned models like Llama and Alpaca variants. Understanding both approaches, and when to use which, is what separates someone who can train a model from someone who can train a good model.

## Lesson 2: Efficient Data Loading

You have a problem. Your training data is 160 gigabytes. Your GPU has 24 gigabytes of memory. Your system RAM is 64 gigabytes. How do you train on data that does not fit in memory?

The naive answer is "load chunks one at a time." The real answer is memory mapping. Let us build it.

### The Problem You Actually Have

Open a file normally in Python:

```python
with open("training_data.bin", "rb") as f:
    data = f.read()  # This loads EVERYTHING into RAM
```

Try this with 160 gigabytes and watch your system freeze. The operating system will start swapping to disk, your training will crawl, and you will learn an expensive lesson about Computational Cost.

Here is what we want instead: a way to treat a file on disk as if it were already in memory, loading only the pieces we actually access. The operating system already knows how to do this. We just need to ask.

### Memory Mapping From Scratch

```python
import mmap
import numpy as np

def create_memory_mapped_dataset(filepath):
    """
    Memory map a binary file containing token IDs.
    Returns a numpy array that reads directly from disk.
    """
    # Open the file in read mode
    f = open(filepath, "rb")
    
    # Get file size
    f.seek(0, 2)  # Seek to end
    file_size = f.tell()
    f.seek(0)  # Back to start
    
    # Create memory map
    mm = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_READ)
    
    # Wrap as numpy array - NO COPY HAPPENS HERE
    tokens = np.frombuffer(mm, dtype=np.uint16)
    
    return tokens, f, mm  # Keep references alive
```

What just happened? The `mmap.mmap` call told your operating system: "I want to treat this file as if it were memory." The OS creates a virtual memory mapping. When you access `tokens[1000000]`, the OS loads just that page from disk. Pages you never touch stay on disk.

This is why a 500-line C program can load 70 billion parameters. The parameters file gets memory mapped. The run file just accesses the weights it needs. The interface between them is the memory map itself, a view into the file that loads on demand.

### Building the Data Loader

Now we need to serve batches efficiently. Here is where batching strategy matters:

```python
class EfficientDataLoader:
    def __init__(self, tokens, batch_size, block_size):
        self.tokens = tokens  # Memory-mapped array
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_tokens = len(tokens)
        
        # Precompute valid starting positions
        self.n_batches = (self.n_tokens - block_size) // batch_size
        
    def get_batch(self, batch_idx):
        """
        Get a specific batch by index.
        Returns (inputs, targets) where targets are inputs shifted by 1.
        """
        # Calculate starting positions for this batch
        start_positions = []
        for i in range(self.batch_size):
            pos = (batch_idx * self.batch_size + i) % (self.n_tokens - self.block_size)
            start_positions.append(pos)
        
        # Extract sequences
        x = np.zeros((self.batch_size, self.block_size), dtype=np.int64)
        y = np.zeros((self.batch_size, self.block_size), dtype=np.int64)
        
        for i, pos in enumerate(start_positions):
            x[i] = self.tokens[pos:pos + self.block_size]
            y[i] = self.tokens[pos + 1:pos + self.block_size + 1]
        
        return x, y
```

Notice we are not loading the entire dataset. We are indexing into a memory-mapped array. Only the pages containing our batch positions get loaded from disk.

### Why Batching Matters

You might wonder: why not just process the entire dataset at once? The Computational Cost makes this impossible for large datasets. But there are deeper reasons.

Processing the entire dataset as a whole eliminates the beneficial noise from stochastic gradient descent. That noise helps escape local minima. It improves generalization. Mini-batch processing gives you multiple weight updates per epoch instead of one. Modern GPUs are optimized for parallel batch processing, not sequential whole-dataset operations.

The batch size is a tradeoff. Larger batches give more stable gradients but fewer updates per epoch. Smaller batches give noisier gradients but more frequent updates. In practice, you pick the largest batch size that fits in GPU memory, then tune from there.

### Shuffling Without Loading Everything

For training, we want random batches. But we cannot shuffle 160 gigabytes in memory. Solution: shuffle indices, not data.

```python
class ShuffledDataLoader(EfficientDataLoader):
    def __init__(self, tokens, batch_size, block_size, seed=42):
        super().__init__(tokens, batch_size, block_size)
        self.rng = np.random.default_rng(seed)
        self.shuffle_indices()
    
    def shuffle_indices(self):
        """Create a shuffled list of valid starting positions."""
        n_valid = self.n_tokens - self.block_size - 1
        self.indices = self.rng.permutation(n_valid)
        self.current_idx = 0
    
    def get_random_batch(self):
        """Get next random batch, reshuffling when exhausted."""
        if self.current_idx + self.batch_size > len(self.indices):
            self.shuffle_indices()
        
        positions = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        x = np.zeros((self.batch_size, self.block_size), dtype=np.int64)
        y = np.zeros((self.batch_size, self.block_size), dtype=np.int64)
        
        for i, pos in enumerate(positions):
            x[i] = self.tokens[pos:pos + self.block_size]
            y[i] = self.tokens[pos + 1:pos + self.block_size + 1]
        
        return x, y
```

The indices array is just integers. A few hundred megabytes at most, even for billions of tokens. We shuffle that, then use it to index into the memory-mapped data.

### Common Mistakes

**Mistake 1: Forgetting to keep references alive.** Memory maps get garbage collected if you lose the reference. Keep the file handle and mmap object alive for the duration of training.

**Mistake 2: Using the wrong dtype.** If your tokenizer produces uint16 tokens but you load as int32, you will read garbage. Match the dtype exactly.

**Mistake 3: Not handling file boundaries.** When extracting a sequence near the end of the file, you can read past the valid data. Always check that `pos + block_size + 1 <= n_tokens`.

**Mistake 4: Copying when you meant to view.** Operations like `np.array(tokens[pos:pos+block_size])` create copies. For the final batch tensor, this is fine. But avoid unnecessary intermediate copies.

### How to Know It Is Working

Your data loader is working correctly when:

1. Memory usage stays constant regardless of dataset size
2. You can iterate through the entire dataset without memory growth
3. Random access to any position returns correct token sequences
4. Training speed is limited by GPU compute, not data loading

Test with a small file first where you can verify the tokens manually. Then scale to your full dataset.

### The Connection to Scale

This is why training runs cost tens or hundreds of millions of dollars. You need very large clusters processing very large datasets. The data loading infrastructure must keep GPUs fed continuously. Any bottleneck in the data pipeline means expensive hardware sitting idle.

Memory mapping is the foundation. On top of this, production systems add prefetching, multi-process loading, and distributed data sharding. But the core insight remains: treat the file as memory, access only what you need, and let the operating system handle the rest.

In the next lesson, we will build the batching and shuffling strategies that sit on top of this foundation.

## Lesson 3: Batching and Shuffling

You have a dataset. Maybe it's a few thousand examples, maybe it's millions. Your model needs to learn from all of it. The naive approach would be to load everything into memory, compute gradients on the entire dataset, and update your weights once. This is called full-batch gradient descent.

It doesn't work.

Not because the math is wrong. The math is fine. It doesn't work because of Memory Constraints. Your GPU has maybe 24GB, maybe 80GB if you're lucky. A serious dataset with millions of tokens, plus model parameters, plus activations, plus optimizer states? You're looking at needing hundreds of gigabytes. The determining factors here are GPU memory capacity, model size, sequence length, and precision. Without mini-batching, even moderately sized models would exhaust GPU memory with datasets containing more than a few thousand examples.

So we batch.

### Why Batching Exists

Let's build a simple batching function:

```python
def create_batches(data, batch_size):
    """Split data into batches."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

# Example usage
tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
batches = create_batches(tokens, batch_size=3)
# Result: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
```

Simple. But this is just splitting. For language model training, we need sequences:

```python
import torch

def get_batch(data, batch_size, block_size, device='cpu'):
    """Get a random batch of sequences for training."""
    # Random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack sequences into batch
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)
```

This function does something important: it samples random positions from your data. Each batch contains `batch_size` sequences, each of length `block_size`. The targets `y` are just the inputs shifted by one position. This is the fundamental setup for next-token prediction.

### The Data Distribution Problem

Here's where shuffling becomes critical.

Imagine your training data is organized by topic. All the cooking recipes come first, then all the sports articles, then all the science papers. If you train in order, your model sees nothing but recipes for the first 10,000 batches. It gets really good at predicting cooking terms. Then suddenly it hits sports content and everything it learned about recipes becomes less useful.

This is a Data Distribution problem. The model's gradient updates are biased toward whatever it saw recently. You want each batch to be a representative sample of your entire dataset.

```python
def shuffle_data(data, seed=None):
    """Shuffle data indices for random access."""
    if seed is not None:
        torch.manual_seed(seed)
    
    indices = torch.randperm(len(data))
    return indices

class ShuffledDataLoader:
    def __init__(self, data, batch_size, block_size):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.indices = None
        self.position = 0
        
    def reset_epoch(self):
        """Shuffle at the start of each epoch."""
        max_start = len(self.data) - self.block_size
        self.indices = torch.randperm(max_start)
        self.position = 0
        
    def next_batch(self):
        if self.indices is None:
            self.reset_epoch()
            
        if self.position + self.batch_size > len(self.indices):
            self.reset_epoch()
            
        batch_indices = self.indices[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        
        x = torch.stack([self.data[i:i+self.block_size] for i in batch_indices])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in batch_indices])
        
        return x, y
```

The key insight: shuffle once per epoch, not once per batch. This ensures every example gets seen exactly once per epoch While Transformers and other architectures benefit from seeing diverse data in each batch.

### Batch Size Selection

Batch size is one of those hyperparameters that seems arbitrary but actually matters a lot. Transformers gained widespread adoption partly due to their robustness to hyperparameter choices, but batch size still requires thought.

Smaller batches:
- More frequent updates
- Noisier gradients (can help escape local minima)
- Lower memory usage
- Potentially slower convergence

Larger batches:
- More stable gradients
- Better GPU utilization
- Higher memory usage
- Can require learning rate adjustment

```python
def calculate_max_batch_size(model, block_size, dtype=torch.float32):
    """Estimate maximum batch size for available GPU memory."""
    if not torch.cuda.is_available():
        return 32  # Conservative default for CPU
    
    # Get available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    # Rough estimate: model params + activations + gradients
    # This is approximate, actual usage depends on many factors
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Activations scale with batch_size * block_size * hidden_dim
    # Start conservative and binary search
    batch_size = 1
    while batch_size < 512:
        try:
            test_input = torch.randint(0, 1000, (batch_size * 2, block_size)).cuda()
            _ = model(test_input)
            batch_size *= 2
            del test_input
            torch.cuda.empty_cache()
        except RuntimeError:
            break
            
    return batch_size
```

### Common Mistakes

**Mistake 1: Not shuffling between epochs**

If you use the same order every epoch, your model might memorize the sequence of examples rather than learning Generalization Beyond the training set.

**Mistake 2: Dropping the last incomplete batch**

```python
# Wrong: loses data
for i in range(0, len(data) - batch_size, batch_size):.

# Right: handles remainder
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]  # Automatically handles shorter final batch
```

**Mistake 3: Shuffling sequences incorrectly for language modeling**

For autoregressive language models, you shuffle starting positions, not individual tokens. Breaking up sequences destroys the context the model needs.

**Mistake 4: Ignoring sequence boundaries in documents**

If your data contains multiple documents, you probably don't want sequences that span document boundaries:

```python
def get_batch_respecting_boundaries(data, boundaries, batch_size, block_size):
    """Sample sequences that don't cross document boundaries."""
    valid_starts = []
    for i, (start, end) in enumerate(boundaries):
        # Only include positions where full sequence fits within document
        for pos in range(start, end - block_size):
            valid_starts.append(pos)
    
    selected = random.sample(valid_starts, batch_size)
    x = torch.stack([data[i:i+block_size] for i in selected])
    y = torch.stack([data[i+1:i+block_size+1] for i in selected])
    return x, y
```

### Success Indicators

You know your batching and shuffling is working when:

1. Training loss decreases smoothly without sudden spikes when transitioning between data regions
2. GPU memory usage is stable and close to your target
3. Each epoch produces slightly different loss curves (because shuffle order differs)
4. Validation performance tracks training performance without massive gaps

The Credit Assignment for what makes training work is often misplaced. People focus on architectures and optimizers. But bad data loading can sabotage everything. If your batches aren't representative of your data distribution, your model learns a distorted view of the world.

For large datasets, random mini-batching serves as an approximation while still benefiting from inter-data-point relationships within batches. This is why these training runs today are many tens or even potentially hundreds of millions of dollars. You need very large clusters, very large datasets, and very careful data preparation.

Next lesson, we'll look at synthetic data generation for storytelling, where you'll learn to create training data that doesn't exist yet.

## Lesson 4: Synthetic Data Generation

You have a problem. You need thousands of story prompts and completions to train your Storyteller, but manually writing them would take months. The solution that transformed the entire field in 2023 was surprisingly simple: use a stronger model to generate your training data.

Let's build a synthetic data generator.

### The Self-Instruct Pattern

Here is the core loop that changed everything:

```python
import json
from pathlib import Path

class StoryDataGenerator:
    def __init__(self, teacher_model, seed_examples):
        """
        teacher_model: API or local model stronger than what we're training
        seed_examples: small set of human-written story prompts and completions
        """
        self.teacher = teacher_model
        self.seeds = seed_examples
        self.generated = []
    
    def expand_prompts(self, num_new=100):
        """Generate new story prompts from seed examples."""
        
        expansion_prompt = """Here are some example story prompts:

{examples}

Generate {n} new, diverse story prompts in the same style. 
Make them varied: different genres, lengths, tones.
Output as JSON list."""
        
        # Sample a few seeds for context
        example_text = "\n".join(
            f"- {ex['prompt']}" 
            for ex in random.sample(self.seeds, min(5, len(self.seeds)))
        )
        
        response = self.teacher.generate(
            expansion_prompt.format(examples=example_text, n=num_new)
        )
        
        new_prompts = json.loads(response)
        return new_prompts
```

This is Self-Instruct Data Generation: using a stronger language model to expand a small set of human prompts into a large, diverse dataset of question-answer pairs without human annotation. The Alpaca model demonstrated that fine-tuning on synthetic data generated by a stronger model can produce capable instruction-following models. That result was kind of shocking at the time.

### Generating Completions

Prompts alone are not enough. You need the stories too:

```python
def generate_story_completion(self, prompt):
    """Have teacher model write a story for the prompt."""
    
    story_instruction = """Write a short story based on this prompt.
    
Prompt: {prompt}

Requirements:
- 200-500 words
- Clear beginning, middle, end
- Engaging characters
- Show don't tell

Story:"""
    
    story = self.teacher.generate(
        story_instruction.format(prompt=prompt),
        max_tokens=1024,
        temperature=0.7
    )
    
    return {
        'prompt': prompt,
        'completion': story,
        'source': 'synthetic',
        'teacher_model': self.teacher.model_name
    }

def generate_dataset(self, target_size=10000):
    """Build full synthetic dataset."""
    
    # Start with seed completions
    dataset = list(self.seeds)
    
    while len(dataset) < target_size:
        # Generate new prompts
        new_prompts = self.expand_prompts(num_new=100)
        
        # Generate completions for each
        for prompt in new_prompts:
            if len(dataset) >= target_size:
                break
                
            example = self.generate_story_completion(prompt)
            
            # Basic quality filter
            if self.passes_quality_check(example):
                dataset.append(example)
                
        print(f"Dataset size: {len(dataset)}/{target_size}")
    
    return dataset
```

### Quality Filtering

Not all synthetic data is good. You need filters:

```python
def passes_quality_check(self, example):
    """Filter out bad synthetic examples."""
    
    completion = example['completion']
    
    # Length check
    word_count = len(completion.split())
    if word_count < 50 or word_count > 2000:
        return False
    
    # Refusal detection - model said no
    refusal_phrases = [
        "I cannot",
        "I'm not able to",
        "I cannot answer that",
        "I'm not programmed to"
    ]
    if any(phrase.lower() in completion.lower() for phrase in refusal_phrases):
        return False
    
    # Repetition check
    sentences = completion.split('.')
    if len(sentences) > 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.7:  # Too much repetition
            return False
    
    # Basic coherence - does it end properly?
    if not completion.strip().endswith(('.', '!', '?', '"')):
        return False
    
    return True
```

The filtering step matters more than you think. A dataset with 5000 Quality Synthetic Data Generated examples beats 20000 unfiltered examples. Bad data actively hurts your model.

### Diversity Through Prompting

The biggest failure mode is generating the same story over and over. Force diversity:

```python
def generate_diverse_prompts(self, num_prompts, categories=None):
    """Generate prompts across specific types."""
    
    if categories is None:
        categories = [
            "fantasy adventure",
            "science fiction", 
            "mystery",
            "romance",
            "horror",
            "comedy",
            "historical fiction",
            "slice of life"
        ]
    
    all_prompts = []
    per_category = num_prompts // len(categories)
    
    for category in categories:
        category_prompt = f"""Generate {per_category} unique story prompts 
for the {category} genre.

Each prompt should suggest:
- A specific setting
- An interesting character
- A conflict or goal

Output as JSON list of strings."""
        
        response = self.teacher.generate(category_prompt)
        prompts = json.loads(response)
        
        for p in prompts:
            all_prompts.append({
                'prompt': p,
                'category': category
            })
    
    return all_prompts
```

### The ShareGPT Insight

Here is something that accelerated progress significantly: ShareGPT provided open language model builders with prompts similar to what users asked ChatGPT. Real user prompts are gold because they show what people actually want, not what researchers imagine they want.

For your Storyteller, this means collecting real story requests when possible:

```python
def incorporate_real_prompts(self, user_prompts_file):
    """Mix real user prompts with synthetic ones."""
    
    real_prompts = []
    with open(user_prompts_file) as f:
        for line in f:
            prompt = line.strip()
            if len(prompt) > 10:  # Basic filter
                real_prompts.append({
                    'prompt': prompt,
                    'source': 'user',
                    'category': 'unknown'
                })
    
    # Generate completions for real prompts
    real_examples = []
    for p in real_prompts:
        example = self.generate_story_completion(p['prompt'])
        example['original_source'] = 'user_submitted'
        if self.passes_quality_check(example):
            real_examples.append(example)
    
    return real_examples
```

### Why This Works

The autoregressive loss function, which predicts the next token in a sequence, does not care whether your training data came from humans or from another model. It just learns the distribution. The transition from human-annotated datasets to synthetic/self-instruct data enabled the rapid proliferation of open fine-tuned models.

But understand the limitation: your student model will be at best as good as your teacher on the specific tasks you generate data for. You are doing a kind of Synthetic Data Distillation, transferring knowledge from a larger model into a smaller one. The student learns a gestalt rather than exact copies of the teacher's behavior.

### Common Mistakes

**Using a weak teacher**: If your teacher model writes bad stories, your training data will be bad. Use the strongest model you can access for generation.

**No diversity control**: Without explicit category forcing, you will generate thousands of variations on the same three story types. The model latches onto whatever patterns are easiest.

**Skipping quality filters**: It feels wasteful to throw away generated data. But bad examples teach bad habits. Filter aggressively.

**Forgetting the human seeds**: Your seed examples anchor the entire distribution. Spend real time writing 50-100 excellent human examples before you start synthetic generation.

### Success Indicators

You know your synthetic data pipeline is working when:

1. Generated prompts cover all your target categories roughly equally
2. Quality filter pass rate is above 70% (below that, fix your generation prompts)
3. Random samples from the dataset read like plausible stories
4. A model trained on your data generates stories that feel different from each other

### Putting It Together

```python
# Full pipeline
generator = StoryDataGenerator(
    teacher_model=load_api_model("gpt-4"),
    seed_examples=load_json("human_seeds.json")  # 100 hand-written examples
)

# Generate diverse synthetic dataset
synthetic = generator.generate_dataset(target_size=10000)

# Mix with any real user prompts you have
if Path("user_prompts.txt").exists():
    real = generator.incorporate_real_prompts("user_prompts.txt")
    synthetic.extend(real)

# Save for training
save_dataset(synthetic, "storyteller_train.jsonl")
```

This is how modern fine-tuning datasets get built. The OpenAssistant project demonstrated that large-scale human data collection is crucial for alignment, but for most applications, synthetic data generation from stronger models has become the dominant approach for creating instruction-tuning datasets.

Next lesson, we bring everything together: loading, batching, shuffling, and synthetic generation into a complete data pipeline for training your Storyteller.

## Lesson 5: Data Quality

You have built a data loader. You have implemented batching and shuffling. You have generated synthetic data. Now comes the part that actually matters most.

Data quality is more important than data quantity. This is not a nice-to-have principle. This is the dominant factor in whether your model learns anything useful. Research teams at large companies often have data teams that significantly outnumber modeling teams. That ratio tells you something about where the real work happens.

Let me put it bluntly: your model's output quality directly reflects your training data quality. Garbage in, garbage out is not a cliche here. It is a mathematical certainty.

### Why Quality Beats Quantity

The scaling laws everyone talks about depend on clean data. You can throw more compute at a problem. You can make your model bigger. But if your data is contaminated with noise, duplicates, or low-quality examples, you hit a ceiling that no amount of scaling will break through.

Think about what next-word prediction actually requires. When you train a language model on a large enough dataset, it learns diverse skills: grammar, lexical semantics, world knowledge, sentiment analysis, translation, spatial reasoning, math. But it can only learn these skills if the training signal is consistent. If your data says "2+2=4" in one place and "2+2=5" in another, the model learns neither. It learns confusion.

Modern AI research has shifted from architectural innovation toward a heavy focus on data engineering and loss optimization. The architecture is mostly solved. The data problem is not.

### Building a Quality Filter

Let us build a practical quality filter for text data. We will start with the simplest useful implementation and then extend it.

```python
import re
from collections import Counter

class DataQualityFilter:
    def __init__(self, min_length=50, max_length=10000):
        self.min_length = min_length
        self.max_length = max_length
        self.stats = Counter()
    
    def check_length(self, text):
        """Filter by reasonable length bounds."""
        length = len(text)
        if length < self.min_length:
            self.stats['too_short'] += 1
            return False
        if length > self.max_length:
            self.stats['too_long'] += 1
            return False
        return True
    
    def check_repetition(self, text, threshold=0.3):
        """Detect excessive repetition."""
        words = text.lower().split()
        if len(words) < 10:
            return True
        
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        repetition_ratio = most_common_count / len(words)
        
        if repetition_ratio > threshold:
            self.stats['too_repetitive'] += 1
            return False
        return True
    
    def check_language_quality(self, text):
        """Basic heuristics for text quality."""
        # Check for reasonable punctuation
        punct_ratio = sum(1 for c in text if c in '. !?;:') / max(len(text), 1)
        if punct_ratio < 0.005 or punct_ratio > 0.15:
            self.stats['bad_punctuation'] += 1
            return False
        
        # Check for excessive special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.2:
            self.stats['too_many_special'] += 1
            return False
        
        return True
    
    def filter(self, text):
        """Apply all quality checks."""
        if not self.check_length(text):
            return False
        if not self.check_repetition(text):
            return False
        if not self.check_language_quality(text):
            return False
        
        self.stats['passed'] += 1
        return True
    
    def report(self):
        """Print filtering statistics."""
        total = sum(self.stats.values())
        print(f"Total processed: {total}")
        for reason, count in self.stats.most_common():
            pct = 100 * count / max(total, 1)
            print(f"  {reason}: {count} ({pct:.1f}%)")
```

This is a starting point. Real data pipelines get much more sophisticated.

### Deduplication: The Hidden Quality Problem

Duplicate data is poison. When your model sees the same example multiple times, it memorizes rather than generalizes. Worse, duplicates skew your loss metrics. You think you are improving, but you are just overfitting to repeated patterns.

```python
import hashlib

class Deduplicator:
    def __init__(self):
        self.seen_hashes = set()
        self.duplicates_found = 0
    
    def get_hash(self, text):
        """Create a hash for deduplication."""
        # Normalize before hashing
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, text):
        """Check if we have seen this text before."""
        h = self.get_hash(text)
        if h in self.seen_hashes:
            self.duplicates_found += 1
            return True
        self.seen_hashes.add(h)
        return False
    
    def deduplicate_dataset(self, texts):
        """Filter duplicates from a list of texts."""
        unique = []
        for text in texts:
            if not self.is_duplicate(text):
                unique.append(text)
        print(f"Removed {self.duplicates_found} duplicates")
        print(f"Kept {len(unique)} / {len(texts)} ({100*len(unique)/len(texts):.1f}%)")
        return unique
```

For large datasets, exact matching is not enough. You need fuzzy deduplication to catch near-duplicates. MinHash and locality-sensitive hashing are the standard tools here, but they are beyond our scope today.

### Balancing Your Dataset

Large language models work best when trained on datasets with roughly uniform concept distributions. Long tails of imbalanced concepts cause problems. If 90% of your storytelling data is romance and 10% is science fiction, your model will be great at romance and terrible at sci-fi.

```python
from collections import defaultdict

class DatasetBalancer:
    def __init__(self, max_per_category=1000):
        self.max_per_category = max_per_category
        self.category_counts = defaultdict(int)
    
    def categorize(self, text):
        """Simple keyword-based categorization."""
        # In practice, you would use a classifier here
        text_lower = text.lower()
        if any(word in text_lower for word in ['love', 'heart', 'romance']):
            return 'romance'
        if any(word in text_lower for word in ['space', 'robot', 'future']):
            return 'scifi'
        if any(word in text_lower for word in ['magic', 'dragon', 'wizard']):
            return 'fantasy'
        return 'other'
    
    def should_include(self, text):
        """Decide whether to include based on category balance."""
        category = self.categorize(text)
        if self.category_counts[category] >= self.max_per_category:
            return False
        self.category_counts[category] += 1
        return True
    
    def balance_dataset(self, texts):
        """Filter to create a more balanced dataset."""
        balanced = [t for t in texts if self.should_include(t)]
        print("Category distribution:")
        for cat, count in sorted(self.category_counts.items()):
            print(f"  {cat}: {count}")
        return balanced
```

### The Full Pipeline

Now let us put it all together:

```python
def build_clean_dataset(raw_texts):
    """Complete data quality pipeline."""
    print("Starting with", len(raw_texts), "examples")
    
    # Step 1: Basic quality filtering
    quality_filter = DataQualityFilter()
    filtered = [t for t in raw_texts if quality_filter.filter(t)]
    print("\nAfter quality filtering:")
    quality_filter.report()
    
    # Step 2: Deduplication
    deduper = Deduplicator()
    unique = deduper.deduplicate_dataset(filtered)
    
    # Step 3: Balance categories
    balancer = DatasetBalancer(max_per_category=500)
    balanced = balancer.balance_dataset(unique)
    
    print(f"\nFinal dataset: {len(balanced)} high-quality examples")
    return balanced
```

### Common Mistakes

The biggest mistake is skipping this step entirely. People get excited about model architecture and training loops. Data cleaning feels boring. But data cleaning, filtering, and synthesizing is the most important work at large language model companies.

Second mistake: filtering too aggressively. If you remove everything that looks slightly weird, you might remove the interesting edge cases that teach your model robustness.

Third mistake: not tracking what you filter. Always keep statistics. When your model fails at something, you want to know whether you accidentally filtered out all the training examples for that capability.

### Success Indicators

You know your data quality pipeline is working when:

1. Your training loss decreases smoothly without strange spikes
2. Your model generates coherent text without repetitive loops
3. The model shows balanced capability across different story types
4. Manual inspection of random samples shows consistently high-quality examples

Data quality is not glamorous work. It does not make for exciting papers. But it is where models are won and lost. The teams that win are the ones that take data seriously.

## Practice

Build efficient data loading with memory mapping, implement batching and shuffling strategies, create synthetic data generation for storytelling, and implement data quality filtering and cleaning.

### Setup

You'll need the following files from previous chapters:
- Your tokenizer from Chapter 06
- A text corpus (start with 1-10MB of plain text, such as public domain stories from Project Gutenberg)

Create a new file `data_pipeline.py` for this exercise.

```python
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
```

### Part 1: Memory-Mapped Dataset

Implement the `MemmapDataset` class from the lesson. Then:

1. Write a preprocessing script that tokenizes a text file and saves it as a binary file of `uint16` tokens
2. Verify your memory-mapped loader works by loading sequences from different positions
3. Confirm memory usage stays constant regardless of file size (use a large file to test)

### Part 2: Batching and Shuffling

Complete the `DataLoader` setup from where the lesson left off:

1. Implement proper shuffling with a fixed random seed for reproducibility
2. Add `num_workers` for parallel data loading
3. Handle the last incomplete batch (drop it or pad it, know why you chose what you chose)
4. Write a function that yields infinite batches for training (cycling through epochs)

### Part 3: Synthetic Story Data

Build a generator that creates simple synthetic stories for testing:

```python
def generate_synthetic_story(min_length=100, max_length=500):
    """Generate a simple synthetic story with predictable structure."""
    # Your implementation here
    pass
```

Your generator should produce stories with:
- A character name
- A setting
- A simple action sequence
- An ending

This is for testing your pipeline, not for training a good model.

### Part 4: Data Quality Filtering

Implement filters that remove or flag problematic training examples:

1. **Length filter**: Remove sequences that are too short or too long
2. **Repetition filter**: Detect and remove text with excessive repetition (e.g. "the the the the")
3. **Language filter**: Basic heuristic to detect non-English text (character frequency analysis)
4. **Deduplication**: Exact-match deduplication using hashing

```python
class DataFilter:
    def __init__(self):
        self.seen_hashes = set()
    
    def is_valid(self, text: str) -> bool:
        """Return True if text passes all quality filters."""
        # Your implementation here
        pass
```

### Success Criteria

- [ ] Memory-mapped dataset loads sequences without loading entire file into RAM
- [ ] `DataLoader` produces batches of shape `[batch_size, sequence_length]`
- [ ] Shuffling produces different orderings with different seeds, same ordering with same seed
- [ ] Synthetic generator produces valid, tokenizable stories
- [ ] Quality filters correctly identify and remove at least 3 types of problematic text
- [ ] Full pipeline runs on a 100MB+ file without memory errors

### Common Pitfalls

**Off-by-one errors in sequence slicing**: Your input `x` and target `y` should be offset by exactly one token. Print the first few tokens of both to verify.

**Forgetting to convert dtypes**: NumPy's `uint16` must become PyTorch's `int64` (or `long`) for embedding lookups. The model will crash or silently produce garbage otherwise.

**Shuffling the wrong thing**: Shuffle starting indices, not the underlying data. The memory-mapped file should remain unchanged.

**Workers and memory maps**: Each DataLoader worker may create its own memory map. This is usually fine but can cause issues on some systems. Test with `num_workers=0` first.

**Filtering too aggressively**: Track what percentage of data each filter removes. If a filter removes more than 10-20%, inspect the rejected samples, your filter may be too strict.

## Assessment

By the end of this module, you should be able to:

- **Implement efficient data loading** using memory-mapped files that can handle datasets far larger than available RAM
- **Design batching strategies** that properly shuffle and organize sequences for training, understanding why random access matters for model convergence
- **Generate and evaluate synthetic data** for bootstrapping training or augmenting limited datasets
- **Assess data quality** by identifying issues like duplication, contamination, encoding errors, and domain imbalance that degrade model performance
- **Curate datasets intentionally** by making principled decisions about what to include, exclude, and how to weight different sources

Success means you can take a raw corpus of text, whether it's a few megabytes of stories or terabytes of web crawl, and transform it into a training pipeline that feeds your model efficiently without running out of memory. You understand that the dataset is not just an input to training but a fundamental design decision that shapes what your model learns. You can look at a trained model's failures and trace them back to data problems. You can make the tradeoff between dataset size and quality, knowing when more data helps and when cleaner data matters more. When someone hands you a new text source, you know exactly how to evaluate it, preprocess it, and integrate it into your pipeline.

---

# Optimization

## Lesson 1: Initialization and Optimization

You have a neural network. You have data. You have a loss function. Now what?

Now you need to make the thing actually learn. And this is where most people hit a wall they don't even see coming. They wire everything up correctly, call `loss.backward()`, update their weights, and... nothing. The loss doesn't move. Or it explodes. Or it drops a tiny bit and then flatlines forever.

The problem isn't your architecture. The problem is that you've ignored two things that seem boring but are actually load-bearing: how you initialize your weights, and how you update them.

Let's fix that.

### The Initial Simplicity

Here's what happens when you initialize weights badly. Let's say you set all your weights to zero. Every neuron in a layer computes the same thing. Every gradient is the same. Every update is the same. Your network has a thousand neurons but they're all doing identical work. You've built a very expensive single neuron.

Okay, so random initialization. You sample from a normal distribution, mean zero, standard deviation one. Now your neurons are different. But watch what happens in a deep network. Each layer multiplies inputs by weights and sums them up. If your weights are too big, activations grow exponentially as you go deeper. By layer 10, you're dealing with numbers so large they overflow to infinity. If your weights are too small, activations shrink exponentially. By layer 10, everything is zero.

This is the Practical Complexity hiding behind the Initial Simplicity. The math of a single layer is trivial. The math of many layers composed together creates dynamics that will destroy your training run before it starts.

### Xavier and Kaiming: The Fix

The insight is simple: we need to keep the variance of activations roughly constant across layers. If the variance stays stable, signals can flow forward without exploding or vanishing, and gradients can flow backward without the same problems.

Xavier initialization (also called Glorot) does this for networks with tanh or sigmoid activations. For a layer with `fan_in` input connections and `fan_out` output connections, you initialize weights from:

```python
std = sqrt(2.0 / (fan_in + fan_out))
W = np.random.randn(fan_in, fan_out) * std
```

Kaiming initialization (also called He) does this for ReLU networks. ReLU kills half your activations (the negative ones), so you need to compensate:

```python
std = sqrt(2.0 / fan_in)
W = np.random.randn(fan_in, fan_out) * std
```

That's it. Two lines of code that make the difference between a network that trains and one that doesn't. The initialization sensitivity discussion from the source materials directly applies here: poor initialization can prevent training entirely in deeper architectures.

### Stochastic Gradient Descent

Now your network can propagate signals. Time to update weights.

The simplest approach is Stochastic Gradient Descent. Compute the gradient of your loss with respect to each weight. Move each weight a little bit in the direction that reduces the loss:

```python
for param in model.parameters():
    param.data -= learning_rate * param.grad
```

This works. It's also slow and noisy. The "stochastic" part means you're computing gradients on random mini-batches, not the full dataset. Each batch gives you a slightly different gradient. Your updates zigzag around instead of heading straight toward the minimum.

### Momentum: The Hockey-Stick Maker

Momentum fixes the zigzag problem by adding memory. Instead of just using the current gradient, you keep a running average of past gradients:

```python
velocity = momentum * velocity + learning_rate * param.grad
param.data -= velocity
```

Think of it like pushing a heavy ball down a loss landscape. The ball accumulates speed in consistent directions and dampens out the back-and-forth oscillations. This is often what creates that hockey-stick shaped loss curve where nothing seems to happen for a while and then suddenly the model takes off.

The momentum coefficient is typically 0.9. This means 90% of your update comes from the accumulated history, 10% from the current gradient.

### Adam: Adaptive Learning Rates

Different parameters need different learning rates. A weight connecting to a frequently-activated neuron gets gradients all the time. A weight connecting to a rarely-activated neuron gets gradients occasionally. If you use the same learning rate for both, you're either updating the rare one too slowly or the frequent one too aggressively.

Adam tracks two things for each parameter: the mean of recent gradients (like momentum) and the mean of recent squared gradients. The squared gradients tell you how much the gradient varies. Parameters with high-variance gradients get smaller effective learning rates. Parameters with low-variance gradients get larger ones.

```python
m = beta1 * m + (1 - beta1) * grad          # momentum
v = beta2 * v + (1 - beta2) * grad**2       # variance
m_hat = m / (1 - beta1**t)                  # bias correction
v_hat = v / (1 - beta2**t)                  # bias correction
param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

The bias correction terms (`1 - beta**t`) fix a cold-start problem. At the beginning, `m` and `v` are initialized to zero, so they're biased toward zero. The correction compensates for this.

Standard values: `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.

### AdamW: The Weight Decay Fix

Here's a subtle bug in the original Adam paper. They implemented weight decay (a regularization technique that shrinks weights toward zero) by adding it to the gradient before the Adam update. This seems natural but it's wrong. The adaptive scaling that makes Adam work also scales the weight decay, which isn't what you want.

AdamW fixes this by applying weight decay directly to the weights, after the Adam update:

```python
param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
param -= learning_rate * weight_decay * param  # separate step
```

This is what you should use. AdamW is the standard optimizer for transformer training. The weight decay coefficient is typically 0.01 to 0.1.

### Where People Get Stuck

The most common failure mode: learning rate too high. Your loss oscillates wildly or explodes to NaN. Start with 1e-4 or 1e-3 for Adam and adjust from there.

Second most common: forgetting to zero gradients. PyTorch accumulates gradients by default. If you don't call `optimizer.zero_grad()` before each backward pass, you're adding new gradients to old ones.

Third: not checking gradient magnitudes. If your gradients are consistently tiny (1e-10) or huge (1e10), something is wrong with your initialization or architecture. The loss landscape navigation techniques from the source materials apply here: monitoring gradient magnitudes during optimization is critical when training from scratch to diagnose issues like vanishing gradients.

### Success Indicators

You're on the right track when:
- Loss decreases smoothly (some noise is fine, wild oscillations are not)
- Gradient magnitudes stay in a reasonable range (roughly 1e-3 to 1e0)
- Activations don't saturate (check that your neurons aren't all outputting the same value)

Remember: backpropagation is a leaky abstraction. The framework handles the mechanics, but you need to understand the internal operations to debug when things go wrong. Building models from scratch requires this understanding, not just stacking layers and hoping.

Next lesson, we'll implement learning rate schedules: warmup and cosine annealing. These take your training from "works" to "works well."

## Lesson 2: Weight Initialization

Let's start with a broken neural network and watch it fail.

```python
import torch
import torch.nn.functional as F

# Simple 3-layer network for character prediction
n_embd = 64
n_hidden = 128
vocab_size = 27  # lowercase letters + space

# Bad initialization: weights too large
W1 = torch.randn(n_embd, n_hidden) * 5.0
b1 = torch.randn(n_hidden) * 5.0
W2 = torch.randn(n_hidden, n_hidden) * 5.0
b2 = torch.randn(n_hidden) * 5.0
W3 = torch.randn(n_hidden, vocab_size) * 5.0
b3 = torch.randn(vocab_size) * 5.0

# Forward pass with random input
x = torch.randn(32, n_embd)  # batch of 32

h1 = torch.tanh(x @ W1 + b1)
h2 = torch.tanh(h1 @ W2 + b2)
logits = h2 @ W3 + b3

print(f"Logits range: {logits.min().item():.1f} to {logits.max().item():.1f}")
print(f"h1 saturation (|value| > 0.99): {(h1.abs() > 0.99).float().mean().item():.1%}")
print(f"h2 saturation (|value| > 0.99): {(h2.abs() > 0.99).float().mean().item():.1%}")
```

Run this. You'll see logits ranging from something like -50 to +50, and saturation rates near 100%. This network is broken before training even starts.

### Why This Matters

At initialization, a neural network's output logits should be close to zero to produce a roughly uniform probability distribution. When you have 27 possible characters, uniform probability means each gets about 1/27 chance. The expected loss at initialization for a classification problem with N classes is approximately -log(1/N). For 27 classes, that's -log(1/27) which is about 3.3.

But with logits ranging from -50 to +50? The softmax will assign probability ~1.0 to whichever class has the highest logit and ~0.0 to everything else. If the correct answer isn't that class, your loss explodes. You're starting with extreme confidence in wrong answers.

Here's the deeper problem. Look at those saturation numbers. When tanh activation outputs are near plus or minus 1, in the flat regions, gradients flowing backward through those units get multiplied by (1 - t squared) which is approximately 0. The gradients vanish. The weights can't update. Learning stops.

### The Dead Neuron Problem

A dead neuron occurs when all training examples activate it in the flat region of its nonlinearity. For tanh, that means outputs near plus or minus 1. For ReLU, that means outputs less than or equal to 0.

With tanh, neurons can recover if some examples push them back into the active region. With ReLU, you can get permanent brain damage. If a high learning rate or large gradient knocks a ReLU neuron into a state where no input ever activates it, that neuron is dead forever. It contributes nothing. It learns nothing.

Leaky ReLU avoids this because it has no completely flat region. Gradients always flow, even for negative inputs. But proper initialization helps with any activation function.

### The Solution: Scale Your Weights

Proper weight initialization aims to keep activations and gradients roughly Gaussian with mean 0 and standard deviation 1 throughout the network. No expansion. No shrinkage. Signal flows cleanly forward and backward.

Here's the math. For a linear layer with Gaussian input and weights, dividing weights by the square root of fan_in preserves output standard deviation at 1, assuming no nonlinearity.

Why? If you have n inputs each with standard deviation 1, and weights with standard deviation 1, the output variance is n times 1 times 1 equals n. Standard deviation is square root of n. To get output standard deviation back to 1, divide weights by square root of n.

```python
# Xavier initialization (for linear layers or tanh)
fan_in = n_embd
W1 = torch.randn(n_embd, n_hidden) / (fan_in ** 0.5)
```

### Kaiming Initialization

Kaiming initialization adjusts the weight scale by a gain factor specific to the nonlinearity to compensate for the contractive effect of the activation.

tanh squashes values. On average, it reduces the standard deviation. ReLU zeros out half the values. These effects compound through layers. Kaiming fixes this with a gain factor: square root of 2 for ReLU, 5/3 for tanh.

```python
# Kaiming initialization for tanh
gain = 5/3  # approximately 1.67
W1 = torch.randn(n_embd, n_hidden) * (gain / (fan_in ** 0.5))

# Kaiming initialization for ReLU
gain = (2 ** 0.5)  # approximately 1.41
W1 = torch.randn(n_embd, n_hidden) * (gain / (fan_in ** 0.5))
```

PyTorch provides these directly:

```python
torch.nn.init.kaiming_normal_(W1, nonlinearity='tanh')
torch.nn.init.kaiming_normal_(W1, nonlinearity='relu')
```

### Fixed Network

```python
# Proper initialization
fan_in = n_embd
gain = 5/3  # for tanh

W1 = torch.randn(n_embd, n_hidden) * (gain / (fan_in ** 0.5))
b1 = torch.zeros(n_hidden)  # biases to zero is safe
W2 = torch.randn(n_hidden, n_hidden) * (gain / (n_hidden ** 0.5))
b2 = torch.zeros(n_hidden)
W3 = torch.randn(n_hidden, vocab_size) * (0.01)  # small for output layer
b3 = torch.zeros(vocab_size)

x = torch.randn(32, n_embd)
h1 = torch.tanh(x @ W1 + b1)
h2 = torch.tanh(h1 @ W2 + b2)
logits = h2 @ W3 + b3

print(f"Logits range: {logits.min().item():.2f} to {logits.max().item():.2f}")
print(f"h1 saturation: {(h1.abs() > 0.99).float().mean().item():.1%}")
print(f"h2 saturation: {(h2.abs() > 0.99).float().mean().item():.1%}")
```

Now logits should range roughly from -0.5 to +0.5. Saturation should be low, maybe 5% or less. The network is ready to learn.

Notice the output layer gets special treatment. We initialize it very small, around 0.01, so initial logits are near zero. This gives us that uniform distribution we want at the start.

### Common Mistakes

**Initializing weights to zero.** Initializing biases to zero is generally safe, but initializing weights exactly to zero breaks symmetry. All neurons in a layer compute the same thing. They all get the same gradient. They all update identically. You effectively have one neuron repeated many times.

**Ignoring the hockey stick.** A hockey-stick loss curve, where you see a steep initial drop, often indicates the network is first correcting poorly initialized logits before learning meaningful patterns. You're spending cycles squashing down weights instead of learning. And the reason for that is because we're spending more time optimizing the neural net to fix initialization problems instead of learning actual patterns from data.

**Assuming depth doesn't matter.** Deeper networks are less forgiving of poor initialization. Problems that cause minor issues in shallow nets can prevent training entirely in deep architectures. Each layer compounds the problem.

### Diagnostic Tools

Inspecting activation histograms and saturation rates is a practical way to diagnose initialization problems:

```python
@torch.no_grad()
def check_activations(model, x):
    """Check activation statistics through the network."""
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight'):
            # Forward hook to capture activations
            print(f"{name}: mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
```

Using torch.no_grad() disables gradient tracking for efficiency when performing inference or evaluation, as no backward pass is needed.

### Modern Context

Modern innovations like residual connections and normalization layers, including batch norm and layer norm, have reduced the sensitivity of training to exact initialization schemes. These architecture modifications help gradients flow and keep activations stable.

But initialization still matters. It determines where you start. Good initialization means faster convergence and more stable training. Bad initialization means wasted compute, or worse, a network that never learns at all.

The simple baseline: Kaiming initialization for hidden layers, small random values for output layers, zeros for biases. Start there. Check your activation statistics. Adjust if needed.

Next lesson, we build the optimizers that actually update these weights.

## Lesson 3: SGD with Momentum

Let's start by watching vanilla SGD fail. Open your notebook and run this:

```python
import torch
import torch.nn as nn

# A simple loss landscape with a narrow ravine
def ravine_loss(w):
    return 0.5 * w[0]**2 + 50 * w[1]**2

# Vanilla SGD
w = torch.tensor([10.0, 1.0], requires_grad=True)
lr = 0.02
history = [w.detach().clone()]

for step in range(100):
    loss = ravine_loss(w)
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()
    history.append(w.detach().clone())

print(f"Final position: {w.data}")
print(f"Final loss: {ravine_loss(w).item():.6f}")
```

Watch what happens. The optimization oscillates wildly in one direction while crawling slowly in another. This is the fundamental problem vanilla SGD faces on real loss landscapes, and it gets dramatically worse when you scale up to billions of parameters where the loss surface becomes increasingly complex.

### Why Vanilla SGD Struggles

The issue is that vanilla SGD treats every step independently. It has no memory. Each gradient update is computed fresh, and if the gradient points in a slightly different direction than the previous step, SGD just follows it blindly.

Real neural network loss landscapes have regions where the gradient changes rapidly in some directions and slowly in others. Think of a long, narrow valley. The gradient points steeply down the valley walls but only gently along the valley floor toward the minimum. Vanilla SGD bounces back and forth between the walls while making minimal progress toward the actual goal.

This is where momentum enters the picture.

### The Momentum Intuition

Imagine rolling a ball down a hilly landscape. The ball doesn't instantly change direction when the slope changes. It has inertia. Previous motion influences current motion.

Momentum adds exactly this property to gradient descent. Instead of updating weights directly with the gradient, we maintain a velocity vector that accumulates gradients over time:

```python
v = beta * v + gradient
w = w - lr * v
```

The `beta` parameter (typically 0.9) controls how much history we remember. With beta=0.9, the velocity is 90% of what it was before plus 10% of the new gradient direction.

Let's implement this and watch the difference:

```python
# SGD with momentum
w = torch.tensor([10.0, 1.0], requires_grad=True)
v = torch.zeros_like(w)
lr = 0.02
beta = 0.9
history_momentum = [w.detach().clone()]

for step in range(100):
    loss = ravine_loss(w)
    loss.backward()
    with torch.no_grad():
        v = beta * v + w.grad
        w -= lr * v
        w.grad.zero_()
    history_momentum.append(w.detach().clone())

print(f"Final position: {w.data}")
print(f"Final loss: {ravine_loss(w).item():.6f}")
```

Run both versions. The momentum version converges faster and oscillates less. The velocity accumulates consistent gradient directions while canceling out oscillations.

### Building a Proper SGD with Momentum Class

Now let's build something we can actually use for training:

```python
class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity buffers for each parameter
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        with torch.no_grad():
            for p, v in zip(self.params, self.velocities):
                if p.grad is None:
                    continue
                # Update velocity
                v.mul_(self.momentum).add_(p.grad)
                # Update parameter
                p.sub_(self.lr * v)
```

Notice the velocity buffers. Each parameter in your model gets its own velocity tensor that persists across training steps. This is the "memory" that vanilla SGD lacks.

### Testing on a Real Network

Let's verify this works on an actual neural network:

```python
# Simple MLP for testing
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.GELU(),
    nn.Linear(256, 10)
)

# Create fake data
x = torch.randn(64, 784)
y = torch.randint(0, 10, (64,))

# Our optimizer
optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(10):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

The loss should decrease smoothly. Compare this to vanilla SGD (set momentum=0) and you'll typically see more erratic behavior.

### The Mathematics

For those who want the formal picture, momentum SGD maintains:

```
v_t = β * v_{t-1} + g_t
θ_t = θ_{t-1} - α * v_t
```

Where `g_t` is the gradient at step t, `β` is the momentum coefficient, and `α` is the learning rate.

The effective step size in any direction depends on gradient consistency. If gradients point the same way for many steps, velocity builds up and you move faster. If gradients oscillate, the velocity stays small because positive and negative contributions cancel.

This is why momentum helps with the ravine problem. Along the valley floor, gradients consistently point toward the minimum, so velocity accumulates. Across the valley walls, gradients flip sign each step, so they cancel out in the velocity.

### Common Mistakes

**Mistake 1: Forgetting to initialize velocities as zeros.** If you initialize with random values, your first few updates will be nonsense.

**Mistake 2: Using momentum that's too high.** With beta=0.99, the optimizer has so much inertia it overshoots minima and takes forever to settle. Start with 0.9.

**Mistake 3: Not adjusting learning rate.** Momentum effectively increases your step size because velocity accumulates. If you switch from vanilla SGD to momentum SGD, you often need to reduce the learning rate.

**Mistake 4: Confusing momentum with learning rate.** They do different things. Learning rate scales the step size. Momentum controls how much history influences the current step.

### Why This Matters for LLMs

When training large language models with billions of parameters, the loss landscape becomes extraordinarily complex. The optimization problem involves navigating high-dimensional spaces where gradients can be noisy due to stochastic sampling of training data.

Momentum provides Empirical Stability by smoothing out this noise. Instead of reacting to every random fluctuation in the gradient, the optimizer maintains a consistent direction based on accumulated evidence.

However, momentum alone isn't enough for modern LLM training. The adaptive learning rate methods we'll cover next, specifically Adam and AdamW, build on momentum but add per-parameter learning rate adaptation. This turns out to be crucial when different parameters need different update scales.

### Success Indicators

You know momentum is working when:

1. Loss decreases more smoothly than vanilla SGD
2. Training is stable at higher learning rates than vanilla SGD could handle
3. Convergence is faster, especially in the early stages of training

You know something is wrong when:

1. Loss oscillates wildly (momentum too low or learning rate too high)
2. Loss decreases extremely slowly (momentum too high, optimizer can't change direction)
3. Loss explodes (learning rate way too high for the effective step size with momentum)

### What Comes Next

Momentum solves the oscillation problem, but it treats all parameters equally. In the next lesson, we'll see how Adam adds adaptive learning rates per parameter, and why AdamW's decoupled weight decay matters for training stability. These optimizers are what actually get used for LLM training, where the scale of parameters makes adaptive methods essential.

So if I just erase these and I now take away the break statement, we can run the optimization with this new momentum implementation and see what losses we record. The improvement over vanilla SGD should be obvious.

## Lesson 4: Adam and AdamW

So you've built SGD with momentum. You've seen how momentum accumulates velocity and smooths out the gradient noise. But if you actually try to train a transformer with SGD, even with momentum, you're going to have a rough time. The learning rate that works for one layer destroys another. Parameters that should update quickly get stuck. Parameters that should update slowly explode.

This is where Adam comes in. And then AdamW, which fixes something subtle but important that Adam gets wrong.

Let's build both.

### The Problem Adam Solves

Here's what happens when you train a neural network with SGD. Some parameters receive gradients that are consistently large. Others receive gradients that are tiny and sparse. Think about an embedding layer: the embedding for "the" gets updated on almost every batch, while the embedding for "xylophone" might go thousands of batches without a single gradient signal.

With SGD, you pick one learning rate. If it's big enough for "xylophone" to learn anything, it's way too big for "the" and you get instability. If it's small enough for "the" to stay stable, "xylophone" never learns.

What you want is adaptive learning rates. Each parameter gets its own effective learning rate based on the history of gradients it's received.

### Building Adam from Scratch

Adam combines two ideas: momentum (which we already built) and something called RMSprop, which tracks the magnitude of recent gradients.

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # timestep
        
        # First moment (momentum)
        self.m = [torch.zeros_like(p) for p in self.params]
        # Second moment (squared gradients)
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

Let me walk through what's happening here. We maintain two running averages for each parameter:

`m` is the first moment, basically momentum. It's an exponential moving average of the gradients themselves. With beta1=0.9, we're keeping 90% of the old momentum and adding 10% of the new gradient.

`v` is the second moment. It's an exponential moving average of the squared gradients. This tracks the magnitude of recent gradients. With beta2=0.999, this changes very slowly, giving us a stable estimate of gradient scale.

The update divides by `sqrt(v_hat)`. This is the key insight: parameters that have been receiving large gradients get divided by a large number, effectively reducing their learning rate. Parameters with small gradients get divided by a small number, boosting their effective learning rate.

### Why Bias Correction Matters

Notice those lines with `m_hat` and `v_hat`? That's bias correction, and it matters a lot in early training.

Here's the issue. We initialize `m` and `v` to zeros. On the first step, `m[i] = 0.1 * g` because we kept 90% of zero. That's way smaller than the actual gradient. The second moment has the same problem but worse because beta2 is 0.999.

The bias correction compensates. On step 1, we divide by `(1 - 0.9^1) = 0.1`, which scales our first moment back up to the right magnitude. As training progresses and t gets large, `beta1^t` approaches zero and the correction disappears.

Without bias correction, early training is sluggish. The model just sits there for hundreds of steps while the moment estimates warm up. With bias correction, you get good updates from step one.

### The Weight Decay Problem

Now here's where things get interesting. Regularization. You want to prevent overfitting by penalizing large weights. The standard approach is L2 regularization: add `lambda * sum(w^2)` to your loss.

When you take the gradient of this regularized loss, you get `grad + lambda * w`. So the gradient has an extra term that pushes weights toward zero.

With SGD, this works fine. The update becomes `w -= lr * (grad + lambda * w)`, which is equivalent to `w -= lr * grad; w *= (1 - lr * lambda)`. The weight shrinks proportionally.

But with Adam, something goes wrong. That extra `lambda * w` term gets fed into the moment estimates. The second moment `v` now includes the regularization term, which means the adaptive learning rate is computed based on regularization strength, not just gradient magnitude. The whole adaptive scaling gets corrupted.

### AdamW: Decoupled Weight Decay

AdamW fixes this by decoupling weight decay from the gradient. Instead of adding regularization to the loss, we apply weight decay directly to the parameters after the Adam update:

```python
class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Update moments with ONLY the gradient (no weight decay here)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Adam update
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            
            # Weight decay applied SEPARATELY
            p.data -= self.lr * self.weight_decay * p.data
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

The difference is subtle but critical. Weight decay happens after the Adam update, directly on the parameters. The moment estimates only see the true gradients from the loss function. The adaptive learning rate works as intended.

This is what the field uses now. As the source materials note, optimizers like AdamW adjust model parameters using gradients to minimize loss, with learning rates tuned based on model size and task. When you see a paper training a transformer, they're almost certainly using AdamW.

### Common Mistakes

**Using Adam instead of AdamW for transformers.** The difference matters. AdamW with weight_decay=0.01 or 0.1 is standard.

**Wrong epsilon.** The default 1e-8 works for float32. If you're training in float16 or bfloat16 (which we'll cover in the precision chapter), you might need 1e-6 or larger to avoid numerical issues.

**Forgetting bias correction.** If you implement this yourself and skip bias correction, your early training will be mysteriously slow.

### Success Indicators

You know your AdamW is working when:

1. Loss drops immediately from step one, not after a warmup period
2. Training is stable across a wide range of learning rates (Adam is more forgiving than SGD)
3. Different layers learn at appropriate rates without manual tuning
4. Weight norms stay bounded (weight decay is working)

Next lesson we'll add learning rate scheduling, because even with AdamW, a constant learning rate isn't optimal. You want to warm up, then decay. The combination of AdamW with cosine annealing and warmup is the Implementation Strategy that actually trains modern transformers.

## Lesson 5: Learning Rate Schedulers

You have Adam. You have AdamW. You have momentum. Your optimizer is ready to go. And yet, if you set a constant learning rate and hit train, you will almost certainly get worse results than you could have. The reason is simple: the optimal learning rate changes as training progresses.

Let me show you what I mean. Here is a training run with a constant learning rate:

```python
import math

def train_constant_lr(model, data, lr=1e-3, steps=10000):
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []
    
    for step in range(steps):
        loss = compute_loss(model, data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    
    return losses
```

This works. But watch what happens when we add a scheduler:

```python
def train_with_scheduler(model, data, max_lr=1e-3, steps=10000, warmup_steps=1000):
    optimizer = AdamW(model.parameters(), lr=max_lr)
    losses = []
    
    for step in range(steps):
        # Compute learning rate for this step
        lr = get_lr(step, max_lr, steps, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        loss = compute_loss(model, data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    
    return losses
```

The difference in final loss can be substantial. So what is `get_lr` doing?

### Why Learning Rates Need to Change

Think about what happens during training. At the start, your parameters are randomly initialized. The gradients are large, the loss surface is unexplored, and you need to make big moves to get anywhere useful. But if your learning rate is too high at the start, you can blow up entirely. The gradients explode, the loss goes to infinity, and your training run is dead.

So you need warmup. Start with a tiny learning rate and gradually increase it. This lets the optimizer find a reasonable region of the loss surface before taking aggressive steps.

But then later in training, you have the opposite problem. You have found a good region. The gradients are smaller. If you keep the same learning rate, you will overshoot the minimum repeatedly, bouncing around instead of settling in. You need to decay the learning rate to make smaller, more precise adjustments.

This is the basic intuition: warmup at the start, decay at the end.

### Implementing Warmup

Warmup is straightforward. Linear warmup is most common:

```python
def linear_warmup(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    return max_lr
```

At step 0, the learning rate is 0. At step `warmup_steps`, it reaches `max_lr`. The ramp is linear. Some people use other warmup schedules but linear works fine.

How many warmup steps? A common heuristic is 1-5% of total training steps. For a 100k step run, 1000-5000 warmup steps. If you see instability in the first few hundred steps, increase warmup. If training seems sluggish at the start, you might decrease it.

### Implementing Cosine Annealing

The most popular decay schedule for transformers is cosine annealing. The learning rate follows a cosine curve from `max_lr` down to some minimum value:

```python
def cosine_decay(step, total_steps, max_lr, min_lr=0):
    if step >= total_steps:
        return min_lr
    
    progress = step / total_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay
```

At step 0, you get `max_lr`. At step `total_steps`, you get `min_lr`. The decay is smooth, spending more time at higher learning rates early on and tapering off gently at the end.

Why cosine? Empirically it works well. The gradual decay at the end helps the model settle into a good minimum. Linear decay works too but cosine seems to give slightly better results in practice.

### Putting It Together

Here is the complete scheduler combining warmup and cosine decay:

```python
def get_lr(step, max_lr, total_steps, warmup_steps, min_lr=1e-5):
    # Warmup phase
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    
    # Decay phase
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    
    progress = decay_step / decay_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return min_lr + (max_lr - min_lr) * cosine_decay
```

The `min_lr` parameter matters. Setting it to 0 means the learning rate goes to zero at the end, which can cause training to stall. A small positive value like 1e-5 or 1e-6 keeps things moving.

### Using PyTorch Schedulers

PyTorch provides built-in schedulers. Here is how you would use them:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

# Warmup scheduler
warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=1e-8/max_lr,  # Start near zero
    end_factor=1.0,            # End at max_lr
    total_iters=warmup_steps
)

# Cosine decay scheduler
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=min_lr
)

# Combine them
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

# Training loop
for step in range(total_steps):
    loss = compute_loss(model, data)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate
    optimizer.zero_grad()
```

The `SequentialLR` runs the warmup scheduler for `warmup_steps`, then switches to cosine decay.

### Common Mistakes

**Forgetting to call scheduler.step().** Your learning rate will never change. Training will work but suboptimally.

**Calling scheduler.step() at the wrong time.** Call it after `optimizer.step()`, once per training step. Some older code called it once per epoch, which is wrong for most modern schedules.

**Too short warmup with high learning rate.** If you see loss spikes or NaN in the first few hundred steps, increase warmup or decrease max learning rate.

**Decaying too fast.** If your learning rate hits minimum too early, you spend most of training at a tiny learning rate. Make sure `total_steps` matches your actual training length.

**Not logging the learning rate.** Always log it. When debugging training issues, the first thing to check is whether the learning rate is what you expect.

### The Bigger Picture

Here is something worth noting from the research: transformers benefit more from adaptive optimizers like Adam, which automatically adjust learning rates per parameter, reducing the need for carefully tuned learning rate annealing schedules. This means the exact shape of your schedule matters less than it did for older architectures like RNNs and CNNs.

That said, warmup plus cosine decay is the standard recipe. It works. You will see it in basically every transformer training codebase. The hyperparameters to tune are:

1. `max_lr`: The peak learning rate after warmup
2. `warmup_steps`: How long to ramp up
3. `min_lr`: The floor for the learning rate

Start with the defaults from papers training similar models. For GPT-style models, max learning rates around 1e-4 to 6e-4 are common, with warmup around 1-2% of training.

### Success Indicators

You know your scheduler is working when:

- Loss decreases smoothly without spikes in early training
- The learning rate logged matches what you expect at each step
- Final loss is better than constant learning rate baseline
- Training remains stable through the decay phase

You know something is wrong when:

- Loss explodes in the first few steps (warmup too short or max_lr too high)
- Loss plateaus early and never improves (learning rate decayed too fast)
- Training is unstable throughout (check your implementation)

Learning rate scheduling is one of those things that seems like a detail but actually matters quite a bit. Get it right and training just works better. The good news is that the standard recipe of warmup plus cosine decay is robust and well understood. Use it.

## Practice

**Exercise:** Implement Xavier/Kaiming initialization, build SGD with momentum, create Adam and AdamW optimizers, and implement learning rate schedulers.

### Setup

Create a file called `optimization.py`. You'll build everything from scratch using only NumPy, no PyTorch, no autograd frameworks. You need to understand what these algorithms actually do, not just how to call them.

Start with a simple test network: a 3-layer MLP with dimensions 784 → 256 → 128 → 10. You'll use this to verify each component works correctly.

### Part 1: Initialization

Implement two functions:

```python
def xavier_init(fan_in, fan_out):
    # Return weight matrix with Xavier/Glorot initialization
    pass

def kaiming_init(fan_in, fan_out):
    # Return weight matrix with Kaiming/He initialization
    pass
```

**Success criteria:** Initialize a 10-layer network (256 units each) and pass random input through it. With Xavier init using tanh activations, the variance of activations at layer 10 should be within 0.5-2.0x the input variance. With Kaiming init using ReLU, same test.

### Part 2: SGD with Momentum

Implement an optimizer class:

```python
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        pass
    
    def step(self, grads):
        # Update params in-place
        pass
```

The momentum update rule: `v = momentum * v + grad`, then `param -= lr * v`.

**Success criteria:** On a simple quadratic loss `f(x) = x²`, SGD without momentum should converge. With momentum=0.9, it should converge faster (fewer steps to reach loss < 0.0001).

### Part 3: Adam and AdamW

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        pass
    
    def step(self, grads):
        pass

class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        pass
    
    def step(self, grads):
        pass
```

Remember: Adam applies weight decay to the gradient (L2 regularization). AdamW applies it directly to the weights (decoupled weight decay). This distinction matters.

**Success criteria:** Train your 3-layer MLP on a toy classification problem (generate random clusters). Adam should converge smoothly. AdamW with weight_decay=0.01 should produce weights with smaller L2 norm than Adam.

### Part 4: Learning Rate Schedulers

```python
class LRScheduler:
    def __init__(self, optimizer):
        pass
    
    def step(self):
        # Update optimizer's learning rate
        pass

class StepLR(LRScheduler):
    # Decay LR by gamma every step_size epochs
    pass

class CosineAnnealingLR(LRScheduler):
    # Cosine decay from initial LR to min_lr over T_max steps
    pass

class WarmupCosineScheduler(LRScheduler):
    # Linear warmup for warmup_steps, then cosine decay
    pass
```

**Success criteria:** Plot the learning rate over 1000 steps for each scheduler. StepLR should show discrete drops. Cosine should show smooth decay. Warmup+Cosine should show linear increase then smooth decay.

### Common Pitfalls

1. **Forgetting bias correction in Adam.** The first few updates will be wrong without `m_hat = m / (1 - beta1^t)` and `v_hat = v / (1 - beta2^t)`.

2. **Applying weight decay to biases.** Convention is to only decay weights, not biases. Your AdamW should handle this.

3. **Off-by-one errors in schedulers.** Be clear about whether you update LR before or after the optimizer step, and whether step count starts at 0 or 1.

4. **Using the wrong fan for Kaiming.** For a weight matrix of shape `(fan_in, fan_out)`, Kaiming uses `fan_in` only. Xavier uses both.

5. **Momentum velocity initialization.** Initialize velocity buffers to zero, not random values.

### Verification

Compare your implementations against PyTorch. For the same random seed and initial weights, your optimizer should produce identical parameter updates (within floating point tolerance) to `torch.optim.SGD`, `torch.optim.Adam`, and `torch.optim.AdamW`.

## Assessment

By the end of this module, you should be able to:

- **Explain why weight initialization matters** and implement both Xavier and Kaiming initialization schemes, knowing when to use each based on your activation functions
- **Implement SGD from scratch** and articulate why vanilla gradient descent fails in practice for deep networks
- **Build momentum into your optimizer** and explain how it helps escape local minima and accelerates convergence through gradient accumulation
- **Implement Adam and AdamW** from first principles, understanding the role of first and second moment estimates, bias correction, and why decoupled weight decay matters for generalization
- **Design learning rate schedules** including warmup, cosine annealing, and step decay, and reason about when each is appropriate
- **Implement gradient clipping** and explain why exploding gradients occur and how clipping prevents training instability

Success in this module means you can take a neural network that refuses to train, loss stuck, gradients exploding, learning plateaued, and diagnose whether the problem is initialization, optimizer choice, learning rate schedule, or gradient dynamics. You should be able to implement Adam from memory, explain to someone else why the bias correction terms exist, and know instinctively that a NaN loss after a few hundred steps probably means you need gradient clipping or a lower learning rate. When you see a training curve that drops fast then flatlines, you should think "learning rate too high, then too low" before you think "bad architecture." The optimizer is where theory meets practice, and mastering it means your networks actually learn.

---

# Need for Speed I,  Device

## Lesson 1: Hardware Understanding

You have been writing Python code that runs on your CPU. That is about to change.

Before we write a single line of CUDA, before we benchmark anything, we need to understand why hardware matters so much for deep learning. This is not abstract knowledge. This is the difference between training a model in three days versus three months.

### Why Hardware Matters

Here is a fact that surprises most people: the best algorithms must be designed to utilize current GPU and hardware capabilities effectively. You cannot separate "the algorithm" from "the hardware it runs on." They are married. A beautiful algorithm that ignores hardware realities will lose to an ugly algorithm that respects them.

Training large language models has become more of a systems engineering problem than an NLP research problem. The scaling law demonstrates that adding more compute yields guaranteed performance improvement. This means if you can make your code run faster on the hardware you have, you get better models. Period. No clever tricks required. Just speed.

This is why language modeling is now more of an engineering discipline than an architectural innovation challenge. The architecture is mostly settled. The question is: how fast can you push data through it?

### The CPU: What You Know

Your CPU is a general-purpose processor. It can do almost anything: run your web browser, compile code, play music, train a neural network. This flexibility comes at a cost.

A modern CPU has maybe 8 to 16 cores. Each core is incredibly sophisticated. It can predict which branch your code will take. It can reorder instructions to hide memory latency. It can execute multiple instructions per clock cycle. Each core is a genius.

But you only have a handful of them.

When you write `for i in range(1000000):` in Python, one core handles that loop. The other cores sit idle. You are paying for 16 geniuses and using one.

### The GPU: What You Need to Learn

A GPU is the opposite philosophy. Instead of a few genius cores, you get thousands of simple cores. An NVIDIA GPU might have 10,000 or more CUDA cores. Each one is dumb compared to a CPU core. No branch prediction. No out-of-order execution. Simple arithmetic, that is all.

But there are thousands of them.

Matrix multiplication is the core operation of deep learning. When you multiply a 1024x1024 matrix by another 1024x1024 matrix, you are doing about a billion floating point operations. On a CPU, one core grinds through this sequentially. On a GPU, thousands of cores attack different parts of the problem simultaneously.

This is parallelism. This is why GPUs dominate deep learning.

### Memory: The Hidden Bottleneck

Here is where most people get confused. They think: "GPU has more compute, so GPU is faster." This is only half the story.

Compute means nothing if you cannot feed it data fast enough.

Your CPU accesses RAM. Your GPU has its own memory called High Bandwidth Memory (or VRAM on consumer cards). The GPU memory is much faster than CPU RAM. But there is a catch: data must travel from CPU memory to GPU memory before the GPU can use it.

This transfer is slow. Painfully slow compared to computation.

Imagine a factory with 10,000 workers but only one narrow door. Workers stand idle waiting for materials to arrive. This is what happens when your code is memory bound rather than compute bound.

Understanding whether your code is compute bound or memory bound is essential. If you are compute bound, you need a faster GPU. If you are memory bound, a faster GPU will not help much. You need to restructure your algorithm to move less data.

### The Numbers That Matter

Let me give you concrete numbers so this becomes real.

A modern GPU can perform around 300 trillion floating point operations per second (for lower precision formats). That is 300 TFLOPS.

GPU memory bandwidth might be 2 terabytes per second.

Do the math: 300 trillion operations divided by 2 trillion bytes means you need to do about 150 operations per byte transferred to keep the GPU fully utilized.

Matrix multiplication of large matrices achieves this. You load each number once and use it many times. This is why matrix multiplication is so GPU-friendly.

Element-wise operations do not achieve this. If you just add two vectors, you load two numbers, do one operation, and store one number. Three memory accesses for one operation. The GPU cores sit idle waiting for memory.

This is why we fuse operations in neural networks. Instead of doing matmul, then add bias, then apply activation as three separate steps (three memory round-trips), we do them all in one kernel. Load once, compute everything, store once.

### Hardware Precision

Modern GPUs can operate at different precisions. Full precision uses 32 bits per number. Half precision uses 16 bits. Some operations can use 8 bits or even 4 bits.

Lower precision means:
- Faster computation (more operations per second)
- Less memory used (fit bigger models)
- Less memory bandwidth needed (faster data movement)

The tradeoff is accuracy. But neural networks are surprisingly tolerant of low precision during training, and even more tolerant during inference. This is why Hardware Precision optimization is so important for Efficiency Gains.

We will explore this deeply in the next module on precision.

### Where Learners Get Stuck

The most common mistake is thinking about hardware too late. People write their model in Python, get it working, then ask "how do I make this fast?" By then, the algorithm may be structured in a way that fights the hardware.

Think about hardware from the start. Ask: "Will this operation be compute bound or memory bound? Can I fuse these steps? Am I moving data unnecessarily?"

Another mistake is Avoiding Python entirely. Python is slow, yes. But Python is not your bottleneck when training neural networks. The bottleneck is the GPU operations themselves. Python just schedules work. The actual computation happens in Heavily Optimized CUDA kernels written in C++.

Development Speed matters. Python lets you iterate quickly. Write it in Python first, profile it, then optimize the hot spots. Do not prematurely optimize.

### Success Indicators

You understand this lesson if you can answer:

1. Why does a GPU with 10,000 simple cores beat a CPU with 16 complex cores for matrix multiplication?
2. What does it mean for an operation to be memory bound versus compute bound?
3. Why does fusing operations help GPU performance?
4. If you double your GPU's compute but memory bandwidth stays the same, will your training speed double?

That last question is a trick. The answer depends on whether you are currently compute bound or memory bound. If memory bound, doubling compute helps nothing. This is the key insight.

### What Comes Next

In the next lesson, we will build actual benchmarks. We will measure CPU versus GPU performance on real operations. We will see the numbers ourselves, not just read about them.

Then we will write our first CUDA kernel. You will see how the parallelism actually works at the code level.

Hardware understanding is not optional for serious deep learning work. The collaboration with systems researchers that makes the best algorithms possible starts with understanding what the hardware can and cannot do efficiently.

Let us go measure some things.

## Lesson 2: CPU vs GPU Benchmarks

Let's actually measure the difference. You've heard that GPUs are faster for neural networks. You've probably seen benchmarks showing 10x, 50x, even 100x speedups. But until you run the numbers yourself, it's just hearsay. So we're going to build benchmarks from scratch, understand what we're measuring, and see exactly where the speedup comes from.

### Why Benchmarks Matter

Here's the thing about Hardware Limitations: they're not abstract. They're concrete walls you'll hit repeatedly when training models. The difference between a training run that takes 3 hours versus 3 days comes down to understanding where your bottlenecks are. And you can't optimize what you can't measure.

The Overall Dynamics of neural network computation involve two main resources: compute (how many operations per second) and Memory Bandwidth (how fast you can move data around). CPUs and GPUs make radically different tradeoffs between these. A CPU has maybe 8-16 powerful cores optimized for complex sequential logic. A GPU has thousands of simpler cores optimized for doing the same operation on many pieces of data simultaneously.

### Setting Up Your First Benchmark

Let's start with matrix multiplication. This is the core operation of deep learning. Every layer in your network, every attention computation, every embedding lookup ultimately reduces to multiplying matrices together.

```python
import time
import numpy as np

def benchmark_matmul_cpu(size, iterations=10):
    """Benchmark matrix multiplication on CPU using NumPy."""
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # Warmup
    _ = np.matmul(a, b)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = np.matmul(a, b)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times)
```

Run this for different matrix sizes. Start small: 128, 256, 512, 1024, 2048, 4096. Record the times. You'll notice something interesting: the time doesn't scale linearly with size. A 2048x2048 matrix has 4x the elements of a 1024x1024 matrix, but the multiplication takes roughly 8x longer. That's because matrix multiplication is O(n³), not O(n²).

Now let's add GPU benchmarking. We'll use PyTorch since it handles the CUDA complexity for us initially:

```python
import torch

def benchmark_matmul_gpu(size, iterations=10):
    """Benchmark matrix multiplication on GPU."""
    device = torch.device('cuda')
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # Warmup and sync
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times)
```

Notice the `torch.cuda.synchronize()` calls. This is critical. GPU operations are asynchronous. When you call `torch.matmul`, Python returns immediately while the GPU works in the background. Without synchronization, you'd measure the time to launch the operation, not the time to complete it. This is a common mistake that leads to wildly optimistic GPU benchmarks.

### What You'll Actually See

Run both benchmarks across sizes. Here's what typically happens:

For small matrices (128x128, 256x256), the CPU might actually win. The overhead of transferring data to the GPU and launching kernels dominates the actual computation time. This surprises people.

Around 512x512 to 1024x1024, you'll see crossover. The GPU starts pulling ahead.

At 2048x2048 and beyond, the GPU absolutely dominates. You might see 20x, 50x, even 100x speedups depending on your specific hardware.

This pattern reveals something fundamental about Hardware Utilization. GPUs are Heavily Optimized for throughput, not latency. They need enough work to hide their startup costs. Small problems don't give them enough parallelism to exploit.

### Measuring What Matters: FLOPS and Bandwidth

Raw time isn't the whole story. We need to understand whether we're hitting Physical Limits or leaving performance on the table.

For matrix multiplication of two NxN matrices, the number of floating point operations is approximately 2N³ (N² elements in the output, each requiring N multiplications and N-1 additions).

```python
def compute_flops(size, time_seconds):
    """Compute FLOPS for matrix multiplication."""
    operations = 2 * (size ** 3)
    return operations / time_seconds

def compute_bandwidth_usage(size, time_seconds):
    """Estimate memory bandwidth for matmul."""
    # Read two NxN matrices, write one NxN matrix
    bytes_moved = 3 * (size ** 2) * 4  # float32 = 4 bytes
    return bytes_moved / time_seconds
```

Compare your measured FLOPS against your hardware's theoretical peak. A modern CPU might hit 500 GFLOPS to 1 TFLOPS. A modern GPU might hit 10-30 TFLOPS for float32. If you're getting 10% of theoretical peak, something's wrong. If you're getting 50-70%, you're doing well. NumPy with a good BLAS library and PyTorch's CUDA kernels are both Heavily Optimized and should get you close to peak for large matrices.

### The Memory Bandwidth Bottleneck

Here's where it gets interesting. Matrix multiplication is "compute bound" for large matrices. The ratio of computation to memory access is high enough that the GPU cores stay busy. But many neural network operations aren't like this.

Consider element-wise operations like adding two vectors or applying an activation function:

```python
def benchmark_elementwise_cpu(size, iterations=100):
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = a + b
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times)
```

For element-wise addition, you read two numbers, add them, write one number. That's 3 memory operations for 1 compute operation. You're "memory bound." The GPU's massive compute power sits idle waiting for data.

This is why High Bandwidth Memory matters so much for modern GPUs. Memory Bandwidth often determines real-world performance more than raw FLOPS.

### Common Mistakes

**Avoiding Python overhead**: When benchmarking, make sure you're measuring the operation, not Python's interpreter. Use large enough problems that the actual computation dominates.

**Forgetting synchronization**: Always synchronize before timing GPU operations. Always.

**Measuring cold performance**: Run warmup iterations. The first run often includes JIT compilation, memory allocation, and other one-time costs.

**Ignoring data transfer**: If you're moving data between CPU and GPU for each operation, include that time. In real training loops, data transfer can dominate.

### Success Indicators

You know you've understood this material when:

1. You can predict roughly which operations will benefit from GPU acceleration
2. You understand why small matrices don't see speedups
3. You can compute theoretical FLOPS and compare against measured performance
4. You recognize the difference between compute-bound and memory-bound operations
5. You know when to use synchronization in GPU benchmarks

### What's Next

These benchmarks used Ready Library implementations. NumPy calls into optimized BLAS. PyTorch calls into cuBLAS. In the next lesson, we'll write our own CUDA kernels and see exactly what's happening at the hardware level. You'll understand why these libraries are Heavily Optimized and what techniques they use to achieve near-peak performance.

The goal isn't to beat these libraries. The goal is to understand them deeply enough that you can reason about performance, predict bottlenecks, and make informed decisions about Hardware Utilization when training your models.

## Lesson 3: CUDA Kernel Basics

You have benchmarked CPU versus GPU. You have seen the numbers. Now we write the code that actually runs on those thousands of GPU cores.

This is where things get real.

### Your First CUDA Kernel

Let us start by writing something that works, then understand why it works.

```cuda
__global__ void add_vectors(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

Type this out. Do not copy paste. The act of typing builds understanding.

What you just wrote is a kernel. The `__global__` keyword tells the compiler this function runs on the GPU but gets called from the CPU. Every thread that executes this kernel gets its own unique `idx` value. If you launch 1024 threads, you get 1024 simultaneous additions.

This is the fundamental shift. On CPU, you write a loop that processes elements one at a time (or maybe 8 at a time with SIMD). On GPU, you write the code for ONE element, then launch thousands of threads that each handle their own element.

### The Threading Model

Here is where most people get confused. Let me break it down.

CUDA organizes threads into blocks, and blocks into a grid. Think of it like this:

```
Grid
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └──. Thread 255
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   └──. Thread 255
└── Block 2
    └──.
```

Each thread knows its position through three built-in variables:
- `threadIdx.x` - which thread am I within my block?
- `blockIdx.x` - which block am I in?
- `blockDim.x` - how many threads per block?

The formula `blockIdx.x * blockDim.x + threadIdx.x` gives each thread a globally unique index. If you have 256 threads per block, thread 0 in block 1 gets index 256. Thread 0 in block 2 gets index 512. And so on.

Why this two-level hierarchy? Hardware Limitations. A single block can have at most 1024 threads, but you might need millions. Blocks let you scale beyond that limit while keeping threads that need to cooperate close together.

### Launching Kernels

From your CPU code (the host), you launch kernels like this:

```cuda
int n = 1000000;
int threads_per_block = 256;
int num_blocks = (n + threads_per_block - 1) / threads_per_block;

add_vectors<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

That `<<<num_blocks, threads_per_block>>>` syntax is CUDA's kernel launch configuration. The division formula with the `+ threads_per_block - 1` is a ceiling division trick. You want enough blocks to cover all n elements, rounding up.

Notice the `if (idx < n)` check in the kernel. This is critical. If n is 1000000 and you launch 3907 blocks of 256 threads, you get 1000192 threads. The last 192 threads have indices beyond your array. Without that bounds check, you corrupt memory.

### Memory: The Real Bottleneck

Here is something that trips up everyone who starts GPU programming: compute is rarely your problem. Memory is.

GPUs have different memory spaces:
- Global memory: large (gigabytes), slow to access
- Shared memory: small (kilobytes per block), fast
- Registers: tiny, fastest

When you write `c[idx] = a[idx] + b[idx]`, each of those array accesses goes to global memory. The addition itself takes maybe 1 cycle. The memory accesses take hundreds of cycles.

This is why Memory Bandwidth matters more than raw FLOPS for most operations. Understanding whether your kernel is compute bound or memory bound determines how you optimize it.

A simple vector addition is heavily memory bound. You do one addition per three memory accesses. No amount of clever math will make it faster. You need to load the data regardless.

Matrix multiplication is different. You load data once and reuse it many times. This is why matmul is the core operation of deep learning. It has high arithmetic intensity, meaning lots of compute per byte loaded.

### Memory Transfer Patterns

Data must travel from CPU to GPU before processing, then back after. This transfer goes over PCIe, which is slow compared to GPU memory bandwidth.

```cuda
// Allocate on GPU
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, n * sizeof(float));
cudaMalloc(&d_b, n * sizeof(float));
cudaMalloc(&d_c, n * sizeof(float));

// Copy input data CPU -> GPU
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel
add_vectors<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

// Copy result GPU -> CPU
cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

The `d_` prefix is convention for device (GPU) pointers. The `h_` prefix is for host (CPU) pointers. Keep them straight or you will get crashes that make no sense.

For small operations, the transfer time dominates. This is why you do not want to bounce data back and forth between CPU and GPU for every little operation. Keep data on GPU, chain operations together, only transfer when necessary.

### Why This Matters for LLMs

Remember from the source materials: tensor operations allow all attention heads to be computed simultaneously in a single batched matrix multiplication, leveraging GPU Parallelism. The reason multi-headed attention uses tensor gymnastics instead of processing head by head is exactly what we are learning here.

Fused Kernel Operations reduce overhead by combining multiple operations into single kernels. Avoiding Python interpreter overhead keeps computation in compiled CUDA code. Heavily Optimized BLAS libraries like cuBLAS exploit everything we are discussing here.

When you later write matrix multiplication kernels, you will see how shared memory, memory coalescing, and tiling transform a naive O(n³) algorithm into something that actually uses the hardware well.

### Common Mistakes

1. Forgetting bounds checks. Your kernel launches more threads than array elements. Check `if (idx < n)`.

2. Mixing host and device pointers. Passing a CPU pointer to a kernel crashes. Passing a GPU pointer to printf crashes. Keep them straight.

3. Forgetting to synchronize. Kernel launches are asynchronous. The CPU continues while the GPU works. If you need results, call `cudaDeviceSynchronize()` or use `cudaMemcpy` which synchronizes implicitly.

4. Using too few threads. GPUs need thousands of threads to hide memory latency. Launching 64 threads wastes the hardware.

### Success Indicators

You understand this lesson when you can:
- Write a kernel that processes array elements in parallel
- Calculate the correct grid and block dimensions for any array size
- Explain why `blockIdx.x * blockDim.x + threadIdx.x` gives unique indices
- Predict whether a kernel is memory bound or compute bound
- Move data between CPU and GPU without crashes

Next lesson, we build matrix multiplication on GPU. That is where you will see why all this infrastructure exists. Matrix multiplication is not just an operation. It is THE operation that makes deep learning work at scale.

# Lesson 4: Matrix Multiplication on GPU

You have written your first CUDA kernel. You understand thread blocks and grids. Now we arrive at the operation that matters most: matrix multiplication.

Here is the fact that changes everything: **matrix multiplication is the fundamental mathematical primitive being scaled up** in language modeling, autograd, and neural network training. The core of transformer-based LLMs is attention mechanisms and feed-forward layers, both dominated by large-scale matrix multiplications across billions of parameters. When you optimize matmul, you optimize everything.

Let us build it.

## The Naive Implementation First

Open a new file. Type this out:

```c
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

This computes C = A × B where A is M×K and B is K×N. Each thread computes one element of the output matrix. The thread at position (row, col) walks along row `row` of A and column `col` of B, multiplying and accumulating.

Run this on 1024×1024 matrices. Measure the time. Now we will make it faster.

## Why This Is Slow

Every thread reads K elements from A and K elements from B. For a 1024×1024 matrix with K=1024, that is 2048 memory reads per thread. You have 1024×1024 = 1,048,576 threads. That is over 2 billion memory reads from global memory.

Global memory on a GPU has high latency. The bandwidth is impressive, maybe 900 GB/s on a modern card, but the latency is 400-600 cycles. Your threads spend most of their time waiting for data.

This is what we call **memory bound**. The compute units sit idle while memory transfers crawl along. Memory Bandwidth becomes the bottleneck, not arithmetic throughput.

## Shared Memory Tiling

The fix: load data into shared memory once, reuse it many times. Shared memory lives on-chip, with latency around 20-30 cycles instead of 400-600.

Here is the tiled version:

```c
#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

What changed? Instead of each thread loading its own data from global memory, the entire thread block cooperates. Each thread loads one element of A and one element of B into shared memory. Then all threads in the block can access all 16×16 = 256 elements of each tile.

The `__syncthreads()` calls are critical. The first one ensures all loads complete before any thread starts computing. The second ensures all computation finishes before any thread overwrites the shared memory with the next tile.

## Measuring the Speedup

Run both versions. On typical hardware you will see 5-10x speedup from the tiled version. This is the difference between memory bound and approaching compute bound.

```c
// Launch configuration for tiled version
dim3 blockDim(TILE_SIZE, TILE_SIZE);
dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
```

The grid has enough blocks to cover the output matrix. Each block is 16×16 = 256 threads, which is a reasonable occupancy for most GPUs.

## Memory Transfer Patterns

Here is where people get stuck. They optimize the kernel but forget about getting data to the GPU in the first place.

```c
// Allocate on device
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, M * K * sizeof(float));
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMalloc(&d_C, M * N * sizeof(float));

// Copy to device
cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

// Run kernel
matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

// Copy result back
cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
```

The transfers between host and device go over PCIe, which is much slower than High Bandwidth Memory on the GPU itself. For a single matmul, this overhead might dominate. For neural network training where you do thousands of matmuls between transfers, it amortizes away.

This is why Batch Size matters. Larger batches mean more compute per memory transfer. You want the GPU doing arithmetic, not waiting for data.

## The Connection to Neural Networks

Remember: training via gradient descent involves repeatedly computing forward passes (matrix multiplications) and backward passes (gradient computations via matrix calculus). Every linear layer is a matmul. Every attention head involves multiple matmuls. A transformer with 70 billion parameters is doing an astronomical number of these operations.

The Efficiency Gains from optimizing matmul compound across every layer, every batch, every training step. A 2x speedup in matmul can mean weeks saved on a large training run. This directly affects Energy Costs and Carbon Footprint of training.

## Common Mistakes

**Forgetting `__syncthreads()`**: Your kernel will produce wrong results intermittently. Some threads will read stale data. This bug is maddening because it does not always manifest.

**Tile size too large**: Shared memory is limited, typically 48KB per block. A 32×32 tile of floats uses 4KB. Two tiles use 8KB. You have headroom, but not infinite.

**Tile size too small**: You get less data reuse. The sweet spot depends on your GPU architecture. 16×16 is a safe starting point.

**Not checking bounds**: When M, N, or K are not multiples of TILE_SIZE, threads at the edges will read garbage. The boundary checks are not optional.

## Success Indicators

You are ready to move on when:

1. Your tiled kernel produces the same output as the naive kernel (within floating point tolerance)
2. You see at least 5x speedup on 1024×1024 matrices
3. You can explain why shared memory helps and what `__syncthreads()` does
4. You understand the difference between memory bound and compute bound

## What Comes Next

This tiled implementation is educational but not production quality. Real libraries like cuBLAS use techniques we have not covered: register tiling, double buffering, warp-level primitives, and Fused Kernel Operations that combine multiple operations to avoid intermediate memory writes.

The principle remains the same: minimize memory movement, maximize arithmetic intensity. Hardware Utilization is the goal. Every cycle the GPU spends waiting for memory is a cycle wasted.

In the next lesson we will see how Hardware Precision affects both speed and Numerical Stability. The same matmul in FP16 runs roughly twice as fast as FP32. But the tradeoffs are subtle.

## Lesson 5: Memory Transfer Patterns

You have written CUDA kernels. You have parallelized matrix multiplication across thousands of threads. And yet, if you have been benchmarking carefully, you may have noticed something frustrating: sometimes your GPU code is barely faster than CPU code. Sometimes it is even slower.

This is not a bug in your code. This is physics.

The dirty secret of GPU programming is that the GPU itself is almost never the bottleneck. The bottleneck is getting data to the GPU and getting results back. Memory transfer patterns determine whether your code flies or crawls. Let us understand why, and then let us fix it.

### The Problem You Can Feel

Try this experiment. Write a simple kernel that adds two vectors:

```python
import torch
import time

# Create data on CPU
a_cpu = torch.randn(100_000_000)
b_cpu = torch.randn(100_000_000)

# Time the full operation including transfer
start = time.time()
a_gpu = a_cpu.cuda()
b_gpu = b_cpu.cuda()
c_gpu = a_gpu + b_gpu
c_cpu = c_gpu.cpu()
torch.cuda.synchronize()
end = time.time()
print(f"With transfers: {end - start:.4f}s")

# Now time just the computation
a_gpu = a_cpu.cuda()
b_gpu = b_cpu.cuda()
torch.cuda.synchronize()

start = time.time()
c_gpu = a_gpu + b_gpu
torch.cuda.synchronize()
end = time.time()
print(f"Computation only: {end - start:.4f}s")
```

Run this. The computation-only time will be perhaps 100x faster than the full operation. All that extra time is memory transfer. You are paying a massive tax just to move bytes around.

### Why This Happens

Your CPU and GPU do not share memory. They are connected by a PCIe bus, which is a relatively narrow pipe compared to the memory bandwidth inside either device.

Think of it this way. Inside your GPU, memory bandwidth might be 900 GB/s. The PCIe 4.0 x16 connection between CPU and GPU is about 32 GB/s. That is a 28x difference. Every byte you transfer across that bus is a byte that could have been processed 28 times over if it was already on the GPU.

This is why the expert's perspective emphasizes that understanding LLMs requires knowing how these systems actually process information. The neural network parameters, those billions of weights, are not like database entries you query. They are numerical values that must physically exist in GPU memory to be useful. Moving them is expensive.

### The Patterns That Matter

**Pattern 1: Transfer Once, Compute Many**

The most important pattern is obvious once you see it: move data to the GPU once, then do as much computation as possible before moving results back.

```python
# Bad: transfer per operation
for i in range(100):
    x_gpu = x_cpu.cuda()
    x_gpu = x_gpu * 2
    x_cpu = x_gpu.cpu()

# Good: transfer once
x_gpu = x_cpu.cuda()
for i in range(100):
    x_gpu = x_gpu * 2
x_cpu = x_gpu.cpu()
```

This seems trivial. But in real code, especially when you are debugging or prototyping, it is easy to accidentally transfer data repeatedly without realizing it.

**Pattern 2: Pinned Memory**

Normal CPU memory can be swapped to disk by the operating system. Before the GPU can read from it, the CUDA driver must first copy it to a special "pinned" region that cannot be swapped. You can eliminate this extra copy by allocating pinned memory directly:

```python
# Regular allocation (requires internal copy)
x = torch.randn(10_000_000)
x_gpu = x.cuda()

# Pinned allocation (direct transfer)
x_pinned = torch.randn(10_000_000, pin_memory=True)
x_gpu = x_pinned.cuda()
```

Pinned memory transfers can be 2x faster. The tradeoff is that pinned memory is a limited resource and cannot be swapped, so do not pin everything.

**Pattern 3: Asynchronous Transfers**

The CPU does not need to wait while data transfers to the GPU. You can overlap transfer with computation:

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Transfer batch 1 while computing batch 0
with torch.cuda.stream(stream1):
    batch1_gpu = batch1_cpu.cuda(non_blocking=True)

with torch.cuda.stream(stream2):
    result0 = model(batch0_gpu)
```

This is how production training pipelines achieve high GPU utilization. While the GPU computes on one batch, the next batch is already transferring.

**Pattern 4: Fused Kernel Operations**

Every kernel launch has overhead. Every intermediate result that gets written to memory and then read back has transfer cost. Fused kernels combine multiple operations into one, eliminating intermediate memory traffic.

This is why efficient attention implementations like FlashAttention matter so much. Standard attention computes Q, K, V separately, writes attention scores to memory, reads them back for softmax, writes softmax output, reads it back for the final matmul. Fused attention does all of this in one kernel, keeping intermediate values in fast on-chip memory.

The efficiency gains from fused kernel operations can be 2-4x for attention alone. This is not about faster math. It is about avoiding memory transfer.

### Compute Bound vs Memory Bound

Your code is either waiting for computation to finish or waiting for memory to arrive. Understanding which one determines your optimization strategy.

Matrix multiplication is compute bound when matrices are large. The O(n³) computation dominates the O(n²) memory transfer. This is good. Your GPU stays busy.

Element-wise operations like ReLU are memory bound. Reading and writing the tensor takes longer than the actual computation. No matter how fast your GPU is, you are waiting for memory.

To check which regime you are in, measure achieved FLOPS versus theoretical peak. If you are at 10% of peak, you are probably memory bound. If you are at 70%+, you are compute bound.

### Common Mistakes

**Mistake 1: Calling.cpu() inside training loops.** Every time you pull a tensor back to CPU for logging or debugging, you force synchronization and transfer. Log scalars only, and do it infrequently.

**Mistake 2: Creating tensors on CPU then moving them.** Use `torch.randn(size, device='cuda')` to create directly on GPU.

**Mistake 3: Ignoring memory layout.** Tensors that are not contiguous in memory require extra copies. Call `.contiguous()` before operations that need it, but understand you are paying for a copy.

**Mistake 4: Transferring full precision when you need half.** Convert to float16 before transfer when possible. Half the bytes means half the transfer time.

### Success Indicators

You know you have good memory transfer patterns when:

1. GPU utilization (check with `nvidia-smi`) stays above 80% during training
2. Your training throughput scales linearly as you increase batch size (until you hit compute limits)
3. Adding more computation (deeper model, more layers) increases time proportionally, not exponentially
4. Your profiler shows overlapped data transfer and computation

### The Bigger Picture

This lesson connects directly to why scale matters so much in LLM development. The expert's core insight is that scale of data, parameters, and compute drives capability, with qualitative leaps emerging from quantitative increases. But scale only works if you can actually feed the GPU fast enough.

Hardware limitations around memory bandwidth are why techniques like context parallel exist for handling 100k+ token contexts. The computational and memory load must be distributed across multiple GPUs because no single device can hold everything and transfer it fast enough.

When you train your own models in later chapters, every efficiency gain from proper memory transfer patterns translates directly into either faster training or the ability to train larger models. The difference between amateur and professional deep learning code is often not algorithmic cleverness. It is understanding where the bytes flow.

## Practice

Build CPU vs GPU benchmarks, implement CUDA kernel basics, create matrix multiplication on GPU, and understand memory transfer patterns.

### Setup

You need a machine with an NVIDIA GPU and CUDA installed. Verify your setup:

```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA compiler version
```

Install required packages:

```bash
pip install torch numpy
```

Create a file called `device_benchmark.py` for your work.

### Part 1: CPU vs GPU Benchmarks

Write functions that time matrix multiplication on both devices:

```python
import torch
import time

def benchmark_matmul(size, device, num_runs=100):
    # Create random matrices on the specified device
    # Time the multiplication (excluding data transfer)
    # Return average time in milliseconds
    pass
```

Run benchmarks for sizes: 128, 256, 512, 1024, 2048, 4096. Plot the results. Calculate the speedup ratio at each size.

### Part 2: CUDA Kernel Basics

Write a simple CUDA kernel using PyTorch's custom extension or raw CUDA. Start with vector addition:

```python
# Using torch.cuda if available, or write raw CUDA in a.cu file
def vector_add_gpu(a, b):
    # Your implementation
    pass
```

Compare your kernel's performance against `torch.add()`.

### Part 3: Matrix Multiplication on GPU

Implement naive matrix multiplication in CUDA:

```python
def naive_matmul_gpu(A, B):
    # Each thread computes one element of the output
    # C[i,j] = sum(A[i,k] * B[k,j] for all k)
    pass
```

Then implement tiled matrix multiplication that uses shared memory. Compare both against `torch.matmul()`.

### Part 4: Memory Transfer Patterns

Measure the cost of moving data between CPU and GPU:

```python
def measure_transfer_time(size):
    cpu_tensor = torch.randn(size, size)
    
    # Time: CPU -> GPU
    # Time: GPU -> CPU
    # Time: computation on GPU
    
    # Return all three times
    pass
```

Find the crossover point: at what matrix size does GPU computation become faster than CPU, including transfer overhead?

### Success Criteria

1. Your benchmark shows GPU speedup of 10x or more for 4096x4096 matrices
2. Your naive CUDA kernel produces correct results (verify against PyTorch)
3. Your tiled kernel is faster than your naive kernel
4. You can identify the matrix size where GPU becomes worthwhile
5. You can explain why small matrices are sometimes faster on CPU

### Common Pitfalls

**Forgetting to synchronize.** GPU operations are asynchronous. Call `torch.cuda.synchronize()` before stopping your timer, or your measurements will be wrong.

**Including transfer time unfairly.** When benchmarking computation, data should already be on the device. When benchmarking end-to-end, include transfers.

**Warmup runs.** The first GPU operation is slow due to CUDA context initialization. Run a few warmup iterations before timing.

**Memory not freed.** Call `torch.cuda.empty_cache()` between experiments if you run out of GPU memory.

**Comparing apples to oranges.** Use the same data types (float32) on both devices. GPU float16 vs CPU float32 is not a fair comparison.

### Deliverables

1. `device_benchmark.py` with all implementations
2. A plot showing CPU vs GPU times across matrix sizes
3. A short writeup (comments in code are fine) explaining:
   - At what size does GPU become faster?
   - What is the maximum speedup you observed?
   - Why does tiled matrix multiplication help?

## Assessment

By the end of this module, you should be able to:

- Explain the fundamental architectural differences between CPUs and GPUs, including why CPUs have few sophisticated cores while GPUs have thousands of simple ones
- Describe how CUDA enables parallel computation and articulate the programming model that makes GPU acceleration possible
- Distinguish between compute-bound and memory-bound operations, and identify which category a given operation falls into
- Explain what memory bandwidth is, why it often matters more than raw compute power, and how it creates bottlenecks in deep learning workloads
- Analyze a simple kernel or operation and reason about whether it will be limited by compute throughput or memory transfer speed
- Articulate why hardware considerations are inseparable from algorithm design in modern deep learning

Success in this module means you can look at a piece of deep learning code and immediately start thinking about where the time actually goes. When someone says "my training is slow," you should be able to ask the right questions: Is it compute-bound or memory-bound? How much data is moving between CPU and GPU? Are we saturating the memory bandwidth or leaving compute idle? You should understand that a GPU is not just "a faster processor" but a fundamentally different machine that demands a different way of thinking. The goal is not to memorize specifications but to build intuition, so that when we start writing CUDA kernels in the next lessons, you understand *why* certain patterns are fast and others are catastrophically slow.

---

# Need for Speed II,  Precision

## Lesson 1: Mixed Precision Training

You have a 70-billion-parameter model. Each parameter stored as a float32 takes 4 bytes. That is 280 gigabytes just for the weights. Your GPU has 80 gigabytes of memory. The math does not work.

But here is the thing. What if you stored those same parameters as float16? Now you are at 140 gigabytes. Still too big, but you are getting somewhere. And the computations themselves run faster on modern hardware when you use smaller numbers.

This is mixed precision training. You use different numerical formats for different parts of the training process. Some things need full precision. Some things do not. Knowing which is which, that is what this lesson is about.

### The Problem You Are Solving

Let me give you a concrete example. Working with a specific example of the Llama 2 70B model, this is a large language model released by Meta AI, and this is basically the Llama series of language models, the second iteration of it, and this is the 70-billion-parameter model of this series. Those 70 billion parameters need to live somewhere during training.

But training is not just storing weights. You also need:
- Gradients (same size as weights)
- Optimizer states (Adam needs two more copies)
- Activations (depends on batch size and sequence length)

With float32 everywhere, you are looking at over a terabyte of memory for a model this size. Model inference is cheap and local while training is extremely expensive. This is part of why.

### The Formats You Have

Before we write any code, you need to understand what you are working with.

**Float32 (single-precision):**
- 32-bit representation with 1 sign bit, 8 exponent bits, 23 mantissa bits
- Range from about 1.2×10⁻³⁸ to 3.4×10³⁸
- Precision of roughly 7 decimal digits
- 4 bytes per parameter
- This is your safe default. Good numerical stability.

**Float16 (half-precision):**
- 16-bit representation with 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range from about 6.1×10⁻⁵ to 6.5×10⁴
- Precision of roughly 3-4 decimal digits
- 2 bytes per parameter
- Faster computation on modern GPUs with tensor cores
- Risk: prone to underflow and overflow, numerical instability

**BFloat16 (Brain Float):**
- 16-bit representation with 1 sign bit, 8 exponent bits, 7 mantissa bits
- Same exponent range as float32
- Lower precision than float16, roughly 2-3 decimal digits
- 2 bytes per parameter
- Better stability than float16 for deep learning because it preserves dynamic range

The constraint of storing parameters as float16 numbers, 2 bytes each, creates optimization trade-offs between model size, inference speed, and precision. You get half the memory. You get faster math. But you lose precision and you lose range.

### The Key Insight

Here is what makes mixed precision work: not everything in your training loop needs the same precision.

Your forward pass? Those matrix multiplications can happen in float16. The hardware is optimized for it. The slight precision loss does not matter much for intermediate activations.

Your master weights? Those need to stay in float32. Small gradient updates accumulate over millions of steps. If you store weights in float16, those tiny updates get rounded to zero. Your model stops learning.

Your loss calculation? Float32. You need the precision.

Your gradients? This is where it gets interesting.

### Loss Scaling: The Trick That Makes It Work

Gradients in deep networks can be tiny. Like 10⁻⁸ tiny. Float16 cannot represent numbers that small. Anything below about 6.1×10⁻⁵ becomes zero. Your gradients vanish. Not because of the vanishing gradient problem you learned about, but because of numerical representation limits.

The solution is loss scaling. Before you call backward, you multiply the loss by a large number, say 1024. This scales all your gradients up by the same factor. Now those 10⁻⁸ gradients become 10⁻⁵ gradients, which float16 can represent.

After the backward pass, you divide the gradients by that same scaling factor before updating weights.

```python
scale = 1024.0

# Forward pass in float16
with autocast():
    output = model(input)
    loss = criterion(output, target)

# Scale the loss
scaled_loss = loss * scale

# Backward pass
scaled_loss.backward()

# Unscale gradients before optimizer step
for param in model.parameters():
    if param.grad is not None:
        param.grad.data /= scale

# Update weights (in float32)
optimizer.step()
```

### Dynamic Loss Scaling

That fixed scale of 1024? Sometimes it is too small and you still get underflow. Sometimes it is too big and you get overflow, your gradients become inf or nan.

Dynamic loss scaling starts with a large scale factor and adjusts. If you see infs or nans in your gradients, you skip that update and reduce the scale. If several updates succeed, you increase the scale.

This is what automatic mixed precision does under the hood.

### Where People Get Stuck

**Mistake 1: Storing master weights in float16.** Your optimizer needs float32 weights. The model can cast to float16 for forward and backward, but the canonical weights that get updated must be float32.

**Mistake 2: Forgetting to unscale gradients.** If you scale the loss but forget to unscale gradients, your effective learning rate becomes learning_rate times scale. Your training explodes.

**Mistake 3: Using float16 for everything on older hardware.** Tensor cores on modern GPUs make float16 fast. On older GPUs, float16 can actually be slower because it requires conversion overhead.

**Mistake 4: Not checking for nans.** Mixed precision training produces more nans than float32 training. You need to detect them and handle them, usually by skipping that batch and reducing your loss scale.

### BFloat16 vs Float16

BFloat16 keeps the same exponent range as float32 but sacrifices mantissa bits. This means it can represent very large and very small numbers, just not as precisely.

For deep learning, this trade-off is usually worth it. Gradient magnitudes vary wildly across layers. BFloat16 can represent them all. Float16 cannot.

If your hardware supports BFloat16, use it. You often do not need loss scaling at all because the dynamic range is sufficient.

### Success Indicators

You are doing mixed precision right when:

1. Memory usage drops by roughly 40-50% compared to pure float32
2. Training speed increases by 1.5x to 3x depending on your GPU
3. Loss curves look identical to float32 training
4. No nan or inf values appearing in your gradients or weights
5. Your loss scale stabilizes rather than constantly decreasing

### The Bigger Picture

Parameters or weights are not like entries in a database. They are numerical values in a massive function that collectively implement a statistical model of language patterns. Training is the process of finding these values.

Mixed precision is about being clever with how you represent those numerical values. Full precision where it matters. Half precision where speed matters more than precision. This is not a hack or a shortcut. This is understanding what your numbers actually need to do at each stage of training.

In the next lesson, we will implement each floating point format from scratch. You will see exactly how bits become numbers and why some formats work better than others for neural network training.

## Lesson 2: Floating Point Representations

Let's start with something that will break your intuitions about numbers.

Open a Python shell and type this:

```python
a = 0.1 + 0.2
print(a == 0.3)
```

You get `False`. The actual value is `0.30000000000000004`. This is not a bug. This is floating point arithmetic working exactly as designed.

When you train neural networks with billions of parameters, every single weight, gradient, and activation is stored as a floating point number. Understanding how these numbers actually work on hardware is the difference between a model that trains successfully and one that produces NaN after a thousand steps.

### Why This Matters for Training

Remember from the previous lesson that mixed precision training lets us use lower precision formats to speed up computation. But we glossed over something important: what actually changes when we go from FP32 to FP16 to BF16?

The answer lives in how computers represent real numbers. And the tradeoffs are brutal.

### The Anatomy of a Float

Every floating point number has three parts: a sign bit, an exponent, and a mantissa (also called the significand or fraction).

Think of it like scientific notation. The number 6.022 × 10²³ has:
- A sign (positive)
- A mantissa (6.022)
- An exponent (23)

Floating point works the same way, but in binary and with fixed bit budgets for each part.

**FP32 (32 bits total):**
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits

**FP16 (16 bits total):**
- 1 sign bit
- 5 exponent bits
- 10 mantissa bits

**BF16 / Brain Float (16 bits total):**
- 1 sign bit
- 8 exponent bits
- 7 mantissa bits

**FP8 (8 bits total):**
- 1 sign bit
- 4 or 5 exponent bits (varies by format)
- 2 or 3 mantissa bits

Let's build some intuition for what these bit allocations actually mean.

### Exponent Bits Control Range

The exponent bits determine how large or small your numbers can get. More exponent bits means you can represent a wider range of magnitudes.

FP32 with 8 exponent bits can represent numbers from roughly 10⁻³⁸ to 10³⁸. That is an enormous range.

FP16 with only 5 exponent bits can represent numbers from roughly 10⁻⁵ to 10⁴. Much smaller range.

Here is why this matters for training. When you compute gradients, you often get very small values. A gradient of 10⁻⁶ is perfectly normal. But if your format cannot represent numbers that small, the gradient becomes zero. Your model stops learning.

This is called underflow, and it is one of the reasons loss scaling exists.

### Mantissa Bits Control Precision

The mantissa bits determine how many significant digits you can represent. More mantissa bits means finer distinctions between nearby numbers.

FP32 with 23 mantissa bits gives you about 7 decimal digits of precision.

FP16 with 10 mantissa bits gives you about 3 decimal digits of precision.

BF16 with 7 mantissa bits gives you about 2 decimal digits of precision.

Here is a concrete example. In FP32, you can distinguish between 1.0000001 and 1.0000002. In FP16, both of those round to the same value. In BF16, you lose even more resolution.

### Why BF16 Exists

Brain Float is a fascinating compromise. Google designed it specifically for deep learning, and the reasoning is clever.

Look at the bit allocation again:
- FP16: 5 exponent, 10 mantissa
- BF16: 8 exponent, 7 mantissa

BF16 sacrifices precision to keep the same range as FP32. Why would you want less precision but more range?

Because in neural network training, underflow kills you faster than rounding errors. When gradients become exactly zero due to underflow, learning stops completely. When gradients are slightly imprecise due to rounding, learning continues, just a bit noisily.

BF16 can represent the same tiny and huge numbers as FP32. It just cannot distinguish between values as finely. For gradient descent, this tradeoff works remarkably well.

### Implementing the Formats

Let's write code that shows these representations:

```python
import struct

def float_to_bits(f):
    """Convert float to its bit representation"""
    return format(struct.unpack('>I', struct.pack('>f', f))[0], '032b')

def analyze_float32(f):
    bits = float_to_bits(f)
    sign = bits[0]
    exponent = bits[1:9]
    mantissa = bits[9:]
    print(f"Value: {f}")
    print(f"Sign: {sign}")
    print(f"Exponent: {exponent} ({int(exponent, 2) - 127})")
    print(f"Mantissa: {mantissa}")

analyze_float32(0.15625)
```

Run this. You will see exactly how 0.15625 gets encoded. The exponent is stored with a bias (127 for FP32) so we can represent both positive and negative exponents.

### The Numerical Precision Problem

Here is where things get dangerous. Remember that source material about floating point precision issues? Let me show you the problem directly:

```python
import torch

# Create a small gradient
small_grad = torch.tensor(1e-5, dtype=torch.float32)
weight = torch.tensor(1.0, dtype=torch.float32)

# This works fine in FP32
updated_fp32 = weight - small_grad
print(f"FP32: {weight.item()} - {small_grad.item()} = {updated_fp32.item()}")

# Now try FP16
weight_fp16 = weight.half()
small_grad_fp16 = small_grad.half()
updated_fp16 = weight_fp16 - small_grad_fp16
print(f"FP16: {weight_fp16.item()} - {small_grad_fp16.item()} = {updated_fp16.item()}")
```

In FP16, that small gradient might get rounded to zero, or the subtraction might not change the weight at all due to limited precision. This is why we keep master weights in FP32 during mixed precision training.

### Machine Epsilon and Optimal Step Sizes

Every floating point format has a machine epsilon: the smallest value you can add to 1.0 and get a different result.

```python
import numpy as np
print(f"FP32 epsilon: {np.finfo(np.float32).eps}")
print(f"FP16 epsilon: {np.finfo(np.float16).eps}")
```

FP32 epsilon is about 1.2e-7. FP16 epsilon is about 9.8e-4. That is a huge difference.

When numerical gradient approximation uses very small step sizes, the error in computing differences can become comparable to or larger than the actual difference. The optimal step size is typically around the square root of machine epsilon. For FP32, that is about 1e-4. For FP16, that is about 3e-2.

This is why analytical gradients via backpropagation are preferred in neural network training. Numerical gradients become unreliable at small step sizes due to these precision limits.

### Common Mistakes

**Mistake 1:** Assuming FP16 is just "smaller FP32." The reduced range causes underflow in places you do not expect.

**Mistake 2:** Forgetting that BF16 has less precision than FP16. If you need accurate accumulation, BF16 can introduce more rounding error.

**Mistake 3:** Not checking for NaN and Inf during training. These often indicate overflow or underflow in your chosen precision.

### Success Indicators

You understand floating point representations when you can:
- Explain why 0.1 + 0.2 does not equal 0.3
- Predict which format will underflow first for a given small value
- Justify why BF16 works better than FP16 for many training scenarios
- Calculate the approximate precision of each format in decimal digits

Next lesson, we build the actual mixed precision training loop and implement loss scaling to handle these numerical challenges.

## Lesson 3: Mixed Precision Training Loop

You have the pieces. You understand FP32, FP16, BF16. You know why each format exists and what tradeoffs it makes. Now we put them together into something that actually trains a neural network faster while using less memory.

Let's build a mixed precision training loop from scratch.

### The Core Pattern

Here is the simplest possible mixed precision training loop:

```python
# Master weights in FP32 (this is your source of truth)
model_fp32 = Model().float()

# Copy to FP16 for forward/backward
model_fp16 = copy_to_fp16(model_fp32)

for batch in dataloader:
    # Forward pass in FP16
    inputs = batch.half()  # Convert inputs to FP16
    outputs = model_fp16(inputs)
    loss = compute_loss(outputs, targets.half())
    
    # Backward pass in FP16
    loss.backward()
    
    # Copy gradients to FP32, update master weights
    copy_gradients_to_fp32(model_fp16, model_fp32)
    optimizer.step()  # Updates FP32 master weights
    
    # Copy updated weights back to FP16
    copy_to_fp16(model_fp32, model_fp16)
    optimizer.zero_grad()
```

Run this. It will probably break. The loss will go to NaN after a few hundred steps. This is expected. We need loss scaling.

### Why It Breaks

Remember from the previous lesson that FP16 has a limited dynamic range. Gradients during backpropagation can be extremely small, especially in deep networks. When a gradient value falls below approximately 6.1×10⁻⁵, FP16 cannot represent it. It becomes zero.

Zero gradients mean no learning. Your model stops improving. Or worse, the accumulated numerical errors cause the loss to explode to infinity or collapse to NaN.

This is the Gradient Flow Management problem. Small gradients underflow, large gradients overflow, and your training diverges.

### Loss Scaling: The Fix

The solution is embarrassingly simple. Before the backward pass, multiply the loss by a large number. This scales all gradients up, pushing them into the representable range of FP16. After you copy gradients to FP32, divide by that same number.

```python
loss_scale = 1024.0  # Start with something reasonable

for batch in dataloader:
    inputs = batch.half()
    outputs = model_fp16(inputs)
    loss = compute_loss(outputs, targets.half())
    
    # Scale the loss before backward
    scaled_loss = loss * loss_scale
    scaled_loss.backward()
    
    # Copy gradients and unscale
    copy_gradients_to_fp32(model_fp16, model_fp32)
    for param in model_fp32.parameters():
        if param.grad is not None:
            param.grad.data /= loss_scale
    
    optimizer.step()
    copy_to_fp16(model_fp32, model_fp16)
    optimizer.zero_grad()
```

This works. But what value should loss_scale be? Too small and gradients still underflow. Too large and they overflow.

### Dynamic Loss Scaling

The answer is to adjust the scale automatically. Start with a large scale. If gradients overflow (contain inf or NaN), skip that update and reduce the scale. If training proceeds without overflow for many steps, increase the scale.

```python
loss_scale = 65536.0  # Start high
scale_growth_interval = 2000  # Steps between scale increases
steps_since_overflow = 0

for batch in dataloader:
    inputs = batch.half()
    outputs = model_fp16(inputs)
    loss = compute_loss(outputs, targets.half())
    
    scaled_loss = loss * loss_scale
    scaled_loss.backward()
    
    copy_gradients_to_fp32(model_fp16, model_fp32)
    
    # Check for overflow
    has_overflow = False
    for param in model_fp32.parameters():
        if param.grad is not None:
            param.grad.data /= loss_scale
            if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                has_overflow = True
                break
    
    if has_overflow:
        # Skip this update, reduce scale
        loss_scale /= 2.0
        steps_since_overflow = 0
        optimizer.zero_grad()
        continue
    
    optimizer.step()
    copy_to_fp16(model_fp32, model_fp16)
    optimizer.zero_grad()
    
    steps_since_overflow += 1
    if steps_since_overflow >= scale_growth_interval:
        loss_scale *= 2.0
        steps_since_overflow = 0
```

This is the complete pattern. Dynamic loss scaling with Numerical Precision Safeguards.

### BF16: When You Can Skip Loss Scaling

Brain Float (BF16) has the same exponent range as FP32. This means gradients that would underflow in FP16 can still be represented in BF16. On hardware that supports it (modern NVIDIA GPUs, TPUs), you can often skip loss scaling entirely:

```python
# BF16 training is simpler
model_bf16 = model_fp32.bfloat16()

for batch in dataloader:
    inputs = batch.bfloat16()
    outputs = model_bf16(inputs)
    loss = compute_loss(outputs, targets.bfloat16())
    loss.backward()
    
    copy_gradients_to_fp32(model_bf16, model_fp32)
    optimizer.step()
    copy_to_bf16(model_fp32, model_bf16)
    optimizer.zero_grad()
```

No loss scaling. The tradeoff is lower precision in the mantissa, roughly 2-3 decimal digits versus 3-4 for FP16. For most deep learning workloads, this precision loss does not hurt final model quality.

### Where Things Go Wrong

**Mistake 1: Forgetting to keep master weights in FP32.**

You might think "why not just train entirely in FP16?" The problem is weight updates. When you have a large weight (say, 1000.0) and a small gradient (say, 0.0001), the update 1000.0 + 0.0001 = 1000.0 in FP16 due to limited precision. The gradient vanishes. Your model stops learning.

Master weights in FP32 preserve these small updates.

**Mistake 2: Scaling gradients instead of loss.**

You might try multiplying gradients by the scale factor after backward. This does not work. The underflow already happened during backpropagation. You need to scale before backward so the intermediate gradient values stay representable.

**Mistake 3: Using the same precision everywhere.**

Some operations are numerically sensitive. Softmax, layer normalization, and loss computation often need FP32 even when the rest of the network uses FP16. This is why it is called mixed precision. You mix formats based on what each operation requires.

### The Implementation Level Pattern

In practice, frameworks like PyTorch provide automatic mixed precision (AMP) that handles these details:

```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        outputs = model(batch)
        loss = compute_loss(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

The autocast context manager automatically selects FP16 or FP32 for each operation. The GradScaler handles dynamic loss scaling. You get the benefits without managing the complexity.

But now you understand what is happening underneath.

### Success Indicators

Your mixed precision training loop is working correctly when:

1. Loss decreases smoothly, similar to FP32 training
2. Loss scale stabilizes after initial adjustments (not constantly halving)
3. Memory usage drops by roughly 50% compared to FP32
4. Training throughput increases (1.5-3x faster on modern GPUs)
5. Final model quality matches FP32 baseline

If your loss scale keeps dropping toward 1.0, something is wrong. Check for operations that should stay in FP32. If training is slower than FP32, check that your hardware actually supports fast FP16 operations.

Mixed precision training is how large language models like Llama 2 70B become practical to train. The 70 billion parameters would require 280GB in FP32. In mixed precision, the memory footprint drops dramatically, making training feasible on available hardware.

You now have the complete pattern. Master weights in FP32. Forward and backward in lower precision. Loss scaling to prevent underflow. This is the foundation for everything that comes next.

## Lesson 4: Loss Scaling

You have your mixed precision training loop running. FP16 forward passes, FP32 master weights, everything looks clean. Then you check your gradients and find something disturbing: they are all zeros.

This is not a bug in your code. This is the underflow problem, and loss scaling is how we fix it.

### The Vanishing Gradient Problem in FP16

Let me show you what happens. Run this:

```python
import torch

# Simulate a small gradient value typical in deep networks
gradient_fp32 = torch.tensor(1e-6, dtype=torch.float32)
gradient_fp16 = gradient_fp32.half()

print(f"FP32 gradient: {gradient_fp32}")
print(f"FP16 gradient: {gradient_fp16}")
print(f"FP16 smallest positive: {torch.finfo(torch.float16).tiny}")
```

Output:
```
FP32 gradient: 9.999999974752427e-07
FP16 gradient: 9.5367431640625e-07
FP16 smallest positive: 6.103515625e-05
```

Wait. That gradient is smaller than FP16's smallest representable positive number. So what happens when gradients get even smaller? They become exactly zero. Not approximately zero. Exactly zero. And when your gradients are zero, your Parameter Updates stop happening. Your model stops learning.

Remember from our earlier discussion of Numerical Precision Safeguards: when values fall below the representable range, they underflow to zero. This is catastrophic for training because gradients in deep networks are often tiny, especially in early layers due to the chain rule compounding many small multiplications.

### The Fix: Scale Up, Then Scale Down

The solution is elegant. We multiply the loss by a large number before the backward pass. This scales up all the gradients proportionally. Then we divide the gradients by that same number before applying them to the weights.

```python
# Without loss scaling
loss = compute_loss(model_output, targets)
loss.backward()  # Gradients might underflow to zero
optimizer.step()

# With loss scaling
scale = 1024.0
loss = compute_loss(model_output, targets)
scaled_loss = loss * scale
scaled_loss.backward()  # Gradients are 1024x larger, no underflow

# Unscale before optimizer step
for param in model.parameters():
    if param.grad is not None:
        param.grad.data /= scale
        
optimizer.step()
```

The math works out perfectly. If the true gradient is `g`, we compute `g * scale` during backward, then divide by `scale` before the update. The optimizer sees the correct gradient value, but we avoided the underflow danger zone during computation.

### Building a Loss Scaler Class

Let's build this properly:

```python
class LossScaler:
    def __init__(self, init_scale=2**16, growth_factor=2.0, 
                 backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.steps_since_growth = 0
        
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def check_for_overflow(self, optimizer):
        """Returns True if gradients contain inf or nan"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False
    
    def update(self, overflow_detected):
        if overflow_detected:
            # Overflow: reduce scale immediately
            self.scale *= self.backoff_factor
            self.steps_since_growth = 0
            return False  # Skip this step
        else:
            # No overflow: maybe increase scale
            self.steps_since_growth += 1
            if self.steps_since_growth >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_growth = 0
            return True  # Proceed with step
```

### Why Dynamic Scaling?

You might ask: why not just pick a fixed scale factor?

The problem is that gradient magnitudes change during training. Early in training, gradients tend to be larger. As the model converges, gradients shrink. A scale factor that prevents underflow at the start might cause overflow later, or vice versa.

Dynamic loss scaling solves this through Gradient Flow Diagnostics. We monitor for overflow (inf or nan values) and adjust:

1. If we detect overflow, we reduce the scale and skip that training step
2. If we go many steps without overflow, we increase the scale to provide more headroom against underflow

This is essentially a feedback control system for Numerical Stability Analysis.

### The Complete Training Loop with Loss Scaling

```python
def train_step_with_loss_scaling(model, optimizer, scaler, inputs, targets):
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
    
    # Scale loss and backward
    scaled_loss = scaler.scale_loss(loss)
    scaled_loss.backward()
    
    # Unscale gradients for overflow check
    scaler.unscale_gradients(optimizer)
    
    # Check for overflow
    overflow = scaler.check_for_overflow(optimizer)
    
    # Update scale and decide whether to step
    should_step = scaler.update(overflow)
    
    if should_step:
        optimizer.step()
        
    return loss.item(), scaler.scale
```

### Common Mistakes

**Mistake 1: Forgetting to unscale before gradient clipping**

If you use gradient clipping (and you should for Gradient Flow Management), you must unscale first:

```python
# Wrong
scaler.scale_loss(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clips scaled grads!
scaler.unscale_gradients(optimizer)

# Correct
scaler.scale_loss(loss).backward()
scaler.unscale_gradients(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clips true grads
```

**Mistake 2: Starting with too high a scale**

If your initial scale is too large, you will overflow immediately and spend many steps backing off. Start conservative, around 2^15 or 2^16.

**Mistake 3: Not skipping steps on overflow**

When overflow happens, those gradients are garbage. Do not apply them. Skip the step entirely.

### Success Indicators

How do you know your loss scaling is working?

1. **Scale stays stable**: After initial adjustment, your scale factor should not oscillate wildly. If it keeps bouncing between values, something else is wrong with your training.

2. **No persistent zeros**: Check gradient histograms occasionally. You should not see large clusters at exactly zero.

3. **Loss decreases**: The ultimate test. Your Loss Curve Analysis should show steady improvement.

```python
# Quick diagnostic
def check_gradient_health(model):
    total_params = 0
    zero_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            total_params += p.grad.numel()
            zero_grads += (p.grad == 0).sum().item()
    
    zero_ratio = zero_grads / total_params
    print(f"Zero gradient ratio: {zero_ratio:.4f}")
    if zero_ratio > 0.5:
        print("WARNING: More than half your gradients are zero!")
```

### The Connection to Numerical Precision

This whole problem traces back to what we discussed about floating point formats. FP16 has limited dynamic range. The gap between the largest and smallest representable numbers is narrower than FP32. Loss scaling is our way of shifting the gradient values into the "safe zone" of that limited range.

Think of it like this: FP16 can represent numbers accurately within a certain window. Loss scaling shifts our gradients into that window during computation, then shifts them back to their true values before we use them.

In the next lesson, we will look at how PyTorch's automatic mixed precision handles all of this for you. But now you understand what is happening underneath, so when something goes wrong, you will know where to look.

## Lesson 5: Automatic Mixed Precision

You have spent the last four lessons building up the pieces: understanding floating point formats, implementing mixed precision training manually, and creating loss scaling to keep gradients from vanishing into the underflow void. Now we put it all together into something you can actually use without thinking about it every single time.

Automatic mixed precision is exactly what it sounds like. You tell the system "figure out which operations should be FP16 and which should stay FP32" and it handles the details. But here is the thing: if you have followed along through this module, you already understand what "automatic" means under the hood. You built the machinery. Now we wrap it in a nice interface.

### Why Automatic?

Let me be direct about the problem. Manual mixed precision training works, but it is exhausting. You have to remember which layers need FP32 master weights. You have to manually cast tensors back and forth. You have to implement loss scaling yourself and tune the scale factor. You have to think about every single operation.

This is fine for learning. This is terrible for production.

The insight behind automatic mixed precision is that most operations in a neural network fall into predictable categories. Matrix multiplications and convolutions benefit enormously from FP16 because they are compute-bound and tensor cores eat them for breakfast. Reductions like softmax and layer normalization need FP32 because they accumulate many small values. Loss functions need FP32 because gradients can get tiny.

So we build a system that knows these categories and applies the right precision automatically.

### The Autocast Context Manager

Here is the core pattern in PyTorch:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

That is it. That is the whole thing. Everything inside the `autocast()` context manager gets automatic precision selection. The `GradScaler` handles loss scaling dynamically.

But you need to understand what is happening here because when it breaks, and it will break, you need to debug it.

### What Autocast Actually Does

Inside the autocast context, PyTorch intercepts every operation and decides its precision based on a whitelist and blacklist system.

**Operations that run in FP16 (the whitelist):**
- Linear layers (matrix multiplication)
- Convolutions
- Batch matrix multiplication
- Most element-wise operations

**Operations that stay in FP32 (the blacklist):**
- Softmax
- Layer normalization
- Loss functions
- Reductions (sum, mean over large dimensions)
- Operations that can overflow or underflow easily

**Operations that preserve input precision:**
- Concatenation
- Indexing
- Reshaping

When you call `model(inputs)` inside autocast, every operation checks this list. If a linear layer receives FP32 input, it casts to FP16, does the matmul on tensor cores, and returns FP16 output. If that output flows into layer normalization, it gets cast back to FP32 for the normalization, then back to FP16 for the next linear layer.

This is exactly what you built manually in Lesson 3. The difference is you do not have to write all those casts yourself.

### The GradScaler in Detail

The `GradScaler` implements dynamic loss scaling. Remember from Lesson 4 that we need to scale the loss up before backward to prevent gradient underflow, then scale gradients down before the optimizer step.

GradScaler does this automatically and also handles the tricky part: adjusting the scale factor based on whether gradients overflow.

```python
scaler = GradScaler(init_scale=65536.0, growth_factor=2.0, 
                    backoff_factor=0.5, growth_interval=2000)
```

The algorithm works like this:
1. Start with a high scale factor (65536)
2. Run training steps normally
3. If gradients contain inf or nan, skip the optimizer step and halve the scale
4. If 2000 consecutive steps have no inf/nan, double the scale
5. Repeat forever

This finds the sweet spot automatically. You do not need to tune it.

### BFloat16: The Alternative Path

Everything I have described uses FP16. But remember from Lesson 1 that BFloat16 exists and has the same exponent range as FP32. This means BFloat16 rarely underflows, which means you often do not need loss scaling at all.

```python
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

loss.backward()  # No scaler needed
optimizer.step()
```

BFloat16 trades precision for simplicity. The 7 mantissa bits (versus FP16's 10) mean slightly less accurate computations, but the training is more stable. Many practitioners prefer BFloat16 for this reason. The Llama series of models, for instance, uses BFloat16 throughout.

The tradeoff is hardware support. BFloat16 requires Ampere or newer NVIDIA GPUs. If you have older hardware, FP16 with loss scaling is your only option.

### Common Mistakes and How to Fix Them

**Mistake 1: Putting the optimizer step inside autocast**

```python
# WRONG
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()  # This should be outside
```

The optimizer needs to work with FP32 master weights. Keep it outside the autocast context.

**Mistake 2: Forgetting to call scaler.update()**

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
# Missing: scaler.update()
```

Without `update()`, the scale factor never adjusts. You will either underflow forever or overflow forever.

**Mistake 3: Custom operations that break the whitelist**

If you write custom CUDA kernels or unusual operations, autocast does not know how to handle them. You need to register them manually or wrap them with explicit casts.

```python
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
def my_custom_forward(input):
    # Your custom operation
    pass
```

**Mistake 4: Gradient accumulation without proper scaling**

When accumulating gradients across multiple batches, you need to be careful:

```python
scaler.scale(loss / accumulation_steps).backward()
# Only step and update after accumulation is complete
if (step + 1) % accumulation_steps == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Success Indicators

How do you know automatic mixed precision is working correctly?

1. **Memory usage drops by roughly 40-50%** compared to pure FP32 training. Not exactly half because optimizer states and some operations still use FP32.

2. **Training speed increases by 1.5-3x** on modern GPUs with tensor cores. If you see no speedup, check that your batch size is large enough to saturate the GPU.

3. **Loss curves look identical** to FP32 training. If your loss diverges or becomes unstable, something is wrong with your loss scaling or you have an operation that should be blacklisted but is not.

4. **The scaler's scale factor stabilizes** after initial oscillation. Check `scaler.get_scale()` periodically. It should settle into a range rather than constantly halving.

5. **No inf or nan in your gradients** during normal training. Occasional inf values are fine because the scaler handles them, but if every step has inf, your scale is too high or you have a numerical bug.

You have now built mixed precision training from first principles and learned how to use the automatic tools. The next module takes us to distributed training, where we spread computation across multiple GPUs. The precision techniques you learned here become even more important when you are trying to fit massive models across device memory.

## Practice

Implement FP32, FP16, BF16, and FP8 representations, build mixed precision training loop, create loss scaling for gradient stability, and implement automatic mixed precision patterns.

### Setup

You need a CUDA-capable GPU for this exercise. The tensor core operations that make mixed precision worthwhile only exist on modern NVIDIA hardware (Volta architecture or newer, so GTX 1080 Ti or better).

```bash
# Verify your GPU supports the operations we need
python -c "import torch; print(torch.cuda.get_device_capability())"
# You want (7, 0) or higher for full FP16 tensor core support
# You want (8, 9) or higher for FP8 support
```

Start with a simple model you can train quickly. A small transformer or even an MLP on MNIST works fine. The point is understanding the precision mechanics, not training a large model.

### Part 1: Implement the Representations

Write functions that manually convert between FP32 and each reduced precision format. Do not use PyTorch's casting, implement the bit manipulation yourself.

```python
def float32_to_fp16(x: float) -> int:
    """Convert a float32 to its FP16 bit representation (as an integer)."""
    # Your implementation here
    pass

def float32_to_bf16(x: float) -> int:
    """Convert a float32 to its BF16 bit representation (as an integer)."""
    # Your implementation here
    pass
```

Test edge cases: very small numbers (underflow), very large numbers (overflow), denormalized numbers, infinity, NaN.

### Part 2: Build the Mixed Precision Training Loop

Create a training loop that:
1. Stores master weights in FP32
2. Casts weights to FP16/BF16 for forward pass
3. Computes loss in FP32
4. Computes gradients in FP16/BF16
5. Updates master weights in FP32

Do this manually first, without `torch.cuda.amp`. You need to understand what the automatic tools are doing.

### Part 3: Implement Loss Scaling

Gradients in FP16 often underflow to zero. Loss scaling fixes this:
1. Multiply the loss by a scale factor (start with 1024 or 65536)
2. Backpropagate (gradients are now scaled up)
3. Unscale gradients before the optimizer step
4. If gradients overflow (contain inf/nan), skip the update and reduce the scale
5. If updates succeed for several steps, increase the scale

Implement dynamic loss scaling that automatically adjusts the scale factor.

### Part 4: Automatic Mixed Precision Patterns

Now use `torch.cuda.amp.autocast()` and `torch.cuda.amp.GradScaler()`. Compare the behavior to your manual implementation. Verify they produce similar results.

Profile both versions. Measure:
- Memory usage (peak and average)
- Training throughput (samples per second)
- Final model accuracy

### Success Criteria

- Your manual FP16/BF16 converters match PyTorch's casting for normal values
- Your manual mixed precision loop trains successfully without diverging
- Loss scaling prevents gradient underflow (verify by checking gradient histograms)
- You can explain why certain operations stay in FP32 (reductions, softmax, loss computation)
- Your AMP implementation achieves at least 1.5x speedup over pure FP32 on appropriate hardware

### Common Pitfalls

**Gradients all become zero.** You forgot loss scaling, or your scale factor is too small. Check gradient magnitudes before and after scaling.

**Training diverges after switching to mixed precision.** Some operations need FP32. Batch normalization statistics, softmax denominators, and loss computation are common culprits. Keep these in full precision.

**No speedup observed.** Your model might be too small, your batch size too small, or your GPU might not have tensor cores. Mixed precision overhead can exceed benefits for tiny models.

**BF16 works but FP16 does not.** This is expected for models with large activation ranges. BF16's wider exponent range handles this; FP16's does not. This is why BF16 became the default for LLM training.

**FP8 produces garbage.** FP8 requires per-tensor scaling and is primarily useful for inference, not training. If you are attempting FP8 training, you need sophisticated scaling strategies that are beyond this exercise.

## Assessment

By the end of this module, you should be able to:

- Explain the bit-level structure of float32, float16, and bfloat16, including how sign, exponent, and mantissa bits determine range and precision
- Calculate memory requirements for model weights, gradients, and optimizer states across different precision formats
- Implement mixed precision training that uses float16 or bfloat16 for forward and backward passes while maintaining float32 master weights
- Diagnose and fix numerical instability issues including gradient underflow, overflow, and loss of precision during accumulation
- Apply loss scaling techniques to prevent gradient underflow in float16 training, including dynamic loss scaling that adjusts automatically
- Choose the appropriate precision format for different scenarios, understanding why bfloat16's dynamic range often matters more than float16's extra mantissa bits

Success means you can take a training loop that runs out of memory or runs too slowly, convert it to mixed precision, and have it train stably without degrading model quality. You understand not just the API calls but the underlying numerical behavior, why gradients vanish at certain magnitudes, why you need a master copy of weights, why some operations must stay in float32. When training goes wrong with NaN losses or degraded convergence, you can trace the problem to its numerical source and fix it.

---

# Need for Speed III,  Distributed

## Lesson 1: Distributed Training

You have one GPU. Training is slow. You want to use more GPUs. This is the problem we are solving.

Let me show you the shape of this problem before we get into mechanics.

### Why Distributed Training Exists

Remember from earlier chapters that training involves computing gradients and updating parameters. Each training step looks like this: take a batch of data, run it through the model, compute the loss, backpropagate to get gradients, update weights. Repeat millions of times.

The bottleneck is compute. A single GPU can only process so many tokens per second. If you want to train faster, you need more compute. If you want to train larger models, you need more memory. Both problems point to the same solution: use multiple GPUs.

Here is the fundamental insight. The pre-training stage that builds world knowledge is extremely expensive. We are talking about months of compute on hundreds or thousands of GPUs. The fine-tuning stage that creates a helpful assistant is comparatively cheap. This asymmetry means that distributed training is not optional for serious work. It is required.

The cost of training has dropped dramatically over the years. What cost $40,000 for GPT-2 might cost $100 today, thanks to hardware improvements, better algorithms, and improved distributed training frameworks. But even with these improvements, training at scale requires coordination across many devices.

### The Two Flavors of Parallelism

There are two fundamental ways to split work across GPUs:

**Data Parallelism**: Every GPU has a complete copy of the model. You split your training data across GPUs. Each GPU computes gradients on its portion of the data. Then you average the gradients across all GPUs and update the model.

**Model Parallelism**: The model itself is split across GPUs. Different parts of the network live on different devices. Data flows through the distributed model.

We will focus on data parallelism in this chapter because it is simpler, more common, and the foundation for more advanced techniques. Model parallelism becomes necessary when your model does not fit on a single GPU, which we will address in later lessons.

### How Data Parallelism Works

Let me walk through the mechanics. Say you have 4 GPUs and a batch size of 32.

Without distributed training, one GPU processes all 32 examples, computes gradients, and updates weights.

With data parallelism, each GPU processes 8 examples. Each GPU computes gradients based on its 8 examples. Then the magic happens: you need to combine those gradients.

The operation that combines gradients is called all-reduce. Every GPU sends its gradients to every other GPU, and they all end up with the average. After all-reduce, every GPU has identical gradients. They each apply the same update to their local copy of the model. The models stay synchronized.

This is the core loop:

```
1. Each GPU gets its portion of the batch
2. Each GPU runs forward pass
3. Each GPU computes loss
4. Each GPU runs backward pass (local gradients)
5. All-reduce: average gradients across all GPUs
6. Each GPU updates its local model with averaged gradients
```

The models start identical and stay identical because they all see the same averaged gradients and apply the same updates.

### Communication Overhead

Here is where people get stuck. All-reduce is not free. Every training step, you are moving gradients across the network connecting your GPUs. For a model with billions of parameters, that is billions of floating point numbers that need to travel between devices.

The Communication Overhead can dominate your training time if you are not careful. You might have 4 GPUs but only get 2x speedup because half your time is spent waiting for gradient synchronization.

The ratio that matters is: compute time versus communication time. If your forward and backward passes take 100ms and your all-reduce takes 10ms, you are in good shape. If your all-reduce takes 200ms, you have a problem.

Several factors affect this ratio:

**Batch Size**: Larger batches mean more compute per step relative to communication. This is why distributed training often uses large batch sizes.

**Network bandwidth**: GPUs connected by NVLink communicate faster than GPUs connected over ethernet. Data center topology matters.

**Model size**: More parameters means more gradients to synchronize. Bigger models have higher communication costs.

**Gradient compression**: You can reduce communication by compressing gradients, though this introduces approximation.

### The Effective Batch Size Question

When you use 4 GPUs with batch size 8 per GPU, your effective batch size is 32. This is not the same as training on one GPU with batch size 32 in terms of memory, but it is equivalent in terms of the gradient you compute.

This matters because Batch Size affects training dynamics. Larger batches give you more stable gradients but can hurt generalization. There is research suggesting that very large batches require learning rate adjustments to train well.

The rule of thumb: when you scale to more GPUs, you typically scale your learning rate proportionally (linear scaling rule) up to a point. Beyond that point, you need more sophisticated techniques like warmup schedules.

### Synchronous vs Asynchronous

Everything I described is synchronous data parallelism. All GPUs wait for each other at the all-reduce step. This keeps models perfectly synchronized but means the slowest GPU determines your speed.

Asynchronous approaches let GPUs proceed without waiting. GPU 1 might be on step 100 while GPU 2 is on step 97. This eliminates waiting but introduces staleness. You are updating with gradients computed on an older version of the model. This can hurt convergence.

In practice, synchronous training dominates because the convergence properties are cleaner and the engineering is simpler. The Computational Heterogeneity problem, where different GPUs have different speeds, is usually solved by ensuring homogeneous hardware rather than going asynchronous.

### Common Mistakes

**Forgetting to average gradients**: If you sum instead of average, your effective learning rate scales with GPU count. Your training will diverge or behave strangely.

**Ignoring random seeds**: Each GPU needs different random data augmentation and shuffling, but the same model initialization. Getting this wrong causes subtle bugs.

**Batch normalization statistics**: If your model uses batch norm, you need to decide whether statistics are computed per-GPU or synchronized. This affects model behavior.

**Not accounting for effective batch size**: Comparing runs with different GPU counts requires adjusting for the batch size change.

### What Success Looks Like

You know distributed training is working when:

1. Loss curves look similar to single-GPU training (maybe smoother due to larger effective batch)
2. Training time scales roughly linearly with GPU count (accounting for communication overhead)
3. All GPUs show similar utilization
4. Model checkpoints from any GPU are interchangeable

The goal is near-linear scaling. With 4 GPUs, you want close to 4x speedup. In practice, you might get 3.5x due to communication overhead. Getting below 3x suggests a bottleneck worth investigating.

### Looking Ahead

Data parallelism works when your model fits on one GPU. What happens when it does not? What about the optimizer states that also consume memory? The next lessons cover gradient synchronization mechanics in detail, then DDP implementation, and finally ZeRO-style optimizer sharding that lets you train models larger than any single GPU can hold.

The distributed training frameworks that enable efficient scaling across multiple devices are one of the key software improvements that have driven down training costs so dramatically. Understanding these systems is not just about speed. It is about what becomes possible when you can coordinate hundreds of GPUs effectively.

## Lesson 2: Data Parallel Training

You have one GPU. Training takes a week. You get four GPUs. Does training take two days? One day? The same week?

The answer depends entirely on how you use those GPUs. And the simplest, most effective approach is data parallelism. Let's build it.

### The Problem We're Solving

Remember from the previous lesson that processing B×T examples in parallel during training has significant computational implications. Your single GPU can handle, say, a batch size of 32. But larger batches give you more stable gradient estimates since the gradient is averaged over more tokens rather than individual examples, reducing variance in parameter updates.

What if you want a batch size of 128? You could buy a bigger GPU. Or you could use four GPUs, each processing 32 examples, then combine the results.

That's data parallelism in one sentence: same model on every GPU, different data on each GPU, combine the gradients.

### Building It From Scratch

Let's start with the simplest possible implementation. Forget PyTorch's DistributedDataParallel for now. We're going to understand what it does by building the pieces ourselves.

```python
import torch
import torch.distributed as dist

def setup(rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group(
        backend='nccl',  # NVIDIA's collective communication library
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
```

Each GPU runs its own Python process. `rank` is which GPU this process controls (0, 1, 2, 3). `world_size` is the total number of GPUs. NCCL is NVIDIA's library for GPU-to-GPU communication, and it's fast because GPUs can talk directly to each other through NVLink or PCIe without going through the CPU.

Now the training loop:

```python
def train_step(model, data, target, optimizer):
    # Forward pass - each GPU processes its own batch
    output = model(data)
    loss = F.cross_entropy(output, target)
    
    # Backward pass - each GPU computes gradients for its batch
    loss.backward()
    
    # HERE'S THE MAGIC: synchronize gradients across all GPUs
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= dist.get_world_size()
    
    # Now all GPUs have the same averaged gradient
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```

That `all_reduce` call is where distributed training happens. Let's understand exactly what it does.

### All-Reduce: The Core Operation

Imagine four GPUs, each with a gradient tensor:

```
GPU 0: [1.0, 2.0, 3.0]
GPU 1: [0.5, 1.5, 2.5]
GPU 2: [1.5, 2.5, 3.5]
GPU 3: [1.0, 1.0, 1.0]
```

After `all_reduce` with SUM:

```
GPU 0: [4.0, 7.0, 10.0]
GPU 1: [4.0, 7.0, 10.0]
GPU 2: [4.0, 7.0, 10.0]
GPU 3: [4.0, 7.0, 10.0]
```

Every GPU ends up with the same sum. We divide by world_size to get the average. Now every GPU has identical gradients, so when they all call `optimizer.step()`, they all update their weights identically. The models stay synchronized.

This is the fundamental insight of data parallel training: if all GPUs start with the same weights and apply the same gradient update, they stay in sync forever. You only need to synchronize gradients, not the weights themselves.

### Communication Patterns

Here's where people get confused. "All-reduce" sounds like it should be expensive. Four GPUs, each sending their entire gradient to every other GPU? That's a lot of data moving around.

NCCL is smarter than that. It uses a ring all-reduce algorithm:

1. Arrange GPUs in a ring: 0 → 1 → 2 → 3 → 0
2. Each GPU splits its gradient into N chunks
3. In round 1, each GPU sends chunk i to the next GPU and receives chunk i-1 from the previous GPU
4. After N-1 rounds of this, each GPU has the sum of one chunk
5. Then N-1 more rounds to broadcast the sums back

The total data transferred per GPU is 2×(N-1)/N × gradient_size. For large N, this approaches 2× the gradient size, regardless of how many GPUs you have. The communication cost doesn't explode with more GPUs.

### Communication Overhead

But wait. While GPUs are talking to each other, they're not computing. This is communication overhead, and it's the main challenge in distributed training.

The solution is to overlap communication with computation. While GPU 0 is sending gradients for layer 10, it can already be computing gradients for layer 9. PyTorch's DistributedDataParallel does this automatically by hooking into the backward pass and starting communication as soon as each gradient is ready.

```python
# This is roughly what DDP does internally
def backward_hook(grad):
    # Start all-reduce immediately, don't wait for full backward pass
    handle = dist.all_reduce(grad, async_op=True)
    return grad, handle
```

The `async_op=True` means "start this communication but don't wait for it to finish." You collect all the handles and wait for them at the end.

### Why This Works

Let's connect this back to why we're doing distributed training in the first place. Processing B×T examples in parallel gives us improved hardware utilization through better parallelism, as modern GPUs have thousands of cores that can process many examples simultaneously.

With data parallelism, we're effectively increasing B without changing T. Four GPUs each processing batch size 32 gives us an effective batch size of 128. The gradient we compute is mathematically identical to what we'd get from a single GPU processing 128 examples, just faster.

This matters because very large B×T products may require gradient accumulation or adaptive learning rates to maintain convergence properties. When you scale to many GPUs, you're implicitly increasing the effective batch size, which changes training dynamics. Learning rate schedules often need adjustment.

### Common Mistakes

**Mistake 1: Forgetting to average gradients.** If you sum but don't divide by world_size, your effective learning rate is multiplied by the number of GPUs. Training will diverge.

**Mistake 2: Initializing models differently on each GPU.** If GPU 0 starts with different random weights than GPU 1, they'll never synchronize. Always seed your random number generator consistently or broadcast initial weights from rank 0.

**Mistake 3: Data loading bottlenecks.** Data loading bottlenecks become more pronounced as the system must feed data fast enough to keep all parallel processing units busy. Each GPU needs its own data loader feeding unique batches. If all GPUs process the same batch, you're not getting any benefit from parallelism.

```python
# Each GPU gets a different subset of the data
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

**Mistake 4: Not handling the loss correctly.** The loss you print on GPU 0 is only for GPU 0's batch. To get the true average loss across all GPUs, you need another all-reduce:

```python
dist.all_reduce(loss, op=dist.ReduceOp.SUM)
loss /= world_size
```

### Success Indicators

How do you know your data parallel training is working correctly?

1. **Loss curves match.** Train with 1 GPU batch size 128 and with 4 GPUs batch size 32 each. The loss curves should be nearly identical.

2. **Scaling efficiency.** Four GPUs should give you at least 3.5x speedup over one GPU. If you're seeing less than 3x, you have a communication or data loading bottleneck.

3. **Weights stay synchronized.** Periodically check that `model.parameters()` are identical across all ranks. Any drift means you have a synchronization bug.

4. **GPU utilization stays high.** Use `nvidia-smi` to check that all GPUs are actually working. If GPU 0 is at 100% and others are at 10%, something is wrong with your data distribution.

### What's Next

Data parallelism has a fundamental limit: every GPU holds a complete copy of the model. For a 70 billion parameter model like Llama 2 70B, that's about 140GB just for the weights in fp16. No single GPU can hold that.

In the next lesson, we'll look at gradient synchronization in more detail, then move to techniques that split the model itself across GPUs. But data parallelism is your workhorse. Most training runs use it, and understanding it deeply will make everything else clearer.

## Lesson 3: Gradient Synchronization

You have multiple GPUs. Each one processes its own batch of data. Each one computes its own gradients. Now what?

This is the question that makes or breaks distributed training. And the answer is deceptively simple: you average the gradients across all GPUs, then update the weights identically on each one. That's it. That's gradient synchronization.

But "simple" and "easy to get right" are different things. Let's build this from scratch.

### The Problem We're Solving

Think back to what we established in the previous lesson about data parallel training. You have N GPUs, each with a complete copy of your model. Each GPU processes a different mini-batch. Each GPU computes gradients for its local batch.

Here's the critical insight: if GPU 0 computes gradients from batch A, and GPU 1 computes gradients from batch B, then the average of those gradients is mathematically equivalent to computing gradients on the combined batch (A + B). This is just linearity of the gradient operator.

So gradient synchronization isn't some approximation or hack. It's mathematically exact. The distributed computation gives you the same result as if you had one giant GPU processing all the data at once.

Let's implement this.

### Building the All-Reduce

The operation we need is called "all-reduce." It takes a tensor that exists on every GPU, applies a reduction operation (in our case, averaging), and distributes the result back to every GPU.

```python
import torch
import torch.distributed as dist

def synchronize_gradients(model):
    """Average gradients across all processes."""
    world_size = dist.get_world_size()
    
    for param in model.parameters():
        if param.grad is not None:
            # Sum gradients across all GPUs
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Divide by number of GPUs to get average
            param.grad.div_(world_size)
```

That's the core of it. After each backward pass, before the optimizer step, you call this function. Every GPU ends up with identical averaged gradients. Every GPU applies the same update. Every GPU stays synchronized.

### Why All-Reduce and Not Something Simpler?

You might think: why not just send all gradients to GPU 0, average them there, and broadcast back? That would work. But it's slow.

Consider the communication pattern. If you gather to one GPU, that single GPU becomes a bottleneck. It receives N times the data, does all the computation, then sends N times the data back out. The time scales linearly with the number of GPUs.

All-reduce is smarter. It uses a ring topology where each GPU only communicates with its neighbors. The algorithm works in two phases:

**Phase 1 (Reduce-Scatter):** Each GPU sends a chunk of its data to the next GPU in the ring, which adds it to its own chunk. After N-1 steps, each GPU has the complete sum of one chunk.

**Phase 2 (All-Gather):** Each GPU sends its summed chunk around the ring. After N-1 more steps, every GPU has all the summed chunks.

Total communication: 2 * (N-1) / N * data_size per GPU. As N grows, this approaches 2 * data_size, regardless of how many GPUs you have. The bandwidth cost is nearly constant.

### The Training Loop with Synchronization

Here's what a complete training step looks like:

```python
def training_step(model, optimizer, data, target):
    # Forward pass (each GPU uses its own batch)
    output = model(data)
    loss = compute_loss(output, target)
    
    # Backward pass (each GPU computes local gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # Synchronize gradients across GPUs
    synchronize_gradients(model)
    
    # Update weights (identical on all GPUs)
    optimizer.step()
    
    return loss.item()
```

The synchronize_gradients call is the only distributed operation in the loop. Everything else happens locally on each GPU.

### When Things Go Wrong

The most common failure mode is GPUs getting out of sync. If one GPU updates its weights differently than another, even slightly, the models diverge. After a few steps, you're training completely different models on each GPU.

How does this happen?

**Missing synchronization:** If you forget to synchronize gradients for even one parameter, that parameter drifts. Common culprits are newly added layers, buffers that got converted to parameters, or conditional code paths that skip synchronization.

**Non-deterministic operations:** Some operations aren't perfectly reproducible across GPUs. Floating point arithmetic can give slightly different results depending on operation order. Usually this doesn't matter because synchronization forces agreement. But if you have operations after synchronization that diverge, trouble follows.

**Different random states:** If your model uses dropout or other stochastic operations, each GPU needs either identical random states or the randomness needs to happen before synchronization. Most frameworks handle this, but custom code can break assumptions.

### Verifying Synchronization

Always verify your gradients are actually synchronized. Here's a simple check:

```python
def verify_sync(model):
    """Check that all GPUs have identical parameters."""
    for name, param in model.named_parameters():
        # Gather parameter from all GPUs
        gathered = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, param)
        
        # Check they're all identical
        for i, p in enumerate(gathered[1:], 1):
            if not torch.allclose(gathered[0], p):
                print(f"Parameter {name} differs between GPU 0 and GPU {i}")
                return False
    return True
```

Run this periodically during training. If it ever returns False, you have a synchronization bug.

### The Bandwidth Reality

Gradient synchronization is where distributed training hits physical limits. Your gradients are the same size as your model. For a 70 billion parameter model like Llama 2 70B, that's 70 billion floats, or about 280 GB in float32.

Every training step, you need to synchronize all of that across your network. Even with efficient all-reduce, this is a lot of data. Modern interconnects like NVLink and InfiniBand exist specifically to handle this load.

This is why you'll hear about "communication-computation overlap." The idea is to start synchronizing gradients for early layers while still computing gradients for later layers. The backward pass naturally processes layers in reverse order, so by the time you finish computing, some synchronization is already done.

### Success Indicators

You know gradient synchronization is working when:

1. Loss curves look identical across all GPUs
2. Parameter checksums match across GPUs after each step
3. Scaling from 1 to N GPUs gives you N times the effective batch size
4. Training speed scales sub-linearly (some overhead is expected) but not catastrophically

You know something is wrong when:

1. Different GPUs report different losses
2. Training becomes unstable as you add more GPUs
3. Results change depending on GPU count (beyond batch size effects)

Gradient synchronization is the foundation that makes everything else in distributed training possible. Get this right, and the rest follows. Get it wrong, and nothing else matters.

Next, we'll implement a complete DDP system that handles all of this automatically.

## Lesson 4: DDP Implementation

You have gradient synchronization working. You understand all-reduce. Now we build the real thing: Distributed Data Parallel training that actually runs across multiple GPUs.

Let's write the code first, then understand why it works.

### The Minimal DDP Implementation

Here is a working DDP implementation in about 80 lines:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model_class, dataset, epochs=10):
    setup(rank, world_size)
    
    # Each GPU gets its own copy of the model
    model = model_class().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Sampler ensures each GPU sees different data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )
    
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Critical for proper shuffling
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(rank), y.to(rank)
            
            optimizer.zero_grad()
            loss = ddp_model(x, y)
            loss.backward()  # Gradients synchronized automatically here
            optimizer.step()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    cleanup()

# Launch with:
# torchrun --nproc_per_node=4 train_ddp.py
```

That is the entire thing. Now let's understand what each piece does.

### Why Each Line Matters

**The setup function** initializes the process group. When you run `torchrun --nproc_per_node=4`, it spawns four separate Python processes. Each process needs to know: who am I (rank), how many of us are there (world_size), and how do we talk to each other (the process group).

The `MASTER_ADDR` and `MASTER_PORT` environment variables tell all processes where to find each other. On a single machine, localhost works fine. On multiple machines, you would use the IP address of one designated master node.

**NCCL** is NVIDIA's collective communication library. It handles the actual GPU-to-GPU communication. You could use `gloo` for CPU training, but for GPUs, NCCL is what you want.

**The DistributedSampler** is subtle but essential. Without it, every GPU would train on the same data. The sampler partitions your dataset so GPU 0 sees batches 0, 4, 8, 12... while GPU 1 sees batches 1, 5, 9, 13... and so on. This is data parallelism: same model, different data.

**set_epoch** on the sampler ensures different shuffling each epoch. Miss this and you get the same data order every epoch, which hurts generalization.

**The DDP wrapper** is where the magic happens. When you call `loss.backward()`, DDP intercepts the backward pass. As gradients are computed for each parameter, DDP immediately starts the all-reduce operation in the background. By the time backward finishes, all GPUs have identical averaged gradients.

### How DDP Actually Works

The Hierarchical Control pattern from our source materials applies here directly. Think of DDP as having two layers:

The **Reactive Layer** handles the actual computation. Each GPU runs forward and backward passes independently, as fast as possible. No waiting, no coordination during the main computation.

The **Deliberative Layer** handles synchronization. DDP schedules all-reduce operations to overlap with backward computation. While GPU 0 is computing gradients for layer 5, it is simultaneously sending gradients from layer 10 to other GPUs.

This overlap is why DDP scales well. The communication happens in parallel with computation, hiding most of the synchronization cost.

### Building DDP From Scratch

To really understand DDP, let's implement the core mechanism ourselves:

```python
class ManualDDP:
    def __init__(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        
        # Register hooks to synchronize gradients
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))
    
    def _make_hook(self, param):
        def hook(grad):
            # All-reduce the gradient across all GPUs
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)  # Average
            return grad
        return hook
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def parameters(self):
        return self.model.parameters()
```

This is the essence of DDP. The hook fires after each gradient is computed, triggers all-reduce, and divides by world_size to get the average. PyTorch's real DDP is more sophisticated with bucketing and overlap, but this captures the core idea.

### Common Mistakes and How to Avoid Them

**Mistake 1: Forgetting to set the device**

```python
# Wrong - model stays on CPU or wrong GPU
model = MyModel()
ddp_model = DDP(model)

# Right - explicitly move to correct GPU first
model = MyModel().to(rank)
ddp_model = DDP(model, device_ids=[rank])
```

**Mistake 2: Using regular DataLoader**

```python
# Wrong - all GPUs see same data
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Right - use DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Mistake 3: Logging from all ranks**

```python
# Wrong - prints 4 times with 4 GPUs
print(f"Loss: {loss.item()}")

# Right - only rank 0 logs
if rank == 0:
    print(f"Loss: {loss.item()}")
```

**Mistake 4: Saving checkpoints from all ranks**

```python
# Wrong - all GPUs try to write same file
torch.save(model.state_dict(), "checkpoint.pt")

# Right - only rank 0 saves
if rank == 0:
    torch.save(ddp_model.module.state_dict(), "checkpoint.pt")
dist.barrier()  # Wait for save to complete before continuing
```

Note the `.module` access. DDP wraps your model, so the actual parameters live in `ddp_model.module`.

### Adaptive Computation in DDP

The source materials mention Adaptive Computation as a technique for adjusting model behavior based on conditions. In DDP, this manifests as gradient accumulation:

```python
accumulation_steps = 4

for batch_idx, (x, y) in enumerate(loader):
    with ddp_model.no_sync() if (batch_idx + 1) % accumulation_steps != 0 else nullcontext():
        loss = ddp_model(x, y)
        loss = loss / accumulation_steps
        loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

The `no_sync()` context manager tells DDP to skip synchronization. You accumulate gradients locally for several steps, then synchronize once. This simulates larger batch sizes without needing more GPU memory.

### Success Indicators

You know your DDP implementation is working when:

1. All GPUs show roughly equal memory usage
2. Training speed scales nearly linearly with GPU count (4 GPUs should be 3.5x to 3.8x faster than 1)
3. Loss curves match single-GPU training (given same total batch size)
4. `nvidia-smi` shows all GPUs actively computing, not waiting

Run `watch -n 0.5 nvidia-smi` in a separate terminal while training. You should see GPU utilization bouncing between 90-100% on all devices. If one GPU is at 100% while others idle, something is wrong with your data distribution.

### What Comes Next

DDP handles data parallelism beautifully, but it requires each GPU to hold a complete copy of the model. For a 70 billion parameter model like Llama 2 70B, that is roughly 140GB just for parameters in float16. No single GPU can hold that.

The next lesson covers ZeRO-style optimizer sharding, which lets you train models larger than any single GPU's memory by distributing the optimizer states, gradients, and even parameters across GPUs.

## Lesson 5: ZeRO Optimizer Sharding

You have DDP working. Your gradients synchronize across GPUs. Training scales. But here is the problem: you are about to hit a wall, and it is not compute. It is memory.

Let me show you what I mean. Take a model with 1 billion parameters. In fp32, that is 4 GB just for the weights. Seems manageable. But during training with Adam optimizer, you also need:

- Gradients: another 4 GB
- Optimizer states (momentum and variance): 8 GB more
- Activations for backpropagation: varies, but often larger than everything else

Suddenly your 1B parameter model needs 16+ GB of memory per GPU, and every single GPU in your DDP setup is holding an identical copy of all this state. That is wasteful. That is the problem ZeRO solves.

### Why Optimizer State Dominates Memory

Here is something that surprises people. With Adam, your optimizer states are twice the size of your model parameters. For every parameter, Adam tracks:

```python
# What Adam stores per parameter
m = torch.zeros_like(param)  # First moment (momentum)
v = torch.zeros_like(param)  # Second moment (variance)
```

So a 7 billion parameter model in fp32 means 28 GB for parameters, 28 GB for gradients, and 56 GB for optimizer states. That is 112 GB before activations. No single GPU holds that.

The insight behind ZeRO is simple: in DDP, every GPU computes gradients on different data, but they all end up with identical optimizer states after the all-reduce. Why store identical copies everywhere? What if each GPU only stored a fraction?

### ZeRO Stage 1: Shard the Optimizer States

Let us build this. Start with the most impactful optimization: sharding optimizer states across GPUs.

```python
import torch
import torch.distributed as dist

class ZeROStage1Optimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Partition parameters across ranks
        self.param_groups = self._partition_params()
        
        # Only create optimizer states for our partition
        self.m = {}
        self.v = {}
        for param in self.param_groups[self.rank]:
            self.m[param] = torch.zeros_like(param.data)
            self.v[param] = torch.zeros_like(param.data)
    
    def _partition_params(self):
        """Assign each parameter to exactly one rank."""
        groups = [[] for _ in range(self.world_size)]
        for i, param in enumerate(self.params):
            owner = i % self.world_size
            groups[owner].append(param)
        return groups
```

The key insight here: each GPU only allocates optimizer states for parameters it "owns." With 8 GPUs, each GPU stores 1/8 of the optimizer states. Memory usage drops dramatically.

### The Step Function: Update Then Broadcast

Here is where it gets interesting. Each GPU updates only its owned parameters, then broadcasts those updates to everyone else:

```python
def step(self):
    self.t += 1
    
    # Each rank updates only its owned parameters
    for param in self.param_groups[self.rank]:
        if param.grad is None:
            continue
            
        grad = param.grad.data
        m = self.m[param]
        v = self.v[param]
        
        # Adam update equations
        m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        
        # Bias correction
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # Update parameter
        param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
    
    # Broadcast updated parameters to all ranks
    for owner_rank, param_group in enumerate(self.param_groups):
        for param in param_group:
            dist.broadcast(param.data, src=owner_rank)
```

Notice the pattern: compute locally, communicate globally. Each GPU does 1/N of the optimizer work, then shares results. The broadcast at the end ensures all GPUs have identical parameters for the next forward pass.

### ZeRO Stage 2: Also Shard Gradients

Stage 1 shards optimizer states. Stage 2 goes further and shards gradients too. Instead of all-reduce (which leaves full gradients on every GPU), we use reduce-scatter:

```python
def reduce_scatter_gradients(self):
    """Each rank ends up with gradients only for its owned params."""
    for param in self.params:
        if param.grad is None:
            continue
        
        # Flatten gradient for communication
        grad_flat = param.grad.data.view(-1)
        chunk_size = (grad_flat.numel() + self.world_size - 1) // self.world_size
        
        # Pad to make evenly divisible
        padded_size = chunk_size * self.world_size
        grad_padded = torch.zeros(padded_size, device=grad_flat.device)
        grad_padded[:grad_flat.numel()] = grad_flat
        
        # Reduce-scatter: sum across ranks, scatter results
        output = torch.zeros(chunk_size, device=grad_flat.device)
        dist.reduce_scatter(output, list(grad_padded.chunk(self.world_size)))
        
        # Store only our chunk
        self.grad_chunks[param] = output
```

After reduce-scatter, each GPU holds only 1/N of the summed gradients. Combined with Stage 1, memory for both gradients and optimizer states scales as 1/N.

### ZeRO Stage 3: Shard Parameters Too

Stage 3 is the most aggressive. Even the model parameters themselves are sharded. Each GPU only holds 1/N of the weights. During forward and backward passes, parameters are gathered on-demand:

```python
def forward_with_param_gather(self, layer, input):
    """Gather full parameters, compute, then discard."""
    # Gather parameters from all ranks
    full_weight = torch.zeros_like(layer.weight_full_shape)
    dist.all_gather_into_tensor(full_weight, layer.weight_shard)
    
    # Compute forward pass
    output = F.linear(input, full_weight, layer.bias)
    
    # Discard gathered parameters (keep only our shard)
    del full_weight
    
    return output
```

This trades communication for memory. You gather parameters, use them, throw them away. Memory usage becomes almost constant regardless of model size, limited mainly by activation memory.

### Common Mistakes

**Forgetting gradient synchronization before sharding.** ZeRO assumes gradients are already summed across data-parallel replicas. If you shard unsummed gradients, each GPU optimizes on partial gradient information. The model diverges.

**Mismatched partitioning schemes.** If GPU 0 thinks it owns parameters 0-99 but GPU 1 thinks GPU 0 owns parameters 0-49, you get silent corruption. Use deterministic partitioning based on parameter index.

**Not handling uneven division.** With 7 billion parameters across 8 GPUs, the last GPU gets a different-sized chunk. Your code must handle padding or uneven splits.

### When to Use Each Stage

Stage 1 gives you 4x memory reduction on optimizer states with minimal communication overhead. Use it first.

Stage 2 adds gradient sharding. Communication increases slightly because you need reduce-scatter instead of all-reduce, but memory drops further.

Stage 3 is for truly massive models that cannot fit even with Stages 1 and 2. The communication overhead is significant since you gather parameters for every layer, every forward and backward pass. But it enables training models that would otherwise be impossible.

### Success Indicators

You know ZeRO is working when:

1. Memory per GPU drops roughly linearly with GPU count
2. Model checkpoints can be reconstructed by gathering shards from all ranks
3. Training loss matches non-sharded baseline (within numerical precision)
4. Gradient norms reported by each rank differ (they each see different shards)

The trade-off between simplified, Transparent Implementation and Heavily Optimized, Ready Library applies here strongly. Our implementation helps you understand the mechanics. For production, use PyTorch FSDP or DeepSpeed, which handle edge cases, overlap communication with compute, and provide battle-tested reliability.

ZeRO transforms what is trainable. A model that needs 112 GB per GPU with naive DDP needs only 14 GB per GPU with ZeRO Stage 1 across 8 GPUs. That is the difference between "impossible" and "fits on commodity hardware."

You now understand distributed training from first principles: data parallelism, gradient synchronization, all-reduce, and memory-efficient sharding. These are the Key Insights that let you scale from a single GPU to a cluster.

## Practice

Build data parallel training, implement gradient synchronization, create DDP implementation, and implement ZeRO-style optimizer sharding.

### Setup

You'll need access to multiple GPUs for this exercise. Options:
- Multi-GPU machine (2+ GPUs)
- Cloud instance (AWS p3.8xlarge, Lambda Labs, etc.)
- For initial development, you can simulate with a single GPU using fake process groups

Start with the model from Chapter 05 (Transformer) or a simple MLP if you want faster iteration cycles.

```python
# Verify your setup
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

### Part 1: Manual Gradient Synchronization

Before using any framework, implement gradient averaging yourself:

1. Initialize the same model on each GPU (same random seed)
2. Split a batch across GPUs manually
3. Run forward and backward passes independently
4. Use `torch.distributed.all_reduce` to average gradients
5. Apply identical optimizer steps

### Part 2: DDP from Scratch

Build your own DistributedDataParallel wrapper:

1. Wrap a model and register hooks on parameters
2. After backward pass, trigger all-reduce on gradients
3. Handle gradient bucketing (combine small tensors before communication)
4. Verify your implementation matches PyTorch's DDP numerically

### Part 3: ZeRO-Style Optimizer Sharding

Implement ZeRO Stage 1 (optimizer state partitioning):

1. Each GPU owns optimizer states for only 1/N of parameters
2. After all-reduce of gradients, each GPU updates only its partition
3. Broadcast updated parameters to all GPUs
4. Measure memory savings compared to standard DDP

### Success Criteria

- [ ] Training loss curves match between single-GPU and multi-GPU runs (given same total batch size and seed)
- [ ] Gradient values are identical across all GPUs after synchronization
- [ ] Your DDP wrapper produces same results as `torch.nn.parallel.DistributedDataParallel`
- [ ] ZeRO implementation reduces per-GPU optimizer memory by ~N (where N = number of GPUs)
- [ ] Throughput scales near-linearly: 2 GPUs ≈ 1.8-1.95x speedup

### Common Pitfalls

**Deadlocks**: All processes must call collective operations (all-reduce, broadcast) in the same order. If one process takes a different code path, everything hangs.

**Random seed mismanagement**: Model initialization must be identical across GPUs. Data sampling must be different. Use separate random generators.

**Gradient accumulation confusion**: If you're accumulating gradients over multiple micro-batches, only synchronize after the final micro-batch.

**Forgetting to average**: `all_reduce` sums by default. Divide by world size to get the mean.

**Memory not actually saved**: For ZeRO, verify with `torch.cuda.memory_allocated()` that you're actually reducing memory, not just moving things around.

### Verification Script

```python
# Run on each GPU to verify synchronization
def check_sync(model, rank):
    for name, param in model.named_parameters():
        tensor = param.data.clone()
        torch.distributed.all_reduce(tensor)
        tensor /= torch.distributed.get_world_size()
        diff = (param.data - tensor).abs().max().item()
        if diff > 1e-6:
            print(f"RANK {rank}: {name} out of sync by {diff}")
```

## Assessment

By the end of this module, you should be able to:

- **Explain data parallelism**: Describe how multiple GPUs each hold a complete copy of the model and process different portions of the training data simultaneously
- **Explain model parallelism**: Understand when and why you would split the model itself across devices, and how this differs from data parallelism
- **Implement gradient synchronization**: Understand why gradients must be combined across GPUs and how the averaging process keeps model copies identical
- **Describe the all-reduce operation**: Explain what all-reduce does, why it's the critical communication primitive in distributed training, and how it enables every GPU to end up with the same averaged gradients
- **Understand ZeRO optimization**: Explain how ZeRO (Zero Redundancy Optimizer) reduces memory redundancy by partitioning optimizer states, gradients, and parameters across GPUs rather than replicating everything everywhere

Success means you can look at a distributed training setup and understand exactly what each GPU is doing at each moment, which data it's processing, what it's computing, when it's communicating with other GPUs, and why the models stay synchronized despite living on separate devices. You should be able to reason about the tradeoffs: why data parallelism scales well but hits memory limits, why all-reduce becomes a bottleneck at scale, and how ZeRO trades communication for memory savings. When someone mentions "we trained on 64 GPUs with DDP," you should immediately understand the mechanics, not as magic, but as a coordination problem you could implement yourself.

---

# Inference I,  KV-Cache

## Lesson 1: KV-Cache Optimization

You have trained a transformer. You understand attention. You have built the architecture from scratch. Now you want to actually use it, to generate text token by token, and you are about to discover something frustrating.

Generation is slow. Really slow. And the reason why is going to teach you one of the most important practical optimizations in all of LLM deployment.

Let me show you the problem first.

### The Wasteful Loop

When you generate text with a transformer, you do it one token at a time. You feed in your prompt, get a probability distribution over the next token, sample from it, append that token to your sequence, and repeat. Simple enough.

Here is what that looks like in pseudocode:

```python
def generate_naive(model, prompt_tokens, max_new_tokens):
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        # Run the ENTIRE sequence through the model
        logits = model(tokens)
        # Only use the last position's logits
        next_token_logits = logits[-1]
        next_token = sample(next_token_logits)
        tokens = tokens + [next_token]
    return tokens
```

Do you see the problem? Every single time through that loop, you are running the entire sequence through the model. When you generate token 100, you are recomputing the attention for tokens 1 through 99. When you generate token 101, you recompute tokens 1 through 100. All that computation for tokens 1 through 99? You already did it. You are throwing it away and doing it again.

This is quadratic behavior hiding inside your generation loop. And it gets worse as your sequence gets longer.

### What Attention Actually Computes

Let us look at what happens inside attention to understand what we can cache.

For each token position, attention computes three things: a query vector Q, a key vector K, and a value vector V. These come from linear projections of the input embeddings:

```python
Q = input @ W_q  # What am I looking for?
K = input @ W_k  # What do I contain?
V = input @ W_v  # What information do I carry?
```

Then the attention mechanism uses Q to look up relevant information from K and V:

```python
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
```

Here is the key insight. When you are generating token 101, the K and V vectors for tokens 1 through 100 do not change. They were computed from those tokens' embeddings, and those embeddings are fixed. The only new computation needed is:

1. Compute Q, K, V for the new token (position 101)
2. Use the new Q to attend over all previous K and V vectors
3. Append the new K and V to your cache for next time

This is the KV-cache. You store the key and value vectors from all previous positions, and you only compute the new ones for each generation step.

### Building the Cache

Here is what KV-cache implementation looks like:

```python
def generate_with_cache(model, prompt_tokens, max_new_tokens):
    # First pass: process entire prompt, initialize cache
    logits, kv_cache = model(prompt_tokens, kv_cache=None)
    
    tokens = prompt_tokens
    next_token = sample(logits[-1])
    tokens = tokens + [next_token]
    
    for _ in range(max_new_tokens - 1):
        # Subsequent passes: only process the NEW token
        logits, kv_cache = model([next_token], kv_cache=kv_cache)
        next_token = sample(logits[-1])
        tokens = tokens + [next_token]
    
    return tokens
```

The model's forward pass changes based on whether a cache is provided. Without a cache, it processes the full sequence. With a cache, it only processes the new token and looks up previous keys and values from the cache.

Inside each attention layer, the cache management looks like this:

```python
def attention_with_cache(x, kv_cache_layer):
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v
    
    if kv_cache_layer is not None:
        # Concatenate new K, V with cached K, V
        cached_k, cached_v = kv_cache_layer
        k = concatenate([cached_k, k], dim=sequence)
        v = concatenate([cached_v, v], dim=sequence)
    
    # Store updated cache for next iteration
    new_cache = (k, v)
    
    # Attention computation uses full K, V
    attn_weights = softmax(q @ k.T / sqrt(d_k))
    output = attn_weights @ v
    
    return output, new_cache
```

### The Memory-Compute Tradeoff

Nothing is free. KV-cache trades memory for compute. You are storing all those K and V vectors instead of recomputing them.

How much memory? For each layer, you store K and V tensors of shape `(sequence_length, d_model)`. With a model that has 32 layers and d_model of 4096, generating a sequence of 2048 tokens requires:

```
2 (K and V) × 32 (layers) × 2048 (tokens) × 4096 (dimensions) × 2 (bytes for fp16)
= ~1 GB per sequence
```

This is why you hear about High Bandwidth Memory being critical for inference. The cache needs to be read every single generation step, and memory bandwidth often becomes the bottleneck, not compute.

For batched generation with multiple sequences, multiply that memory by your batch size. This creates real Latency Tradeoffs in production systems.

### Common Mistakes

**Forgetting positional encoding updates.** When you process only the new token, you need to give it the correct position index, not position 0. The new token at position 101 needs position encoding 101.

**Cache shape mismatches.** The cache grows by one position each step. Make sure your concatenation happens along the sequence dimension, not the batch or feature dimension.

**Not handling the prompt phase separately.** The first forward pass processes all prompt tokens at once and initializes the cache. Subsequent passes process one token at a time. These are different code paths.

**Ignoring memory limits.** For very long sequences, you may need to implement cache eviction strategies, keeping only recent tokens or using techniques like sliding window attention.

### Why This Matters Beyond Speed

Understanding KV-cache connects to deeper principles about how we deploy these systems. Remember that LLM development involves two main stages: pre-training to build world knowledge, followed by fine-tuning to create a helpful conversational assistant. But there is a third stage that matters enormously for real applications: inference optimization.

Pre-training is extremely expensive but happens once. Inference happens every single time someone uses your model. The efficiency gains from KV-cache mean the difference between a model that costs pennies per query and one that costs dollars. This has real implications for Environmental Concerns and Carbon Footprint of deployed systems.

This is also why the distinction between training and inference is so sharp in practice. During training, caching of vector comparisons is generally not used because embeddings are continuously updated with each optimization step. The cache would become stale immediately. But during inference, everything is frozen, and caching becomes not just possible but essential.

### Success Indicators

You understand KV-cache when you can:

1. Explain why naive generation is quadratic and cached generation is linear
2. Implement a forward pass that correctly handles both the initial prompt and incremental token generation
3. Calculate the memory requirements for caching a given model and sequence length
4. Debug common issues like position encoding mismatches

In the next lesson, we will build the complete incremental generation loop, handling all the edge cases that arise in practice. You will see how this optimization transforms generation from painfully slow to actually usable.

## Lesson 2: KV-Cache Implementation

Let's build a KV-cache from scratch. We're going to write the actual code, make mistakes, fix them, and understand why every piece matters. By the end of this lesson, you'll have a working implementation that makes generation fast.

### The Problem We're Solving

Remember from the last lesson: without caching, generating 100 tokens means computing attention over all previous tokens 100 times. That's roughly 5,050 attention computations instead of 100. The Compute Cost grows quadratically with sequence length, which is brutal.

Here's what we're going to build: a cache that stores the Key and Value tensors from each layer, so we only compute attention for new tokens.

### Starting Simple: A Single-Layer Cache

Let's write the most basic version first:

```python
class KVCache:
    def __init__(self, max_seq_len, num_heads, head_dim):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate memory for K and V
        self.k_cache = torch.zeros(max_seq_len, num_heads, head_dim)
        self.v_cache = torch.zeros(max_seq_len, num_heads, head_dim)
        self.seq_len = 0  # Current position in cache
    
    def update(self, k_new, v_new):
        # k_new, v_new: shape (1, num_heads, head_dim) for single token
        pos = self.seq_len
        self.k_cache[pos] = k_new.squeeze(0)
        self.v_cache[pos] = v_new.squeeze(0)
        self.seq_len += 1
        
        # Return all cached K, V up to current position
        return self.k_cache[:self.seq_len], self.v_cache[:self.seq_len]
```

Stop. Before you move on, trace through what happens when we generate three tokens. First token: cache is empty, we store K1, V1 at position 0. Second token: we store K2, V2 at position 1, return K1:2 and V1:2. Third token: store at position 2, return all three. See how the cache grows?

### Integrating with Attention

Now we need to modify our attention computation to use the cache. The key insight is this: the Query comes from only the new token, but Keys and Values come from the entire cached history.

```python
def attention_with_cache(q, k, v, cache):
    # q: (1, num_heads, head_dim) - just the new token
    # k, v: (1, num_heads, head_dim) - new token's K, V
    
    # Update cache and get full history
    k_full, v_full = cache.update(k, v)
    
    # k_full, v_full: (seq_len, num_heads, head_dim)
    # Compute attention: new query attends to all cached keys
    scores = torch.matmul(q, k_full.transpose(-2, -1))  # (1, num_heads, seq_len)
    scores = scores / math.sqrt(cache.head_dim)
    weights = F.softmax(scores, dim=-1)
    
    output = torch.matmul(weights, v_full)  # (1, num_heads, head_dim)
    return output
```

This is where most people get confused. The Query is small, just one token. But the Keys and Values are the full sequence. You're asking "what should this new token attend to?" and the answer involves everything that came before.

### Multi-Layer Cache

Real transformers have multiple layers. Each layer needs its own K and V cache because each layer's attention patterns are different. Here's how we handle that:

```python
class MultiLayerKVCache:
    def __init__(self, num_layers, max_seq_len, num_heads, head_dim):
        self.caches = [
            KVCache(max_seq_len, num_heads, head_dim) 
            for _ in range(num_layers)
        ]
    
    def update(self, layer_idx, k_new, v_new):
        return self.caches[layer_idx].update(k_new, v_new)
    
    def reset(self):
        for cache in self.caches:
            cache.seq_len = 0
```

The Memory Tricks here matter. We pre-allocate all the memory upfront because dynamic allocation during generation is slow. The tradeoff is that we're reserving High Bandwidth Memory even for sequences shorter than max length.

### The Generation Loop

Now let's put it all together in a generation loop:

```python
def generate_with_cache(model, prompt_tokens, max_new_tokens):
    cache = MultiLayerKVCache(
        num_layers=model.num_layers,
        max_seq_len=len(prompt_tokens) + max_new_tokens,
        num_heads=model.num_heads,
        head_dim=model.head_dim
    )
    
    # Phase 1: Process prompt (prefill)
    # We can process the whole prompt at once
    logits = model.forward_prefill(prompt_tokens, cache)
    next_token = sample(logits[-1])
    generated = [next_token]
    
    # Phase 2: Generate new tokens one at a time
    for _ in range(max_new_tokens - 1):
        logits = model.forward_single(next_token, cache)
        next_token = sample(logits)
        generated.append(next_token)
    
    return generated
```

Notice the two phases. Prefill processes all prompt tokens together, populating the cache. Then incremental decoding generates one token at a time, using and updating the cache. This Inference Asymmetry is fundamental to understanding why generation has different performance characteristics than training.

### Common Mistakes

Here's where people mess up:

**Mistake 1: Forgetting to reset the cache.** If you generate multiple sequences without resetting, you'll get garbage. The cache still has old K, V values mixed in.

```python
# WRONG
output1 = generate_with_cache(model, prompt1, 50)
output2 = generate_with_cache(model, prompt2, 50)  # Cache contaminated!

# RIGHT
cache.reset()
output1 = generate_with_cache(model, prompt1, 50)
cache.reset()
output2 = generate_with_cache(model, prompt2, 50)
```

**Mistake 2: Shape mismatches.** The cache stores (seq_len, num_heads, head_dim) but your model might output (batch, seq_len, num_heads, head_dim). Watch your dimensions carefully.

**Mistake 3: Not handling the prefill phase.** You can't just process prompt tokens one at a time like generation tokens. Well, you can, but it's slow. Prefill should process the whole prompt in parallel.

### Memory Management

The Efficiency Gains from KV-cache come with a cost: memory. For a model with 32 layers, 32 attention heads, 128-dimensional heads, and a sequence length of 2048, you need:

```
2 (K and V) × 32 (layers) × 2048 (seq_len) × 32 (heads) × 128 (head_dim) × 2 (bytes for fp16)
= 1,073,741,824 bytes
= 1 GB per sequence
```

This is why batched generation is tricky. Ten concurrent users means 10 GB just for KV-cache. The Hardware Limitations around memory become the bottleneck, not compute.

### Success Indicators

You know your implementation works when:

1. Generation speed is roughly constant per token, not increasing with sequence length
2. Output quality matches non-cached generation exactly (the cache shouldn't change results, only speed)
3. Memory usage grows linearly with sequence length, not quadratically

Test it:

```python
# Without cache: time should increase with each token
# With cache: time should stay roughly constant
for seq_len in [100, 200, 400, 800]:
    start = time.time()
    generate_with_cache(model, prompt, seq_len)
    print(f"Length {seq_len}: {time.time() - start:.2f}s")
```

### What We Built

You now have a working KV-cache implementation. The Overall Dynamics are simple: store what you've computed, reuse it for future tokens. But the implementation details matter enormously for real-world performance.

In the next lesson, we'll handle batched generation, where multiple sequences share the model but each needs its own cache. That's where things get interesting with memory management and scheduling.

## Lesson 3: Incremental Generation Loop

You have a KV-cache. You understand why it exists. Now let's actually use it to generate text efficiently.

Here's the thing that trips people up: the generation loop in a language model works by iteratively predicting and appending new tokens while maintaining context through the cache. Each iteration does three things. Predict the next token. Append it. Update the cache. That's the whole loop. But getting the details right is where most implementations break.

Let's build it.

### The Basic Loop Structure

```python
def generate_incremental(model, prompt_tokens, max_new_tokens, kv_cache=None):
    """Generate tokens one at a time, using KV-cache for efficiency."""
    
    # Step 1: Process the entire prompt (prefill phase)
    if kv_cache is None:
        kv_cache = create_empty_cache(model)
    
    # Run prompt through model, populate cache
    logits, kv_cache = model.forward(prompt_tokens, kv_cache, use_cache=True)
    
    # Get prediction for first new token (last position's logits)
    next_token_logits = logits[:, -1,:]
    next_token = sample_token(next_token_logits)
    
    generated = [next_token]
    
    # Step 2: Generate remaining tokens one at a time (decode phase)
    for _ in range(max_new_tokens - 1):
        # Only feed the single new token, not the whole sequence
        logits, kv_cache = model.forward(
            next_token.unsqueeze(0),  # Shape: [1, 1]
            kv_cache,
            use_cache=True
        )
        
        next_token_logits = logits[:, -1,:]
        next_token = sample_token(next_token_logits)
        generated.append(next_token)
        
        # Stop if we hit end-of-sequence
        if next_token == EOS_TOKEN:
            break
    
    return generated, kv_cache
```

Two distinct phases here. The prefill phase processes your entire prompt in one forward pass. This is where you pay the cost of attending to all prompt tokens. The decode phase generates new tokens one at a time, and this is where the cache saves you. Instead of reprocessing everything, you process exactly one token per step.

### Why This Matters: The Compute Cost Difference

Without caching, generating 100 tokens from a 1000-token prompt requires:
- Step 1: Process 1001 tokens
- Step 2: Process 1002 tokens
- Step 3: Process 1003 tokens
-.
- Step 100: Process 1100 tokens

Total attention computations scale quadratically. You're recomputing the same key-value pairs over and over.

With caching:
- Prefill: Process 1000 tokens once
- Step 1: Process 1 token, attend to 1000 cached + 1 new
- Step 2: Process 1 token, attend to 1001 cached + 1 new
-.

The compute cost asymmetry is dramatic. You transform quadratic into linear for the generation phase. This is why every production system uses KV-cache.

### The Forward Pass Must Know About the Cache

Your model's forward function needs to handle both modes:

```python
def forward(self, tokens, kv_cache=None, use_cache=False):
    batch_size, seq_len = tokens.shape
    
    # Get embeddings
    x = self.embed(tokens)
    
    # Determine position offset from cache
    if kv_cache is not None and kv_cache[0] is not None:
        cache_len = kv_cache[0][0].shape[2]  # Cached sequence length
        positions = torch.arange(cache_len, cache_len + seq_len)
    else:
        positions = torch.arange(seq_len)
    
    x = x + self.pos_embed(positions)
    
    new_cache = []
    for layer_idx, layer in enumerate(self.layers):
        layer_cache = kv_cache[layer_idx] if kv_cache else None
        x, updated_cache = layer(x, layer_cache, use_cache)
        new_cache.append(updated_cache)
    
    logits = self.output_proj(self.norm(x))
    
    return logits, new_cache if use_cache else None
```

The position offset is critical. When you're generating token 1001, it needs position embedding 1001, not position 0. The cache tells you where you are in the sequence. Getting this wrong produces garbage because the model thinks every new token is at position 0.

### Context Maintenance Through the Cache

The generation loop maintains context through what the cache stores. Each layer's cache contains the key and value projections for all previous tokens. When attention computes for the new token, it queries against this full history.

```python
def attention_with_cache(self, q, k, v, layer_cache, use_cache):
    # q, k, v are projections of current input only
    
    if layer_cache is not None:
        cached_k, cached_v = layer_cache
        # Concatenate new k, v with cached versions
        k = torch.cat([cached_k, k], dim=2)
        v = torch.cat([cached_v, v], dim=2)
    
    # Standard attention computation
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    
    # Causal mask only needed for prefill with multiple tokens
    # During decode, we process one token attending to all previous
    if q.shape[2] > 1:
        scores = scores.masked_fill(self.causal_mask[:,:,:q.shape[2],:k.shape[2]] == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    
    # Return updated cache
    new_cache = (k, v) if use_cache else None
    return output, new_cache
```

This concatenation is the context maintenance mechanism. The new token's queries attend to all cached keys, which represent the full history of the conversation.

### Common Mistakes That Break Everything

**Mistake 1: Forgetting to update positions**

Every new token needs the correct position index. If you always use positions 0, 1, 2... for each decode step, your model loses track of where it is in the sequence.

**Mistake 2: Applying causal mask during single-token decode**

When generating one token at a time, there's nothing to mask. The single query attends to all cached keys. Applying a causal mask here either does nothing or causes shape mismatches.

**Mistake 3: Not handling batch dimension correctly**

During decode, your input shape is [batch_size, 1]. The cache shape is [batch_size, num_heads, cached_len, head_dim]. Concatenation happens on the sequence dimension (dim 2), not the batch dimension.

**Mistake 4: Returning stale cache**

The cache must be updated every step. If you return the old cache instead of the concatenated version, you lose all context beyond the prompt.

### Efficiency Considerations for Production

The basic loop works but leaves performance on the table. Production systems add:

**Stopping conditions**: Check for end-of-sequence token, maximum length, or custom criteria like detecting a complete sentence.

**Temperature and sampling**: The `sample_token` function should support temperature scaling, top-k, and top-p sampling. This affects generation quality dramatically.

**Memory pre-allocation**: Instead of concatenating tensors each step, pre-allocate cache memory for max_length and write into it. Avoids memory fragmentation.

```python
# Pre-allocated cache approach
cache_k = torch.zeros(batch, heads, max_len, head_dim)
cache_v = torch.zeros(batch, heads, max_len, head_dim)
cache_position = 0

# During generation
cache_k[:,:, cache_position:cache_position+1,:] = new_k
cache_v[:,:, cache_position:cache_position+1,:] = new_v
cache_position += 1
```

### Success Indicators

You know your incremental generation loop is working when:

1. Generated text is coherent and continues naturally from the prompt
2. Generation speed is roughly constant per token after prefill
3. Memory usage grows linearly with sequence length, not quadratically
4. Position embeddings are correct (test by generating the same prompt twice, should get similar outputs)
5. The model can generate up to its context window limit without errors

Test with a simple prompt and verify the output makes sense. Then time 100 tokens of generation. If each token takes roughly the same time, your caching is working. If later tokens take progressively longer, something is being recomputed.

Next lesson, we'll extend this to batched generation, where multiple sequences generate simultaneously while sharing compute. That's where the real throughput gains come from.

## Lesson 4: Batched Generation

You have a working KV-cache. You can generate tokens one at a time without recomputing everything. That is good. But here is the problem you will hit immediately in production: you have eight users waiting for responses, and you are processing them one at a time like it is 1995.

Batched generation with cache is how you go from serving one user to serving many users simultaneously. The mechanics are trickier than you might expect, and the implementation details matter enormously for memory and speed.

Let us build it.

### The Core Problem

When you generate text for a single sequence, your cache grows linearly with sequence length. Simple. But when you batch multiple sequences together, you hit a fundamental tension: different sequences have different lengths, and they finish generating at different times.

Consider this scenario. User A sends "Write a haiku about" and User B sends "Explain quantum computing in detail, covering superposition, entanglement, and". User A's prompt is 5 tokens. User B's prompt is 12 tokens. User A will probably finish generating in 20 tokens. User B might need 500.

How do you batch these together efficiently?

### Padding and Attention Masks

The naive approach is padding. You pad shorter sequences to match the longest one, then process them together. Your batch looks like:

```python
# Padded batch (conceptually)
batch = [
    [PAD, PAD, PAD, PAD, PAD, PAD, PAD, "Write", "a", "haiku", "about"],
    ["Explain", "quantum", "computing", "in", "detail", ",", "covering", "superposition", ",", "entanglement", ","]
]
```

This works, but you need attention masks to tell the model which positions are real tokens and which are padding. The attention mechanism should not attend to padding tokens.

```python
def create_attention_mask(sequence_lengths, max_length):
    batch_size = len(sequence_lengths)
    mask = torch.zeros(batch_size, max_length)
    for i, length in enumerate(sequence_lengths):
        mask[i,:length] = 1
    return mask
```

Here is where Hugging Face's text generation implementation has several specific quirks. When generating text for multiple prompts with different lengths, there can be inefficiencies due to padding and attention mask handling that are not immediately obvious from the API. The key-value cache management for transformer models has some implementation-specific behaviors that can affect generation speed and memory usage in edge cases.

### Building the Batched Cache

Your cache now needs an extra dimension. Instead of storing keys and values for one sequence, you store them for all sequences in the batch.

```python
class BatchedKVCache:
    def __init__(self, batch_size, num_layers, num_heads, head_dim, max_length, device):
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Pre-allocate cache: [batch, layers, heads, max_length, head_dim]
        self.keys = torch.zeros(
            batch_size, num_layers, num_heads, max_length, head_dim,
            device=device
        )
        self.values = torch.zeros(
            batch_size, num_layers, num_heads, max_length, head_dim,
            device=device
        )
        
        # Track current position for each sequence
        self.positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    def update(self, layer_idx, new_keys, new_values, sequence_mask=None):
        # new_keys shape: [batch, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = new_keys.shape
        
        for b in range(batch_size):
            if sequence_mask is not None and not sequence_mask[b]:
                continue  # Skip finished sequences
            
            pos = self.positions[b].item()
            self.keys[b, layer_idx,:, pos:pos+seq_len,:] = new_keys[b]
            self.values[b, layer_idx,:, pos:pos+seq_len,:] = new_values[b]
        
        # Update positions for active sequences
        if sequence_mask is not None:
            self.positions += seq_len * sequence_mask.long()
        else:
            self.positions += seq_len
```

### Handling Variable Completion Times

This is where batched generation gets interesting. When User A finishes generating (hits an end token), User B is still going. You have three options:

**Option 1: Wait for everyone.** Keep generating for all sequences until the longest one finishes. Wasteful, but simple.

**Option 2: Remove finished sequences.** When a sequence finishes, remove it from the batch and continue with the rest. This requires reorganizing your cache, which is expensive.

**Option 3: Mask finished sequences.** Keep finished sequences in the batch but skip their computation. They consume memory but not compute.

Option 3 is what most production systems use:

```python
def batched_generate(model, cache, input_ids, max_new_tokens, eos_token_id):
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Track which sequences are still generating
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated = [[] for _ in range(batch_size)]
    
    # Initial forward pass with full prompts
    logits = model.forward_with_cache(input_ids, cache, is_prefill=True)
    
    for step in range(max_new_tokens):
        # Sample next tokens (only for active sequences)
        next_tokens = sample_tokens(logits[:, -1,:])
        
        # Store generated tokens
        for b in range(batch_size):
            if active_mask[b]:
                generated[b].append(next_tokens[b].item())
        
        # Check for completion
        finished = (next_tokens == eos_token_id)
        active_mask = active_mask & ~finished
        
        # Stop if everyone is done
        if not active_mask.any():
            break
        
        # Forward pass with new tokens (incremental)
        # Only compute for active sequences, but maintain batch structure
        logits = model.forward_with_cache(
            next_tokens.unsqueeze(1), 
            cache, 
            is_prefill=False,
            active_mask=active_mask
        )
    
    return generated
```

### Memory Management Considerations

Batched generation multiplies your memory requirements. If a single sequence needs 2GB of cache memory, a batch of 8 needs 16GB. This is the fundamental memory-compute tradeoff you will wrestle with constantly.

Pre-allocation is critical. Allocating memory during generation causes GPU stalls. Allocate your maximum possible cache size upfront:

```python
# Calculate cache memory requirement
cache_memory_per_sequence = (
    num_layers * 2 *  # keys and values
    num_heads * 
    max_sequence_length * 
    head_dim * 
    bytes_per_element
)

total_cache_memory = cache_memory_per_sequence * batch_size
```

For a model like Llama 2 70B with 80 layers, 64 heads, 128 head dim, and 4096 max length in float16, that is about 5GB per sequence. Batch size of 8 means 40GB just for the cache.

### Common Mistakes

**Forgetting the attention mask.** Without proper masking, your model attends to padding tokens and produces garbage. This is subtle because it might work on short sequences and fail on long ones.

**Not tracking sequence positions independently.** Each sequence in your batch has its own position counter. If you use a single counter, your cache indices get corrupted.

**Ignoring finished sequences in loss computation.** If you are doing any kind of online learning or evaluation, make sure finished sequences do not contribute to metrics after they complete.

### Success Indicators

You know batched generation is working correctly when:

1. Generating 8 sequences takes less than 8x the time of generating 1 sequence
2. Shorter sequences finish earlier and do not block longer ones
3. Memory usage scales linearly with batch size, not quadratically
4. Output quality is identical whether you generate in batch or one at a time

Run this test: generate the same prompt 8 times, once as a batch of 8 and once as 8 individual generations with the same random seed. The outputs should be identical. If they differ, your batching logic has a bug.

Next lesson, we handle cache memory pressure when you cannot fit everything in GPU memory.

## Lesson 5: Cache Memory Management

You have built a KV-cache. You have made it work with batched generation. And now you are about to discover why production systems spend enormous engineering effort on something that sounds boring: memory management.

Here is the problem. Your GPU has finite memory. Your cache grows with every token you generate. At some point, these two facts collide. What happens next determines whether your system gracefully handles thousands of concurrent users or crashes spectacularly under load.

Let us build the memory management system.

### The Memory Pressure Problem

Start with a concrete calculation. You have a model with 32 layers, hidden dimension 4096, and you want to support sequences up to 8192 tokens. Each KV-cache entry stores keys and values for every layer.

```python
def calculate_cache_memory(
    num_layers: int,
    hidden_dim: int,
    max_seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2  # float16
) -> int:
    """Calculate memory needed for KV-cache in bytes."""
    # Keys: [batch, layers, seq_len, hidden_dim]
    # Values: [batch, layers, seq_len, hidden_dim]
    keys_memory = batch_size * num_layers * max_seq_len * hidden_dim * dtype_bytes
    values_memory = keys_memory  # Same shape
    return keys_memory + values_memory

# For our example model
memory_per_request = calculate_cache_memory(
    num_layers=32,
    hidden_dim=4096,
    max_seq_len=8192,
    batch_size=1
)
print(f"Memory per request: {memory_per_request / 1e9:.2f} GB")
# Output: Memory per request: 4.29 GB
```

Over four gigabytes per request. If you want to serve 10 concurrent users, that is 43 GB just for the cache. Your model weights take additional memory. Your activations during inference take memory. You see the problem.

### Building a Cache Memory Manager

The solution is a memory manager that tracks allocations, reuses freed memory, and enforces limits. Here is a working implementation:

```python
class CacheMemoryManager:
    def __init__(
        self,
        max_memory_bytes: int,
        num_layers: int,
        hidden_dim: int,
        dtype=torch.float16
    ):
        self.max_memory = max_memory_bytes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        
        # Track active allocations: request_id -> (cache_tensor, seq_len)
        self.active_caches = {}
        self.current_memory = 0
        
        # Pool of freed cache tensors for reuse
        self.free_pool = []  # List of (tensor, max_seq_capacity)
    
    def bytes_per_token(self) -> int:
        """Memory for one token across all layers (K and V)."""
        dtype_bytes = 2 if self.dtype == torch.float16 else 4
        return 2 * self.num_layers * self.hidden_dim * dtype_bytes
    
    def allocate(self, request_id: str, max_seq_len: int) -> dict:
        """Allocate cache for a new request."""
        needed_bytes = max_seq_len * self.bytes_per_token()
        
        # Check if we can reuse from pool
        for i, (tensor, capacity) in enumerate(self.free_pool):
            if capacity >= max_seq_len:
                self.free_pool.pop(i)
                cache = {
                    'keys': tensor['keys'],
                    'values': tensor['values'],
                    'current_len': 0,
                    'max_len': capacity
                }
                self.active_caches[request_id] = cache
                return cache
        
        # Need fresh allocation
        if self.current_memory + needed_bytes > self.max_memory:
            raise MemoryError(
                f"Cannot allocate {needed_bytes} bytes. "
                f"Current: {self.current_memory}, Max: {self.max_memory}"
            )
        
        cache = {
            'keys': torch.zeros(
                self.num_layers, max_seq_len, self.hidden_dim,
                dtype=self.dtype, device='cuda'
            ),
            'values': torch.zeros(
                self.num_layers, max_seq_len, self.hidden_dim,
                dtype=self.dtype, device='cuda'
            ),
            'current_len': 0,
            'max_len': max_seq_len
        }
        
        self.active_caches[request_id] = cache
        self.current_memory += needed_bytes
        return cache
    
    def release(self, request_id: str):
        """Release cache back to pool for reuse."""
        if request_id not in self.active_caches:
            return
        
        cache = self.active_caches.pop(request_id)
        # Zero out for security, add to pool
        cache['keys'].zero_()
        cache['values'].zero_()
        self.free_pool.append((
            {'keys': cache['keys'], 'values': cache['values']},
            cache['max_len']
        ))
    
    def memory_stats(self) -> dict:
        """Return current memory statistics."""
        return {
            'active_requests': len(self.active_caches),
            'pooled_tensors': len(self.free_pool),
            'current_bytes': self.current_memory,
            'max_bytes': self.max_memory,
            'utilization': self.current_memory / self.max_memory
        }
```

### Why Pooling Matters

The free pool is critical. GPU memory allocation is slow. Like, really slow. Every time you call `torch.zeros()` on CUDA, you pay a synchronization cost. During Training, you can absorb this cost because you are doing massive batch computations anyway. During Inference, you are generating one token at a time, and allocation overhead becomes a significant fraction of your total latency.

By pooling freed tensors and reusing them, you avoid repeated allocations. A request finishes, its cache goes to the pool. A new request arrives, it grabs a cache from the pool. No allocation, no synchronization, fast turnaround.

### Handling Memory Pressure

What happens when you run out of memory? You have options, and each involves tradeoffs:

```python
class EvictionPolicy:
    """Strategies for handling memory pressure."""
    
    @staticmethod
    def evict_oldest(manager: CacheMemoryManager, needed_bytes: int) -> bool:
        """Evict oldest requests until we have space."""
        # This is brutal but simple
        sorted_requests = sorted(
            manager.active_caches.items(),
            key=lambda x: x[1].get('created_at', 0)
        )
        
        freed = 0
        for request_id, cache in sorted_requests:
            if freed >= needed_bytes:
                return True
            manager.release(request_id)
            freed += cache['max_len'] * manager.bytes_per_token()
        
        return freed >= needed_bytes
    
    @staticmethod
    def evict_longest(manager: CacheMemoryManager, needed_bytes: int) -> bool:
        """Evict requests with longest sequences first."""
        # Targets memory hogs
        sorted_requests = sorted(
            manager.active_caches.items(),
            key=lambda x: x[1]['current_len'],
            reverse=True
        )
        
        freed = 0
        for request_id, cache in sorted_requests:
            if freed >= needed_bytes:
                return True
            manager.release(request_id)
            freed += cache['max_len'] * manager.bytes_per_token()
        
        return freed >= needed_bytes
```

Eviction is painful. You are killing someone's in-progress generation. Production systems track priority levels, implement graceful degradation, or simply reject new requests when memory is tight. The right choice depends on your application.

### Common Mistakes

**Mistake 1: Not pre-allocating for maximum sequence length.** If you allocate cache incrementally as the sequence grows, you fragment GPU memory. Allocate for your maximum supported length upfront, even if most sequences are shorter.

**Mistake 2: Forgetting to release caches.** Memory leaks in cache management are silent killers. Your system works fine for an hour, then crashes. Always release caches when requests complete, error out, or timeout.

**Mistake 3: Ignoring the Compute Cost of cache operations.** Copying cache data between tensors, zeroing out released caches, managing the pool structure all take time. Profile your memory management code. It should be negligible compared to model forward passes.

### Success Indicators

You know your cache memory management works when:

1. Memory utilization stays stable under sustained load, not growing unboundedly
2. Allocation latency is near zero after warmup because you are reusing pooled tensors
3. You can report accurate memory statistics at any moment
4. Your system gracefully handles the case where memory runs out rather than crashing
5. No memory leaks over hours of operation

### The Bigger Picture

Cache memory management is where the rubber meets the road for production LLM systems. The Architectural Optimization decisions you make here determine your cost per request, your maximum concurrent users, and your tail latency. This is not glamorous work. It is essential work.

The memory-compute tradeoff is fundamental. You can always recompute KV values instead of caching them, trading compute for memory. You can quantize your cache to smaller data types, trading precision for memory. You can limit sequence lengths, trading capability for memory. Every production system navigates these tradeoffs based on their specific constraints.

You have now completed the KV-cache module. You understand why the cache exists, how to implement it, how to extend it for batched generation, and how to manage memory under pressure. This is the foundation for efficient LLM inference. In the next module, we will build on this foundation with quantization techniques that further reduce memory requirements while maintaining generation quality.

## Practice

Build KV-cache implementation, create incremental generation loop, implement batched generation with cache, and manage cache memory.

### Setup

Start with your working transformer from Chapter 05. You'll need:
- A trained (or partially trained) model that can generate coherent-ish text
- Your attention implementation with separate Q, K, V projections
- A simple sampling function (argmax or temperature sampling)

Create a new file `kv_cache.py` for this work.

### Part 1: Basic KV-Cache Structure

Build a cache class that stores K and V tensors for each layer:

```python
class KVCache:
    def __init__(self, num_layers, max_seq_len, num_heads, head_dim):
        # Initialize empty cache storage
        # Shape per layer: (batch_size, num_heads, seq_len, head_dim)
        pass
    
    def update(self, layer_idx, new_k, new_v):
        # Append new K, V to existing cache
        # Return full K, V for attention computation
        pass
    
    def get_seq_len(self):
        # Return current cached sequence length
        pass
```

### Part 2: Incremental Generation Loop

Modify your model's forward pass to accept and update a KV-cache:

```python
def forward(self, tokens, kv_cache=None):
    # If kv_cache exists, only process the NEW tokens
    # Use cached K, V for attention over previous positions
    # Update cache with new K, V
    pass
```

Then build the generation loop:

```python
def generate(model, prompt_tokens, max_new_tokens):
    # 1. Prefill: process entire prompt, populate cache
    # 2. Decode: generate one token at a time using cache
    pass
```

### Part 3: Batched Generation

Extend your implementation to handle multiple sequences simultaneously:
- Different sequences may have different prompt lengths
- Cache must track per-sequence positions
- Handle padding correctly in attention masks

### Part 4: Memory Management

Implement cache memory controls:
- Pre-allocate cache to maximum sequence length
- Track actual vs. allocated memory
- Implement cache clearing/reset for new generations

### Success Criteria

1. **Correctness**: Cached generation produces identical outputs to naive generation (test this explicitly)

2. **Speed**: Generation of 100 tokens should be measurably faster with cache. Time both approaches:
   ```python
   # Should see ~10x or more speedup for longer sequences
   naive_time = time_generation(generate_naive, prompt, 100)
   cached_time = time_generation(generate_cached, prompt, 100)
   ```

3. **Memory**: Cache memory usage should be O(seq_len), not O(seq_len²)

4. **Batching**: Generate 4 sequences simultaneously with different prompt lengths

### Common Pitfalls

**Position encoding mismatch**: When processing only the new token, you must use its actual position (e.g. 101), not position 0. Your position embeddings need the real index.

**Attention mask errors**: The new token should attend to all cached positions plus itself. Get the mask shape wrong and you'll get garbage or crashes.

**Cache dimension ordering**: Decide on (batch, heads, seq, dim) vs (batch, seq, heads, dim) and be consistent. Mismatched shapes cause silent broadcasting bugs.

**Forgetting to detach**: If you store tensors with gradients attached, you'll accumulate a massive computation graph and run out of memory.

**Off-by-one in prefill vs decode**: The prefill phase processes positions 0 to N-1. The first decode step processes position N. Easy to mess up.

### Verification Test

```python
def test_cache_correctness():
    prompt = [1, 2, 3, 4, 5]  # Your token IDs
    
    # Generate without cache
    output_naive = generate_naive(model, prompt, max_new_tokens=20)
    
    # Generate with cache
    output_cached = generate_with_cache(model, prompt, max_new_tokens=20)
    
    assert output_naive == output_cached, "Cache changed model outputs!"
```

Run this test after every change. If cached and naive outputs diverge, you have a bug.

## Assessment

By the end of this module, you should be able to:

- **Explain why naive autoregressive generation is quadratically wasteful** and identify exactly which computations are being redundantly performed at each generation step
- **Implement a KV-cache from scratch** that stores key and value vectors across generation steps and correctly updates them with each new token
- **Describe the memory-compute tradeoff** inherent in KV-caching: you are spending memory to store past computations in order to avoid recomputing them, and you should be able to calculate how much memory a KV-cache requires for a given model size and sequence length
- **Modify a transformer's forward pass** to support both full-sequence processing (for the initial prompt) and incremental single-token processing (for generation with cache)
- **Analyze when KV-cache becomes essential** versus when the overhead might not be worth it, understanding how sequence length, batch size, and model dimensions affect this decision

Success means you can take a working transformer and make its generation dramatically faster without changing its outputs at all. You should be able to look at a generation loop and immediately see the wasted computation. You should be able to calculate the memory footprint of your cache and understand why, at some point, you might run out of GPU memory even though your model fits comfortably. Most importantly, you should understand that this is not a clever trick but a fundamental insight about what information persists across generation steps, an insight that will recur when we discuss quantization, speculative decoding, and serving systems at scale.

---

# Inference II,  Quantization

## Lesson 1: Quantization Fundamentals

You have a 70-billion-parameter model. It takes up 140 gigabytes as float32 weights. You want to run it on your laptop. How do you fit an elephant through a keyhole?

You make the elephant smaller.

This is quantization. And before we get into the theory, let's see what we're actually doing.

### The Problem We're Solving

Remember what Andrej Karpathy said about the Llama 2 70B model: it's really just two files on your file system. The parameters file and the run file. That parameters file is where all the knowledge lives, compressed from internet-scale data into billions of floating point numbers.

Here's the math that hurts: 70 billion parameters times 4 bytes per float32 equals 280 gigabytes. Even at float16, that's 140 gigabytes. Your GPU has maybe 24 gigabytes of VRAM if you're lucky.

So you have three options. Buy more GPUs. Use a smaller model. Or make each parameter take up less space.

Quantization is option three.

### What Quantization Actually Does

Let's start with something concrete. Open a Python shell and type this:

```python
import numpy as np

# A typical weight from a neural network
weight = 0.0234375

# As float32: 4 bytes, 32 bits
print(f"Float32: {np.float32(weight)}")

# As int8: 1 byte, 8 bits  
# But wait - int8 can only represent -128 to 127
# How do we fit 0.0234375 in there?
```

This is the core problem. Float32 can represent numbers like 0.0234375 precisely. Int8 can only represent integers from -128 to 127. How do we map one to the other without losing everything?

The answer is scaling. If your weights range from -1.0 to 1.0, you can map that range to -127 to 127. Multiply by 127, round to the nearest integer, store as int8. When you need the actual value back, divide by 127.

```python
# Simple quantization
scale = 127.0
quantized = np.int8(np.round(weight * scale))
dequantized = quantized / scale

print(f"Original: {weight}")
print(f"Quantized (int8): {quantized}")
print(f"Dequantized: {dequantized}")
print(f"Error: {abs(weight - dequantized)}")
```

Run this. You'll see the error is tiny. We went from 4 bytes to 1 byte and lost almost nothing.

This is the Quantization Benefits in action: 4x storage compression, 4x memory bandwidth reduction, and on hardware that supports it, faster computation.

### Why This Works (And When It Doesn't)

Neural network weights have a beautiful property: they're mostly small numbers clustered around zero. The distribution looks like a bell curve. Most weights are close to zero, with a few outliers.

This matters because quantization error is proportional to the range you're mapping. If your weights span from -0.5 to 0.5, you get finer granularity than if they span from -10 to 10. Neural networks, through the magic of training dynamics and normalization, tend to keep their weights in reasonable ranges.

But here's where people get stuck: not all layers are equal.

Some layers have outliers. A few weights that are way larger than the rest. If you set your scale based on the maximum value, you waste most of your 256 int8 levels on the empty space between the outliers and the bulk of the weights.

This is why calibration matters. You need to look at the actual distribution of weights in each layer and choose your quantization parameters wisely.

### The Discretization Framework

Let me connect this to something you might have seen before. In audio processing, there's a technique called vector quantization where you map continuous embeddings to discrete codes via k-means clustering. The number of discrete codes is critical: too few codes lose important information through averaging, while too many codes introduce noise and reduce robustness.

Quantization for neural networks follows the same principle. INT8 gives you 256 discrete levels. INT4 gives you 16. The question is always: how many levels do you need to preserve the information that matters?

This is discretization enables sequence modeling applied to weights instead of audio. You're taking a continuous space and mapping it to a finite vocabulary of values.

### Storage Compression in Practice

Let's do the math for real models:

| Model | Float32 | Float16 | INT8 | INT4 |
|-------|---------|---------|------|------|
| 7B params | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B params | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B params | 280 GB | 140 GB | 70 GB | 35 GB |

At INT4, that 70B model fits on a single high-end consumer GPU. At INT8, you can run 13B models on a gaming laptop. This is why quantization matters for deployment.

But storage is only half the story. Memory bandwidth is often the bottleneck during inference. When you're generating tokens one at a time, you need to load all those weights from memory for every single token. Smaller weights mean faster loading.

### Common Mistakes

**Mistake 1: Quantizing everything uniformly**

Different layers have different sensitivity to quantization error. The embedding layer and the final output layer are usually more sensitive than middle layers. A good quantization scheme treats layers differently.

**Mistake 2: Ignoring the outliers**

If 99.9% of your weights are between -0.5 and 0.5, but 0.1% are at 2.0, naive quantization will waste precision. Techniques like clipping the outliers or using per-channel scales help.

**Mistake 3: Not validating on real tasks**

Your quantized model might have low reconstruction error on the weights but still perform badly on downstream tasks. Always evaluate on actual benchmarks, not just weight statistics.

**Mistake 4: Forgetting about activations**

We've been talking about weight quantization, but activations during inference also take up memory. Quantizing activations is harder because their range changes with each input.

### Success Indicators

You know your quantization is working when:

1. Model size drops by the expected factor (4x for INT8, 8x for INT4)
2. Inference speed improves (especially on hardware with INT8/INT4 support)
3. Perplexity or task accuracy drops by less than 1-2%
4. Generation quality remains coherent for multi-paragraph outputs

The last point is subtle. Quantization errors can compound during autoregressive generation. A model that looks fine on single-token prediction might drift badly over long sequences.

### Where We're Going

This lesson gave you the intuition. In the next lessons, we'll build actual quantization code. You'll implement post-training quantization where you take a trained model and compress it. Then quantization-aware training where the model learns to be robust to quantization during training. Then we'll go deep on INT8 and INT4 implementations.

By the end of this module, you'll understand why Knowledge Distillation Advantages sometimes beat raw quantization, when Weight Pruning Opportunities complement quantization, and how Parameter Sparsity and Conditional Computation create different tradeoffs.

The goal is simple: take models that need server farms and make them run on laptops. Take inference that costs dollars and make it cost pennies. Make the elephant fit through the keyhole.

Let's get to it.

## Lesson 2: Post-Training Quantization

You have a trained model. It works. The parameters are sitting there as float16 numbers, two bytes each. Now you want to make it smaller and faster without retraining anything. This is post-training quantization, and it is the most practical path to deploying models on constrained hardware.

Let me show you what we are actually doing here.

### The Basic Operation

Take a weight matrix from your transformer. It contains floating point values distributed across some range, maybe -0.5 to 0.5 for a well-trained layer. We want to represent these values using integers instead of floats.

```python
import torch

def quantize_tensor(tensor, bits=8):
    # Find the range of values
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Calculate scale and zero point
    qmin = 0
    qmax = 2**bits - 1
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    zero_point = torch.clamp(torch.round(zero_point), qmin, qmax).int()
    
    # Quantize
    q_tensor = torch.clamp(
        torch.round(tensor / scale + zero_point), 
        qmin, 
        qmax
    ).to(torch.uint8)
    
    return q_tensor, scale, zero_point

def dequantize_tensor(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)
```

Run this on an actual weight matrix. Create one, quantize it, dequantize it, measure the error:

```python
# Simulate a weight matrix from a trained model
weights = torch.randn(512, 512) * 0.1

# Quantize to INT8
q_weights, scale, zp = quantize_tensor(weights, bits=8)

# Dequantize back
reconstructed = dequantize_tensor(q_weights, scale, zp)

# Measure error
error = (weights - reconstructed).abs().mean()
print(f"Mean absolute error: {error:.6f}")
print(f"Storage: {weights.numel() * 2} bytes -> {q_weights.numel()} bytes")
```

You will see something like 0.0002 mean absolute error. The Storage Compression is 2x because we went from 2 bytes per parameter (float16) to 1 byte per parameter (INT8). The Quantization Benefits are immediate and measurable.

### Why This Works

The insight is that neural network weights do not need full floating point precision. They cluster in relatively narrow ranges. When we trained the model, gradient descent found solutions where small perturbations to weights do not catastrophically change outputs. This robustness to noise is exactly what makes quantization possible.

Think about what the expert said about model training versus inference: "So to obtain the parameters, basically the model training, as we call it, is a lot more involved than model inference." During training we need precise gradients flowing backward. During inference we just need forward passes that produce reasonable outputs. The precision requirements are fundamentally different.

### Calibration: The Critical Step

Here is where people get stuck. Naive quantization like I showed above uses the min and max of the weight tensor to set the scale. But what if you have outliers? One weight at -5.0 when everything else is between -0.5 and 0.5 will waste most of your quantization range.

Calibration means running representative data through your model and collecting statistics about activation ranges, not just weight ranges.

```python
class CalibrationObserver:
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.running_mean = 0
        self.running_var = 0
        self.count = 0
    
    def observe(self, tensor):
        self.min_val = min(self.min_val, tensor.min().item())
        self.max_val = max(self.max_val, tensor.max().item())
        
        # Track statistics for percentile-based clipping
        batch_mean = tensor.mean().item()
        batch_var = tensor.var().item()
        
        self.count += 1
        delta = batch_mean - self.running_mean
        self.running_mean += delta / self.count
        self.running_var += delta * (batch_mean - self.running_mean)
    
    def get_range(self, percentile=0.999):
        # Use statistics to clip outliers
        std = (self.running_var / max(1, self.count - 1)) ** 0.5
        clipped_min = self.running_mean - 3 * std
        clipped_max = self.running_mean + 3 * std
        return max(self.min_val, clipped_min), min(self.max_val, clipped_max)
```

You run maybe 100-1000 samples through your model with these observers attached to each layer. Then you use the collected statistics to determine quantization parameters. This is why the Data Requirements for post-training quantization are minimal compared to full training, but not zero.

### INT8 vs INT4

INT8 gives you 256 possible values per weight. INT4 gives you 16. The math for going lower:

```python
def quantize_int4(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    scale = (max_val - min_val) / 15  # 4 bits = 16 values (0-15)
    zero_point = torch.round(-min_val / scale).clamp(0, 15).int()
    
    q_tensor = torch.clamp(
        torch.round(tensor / scale + zero_point),
        0, 15
    ).to(torch.uint8)
    
    return q_tensor, scale, zero_point
```

INT4 is aggressive. You lose more information. But the Storage Compression goes from 2x to 4x compared to float16. For a 70 billion parameter model like Llama 2, that is the difference between needing 140GB and needing 35GB. Suddenly the model fits on consumer hardware.

The Training Dynamics during the original model creation affect how well post-training quantization works. Models trained with regularization, with well-behaved weight distributions, quantize better than models with many outliers.

### Common Mistakes

**Mistake 1: Quantizing everything uniformly.** Different layers have different sensitivity. The first and last layers of transformers often need higher precision. Attention layers behave differently than feed-forward layers.

**Mistake 2: Ignoring activation quantization.** Weights are only half the story. The intermediate activations during inference also need quantization for full INT8 inference speedups. Calibration must capture activation ranges too.

**Mistake 3: Not validating on real tasks.** A model can have low reconstruction error on weights but still perform badly on downstream tasks. Always measure task performance, not just weight error.

### Putting It Together

Here is the full post-training quantization pipeline:

```python
def post_training_quantize(model, calibration_loader, bits=8):
    # Step 1: Attach observers
    observers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            observers[name] = CalibrationObserver()
    
    # Step 2: Run calibration data
    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            _ = model(batch)
            for name, module in model.named_modules():
                if name in observers and hasattr(module, 'weight'):
                    observers[name].observe(module.weight)
    
    # Step 3: Quantize weights using calibrated ranges
    quantized_state = {}
    for name, module in model.named_modules():
        if name in observers and hasattr(module, 'weight'):
            min_val, max_val = observers[name].get_range()
            q_weight, scale, zp = quantize_with_range(
                module.weight, min_val, max_val, bits
            )
            quantized_state[name] = {
                'weight': q_weight,
                'scale': scale,
                'zero_point': zp
            }
    
    return quantized_state
```

### Success Indicators

You know post-training quantization worked when:

1. Model size decreased by expected factor (2x for INT8, 4x for INT4)
2. Perplexity increased by less than 1% for INT8, less than 5% for INT4
3. Downstream task accuracy within 1-2% of original
4. Inference actually runs faster on target hardware

The Knowledge Distillation Advantages become relevant when post-training quantization causes too much degradation. You can use the original model as a teacher to guide a smaller quantized student. But that is no longer "post-training" quantization, that is training again.

Post-training quantization is your first tool. It is fast, requires minimal data, and often works well enough. When it does not work well enough, you move to quantization-aware training, which we cover next.

## Lesson 3: Quantization-Aware Training

In the last lesson, you took a trained model and quantized it after the fact. Post-training quantization is fast and simple, but you probably noticed something: the model got worse. Sometimes a little worse, sometimes a lot worse. The weights weren't expecting to be crammed into INT8 or INT4, so they complained.

What if we could train the model knowing it will be quantized? What if the weights could learn to live comfortably in low precision from the start?

That's quantization-aware training. Let's build it.

### The Core Problem

Here's what happens with post-training quantization. You train a model in FP32, where weights can be any value like 0.0847291 or -1.2938475. Then you come along and say "sorry, you can only be one of 256 values now" (for INT8) or "you can only be one of 16 values" (for INT4). The weight has to round to the nearest allowed value, and that rounding introduces error.

The Quantization Benefits are clear: smaller models, faster inference, lower memory. But the cost is accuracy. Post-training quantization just accepts this cost.

Quantization-aware training does something clever. During training, we simulate what quantization will do to the weights. The model experiences the rounding errors while it's still learning, so it can adjust its weights to be robust to those errors.

### Building the Fake Quantization Operation

The trick is something called "fake quantization" or "simulated quantization." During the forward pass, we quantize the weights just like we would for inference. During the backward pass, we pretend we didn't.

Here's why that works. Quantization is a step function, which has zero gradient almost everywhere. If we tried to backpropagate through actual quantization, gradients would vanish. So we use the "straight-through estimator": forward pass uses quantized values, backward pass uses the original gradient as if no quantization happened.

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, num_bits=8):
        # Quantize: round to nearest integer level
        qmin = 0
        qmax = 2**num_bits - 1
        
        x_scaled = x / scale + zero_point
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_rounded = torch.round(x_clamped)
        
        # Dequantize back to float
        x_fake_quant = (x_rounded - zero_point) * scale
        return x_fake_quant
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output, None, None, None
```

This is the heart of quantization-aware training. The forward pass sees quantized weights, so the loss reflects quantization error. The backward pass flows gradients through, so the optimizer can still update weights.

### The Training Loop

Now we integrate this into actual training. Every layer that will be quantized at inference time needs fake quantization during training.

```python
class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, num_bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.num_bits = num_bits
        
        # Learnable quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_zero_point = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Apply fake quantization to weights
        w_quant = FakeQuantize.apply(
            self.weight, 
            self.weight_scale, 
            self.weight_zero_point,
            self.num_bits
        )
        return F.linear(x, w_quant, self.bias)
```

Notice that scale and zero_point are learnable parameters. The model doesn't just learn weights that survive quantization, it learns the best quantization parameters for those weights. This is where quantization-aware training really shines.

### Calibration During Training

Remember from the last lesson that Calibration means finding the right scale and zero_point for each layer. In post-training quantization, we ran calibration data through the model once. In quantization-aware training, calibration happens continuously.

The scale and zero_point parameters get updated by the optimizer just like regular weights. Early in training, they might be way off. That's fine. The model learns to adjust them as it trains.

A common approach is to initialize these parameters using a few batches of data, similar to post-training calibration, then let training refine them:

```python
def initialize_qat_params(model, calibration_loader):
    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            # Run forward pass to collect activation statistics
            _ = model(batch)
            break  # Just need one batch for initialization
    
    for module in model.modules():
        if isinstance(module, QATLinear):
            # Initialize scale based on weight range
            w_min, w_max = module.weight.min(), module.weight.max()
            module.weight_scale.data = (w_max - w_min) / (2**module.num_bits - 1)
            module.weight_zero_point.data = -w_min / module.weight_scale.data
```

### INT8 vs INT4: The Trade-off

For INT8 inference, quantization-aware training usually recovers most of the accuracy lost by post-training quantization. You might lose 0.1% accuracy instead of 1%. The model learns to keep weights near the quantization grid points.

For INT4 inference, things get harder. With only 16 possible values per weight, the quantization grid is coarse. The model has to work much harder to maintain accuracy. This is where the principle from our source materials becomes critical: "too few codes lose important information through averaging."

When targeting INT4, you often need to train longer and use techniques like mixed precision, where some sensitive layers stay at INT8 while others go to INT4.

### Common Mistakes

**Starting QAT too early.** If you apply fake quantization from the very first batch, the model struggles to learn anything because the quantization noise dominates. Better approach: train normally for a while, then enable QAT for the final portion of training. Think of it as letting the model find a good region of weight space first, then fine-tuning it to be quantization-friendly.

**Forgetting to quantize activations.** We focused on weight quantization, but activations also need quantization for full INT8 inference. Add fake quantization after each activation function too.

**Using the same num_bits everywhere.** Some layers are more sensitive than others. The first and last layers often need higher precision. Experiment with mixed-precision QAT where different layers use different bit widths.

### When to Use QAT vs Post-Training Quantization

Post-training quantization is fast. You can quantize a model in minutes. Use it when:
- You're targeting INT8 and can tolerate small accuracy drops
- You don't have compute budget for retraining
- You're experimenting and need quick iterations

Quantization-aware training takes longer, since you're essentially retraining the model. Use it when:
- You're targeting INT4 or lower
- Accuracy is critical and you can't afford any degradation
- You have the compute budget for training

Remember from our source materials: model inference is computationally cheap and can be done locally, but training is extremely expensive and requires massive computational resources. QAT sits in between, cheaper than training from scratch but more expensive than post-training quantization.

### Success Indicators

You know QAT is working when:
- The training loss with fake quantization is only slightly higher than without
- The learned scale parameters stabilize rather than oscillating wildly
- Accuracy on validation data matches or exceeds post-training quantized accuracy
- When you actually deploy with INT8/INT4, performance matches what you saw during training

The gap between "training with fake quantization" and "inference with real quantization" should be minimal. If there's a big gap, something is wrong with your fake quantization implementation.

### What's Next

You've now built both post-training quantization and quantization-aware training. In the next lesson, we'll look at INT8 and INT4 implementations in detail, including the actual inference kernels that run on hardware. The fake quantization you built here simulates what those kernels do, but seeing the real thing will make the connection concrete.

## Lesson 4: INT8 and INT4 Implementations

Alright, so we've covered the theory of quantization and we've built post-training quantization and quantization-aware training. Now we're going to actually implement the two precision levels that matter most in practice: INT8 and INT4. This is where the rubber meets the road.

Let me just say upfront: INT8 is the workhorse of production inference. INT4 is where things get interesting and a bit dangerous. Both require you to understand what's actually happening to your numbers, and that's what we're going to build today.

### Why These Two Precision Levels

So you might be wondering, why INT8 and INT4 specifically? Why not INT6 or INT5?

The answer is Hardware Optimization. Modern GPUs and specialized inference chips have dedicated INT8 tensor cores. They're designed to blast through INT8 matrix multiplications at ridiculous speeds. INT4 is trickier because hardware support is less universal, but the memory savings are so dramatic that people have figured out clever ways to make it work.

Remember from our earlier discussion that the Llama 2 70B model is really just two files on your file system: the parameters file and the run file. When you're serving millions of requests, that parameters file needs to fit in memory and move through your hardware fast. INT8 cuts your memory in half compared to FP16. INT4 cuts it in half again.

### Building INT8 Inference

Let's start with INT8 because it's more straightforward. The core idea is that we're mapping floating point values to 256 discrete levels (since 2^8 = 256).

```python
import numpy as np

class INT8Linear:
    def __init__(self, weight_fp32, bias_fp32=None):
        # Find the scale factor
        self.weight_scale = np.abs(weight_fp32).max() / 127.0
        
        # Quantize weights to INT8
        self.weight_int8 = np.round(
            weight_fp32 / self.weight_scale
        ).astype(np.int8)
        
        # Keep bias in higher precision
        self.bias = bias_fp32
    
    def forward(self, x_fp32):
        # Quantize input on the fly
        x_scale = np.abs(x_fp32).max() / 127.0
        x_int8 = np.round(x_fp32 / x_scale).astype(np.int8)
        
        # Integer matrix multiplication
        # This is where the speed comes from
        out_int32 = np.dot(x_int8.astype(np.int32), 
                          self.weight_int8.T.astype(np.int32))
        
        # Dequantize back to float
        out_fp32 = out_int32 * (x_scale * self.weight_scale)
        
        if self.bias is not None:
            out_fp32 += self.bias
            
        return out_fp32
```

Notice something important here: we're doing the matrix multiplication in INT32. Why? Because when you multiply two INT8 numbers, you can get a result up to 127 * 127 = 16129, which overflows INT8. The accumulation of many such products definitely overflows. So the actual computation happens in INT32, but the memory storage and memory bandwidth use INT8.

This is the key insight for Hardware Optimization: the bottleneck in transformer inference is often memory bandwidth, not compute. Moving INT8 values instead of FP16 values doubles your effective memory bandwidth.

### The INT4 Challenge

INT4 is where things get spicy. You only have 16 discrete values to represent your weights. That's... not a lot. The number of discrete codes is critical here: too few codes lose important information through averaging.

Here's a basic INT4 implementation:

```python
class INT4Linear:
    def __init__(self, weight_fp32, group_size=128):
        self.group_size = group_size
        
        # Reshape for group-wise quantization
        # This is crucial for INT4 to work at all
        out_features, in_features = weight_fp32.shape
        num_groups = in_features // group_size
        
        weight_grouped = weight_fp32.reshape(
            out_features, num_groups, group_size
        )
        
        # Per-group scales
        self.scales = np.abs(weight_grouped).max(axis=2) / 7.0
        
        # Quantize each group independently
        weight_scaled = weight_grouped / self.scales[:,:, np.newaxis]
        self.weight_int4 = np.round(weight_scaled).clip(-8, 7).astype(np.int8)
        
        # Pack two INT4 values into one INT8 for storage
        # This is where the memory savings actually happen
        self.weight_packed = self._pack_int4(self.weight_int4)
    
    def _pack_int4(self, data):
        # Take pairs of INT4 values and pack into INT8
        flat = data.reshape(-1)
        even = flat[0::2] & 0x0F
        odd = (flat[1::2] & 0x0F) << 4
        return (even | odd).astype(np.uint8)
    
    def _unpack_int4(self, packed):
        even = (packed & 0x0F).astype(np.int8)
        odd = ((packed >> 4) & 0x0F).astype(np.int8)
        # Sign extend
        even = np.where(even > 7, even - 16, even)
        odd = np.where(odd > 7, odd - 16, odd)
        return np.stack([even, odd], axis=-1).reshape(-1)
```

The group_size parameter is doing a lot of work here. Instead of one scale for the entire weight matrix, we use one scale per group of 128 weights. This is a form of Conditional Computation in the sense that different parts of the weight matrix get different treatment.

### Calibration for Production

Here's where people get stuck: the scale factors matter enormously. Bad scales mean bad accuracy. Good scales require seeing representative data.

```python
class CalibratedQuantizer:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.max_val = 2 ** (num_bits - 1) - 1
        self.activation_ranges = {}
    
    def calibrate(self, model, calibration_data):
        """Run calibration data through model to find activation ranges"""
        
        def hook_fn(name):
            def fn(module, input, output):
                if name not in self.activation_ranges:
                    self.activation_ranges[name] = {
                        'min': float('inf'),
                        'max': float('-inf')
                    }
                
                # Track running min/max
                self.activation_ranges[name]['min'] = min(
                    self.activation_ranges[name]['min'],
                    output.min().item()
                )
                self.activation_ranges[name]['max'] = max(
                    self.activation_ranges[name]['max'],
                    output.max().item()
                )
            return fn
        
        # Register hooks on all layers
        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Run calibration data
        for batch in calibration_data:
            model(batch)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return self.activation_ranges
```

This calibration process is essential. You're basically asking: "What range of values will this layer actually see in practice?" Then you set your quantization scale to cover that range efficiently.

### Common Mistakes

First mistake: using the same scale for weights and activations. Weights are static, you can calibrate them once. Activations change with every input. They need dynamic scaling or careful calibration.

Second mistake: ignoring outliers. Some activations have extreme outliers that blow up your scale factor, wasting precision on rare values. Techniques like clipping the top 0.1% of values before computing scales can help dramatically.

Third mistake: quantizing everything uniformly. The first and last layers of transformers are often more sensitive to quantization error. Keep them in higher precision. This is a form of Memory Optimization where you spend your precision budget where it matters most.

### Success Indicators

How do you know your quantization is working? Run your quantized model on a validation set and compare perplexity to the FP16 baseline. For INT8, you should see less than 0.5% perplexity increase. For INT4, you might see 1-2% increase, maybe more depending on the model.

Also check generation quality manually. Sometimes perplexity looks fine but the model starts repeating itself or producing grammatically weird outputs. Trust your eyes as well as your metrics.

The goal here is that your quantized model becomes like what was described: the parameters file shrinks dramatically while the run file stays basically the same. You're compressing the knowledge, not changing how it's used.

## Lesson 5: Calibration and Error Analysis

You have quantized your model. The weights went from 32-bit floats down to 8 bits, maybe even 4 bits. The model still runs. It still generates text. But here is the question that keeps practitioners up at night: how much did you actually break?

This lesson is about measuring the damage and fixing what you can.

### The Calibration Problem

Let me show you something that will make the problem concrete. Take your INT8 quantized model and run it on a batch of inputs. Compare the activations at each layer to what the full-precision model produces. You will see drift. Small errors in early layers compound into larger errors in later layers. By the time you reach the output, your probability distribution over next tokens looks different from what the original model would have produced.

This is the calibration problem. Your quantized model is miscalibrated. It assigns probabilities that do not match the true model's beliefs.

Here is a simple diagnostic:

```python
def measure_calibration_drift(fp_model, quant_model, calibration_data):
    """Compare activation statistics between models."""
    drift_per_layer = {}
    
    fp_activations = {}
    quant_activations = {}
    
    # Hook to capture activations
    def make_hook(storage, name):
        def hook(module, input, output):
            storage[name] = output.detach()
        return hook
    
    # Register hooks on both models
    for name, module in fp_model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(make_hook(fp_activations, name))
    
    for name, module in quant_model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(make_hook(quant_activations, name))
    
    # Run calibration data through both
    for batch in calibration_data:
        with torch.no_grad():
            fp_model(batch)
            quant_model(batch)
        
        for name in fp_activations:
            fp_act = fp_activations[name]
            q_act = quant_activations[name]
            
            # Mean squared error between activations
            mse = ((fp_act - q_act) ** 2).mean().item()
            
            if name not in drift_per_layer:
                drift_per_layer[name] = []
            drift_per_layer[name].append(mse)
    
    return {k: sum(v)/len(v) for k, v in drift_per_layer.items()}
```

Run this. You will see which layers suffer the most from quantization. This is your roadmap for where to focus calibration efforts.

### Why Calibration Data Matters

The number of discrete codes in vector quantization is critical: too few codes lose important information through averaging, while too many codes introduce noise and reduce robustness. This principle from the source materials applies directly to weight quantization. Your scale factors and zero points are only as good as the data you used to compute them.

Bad calibration data leads to bad scale factors. If your calibration set does not cover the actual distribution of inputs your model will see, your quantization ranges will be wrong. Activations will clip. Information will be lost.

Here is how to build a proper calibration set:

```python
def build_calibration_set(tokenizer, num_samples=512, seq_length=128):
    """Build diverse calibration data."""
    calibration_texts = []
    
    # Mix of different content types
    sources = [
        load_wikipedia_samples(num_samples // 4),
        load_code_samples(num_samples // 4),
        load_conversation_samples(num_samples // 4),
        load_technical_samples(num_samples // 4),
    ]
    
    for source in sources:
        for text in source:
            tokens = tokenizer.encode(text)[:seq_length]
            if len(tokens) == seq_length:
                calibration_texts.append(tokens)
    
    return torch.tensor(calibration_texts)
```

The key insight: diversity matters more than volume. 512 diverse samples will calibrate better than 10,000 samples from a single domain.

### Error Analysis with Diagnostic Classifiers

Now we get systematic. Diagnostic Classifiers let you probe what information survives quantization. The idea is simple: train a small classifier on the activations to predict some property, then compare performance between the full-precision and quantized models.

```python
class DiagnosticProbe(nn.Module):
    """Linear probe for activation analysis."""
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, activations):
        # Pool over sequence dimension
        pooled = activations.mean(dim=1)
        return self.classifier(pooled)

def compute_probe_accuracy(model, probe, eval_data, labels):
    """Measure how much task-relevant information survives."""
    correct = 0
    total = 0
    
    for batch, label in zip(eval_data, labels):
        with torch.no_grad():
            # Get activations from a specific layer
            activations = get_layer_activations(model, batch, layer_idx=-4)
            predictions = probe(activations).argmax(dim=-1)
            correct += (predictions == label).sum().item()
            total += len(label)
    
    return correct / total
```

Train the probe on full-precision activations. Then evaluate it on quantized activations without retraining. The accuracy drop tells you how much task-relevant information you lost.

### Representational Similarity Analysis

Representational Similarity Analysis gives you a different view. Instead of asking "can we decode information X from the activations," it asks "do the two models represent inputs in the same geometric structure?"

```python
def compute_rsa(activations_1, activations_2):
    """Compare representational geometry between two models."""
    # Compute pairwise distances in each space
    def pairwise_distances(acts):
        acts_flat = acts.reshape(acts.shape[0], -1)
        dists = torch.cdist(acts_flat, acts_flat)
        return dists
    
    dists_1 = pairwise_distances(activations_1)
    dists_2 = pairwise_distances(activations_2)
    
    # Flatten upper triangle
    triu_idx = torch.triu_indices(dists_1.shape[0], dists_1.shape[1], offset=1)
    vec_1 = dists_1[triu_idx[0], triu_idx[1]]
    vec_2 = dists_2[triu_idx[0], triu_idx[1]]
    
    # Spearman correlation
    correlation = spearman_correlation(vec_1, vec_2)
    return correlation
```

An RSA correlation above 0.95 means your quantized model preserves the essential geometry. Below 0.90 and you should worry.

### Common Mistakes

First mistake: calibrating on random noise or synthetic data. Your model has never seen random noise during training. Its activations on random inputs tell you nothing useful about how it behaves on real inputs.

Second mistake: using too little calibration data. You need enough samples to estimate the true range of activations. With INT8, you have 256 bins. You need enough data to populate those bins meaningfully.

Third mistake: ignoring outliers. A single outlier activation can blow up your scale factor and waste most of your quantization range on values that almost never occur. Use percentile clipping:

```python
def compute_robust_scale(activations, percentile=99.9):
    """Clip outliers before computing scale."""
    abs_max = torch.quantile(activations.abs(), percentile / 100)
    return abs_max / 127  # For INT8
```

### Success Indicators

How do you know your calibration worked? Here are the metrics that matter:

1. Perplexity increase under 5% relative to full precision
2. Probe Accuracy drop under 2% on downstream tasks
3. RSA correlation above 0.95 between full and quantized activations
4. No layer showing more than 10x the average activation drift

If you hit these numbers, ship it. Your quantized model is ready for production.

### The Self-Consistency Check

One final technique. Self-consistency is naturally self-calibrated: higher consistency across samples indicates higher accuracy. Run the same prompt through your quantized model multiple times with temperature sampling. If the outputs are consistent, your model is well-calibrated. If they diverge wildly, something is wrong.

```python
def self_consistency_check(model, prompt, num_samples=10, temperature=0.7):
    """Check output consistency as calibration signal."""
    outputs = []
    for _ in range(num_samples):
        output = model.generate(prompt, temperature=temperature, max_tokens=50)
        outputs.append(output)
    
    # Measure agreement
    unique_outputs = len(set(outputs))
    consistency = 1 - (unique_outputs - 1) / (num_samples - 1)
    return consistency
```

High consistency means your quantized model has stable beliefs. Low consistency means the quantization noise is overwhelming the signal.

You now have the tools to quantize responsibly. Measure everything. Trust nothing until you have verified it. The difference between a quantized model that works and one that subtly fails is calibration.

## Practice

Build post-training quantization, implement quantization-aware training, create INT8 and INT4 implementations, and perform calibration and error analysis.

### Setup

You'll need a trained model to quantize. Use your transformer from Chapter 05, or load a small pretrained model:

```python
import torch
import torch.nn as nn
import numpy as np

# Start with a simple linear layer to understand the mechanics
layer = nn.Linear(256, 256)
weights = layer.weight.data.clone()

print(f"Weight shape: {weights.shape}")
print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"Memory (float32): {weights.numel() * 4} bytes")
```

### Part 1: Post-Training Quantization (PTQ)

Implement symmetric and asymmetric quantization:

```python
def quantize_symmetric(tensor, num_bits=8):
    """Map float tensor to int range [-2^(n-1)+1, 2^(n-1)-1]"""
    # Your implementation:
    # 1. Find the absolute maximum value
    # 2. Compute scale = max_val / (2^(n-1) - 1)
    # 3. Quantize: round(tensor / scale)
    # 4. Return quantized tensor and scale
    pass

def quantize_asymmetric(tensor, num_bits=8):
    """Map float tensor to int range [0, 2^n - 1] with zero-point"""
    # Your implementation:
    # 1. Find min and max values
    # 2. Compute scale = (max - min) / (2^n - 1)
    # 3. Compute zero_point = round(-min / scale)
    # 4. Quantize: round(tensor / scale) + zero_point
    # 5. Return quantized tensor, scale, and zero_point
    pass
```

### Part 2: INT8 and INT4 Implementations

Build a quantized linear layer that stores weights in low precision but computes in float:

```python
class QuantizedLinear(nn.Module):
    def __init__(self, original_layer, num_bits=8):
        super().__init__()
        # Store quantized weights and scales
        # Implement forward pass with dequantization
        pass
    
    def forward(self, x):
        # Dequantize weights, compute matmul, return result
        pass
```

Extend to INT4. The challenge: PyTorch doesn't have int4. Pack two int4 values into one int8:

```python
def pack_int4(tensor_int4):
    """Pack two int4 values into one int8"""
    pass

def unpack_int4(tensor_int8):
    """Unpack int8 back to two int4 values"""
    pass
```

### Part 3: Calibration

Run sample data through your model to find optimal scales per layer:

```python
def calibrate_model(model, calibration_data, num_bits=8):
    """
    Run calibration data through model.
    For each layer, track min/max activations.
    Return optimal scales for each layer.
    """
    pass
```

Try different calibration strategies:
- Min-max: use observed min/max
- Percentile: use 99.9th percentile to ignore outliers
- MSE: find scale that minimizes reconstruction error

### Part 4: Quantization-Aware Training (QAT)

Implement fake quantization for training:

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, num_bits):
        # Quantize then immediately dequantize
        # This simulates quantization error during training
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unchanged
        pass
```

Fine-tune your model with fake quantization inserted after weights and activations.

### Part 5: Error Analysis

Measure the damage:

```python
def analyze_quantization_error(original_model, quantized_model, test_data):
    """
    Compare outputs layer-by-layer.
    Report: MSE, max error, cosine similarity.
    Identify which layers suffer most.
    """
    pass
```

### Success Criteria

1. **PTQ works**: INT8 model produces similar outputs to float32 (cosine similarity > 0.99)
2. **Memory reduced**: Verify 4x reduction for INT8, 8x for INT4
3. **INT4 functional**: Model still generates coherent text (some quality loss acceptable)
4. **Calibration improves accuracy**: Calibrated quantization beats naive min-max
5. **QAT recovers quality**: Fine-tuned quantized model approaches float32 performance
6. **Error analysis complete**: You can identify which layers are most sensitive

### Common Pitfalls

**Outliers destroy everything.** One weight at 10.0 when others are at 0.01 wastes your entire dynamic range. Use per-channel quantization or clip outliers.

**Activations matter too.** Quantizing only weights gets you halfway. For full INT8 inference, you need to quantize activations, which requires calibration data.

**INT4 is fragile.** Only 16 possible values. Some layers (especially the first and last) need higher precision. Mixed-precision quantization keeps sensitive layers at INT8.

**Don't quantize LayerNorm.** The running statistics need float precision. Same for the final output projection in many cases.

**Symmetric vs asymmetric.** Weights are usually symmetric around zero, use symmetric. Activations after ReLU are all positive, use asymmetric.

### Stretch Goals

- Implement GPTQ (optimal brain quantization for each layer)
- Add per-channel quantization for weights
- Build a mixed-precision quantizer that automatically chooses bits per layer
- Measure actual inference speedup on your hardware

## Assessment

By the end of this module, you should be able to:

- **Explain quantization fundamentals**: Describe why quantization works for neural networks, how scaling maps floating-point ranges to integer ranges, and calculate the storage savings from different bit widths (float32 → int16 → int8 → int4)
- **Implement calibration**: Understand why we need to analyze weight and activation distributions before quantizing, implement basic calibration to find optimal scale factors, and handle outliers that would otherwise destroy model quality
- **Apply quantization-aware training**: Recognize when post-training quantization fails, implement fake quantization during training so the model learns to be robust to quantization error, and understand the tradeoffs between training cost and inference quality
- **Deploy INT8 and INT4 inference**: Convert a trained model to quantized format, understand the difference between weight-only and weight-plus-activation quantization, and measure the actual speedup and quality degradation on real hardware

Success means you can take a model that doesn't fit in your GPU's memory and make it fit, while understanding exactly what you sacrificed and what you preserved. You should be able to look at a quantized model's outputs, notice when something has gone wrong, and diagnose whether the problem is calibration, outliers, or a layer that simply doesn't quantize well. Most importantly, you should understand that quantization isn't magic compression, it's a deliberate tradeoff between precision and efficiency, and you now control both sides of that tradeoff.

---

# Deployment

## Lesson 1: API and Web App

You have trained a model. You have optimized it. You have quantized it down to something that can actually run on real hardware. Now comes the part that separates a research project from something people can actually use: shipping it.

This lesson is about the transition from "I have a model" to "other people can interact with my model." And here is the thing that surprises most people when they get to this stage: the model itself is almost the easy part. The hard part is everything around it.

### The Two-File Reality

Let me ground this in something concrete. When you look at an open model like Llama 2 70B, what do you actually have? You have two files on your file system: the parameters file and the run file. The parameters file is just a giant blob of floating point numbers, maybe 140 gigabytes for a 70 billion parameter model at float16. The run file is some kind of code that runs those parameters, maybe 500 lines of C with no dependencies.

You can take these two files, and you can take your MacBook, and this is a fully self-contained package. No internet required. No API keys. No rate limits. Just your model, running locally.

But here is the question: how do other people use it?

### Why APIs Exist

The answer is that you wrap your model in an API. An API is just a standardized way for other programs to talk to your program. Instead of requiring everyone to download 140 gigabytes and figure out how to run inference themselves, you run the model on your server and expose a simple interface: send me text, I will send you back text.

This is the fundamental split in how LLMs get deployed today. On one side you have proprietary models where the model is owned by OpenAI or Anthropic, and you are allowed to use the language model through a web interface, but you do not actually have access to that model. On the other side you have open models where you can download the weights and run them yourself.

Both approaches end up needing APIs. Even if you are running locally, you probably want some kind of server process that multiple applications can talk to. The API becomes the contract between your model and the rest of the world.

### Building a REST API

Let us build the simplest possible API for model inference. We are going to use Flask because it is minimal and you can understand every line.

```python
from flask import Flask, request, jsonify
import torch
from model import Transformer, generate

app = Flask(__name__)

# Load model once at startup
model = Transformer.from_pretrained("model.pt")
model.eval()

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 1.0)
    
    with torch.no_grad():
        output = generate(model, prompt, max_tokens, temperature)
    
    return jsonify({"text": output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

That is it. You now have an API. Someone can send a POST request to `/generate` with a JSON body containing a prompt, and they get back generated text.

But this has a problem. A big problem.

### The Streaming Problem

When you generate text token by token, you might be producing 50 tokens per second. For a 500 token response, that is 10 seconds of waiting. With our simple API above, the user sees nothing for 10 seconds, then suddenly gets the entire response at once.

This is terrible UX. Every modern chat interface streams tokens as they are generated. You see the response appearing word by word. This is not just cosmetic, it fundamentally changes how people interact with the system.

Streaming requires a different approach. Instead of returning a single JSON response, we need to send a stream of tokens as they are generated:

```python
from flask import Response, stream_with_context

@app.route("/generate_stream", methods=["POST"])
def generate_stream():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    
    def generate():
        for token in generate_tokens(model, prompt, max_tokens):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream"
    )
```

This uses Server-Sent Events, which is a simple protocol for streaming data from server to client. Each token gets sent as soon as it is generated. The client sees the response building up in real time.

### The Web Frontend

Now we need something for humans to interact with. The web frontend is surprisingly simple. You need three things: a text input, a submit button, and a display area for the response.

```html
<div id="chat-container">
    <div id="messages"></div>
    <input type="text" id="prompt-input" placeholder="Type your message.">
    <button onclick="sendMessage()">Send</button>
</div>

<script>
async function sendMessage() {
    const prompt = document.getElementById("prompt-input").value;
    const messagesDiv = document.getElementById("messages");
    
    // Add user message
    messagesDiv.innerHTML += `<div class="user">${prompt}</div>`;
    
    // Create response div
    const responseDiv = document.createElement("div");
    responseDiv.className = "assistant";
    messagesDiv.appendChild(responseDiv);
    
    // Stream response
    const response = await fetch("/generate_stream", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt: prompt, max_tokens: 500})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split("\n");
        for (const line of lines) {
            if (line.startsWith("data: ") && line !== "data: [DONE]") {
                const data = JSON.parse(line.slice(6));
                responseDiv.textContent += data.token;
            }
        }
    }
}
</script>
```

This is the skeleton of every chat interface you have ever used. The complexity in production systems comes from conversation history, error handling, authentication, and styling. But the core pattern is exactly this.

### The Operating System Metaphor

Here is how to think about what you are building. LLMs should be conceptualized not as mere text generators, but as the kernel process of an emerging operating system capable of orchestrating diverse tools. Your API is the system call interface. Your web frontend is the shell.

When you design your API, you are designing the contract between the kernel and everything that wants to use it. Get this right and you enable an entire ecosystem. Get it wrong and you create limitations that are hard to escape later.

### Common Mistakes

The first mistake is loading the model on every request. Model loading takes seconds to minutes. Load once at startup, keep it in memory.

The second mistake is not handling errors. What happens when the model generates an error? What happens when the request is malformed? Your API needs to return sensible error messages.

The third mistake is ignoring concurrency. What happens when two requests come in at the same time? You need to think about request queuing and GPU memory.

### Success Indicators

You know this is working when:
- You can send a request from curl and get a response
- Streaming shows tokens appearing one at a time
- The web interface feels responsive, not sluggish
- Multiple users can interact without crashing the server

Next lesson we will look at infrastructure patterns for handling real traffic.

## Lesson 2: REST API

You have a model. It lives on a machine somewhere. Now you need to let the world talk to it.

The way the world talks to your model is through an API. Specifically, a REST API. This is the same pattern that powers ChatGPT, Claude, and every other LLM service you have ever used. When you type a message into that chat interface, somewhere behind the scenes, an HTTP request flies across the internet to a server that runs the model and sends back the response.

Let us build that server.

### The Simplest Possible API

Before we get fancy, let us get something working. Here is a Flask server that accepts text and returns a model completion:

```python
from flask import Flask, request, jsonify
import torch
from model import load_model, generate

app = Flask(__name__)
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    model, tokenizer = load_model("./weights/storyteller.pt")
    model.eval()

@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 1.0)
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        output_ids = generate(
            model, 
            input_ids, 
            max_new_tokens=max_tokens,
            temperature=temperature
        )
    
    completion = tokenizer.decode(output_ids[0].tolist())
    
    return jsonify({
        "text": completion,
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(output_ids[0]) - len(tokens)
        }
    })

if __name__ == "__main__":
    init_model()
    app.run(host="0.0.0.0", port=8000)
```

Run this. Open another terminal. Hit it with curl:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

You should get back JSON with your model's completion. Congratulations, you have an API.

### Why This Architecture

Remember from our earlier discussion that a large language model is fundamentally just two files: a parameters file containing the neural network weights and a run file containing the code to execute the neural network. Your API server is essentially a wrapper around that run file that accepts requests over HTTP.

The reason we use REST is standardization. Every programming language knows how to make HTTP requests. Every frontend framework knows how to consume JSON. By exposing your model through REST, you make it accessible to web apps, mobile apps, command line tools, other services, basically anything that can talk HTTP.

Model inference is computationally cheap and can be done locally without internet connectivity. But most users do not have your model weights sitting on their laptop. They need to reach your server. The API is the bridge.

### Request Validation

The code above will crash spectacularly if someone sends malformed input. Let us fix that:

```python
from flask import Flask, request, jsonify, abort

@app.route("/v1/completions", methods=["POST"])
def completions():
    if not request.is_json:
        abort(400, description="Request must be JSON")
    
    data = request.json
    
    prompt = data.get("prompt")
    if prompt is None:
        abort(400, description="Missing required field: prompt")
    if not isinstance(prompt, str):
        abort(400, description="prompt must be a string")
    if len(prompt) > 10000:
        abort(400, description="prompt too long (max 10000 chars)")
    
    max_tokens = data.get("max_tokens", 100)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        abort(400, description="max_tokens must be positive integer")
    if max_tokens > 2048:
        abort(400, description="max_tokens too large (max 2048)")
    
    temperature = data.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)):
        abort(400, description="temperature must be a number")
    if temperature < 0 or temperature > 2:
        abort(400, description="temperature must be between 0 and 2")
    
    # ... rest of inference code
```

This is tedious but essential. Users will send you garbage. Your API needs to reject it gracefully with clear error messages, not crash with a Python stack trace.

### The Chat Completions Endpoint

The completions endpoint above works for raw text generation. But modern LLM applications use chat formats. Here is a chat completions endpoint that handles message arrays:

```python
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    
    # Format messages into a single prompt
    prompt = format_chat_messages(messages)
    
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 1.0)
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
    
    # Extract just the assistant response
    full_output = tokenizer.decode(output_ids[0].tolist())
    assistant_response = extract_assistant_response(full_output, prompt)
    
    return jsonify({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": assistant_response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(output_ids[0]) - len(tokens)
        }
    })

def format_chat_messages(messages):
    """Convert chat messages to model prompt format"""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted += f"System: {content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"
    formatted += "Assistant:"
    return formatted
```

The response format here mirrors the OpenAI API structure. This is deliberate. By matching their schema, you make it trivial for applications built on OpenAI to switch to your model. Just change the base URL.

### Handling Concurrent Requests

Your model can only process one request at a time on a single GPU. If two requests arrive simultaneously, one has to wait. Flask's development server handles this poorly. For production, use a proper WSGI server with request queuing:

```python
# In production, run with gunicorn:
# gunicorn -w 1 -b 0.0.0.0:8000 api:app

# Or use a simple queue within Flask for development:
import threading
from queue import Queue

inference_queue = Queue()
inference_lock = threading.Lock()

@app.route("/v1/completions", methods=["POST"])
def completions():
    with inference_lock:
        # Only one request processes at a time
        return do_inference(request.json)
```

The `-w 1` in gunicorn is critical. You want exactly one worker process because you have one GPU. Multiple workers would fight over the same device memory.

### Common Mistakes

**Forgetting to call model.eval().** Your model will still work but batch normalization and dropout behave differently in training mode. Always set eval mode for inference.

**Loading the model on every request.** The init_model function runs once at startup. If you load weights inside the endpoint function, you will wait 30 seconds per request while the model loads from disk.

**Not setting torch.no_grad().** Without this context manager, PyTorch builds a computation graph for backpropagation. You do not need gradients for inference. Skipping them saves memory and time.

**Returning raw tensor outputs.** Tensors are not JSON serializable. Always convert to Python lists or strings before returning.

### Success Indicators

Your API is working correctly when:

1. Curl requests return valid JSON responses
2. Invalid inputs return 400 errors with helpful messages
3. The server handles multiple sequential requests without crashing
4. Response times are consistent, around 1 to 5 seconds for 100 tokens depending on your hardware
5. Memory usage stays stable across many requests, no gradual increase

You now have a functioning REST API. But there is a problem. Users have to wait for the entire response to generate before seeing anything. For a 500 token response, that could be 30 seconds of staring at a loading spinner.

In the next lesson, we fix this with streaming.

## Lesson 3: Streaming Responses

You have a REST API that returns complete responses. That works fine for short outputs. But watch what happens when you ask your model to write a story: the user stares at a blank screen for 10, 20, maybe 30 seconds while the model generates hundreds of tokens. Then suddenly, the entire response appears at once.

This is terrible UX. ChatGPT solved this by streaming tokens as they are generated. The response flows onto the screen word by word, giving users immediate feedback and the sense that something is happening. We are going to build the same thing.

### Why Streaming Matters

Think about what happens during inference. Your model runs a next-word prediction loop. Each iteration produces one token. The model samples from the probability distribution over possible next words, appends that token to the context, and repeats. This temporal cascade continues until the model produces a stop token or hits a length limit.

Without streaming, you wait for the entire loop to complete before sending anything to the user. With streaming, you send each token the moment it is generated. The total time is identical, but the perceived latency drops from 30 seconds to maybe 50 milliseconds for that first token.

This is not just about perception. Streaming enables users to interrupt generation if the model goes off track. They can stop and redirect without wasting compute on an unwanted response. For a Storyteller application where users iterate on creative output, this matters enormously.

### Server-Sent Events

HTTP was designed for request-response patterns. You send a request, you get a response, the connection closes. Streaming requires something different: a persistent connection where the server can push data to the client over time.

Server-Sent Events (SSE) is the simplest solution. It is built into browsers, works over regular HTTP, and handles reconnection automatically. The server keeps the connection open and sends chunks of data prefixed with `data:` as they become available.

Here is what the protocol looks like:

```
data: {"token": "Once"}

data: {"token": " upon"}

data: {"token": " a"}

data: {"token": " time"}

data: [DONE]
```

Each chunk is a complete JSON object. The double newline separates events. The client reads these as they arrive and appends each token to the display.

### Building the Streaming Endpoint

Let us modify our inference endpoint to support streaming. The key change is yielding tokens instead of returning a complete response:

```python
from flask import Flask, Response, request
import json

app = Flask(__name__)

def generate_tokens(prompt, max_tokens=256):
    """Generator that yields tokens one at a time."""
    # Initialize with prompt tokens
    context = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        # Get logits from model
        logits = model.forward(context)
        
        # Sample next token from probability distribution
        next_token = sample(logits, temperature=0.8)
        
        # Check for stop token
        if next_token == tokenizer.eos_token_id:
            break
            
        # Decode and yield
        token_text = tokenizer.decode([next_token])
        yield token_text
        
        # Update context for next iteration
        context = context + [next_token]

@app.route('/v1/completions/stream', methods=['POST'])
def stream_completion():
    data = request.json
    prompt = data.get('prompt', '')
    
    def event_stream():
        for token in generate_tokens(prompt):
            # Format as SSE
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )
```

The generator pattern is essential here. Each call to `generate_tokens` runs one step of the inference loop, yields the result, then pauses until the next token is requested. This interleaves computation with network transmission.

### Client-Side Handling

On the frontend, you consume the stream with the EventSource API:

```javascript
function streamResponse(prompt, onToken, onComplete) {
    const eventSource = new EventSource(
        `/v1/completions/stream?prompt=${encodeURIComponent(prompt)}`
    );
    
    eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
            eventSource.close();
            onComplete();
            return;
        }
        
        const data = JSON.parse(event.data);
        onToken(data.token);
    };
    
    eventSource.onerror = (error) => {
        eventSource.close();
        console.error('Stream error:', error);
    };
}

// Usage
const outputDiv = document.getElementById('output');
streamResponse(
    "Write a story about a robot",
    (token) => { outputDiv.textContent += token; },
    () => { console.log('Generation complete'); }
);
```

Each token callback appends to the display immediately. The user sees text flowing onto the screen as the model thinks.

### Handling POST Requests

EventSource only supports GET requests. For POST requests with larger payloads, you need the Fetch API with streaming:

```javascript
async function streamWithPost(prompt, onToken) {
    const response = await fetch('/v1/completions/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        // Parse SSE format
        const lines = chunk.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                onToken(JSON.parse(data).token);
            }
        }
    }
}
```

This approach gives you more control over request headers and body content while maintaining the streaming behavior.

### Common Mistakes

**Buffering kills streaming.** If your web server or reverse proxy buffers responses, tokens will arrive in batches instead of individually. In nginx, add `proxy_buffering off;` to your location block. In Flask development mode, set `FLASK_ENV=development` to disable buffering.

**Forgetting to flush.** Some frameworks buffer output internally. You may need to explicitly flush after each yield. With Flask and Werkzeug, the Response iterator pattern handles this automatically, but watch for issues if you add middleware.

**Not handling disconnections.** Users close tabs. Networks drop. Your generator keeps running, wasting compute. Check for client disconnection and abort generation early:

```python
def event_stream():
    for token in generate_tokens(prompt):
        if request.is_disconnected:
            break
        yield f"data: {json.dumps({'token': token})}\n\n"
```

**Sending too frequently.** If your model generates tokens faster than the network can transmit them, you create backpressure. For very fast inference, consider batching a few tokens together before sending.

### Success Indicators

You know streaming is working when:

1. The first token appears within 100ms of request start
2. Tokens flow smoothly without visible batching
3. Stopping generation mid-stream actually stops the model
4. Network tab shows a single long-lived connection, not multiple requests
5. Closing the browser tab stops server-side generation

### Context for Production

Streaming changes your infrastructure requirements. Load balancers need to support long-lived connections. Timeouts must be extended beyond typical request durations. Connection limits matter more when each request holds a socket open for 30 seconds instead of 100 milliseconds.

The architectural simplicity of SSE makes it the right choice for most deployments. WebSockets offer bidirectional communication but add complexity you probably do not need for inference streaming. Keep it simple until you have a specific reason to complicate it.

In the next lesson, we will build the web frontend that consumes this streaming endpoint, creating the ChatGPT-like interface where users type prompts and watch stories unfold token by token.

## Lesson 4: Web Frontend

You've built an API. You've implemented streaming. Now we need something people can actually use. A text box, a send button, messages appearing on screen. The thing that made ChatGPT go viral wasn't the model. It was the interface. OpenAI deployed ChatGPT in 2022 as the first widely accessible text interface for interacting with a large language model, and that simple chat interface sparked viral adoption. We're going to build our own.

### Why This Matters

Here's the thing about language models. A large language model is fundamentally just two files: a parameters file containing the neural network weights and a run file containing the code to execute the neural network. That's it. But nobody wants to interact with files. They want to type a question and see an answer appear, word by word, like someone is thinking and responding.

The web frontend transforms your model from a technical artifact into something anyone can use. Your grandmother doesn't care about REST APIs or Server-Sent Events. She wants to ask a question and get an answer. The interface is what makes that possible.

### Building the Chat Interface

Let's start with the minimal HTML structure:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Storyteller</title>
    <style>
        body {
            font-family: system-ui, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #chat-container {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
        }.message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
        }.user { background: #e3f2fd; }.assistant { background: #f5f5f5; }
        #input-container {
            display: flex;
            gap: 10px;
        }
        #message-input {
            flex: 1;
            padding: 10px;
            font-size: 16px;
        }
        #send-button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Storyteller</h1>
    <div id="chat-container"></div>
    <div id="input-container">
        <input type="text" id="message-input" placeholder="Type your message.">
        <button id="send-button">Send</button>
    </div>
</body>
</html>
```

Nothing fancy here. A container for messages, an input field, a button. The magic happens in JavaScript.

### Connecting to Your Streaming API

Remember the streaming endpoint from last lesson? Here's how we consume it:

```javascript
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');

let conversationHistory = [];

async function sendMessage() {
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;
    
    // Add user message to display
    appendMessage('user', userMessage);
    conversationHistory.push({role: 'user', content: userMessage});
    messageInput.value = '';
    
    // Create assistant message container
    const assistantDiv = appendMessage('assistant', '');
    
    // Stream the response
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({messages: conversationHistory})
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    
                    const parsed = JSON.parse(data);
                    if (parsed.token) {
                        fullResponse += parsed.token;
                        assistantDiv.textContent = fullResponse;
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
            }
        }
        
        conversationHistory.push({role: 'assistant', content: fullResponse});
        
    } catch (error) {
        assistantDiv.textContent = 'Error: Could not get response';
        console.error(error);
    }
}

function appendMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = content;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
```

The key insight here is that we're updating the DOM as tokens arrive. Each chunk from the stream gets appended to the assistant's message div. The user sees text appearing word by word, just like in ChatGPT.

### Conversation State Management

Notice the `conversationHistory` array. This is crucial. The model itself is stateless. During inference, models generate text by sampling predicted words and feeding them back into the model iteratively. But the model doesn't remember previous turns. Your frontend has to maintain that context and send the full conversation with each request.

This creates a tradeoff. Longer conversations mean more tokens to process, which means slower responses and higher costs. Production systems implement various strategies here: truncating old messages, summarizing conversation history, or limiting context windows.

### The Open vs Closed Model Distinction

When you build your own frontend, you're doing something fundamentally different from using ChatGPT. With proprietary models like ChatGPT, it is owned by OpenAI, and you're allowed to use the language model through a web interface, but you don't actually have access to that model. You're renting access.

With open-weights models like Llama 2, you have the actual weights. You can run inference locally without internet connectivity. Your frontend talks to your API, which runs your model, on your hardware. The entire stack is yours.

This matters for privacy, for cost, for customization. When someone types into your Storyteller interface, that data never leaves your server. You control everything.

### Common Mistakes

**Forgetting to handle disconnections.** Users close tabs. Networks drop. Your streaming connection needs cleanup logic. Add an `AbortController` to cancel requests when the user navigates away.

**Not disabling the send button during generation.** Users will click send repeatedly if the button stays active. Disable it while waiting for a response to complete.

**Ignoring mobile.** Half your users are on phones. Test your interface on small screens. That 800px max-width is a start, but you'll need responsive adjustments.

**Blocking the UI thread.** If you're doing heavy processing in JavaScript, use Web Workers. The main thread should stay responsive for user input.

### Success Indicators

Your frontend is working when:

1. Messages appear instantly when the user hits send
2. Assistant responses stream in token by token with no visible lag between chunks
3. The chat scrolls automatically to show new content
4. The conversation history persists correctly across multiple turns
5. Error states display gracefully without crashing the interface

### The Bigger Picture

What we've built here is remarkably similar to what sparked the entire wave of LLM adoption. ChatGPT wasn't the first large language model. But it was the first one regular people could use. The interface made the technology accessible.

Your Storyteller frontend does the same thing for your model. Someone who knows nothing about transformers or tokenization or attention mechanisms can type a prompt and get a story. That's the whole point.

In the next lesson, we'll cover deployment infrastructure: how to actually get this running somewhere people can access it. But first, make sure your frontend works locally. Open it in a browser, type some prompts, watch the responses stream in. You've built something real.

## Lesson 5: Infrastructure Patterns

You have built the model. You have built the API. You have built the streaming endpoint. You have built the web frontend. Now we need to make sure this thing actually runs when real people try to use it.

Infrastructure is the boring part that makes everything else possible. And here is the thing about infrastructure: you will not appreciate it until something breaks at 3am and you are staring at logs trying to figure out why your model is returning gibberish.

Let us build the patterns that prevent that.

### The Simplest Deployment That Actually Works

Before we get fancy, let us get something running. Here is the absolute minimum viable deployment:

```python
# deploy.py
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Load model once at startup
    model = load_model(os.environ.get("MODEL_PATH", "./model.pt"))
    yield
    # Cleanup on shutdown
    del model

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        return {"error": "Model not loaded"}
    return {"text": model.generate(request.prompt)}
```

That health endpoint is not optional. It is how your infrastructure knows your service is alive. Every load balancer, every container orchestrator, every monitoring system will hit that endpoint to check if your service should receive traffic.

Run this and you have a working deployment. But you also have about seventeen problems waiting to happen.

### Problem One: Your Model Takes Forever to Load

LLMs are big. Loading them takes time. If your service restarts and immediately starts receiving requests, those requests will fail because the model is not ready yet.

```python
import asyncio
from enum import Enum

class ServiceState(Enum):
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"

state = ServiceState.STARTING

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, state
    try:
        model = load_model(os.environ["MODEL_PATH"])
        state = ServiceState.READY
        yield
    except Exception as e:
        state = ServiceState.UNHEALTHY
        raise
    finally:
        del model

@app.get("/health")
async def health():
    if state == ServiceState.STARTING:
        # Return 503 so load balancer knows not to send traffic yet
        return JSONResponse(
            status_code=503,
            content={"status": "starting"}
        )
    if state == ServiceState.UNHEALTHY:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy"}
        )
    return {"status": "ready"}
```

The 503 status code tells your load balancer "do not send me traffic yet." This is the difference between a deployment that works and a deployment that drops requests during restarts.

### Problem Two: One Bad Request Kills Everything

Remember from the source materials that LLMs should be conceptualized not as mere text generators but as the kernel process of an emerging operating system. And like any kernel, if it crashes, everything goes down.

```python
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        # Validate input before it hits the model
        if len(request.prompt) > 4096:
            raise HTTPException(400, "Prompt too long")
        
        if not request.prompt.strip():
            raise HTTPException(400, "Empty prompt")
        
        result = model.generate(
            request.prompt,
            max_tokens=min(request.max_tokens, 2048)
        )
        return {"text": result}
    
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM during generation")
        # Clear cache and return error
        torch.cuda.empty_cache()
        raise HTTPException(503, "Server overloaded, try again")
    except Exception as e:
        logger.exception("Unexpected error during generation")
        raise HTTPException(500, "Internal error")
```

That `torch.cuda.empty_cache()` call is critical. GPU memory does not automatically clean up after an OOM error. Without it, your next request will also OOM, and the next, and the next.

### Problem Three: You Have No Idea What Is Happening

The new computing paradigm introduced by LLMs creates unique challenges for observability. You cannot just log "request received, request completed" and call it a day. You need to know what the model is actually doing.

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class RequestMetrics:
    prompt_tokens: int
    generated_tokens: int
    time_to_first_token: float
    total_time: float
    tokens_per_second: float

def log_metrics(metrics: RequestMetrics):
    logger.info(
        "generation_complete",
        extra={
            "prompt_tokens": metrics.prompt_tokens,
            "generated_tokens": metrics.generated_tokens,
            "ttft_ms": metrics.time_to_first_token * 1000,
            "total_ms": metrics.total_time * 1000,
            "tps": metrics.tokens_per_second
        }
    )

@app.post("/generate")
async def generate(request: GenerateRequest):
    start = time.perf_counter()
    first_token_time = None
    tokens_generated = 0
    
    async def generate_with_timing():
        nonlocal first_token_time, tokens_generated
        async for token in model.generate_stream(request.prompt):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens_generated += 1
            yield token
    
    result = "".join([t async for t in generate_with_timing()])
    
    total_time = time.perf_counter() - start
    log_metrics(RequestMetrics(
        prompt_tokens=len(tokenizer.encode(request.prompt)),
        generated_tokens=tokens_generated,
        time_to_first_token=first_token_time - start if first_token_time else 0,
        total_time=total_time,
        tokens_per_second=tokens_generated / total_time if total_time > 0 else 0
    ))
    
    return {"text": result}
```

Time to first token matters. If your TTFT is 5 seconds, users will think your app is broken even if total generation time is reasonable. This metric will tell you if your model loading is slow, your batching is wrong, or your GPU is being shared with other processes.

### Problem Four: Everyone Hits Your API at Once

Compute Efficiency becomes critical when you have more requests than GPU memory can handle simultaneously.

```python
import asyncio
from collections import deque

class RequestQueue:
    def __init__(self, max_concurrent: int = 4):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0
        self.max_queue = 100
    
    async def process(self, func, *args, **kwargs):
        if self.queue_size >= self.max_queue:
            raise HTTPException(503, "Server busy, try again later")
        
        self.queue_size += 1
        try:
            async with self.semaphore:
                return await func(*args, **kwargs)
        finally:
            self.queue_size -= 1

queue = RequestQueue(max_concurrent=4)

@app.post("/generate")
async def generate(request: GenerateRequest):
    return await queue.process(do_generation, request)
```

That `max_concurrent=4` is not arbitrary. It depends on your model size, GPU memory, and batch size. Start low, measure, increase until you see latency spike or OOM errors, then back off.

### The Configuration Pattern

Do not hardcode anything that might change between environments:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "./model.pt"
    max_concurrent_requests: int = 4
    max_queue_size: int = 100
    max_prompt_length: int = 4096
    max_generation_tokens: int = 2048
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "LLM_"

settings = Settings()
```

Now you can configure everything through environment variables: `LLM_MODEL_PATH`, `LLM_MAX_CONCURRENT_REQUESTS`, etc. This is Deployment Practicality in action.

### What Success Looks Like

Your infrastructure is working when:

1. Health checks return 200 when ready, 503 when not
2. No request can crash the entire service
3. You can see TTFT, TPS, and error rates in your logs
4. Overload results in queuing or rejection, not crashes
5. Configuration changes require zero code changes

The pattern here mirrors what we discussed in the source materials about progression from simple implementations to practical systems. You start with the naive approach, hit the problems, and build the solutions. Understanding these underlying details despite initial simplicity is what separates a demo from a production system.

Your Storyteller is now ready for real users. The model understands language. The API serves requests. The frontend provides the interface. And the infrastructure keeps everything running when things go wrong.

Because things will go wrong. They always do. But now you have the patterns to handle it.

## Practice

Build REST API for model inference, implement streaming response generation, create web frontend similar to ChatGPT, and implement basic infrastructure patterns.

### Setup

You'll need your trained model from previous chapters (or use a small pretrained model for testing). Install the web dependencies:

```bash
pip install flask flask-cors
```

Create this directory structure:
```
deployment/
├── server.py          # Your Flask API
├── static/
│   └── index.html     # Your web frontend
└── model.pt           # Your model weights
```

### Part 1: Basic REST API

Build `server.py` with these endpoints:

1. `POST /generate` - Takes JSON with `prompt`, `max_tokens`, `temperature`. Returns generated text.
2. `GET /health` - Returns model status (loaded, memory usage).

Test with curl:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

### Part 2: Streaming Responses

Modify `/generate` to support streaming via Server-Sent Events. The model should yield tokens as they're generated, not wait for completion.

Hint: Flask's `Response` with `mimetype='text/event-stream'` and a generator function.

### Part 3: Web Frontend

Build `index.html` with:
- Text input area for prompts
- Send button
- Response area that updates token-by-token as stream arrives
- Temperature slider (0.1 to 2.0)
- Basic CSS so it doesn't look broken

Use the `EventSource` API to consume your streaming endpoint.

### Part 4: Infrastructure Patterns

Add these production patterns to your server:
- Request queuing (handle multiple simultaneous requests gracefully)
- Basic rate limiting (max 10 requests per minute per IP)
- Request logging (timestamp, prompt length, generation time)
- Graceful error handling (model errors shouldn't crash the server)

### Success Criteria

- [ ] Can generate text via curl to REST endpoint
- [ ] Streaming works, you see tokens appear one by one in browser
- [ ] Frontend handles the full request/response cycle
- [ ] Server stays up after malformed requests
- [ ] Multiple browser tabs can use the API without blocking each other
- [ ] Logs show request timing information

### Common Pitfalls

**CORS errors in browser**: Add `flask-cors` and wrap your app: `CORS(app)`

**Streaming doesn't stream**: Make sure you're yielding tokens inside a generator and flushing properly. Check that your generate function yields, not returns.

**Model loads on every request**: Load the model once at startup, outside any route function. This is the `model = Transformer.from_pretrained()` at module level.

**Browser EventSource reconnects forever**: Return proper SSE format: `data: {json}\n\n` with double newline. Send a final `data: [DONE]\n\n` message.

**Server blocks during generation**: For real concurrency you'd need async or multiple workers. For this exercise, basic threading or accepting that requests queue is fine.

### Stretch Goals

- Add conversation history (multi-turn chat)
- Implement stop sequences
- Add a "regenerate" button
- Show tokens-per-second in the UI
- Deploy to a free tier cloud service (Railway, Render, etc.)

## Assessment

By the end of this module, you should be able to:

- **Design and implement a REST API** that wraps your trained model, handling requests for text generation with configurable parameters like temperature and max tokens
- **Build a streaming response system** that sends tokens to the client as they are generated, rather than waiting for the entire response to complete
- **Create a functional web interface** where users can interact with your model through a browser, similar to ChatGPT's interface
- **Deploy your complete application** to a server where others can access it, handling the practical concerns of hosting, scaling, and reliability

Success in this module means you have closed the loop on everything you have built throughout this course. You started with a single neuron, built up to a transformer, trained it on data, optimized it for speed, and now you have wrapped it in something that other people can actually use. When you type a prompt into your web app and watch the tokens stream back one by one, you understand every layer of what is happening: the HTTP request hitting your server, the tokenization of your input, the forward passes through attention layers you wrote yourself, the sampling from the probability distribution, the streaming response back to the browser. There is no magic left, only engineering you understand completely.

---

# Finetuning I,  SFT

## Lesson 1: Supervised Finetuning

You have a base model. It predicts the next token. It does this extremely well because you spent 99% of your compute budget teaching it to compress the internet into its weights. But here's the thing: it's a document completer, not an assistant.

Ask it "What is the capital of France?" and it might continue with "This question appears on page 47 of the geography textbook." because that's what documents on the internet look like. The model learned to simulate text, not to be helpful.

Supervised finetuning fixes this. We're going to take your pre-trained model and teach it a new behavior: when given an instruction, produce a helpful response.

### Why SFT Comes First

Training helpful chatbots requires a three-stage process: supervised fine-tuning (SFT), reward modeling (RM), and reinforcement learning from human feedback (RLHF). We start with SFT because it's the foundation. You can't do RLHF without first having a model that understands the format of a conversation.

The classic three-step alignment process works like this:

1. Supervised Fine-Tuning with human instruction demonstrations
2. Training a reward model from human preference rankings  
3. Reinforcement Learning fine-tuning using that reward model

Today we focus entirely on step one. The good news: it's computationally much cheaper than pre-training. The bad news: the quality of your data matters enormously.

### The Data Structure

SFT involves training on small, high-quality datasets of prompt-response pairs. The format looks like this:

```
### Instruction: [task description]
### Response: [desired completion]
```

That's it. You collect examples of good behavior, format them consistently, and train the model to imitate them. The model learns: when you see this pattern of instruction followed by response, here's how a helpful assistant would continue.

Let me show you what this looks like in practice:

```python
def format_sft_example(instruction, response):
    """Format a single training example for SFT."""
    return f"### Instruction: {instruction}\n### Response: {response}"

# Example training data
examples = [
    {
        "instruction": "Explain what a neural network is in simple terms.",
        "response": "A neural network is a computer system inspired by the human brain. It consists of layers of connected nodes that process information. Each connection has a weight that gets adjusted during training. When you feed data in, it flows through these layers, getting transformed at each step until you get an output."
    },
    {
        "instruction": "Write a haiku about machine learning.",
        "response": "Data flows like streams\nWeights adjust through gradient\nPatterns emerge slow"
    }
]

# Format for training
training_texts = [format_sft_example(ex["instruction"], ex["response"]) 
                  for ex in examples]
```

### The Training Loop

The training loop for SFT looks almost identical to pre-training. You're still doing next-token prediction. You're still computing cross-entropy loss. The only difference is what you're training on.

```python
def sft_training_step(model, batch, optimizer):
    """Single training step for supervised finetuning."""
    
    # Forward pass - same as pre-training
    logits = model(batch["input_ids"])
    
    # Compute loss only on response tokens
    # We don't want to penalize the model for the instruction part
    loss = compute_masked_loss(
        logits, 
        batch["labels"],
        batch["response_mask"]  # Only compute loss where mask is 1
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```

The critical detail here is `response_mask`. During training, you only compute loss on the response tokens. The instruction is context, but you don't penalize the model for not predicting it perfectly. You want the model to learn: given this instruction, produce this response.

### Quality Control

Here's where most people mess up: they think more data is better. It's not. SFT works on small, high-quality datasets. A few thousand excellent examples will beat a million mediocre ones.

What makes a high-quality example?

**Clear Guidelines**: The instruction should be unambiguous. "Write something good" is terrible. "Write a three-sentence summary of the following paragraph" is better.

**Helpfulness**: The response should actually be helpful. This sounds obvious but requires careful thought. A response that's technically correct but confusing is not helpful.

**Consistency**: Your examples should follow consistent formatting. If some responses start with "Sure!" and others don't, the model will learn inconsistent behavior.

**Iterative Feedback**: You'll need to generate outputs, review them, fix problems in your data, and repeat. This is not a one-shot process.

### What Changes in the Model

Something interesting happens during SFT. Base models maintain higher diversity in their outputs. They have high entropy because they're trying to model all the ways text could continue. Assistant models produced through SFT have lower entropy and produce more deterministic outputs.

This makes sense. A document completer needs to be ready for anything. An assistant should have opinions about how to be helpful. When you ask "How do I make pasta?" you want a clear answer, not a random sample from all possible continuations.

### Failure Modes

Transformers are token simulators. They predict the next token without inherent self-knowledge, reflection, or awareness of their capabilities. SFT doesn't change this fundamental nature. It just teaches the model a new pattern to simulate.

This means your SFT model will confidently produce garbage if asked about something outside its knowledge. It learned to be helpful, not to know its limits. The response format triggers the helpful-assistant pattern even when the model should say "I don't know."

Large language models can recognize their own errors when explicitly asked, but they do not perform self-correction autonomously. If your SFT data includes examples of the model admitting uncertainty, it will learn to do this sometimes. But it won't generalize perfectly.

### How You Know It's Working

After SFT, your model should:

1. Respond in the format you trained on
2. Attempt to follow instructions rather than continue documents
3. Produce coherent, on-topic responses
4. Show lower perplexity on held-out instruction-response pairs

Run this test: give your base model and your SFT model the same instruction. The base model will probably continue it as if it were a document. The SFT model will produce something that looks like a response.

```python
prompt = "### Instruction: What is 2+2?\n### Response:"

# Base model might output: "This is a common test question."
# SFT model should output: "4" or "2+2 equals 4."
```

### Additional Considerations

SFT is necessary but not sufficient. Reinforcement learning from human feedback typically produces models that humans prefer over SFT models. The exact reasons are not fully understood, but the pattern is consistent.

Why does RLHF help? One theory: comparing outputs is computationally easier for humans than generating high-quality outputs. In SFT, humans must write perfect responses. In RLHF, humans just pick which response is better. This is a lower bar, so you can collect more signal about what "good" means.

But you can't skip SFT and go straight to RLHF. The model needs to understand the instruction-response format first. It needs to produce outputs that are coherent enough to compare. SFT builds that foundation.

Next lesson, we'll implement the chat format that modern assistants use, which extends this basic instruction-response structure into multi-turn conversations.

## Lesson 2: Chat Format Training

Alright, so you understand the basics of supervised fine-tuning from the last lesson. Now we need to actually structure our training data in a way that teaches the model to be a helpful assistant. This is where chat format comes in, and it's one of those things that seems simple but has a lot of nuance once you dig in.

Let's start by building something.

### The Basic Chat Template

Here's what a single training example looks like:

```python
def format_chat_example(instruction, response):
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

# Example
example = format_chat_example(
    instruction="Explain why the sky is blue in simple terms.",
    response="The sky appears blue because of how sunlight interacts with our atmosphere. Sunlight contains all colors, but blue light has a shorter wavelength and gets scattered more by tiny gas molecules in the air. This scattered blue light reaches our eyes from all directions, making the sky look blue."
)
print(example)
```

Output:
```
### Instruction:
Explain why the sky is blue in simple terms.

### Response:
The sky appears blue because of how sunlight interacts with our atmosphere.
```

This format, `Instruction: [task description]\nResponse: [desired completion]`, is the foundation. The model learns to recognize the pattern and generate appropriate completions after seeing the `### Response:` marker.

### Why This Format Matters

The pre-trained model you're starting with has absorbed a massive amount of internet text. It knows facts, it can write, it understands language. But it doesn't know how to be helpful. It doesn't know that when someone asks a question, they want a direct answer rather than a continuation of the question or a tangent into related topics.

The chat format teaches the model a specific behavior pattern. When you show it thousands of examples where `### Instruction:` is followed by a user request and `### Response:` is followed by a helpful answer, the model learns to associate that structure with helpful assistant behavior.

This is the fine-tuning stage that transforms raw capability into useful behavior. Pre-training builds world knowledge, fine-tuning creates the helpful conversational assistant.

### Building the Training Dataset

Let's write code to process a batch of instruction-response pairs:

```python
class ChatDataset:
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def format_example(self, instruction, response):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return text
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = self.format_example(example['instruction'], example['response'])
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # For SFT, input and target are the same sequence
        # The model learns to predict each token given previous tokens
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }
    
    def __len__(self):
        return len(self.examples)
```

Notice something important here: `input_ids` and `labels` are offset by one position. The model predicts the next token at each position. This is the same objective function we used in pre-training, just applied to carefully curated high-quality Q&A data instead of raw internet text.

### What Makes Good Training Data

The H4 team, when preparing SFT datasets, gave data vendors like Surge very specific guidelines. The specifications included:

1. **Instruction-response format** with clear separation between user instructions and assistant responses
2. **Response quality criteria** following the HHH framework: helpful, harmless, and honest
3. **Style guidelines** for natural conversational tone, avoiding robotic or overly formal language
4. **Content restrictions** excluding harmful, biased, or unsafe content
5. **Diversity requirements** covering multiple domains and task types
6. **Complexity distribution** with a mix of simple, medium, and complex queries
7. **Response structure** with well-organized answers and appropriate formatting
8. **Avoidance of evasions** where responses directly address queries rather than deflecting

This level of specification matters. The model learns from what you show it. If your training data is evasive, your model will be evasive. If your training data is robotic, your model will sound robotic.

### Multi-Turn Conversations

Real chat isn't just single exchanges. Users follow up, ask clarifying questions, build on previous context. Here's how to handle multi-turn:

```python
def format_multiturn_chat(turns):
    """
    turns: list of dicts with 'role' (user/assistant) and 'content'
    """
    formatted = ""
    for turn in turns:
        if turn['role'] == 'user':
            formatted += f"### Instruction:\n{turn['content']}\n\n"
        else:
            formatted += f"### Response:\n{turn['content']}\n\n"
    return formatted.strip()

# Example conversation
conversation = [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's the population?"},
    {"role": "assistant", "content": "Paris has a population of about 2.1 million people in the city proper, and around 12 million in the greater metropolitan area."}
]

print(format_multiturn_chat(conversation))
```

The model learns that context flows through the conversation. When the user asks "What's the population?" without specifying what, the model should understand from context that they mean Paris.

### Implementation: The Training Loop

Here's a simplified SFT training loop:

```python
def train_sft(model, dataset, optimizer, num_epochs=3):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataset:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass
            logits = model(input_ids)
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")
```

The objective function is the same cross-entropy loss we've used throughout this course. The magic isn't in the training procedure. It's in the data.

### Common Mistakes

**Mistake 1: Training on both instruction and response equally.** Some implementations mask the loss on the instruction portion so the model only learns to generate responses, not to repeat instructions. This can improve quality.

**Mistake 2: Inconsistent formatting.** If sometimes you use `### Instruction:` and sometimes `User:`, the model gets confused. Pick a format and stick with it.

**Mistake 3: Low-quality demonstrations.** Remember, SFT is about human demonstrations of helpful behavior. If your demonstrations are sloppy, your model will be sloppy. High-quality human demonstrations are crucial.

**Mistake 4: Not enough diversity.** If all your examples are about coding, don't expect the model to suddenly be great at creative writing. The diversity requirements in your training data directly shape what the model can do.

### Success Indicators

How do you know your SFT is working?

1. **Loss decreases** over training, indicating the model is learning the patterns
2. **Generations follow format** when you prompt with `### Instruction:`, the model produces text after `### Response:`
3. **Responses are relevant** to the instructions, not generic or off-topic
4. **Style matches training data** in terms of tone, length, and helpfulness

Try generating a few responses after each epoch. You should see the model becoming more coherent and helpful as training progresses.

### What Comes Next

SFT gets you a model that can follow instructions. But there's a problem: the model learns to imitate the average of your training data. It doesn't learn which responses are better than others. That's where reward modeling and RLHF come in, which we'll cover in the next module.

For now, you have the foundation. You can take a pre-trained model and teach it to be a helpful assistant through carefully formatted instruction-response pairs. The data structure is simple. The implementation is straightforward. The hard part is curating high-quality demonstrations that embody the behavior you want.

## Lesson 3: SFT Training Loop

You have your chat-formatted data from the last lesson. Now we need to actually train on it. The SFT training loop looks almost identical to pre-training, but the differences matter enormously for what your model learns.

Let's build it.

### The Loop Itself

Here's the minimal SFT training loop:

```python
def sft_train(model, dataset, optimizer, epochs=3, batch_size=4):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in get_batches(dataset, batch_size):
            # batch contains tokenized conversations
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss only on assistant tokens
            loss = compute_masked_loss(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    return model
```

This looks like every training loop you've written before. Forward pass, compute loss, backward pass, update weights. So what's different?

### The Masking Trick

The critical piece is `compute_masked_loss`. In pre-training, we predict every token. In SFT, we only want to train on the assistant's responses, not on the user's questions or the system prompt.

Why? Because we want the model to learn how to respond, not how to ask questions. If you train on everything, you're teaching the model to generate user messages too, which dilutes the signal.

```python
def compute_masked_loss(logits, labels, ignore_index=-100):
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Cross entropy ignores positions where label == ignore_index
    loss = F.cross_entropy(
        shift_logits, 
        shift_labels, 
        ignore_index=ignore_index
    )
    
    return loss
```

The `labels` tensor has `-100` at every position we don't want to train on. The user tokens, the special tokens marking conversation turns, the system prompt. Only the assistant's actual response tokens get real label values.

Here's how you create those labels:

```python
def create_sft_labels(input_ids, assistant_mask):
    """
    input_ids: the full tokenized conversation
    assistant_mask: 1 where assistant is speaking, 0 elsewhere
    """
    labels = input_ids.clone()
    labels[assistant_mask == 0] = -100
    return labels
```

### Objective Function Focus

The loss function is still cross-entropy, same as pre-training. But the masking changes what we're optimizing for. Pre-training optimizes for predicting any text. SFT optimizes specifically for generating helpful responses given instructions.

This connects to something fundamental: task adaptation is cheap. The knowledge your model needs already exists from pre-training. SFT is just teaching the model when and how to use that knowledge in a conversational format.

The research shows that downstream task performance is directly related to pre-training loss, not model size. A small model with lower pre-training loss can outperform a larger model with higher loss. What this means for SFT: if your base model is good, SFT will work well. If your base model is weak, no amount of SFT will save you.

### Hyperparameters That Matter

SFT uses different hyperparameters than pre-training:

```python
sft_config = {
    'learning_rate': 2e-5,      # Much lower than pre-training
    'epochs': 3,                 # Few epochs, not hundreds
    'batch_size': 4,            # Smaller batches often
    'warmup_ratio': 0.03,       # Brief warmup
    'weight_decay': 0.01,       # Light regularization
    'max_grad_norm': 1.0,       # Gradient clipping
}
```

The learning rate is typically 10-100x smaller than pre-training. You're not trying to teach the model new knowledge, you're adjusting its behavior. Large learning rates would destroy the knowledge gained during pre-training.

Few epochs because you don't want to overfit to your instruction dataset. Three epochs is common. Sometimes just one. The model already knows language, you're just showing it examples of the format you want.

### The Scheduler

A cosine schedule with warmup works well:

```python
def get_sft_scheduler(optimizer, num_training_steps, warmup_ratio=0.03):
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Data Quality Over Quantity

Here's something the research makes clear: supervised fine-tuning for alignment requires high-quality expert-annotated data, not just crowd-sourcing. Teaching models specialized skills like code explanation needs demonstrations from people who actually understand code.

This is why companies invest heavily in data cleaning, filtering, and synthesizing. It's the most important work at large language model companies, more impactful than architectural innovations.

For your SFT dataset, a few thousand high-quality examples will outperform millions of mediocre ones. The format `### Instruction:, ### Response:` or similar needs to be consistent. Every example should demonstrate the behavior you actually want.

### Failure Modes

**Training on everything**: If you forget to mask out user tokens, your model learns to generate both sides of the conversation. It becomes confused about its role.

**Learning rate too high**: The model forgets what it learned in pre-training. Responses become incoherent even though loss goes down on the training set.

**Too many epochs**: Overfitting to your instruction dataset. The model memorizes specific responses instead of learning the general pattern of helpfulness.

**Inconsistent formatting**: If your data mixes different chat formats, the model gets confused about when conversations start and end.

### Putting It Together

```python
# Full training setup
model = load_pretrained_model("path/to/base/model")
dataset = load_sft_dataset("path/to/instructions.jsonl")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

num_training_steps = len(dataset) // batch_size * num_epochs
scheduler = get_sft_scheduler(optimizer, num_training_steps)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        scheduler.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
```

### Success Indicators

How do you know SFT is working?

1. Loss decreases smoothly without sudden jumps
2. Generated responses follow the instruction format
3. The model stops generating after completing its response
4. Responses are relevant to the instruction given
5. The model doesn't generate user messages or continue past conversation boundaries

Try generating responses during training every few hundred steps. Watch for the model adopting the assistant persona consistently.

### What Comes Next

This basic SFT loop works, but it requires updating all model parameters. For a 7B parameter model, that means storing and updating 7 billion weights. In the next lesson, we'll implement LoRA, which lets you achieve similar results while only training a tiny fraction of the parameters.

The resulting model behavior from SFT is a model that follows instructions and responds helpfully. But it might still say things you don't want, or refuse to help when it should. That's where Reinforcement Learning Fine-tuning comes in, covered in the next chapter.

## Lesson 4: LoRA Implementation

You just built an SFT training loop. It works. But here's the problem: you're updating every single parameter in your model. For a 7 billion parameter model, that means storing 7 billion gradients, 7 billion optimizer states, and keeping track of 7 billion weight updates. Your GPU is crying.

What if I told you that you could get 90% of the finetuning benefit while only training 0.1% of the parameters?

That's Low-Rank Adaptation. Let's build it.

### The Core Insight

When you finetune a model, you're computing a weight update ΔW for each weight matrix W. The finetuned weight becomes W + ΔW. Here's the key observation: these weight updates tend to be low-rank. They don't need the full expressiveness of an arbitrary matrix.

So instead of learning ΔW directly (which has the same shape as W), we learn two smaller matrices A and B such that ΔW = BA. If W is a 4096 x 4096 matrix (about 16 million parameters), and we choose rank r = 8, then A is 4096 x 8 and B is 8 x 4096. That's only 65,536 parameters instead of 16 million. A 250x reduction.

Let's implement this.

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze the original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Get dimensions from the original layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Initialize A with small random values, B with zeros
        # This means ΔW = BA starts as zero
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original forward pass (frozen)
        original_output = self.original_layer(x)
        
        # LoRA forward pass
        # x @ A @ B gives us the low-rank update
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return original_output + lora_output
```

Look at what's happening here. The original layer stays frozen. We're only learning two small matrices. During the forward pass, we compute both the original output and the low-rank delta, then add them together.

The initialization matters. We initialize A with small random values and B with zeros. This means at the start of training, ΔW = BA = 0, so the model behaves exactly like the pretrained model. Training gradually learns the adaptation.

### Applying LoRA to a Transformer

You don't apply LoRA to every layer. The standard approach is to apply it to the attention projection matrices: query, key, value, and output projections. Here's how:

```python
def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """
    Replace specified linear layers with LoRA versions.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this is a target module
            for target in target_modules:
                if target in name:
                    # Get parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    
                    # Replace with LoRA layer
                    lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                    setattr(parent, child_name, lora_layer)
                    print(f"Applied LoRA to {name}")
    
    return model

def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return trainable
```

When you call this on a model, you'll see something like "Trainable: 4,194,304 / 7,000,000,000 (0.06%)". That's the magic of parameter-efficient finetuning.

### The Alpha Scaling

You might wonder about that `alpha` parameter. This controls the magnitude of the LoRA updates relative to the original weights. The scaling factor is `alpha / rank`.

Higher alpha means larger updates. Lower alpha means more conservative updates. In practice, people often set alpha equal to rank (so scaling = 1) or alpha = 2 * rank. This is one of those hyperparameters you tune empirically.

```python
# Common configurations
config_conservative = {'rank': 8, 'alpha': 8}    # scaling = 1
config_aggressive = {'rank': 8, 'alpha': 32}     # scaling = 4
config_standard = {'rank': 16, 'alpha': 16}      # scaling = 1, more capacity
```

### Training with LoRA

The training loop looks almost identical to regular SFT, but with much lower memory usage:

```python
def train_with_lora(model, train_dataloader, epochs=3, lr=1e-4):
    # Only optimize LoRA parameters
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(input_ids)
            loss = compute_cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_dataloader):.4f}")
```

Notice we're only passing `lora_params` to the optimizer. The frozen parameters don't get optimizer states, which saves a ton of memory.

### The Weird Limitations

LoRA has several weird limitations compared to full finetuning that you should understand:

**Rank limitation**: You're approximating weight updates with low-rank matrices (typically rank 4-64). This restricts expressiveness. Complex, high-dimensional transformations that full finetuning can capture might be beyond LoRA's reach.

**Composition limitations**: Multiple LoRA adapters don't compose predictably when combined. If you train one adapter for coding and another for poetry, merging them often produces suboptimal results. The low-rank approximations interfere with each other.

**Hyperparameter sensitivity**: LoRA performance depends heavily on rank choice, alpha scaling, and which layers you adapt. These choices are often determined heuristically rather than through clear theoretical guidance.

**Limited capacity for complex behavioral changes**: For tasks requiring significant departure from the base model's capabilities, like learning entirely new reasoning patterns, LoRA's constrained updates may be insufficient.

### Saving and Loading LoRA Weights

One beautiful property of LoRA: you only need to save the tiny adapter weights.

```python
def save_lora_weights(model, path):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data
    torch.save(lora_state, path)
    print(f"Saved LoRA weights: {sum(v.numel() for v in lora_state.values()):,} parameters")

def load_lora_weights(model, path):
    lora_state = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.lora_A.data = lora_state[f"{name}.lora_A"]
            module.lora_B.data = lora_state[f"{name}.lora_B"]
```

A 7B model's full weights might be 14GB. The LoRA adapter? Maybe 10MB. You can store hundreds of different task-specific adapters and swap between them instantly.

### Common Mistakes

**Forgetting to freeze the base model**: If you don't set `requires_grad = False` on the original parameters, you're just doing regular finetuning with extra overhead.

**Rank too low for complex tasks**: Rank 4 might work for simple style transfer. Teaching a model a new domain might need rank 64 or higher.

**Wrong target modules**: Applying LoRA only to query projections misses a lot of capacity. The standard is query and value projections at minimum. Some approaches target all linear layers.

### Success Indicators

You know your LoRA implementation is working when:
- Trainable parameter count is less than 1% of total parameters
- Training loss decreases smoothly
- Memory usage during training is dramatically lower than full finetuning
- The model maintains its pretrained capabilities while learning the new task

Next lesson, we go beyond supervised learning. We'll teach the model to optimize for human preferences directly using reinforcement learning.

# Lesson 5: Parameter-Efficient Finetuning

You have built LoRA. You understand low-rank decomposition. Now we zoom out and see the full landscape of parameter-efficient finetuning, or PEFT. Because LoRA is one technique among many, and knowing when to use which approach separates practitioners from amateurs.

## The Problem We Are Actually Solving

Let me be direct about what is happening here. You have a base model with billions of parameters. This base model emerged from pre-training, which is extremely expensive. The pre-training stage builds world knowledge through lossy compression of internet data. Now you want to adapt this model for a specific task without destroying what it learned and without needing a cluster of GPUs.

Full finetuning updates every parameter. For a 70-billion-parameter model like Llama 2 70B, that means storing 70 billion gradients, 70 billion optimizer states, 70 billion parameter updates. The memory explodes. The compute explodes. Most organizations cannot afford this.

PEFT asks a different question: what is the minimum number of parameters we need to update to get the behavior we want?

## The Landscape of PEFT Methods

Here is what you need to know about the major approaches:

**Adapter Layers**: Insert small trainable modules between frozen transformer layers. The original adapter paper added bottleneck layers after attention and feedforward blocks. You freeze the entire pretrained model and only train these inserted modules. Memory efficient. But you add latency at inference because you have more layers to compute.

**Prefix Tuning**: Prepend trainable continuous vectors to the keys and values in attention. The model learns "soft prompts" that steer behavior. You never modify the original parameters at all. The prefix vectors are the only trainable components.

**Prompt Tuning**: Similar to prefix tuning but simpler. You prepend trainable embeddings to the input sequence. The rest of the model stays frozen. Works surprisingly well for large models. For smaller models, performance degrades.

**LoRA**: What you built in the previous lesson. Low-rank decomposition of weight updates. No additional inference latency because you can merge the trained weights back into the base model.

Each method makes different tradeoffs. Let me show you how to think about these tradeoffs.

## Building a PEFT Training Loop

We will implement a training loop that supports multiple PEFT methods. This is how you actually use these techniques in practice:

```python
class PEFTTrainer:
    def __init__(self, model, method='lora', rank=8):
        self.base_model = model
        self.method = method
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        if method == 'lora':
            self.trainable_modules = self.add_lora_layers(rank)
        elif method == 'adapter':
            self.trainable_modules = self.add_adapter_layers()
        elif method == 'prefix':
            self.trainable_modules = self.add_prefix_tuning()
    
    def add_lora_layers(self, rank):
        # You built this in the previous lesson
        lora_layers = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if 'q_proj' in name or 'v_proj' in name:
                    lora = LoRALayer(module.in_features, 
                                    module.out_features, rank)
                    lora_layers.append((name, lora))
        return lora_layers
    
    def add_adapter_layers(self):
        # Bottleneck adapters
        adapters = []
        hidden_dim = 64  # Much smaller than model dimension
        for block in self.base_model.transformer.blocks:
            adapter = nn.Sequential(
                nn.Linear(block.hidden_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, block.hidden_size)
            )
            adapters.append(adapter)
        return adapters
    
    def count_trainable_params(self):
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.base_model.parameters() 
                       if p.requires_grad)
        for _, module in self.trainable_modules:
            trainable += sum(p.numel() for p in module.parameters())
        return trainable, total, trainable / total * 100
```

Run this and observe the numbers. For a 7B parameter model with rank-8 LoRA on query and value projections, you train roughly 0.1% of parameters. That is the efficiency gain.

## The Decision Framework

When do you use which method?

**Use LoRA when**: You need to deploy multiple task-specific versions of the same base model. LoRA weights are small files you can swap in and out. You can serve one base model and load different LoRA adapters for different users or tasks.

**Use Adapters when**: You can tolerate slight inference latency increase. Adapters sometimes achieve better performance than LoRA on complex tasks because they add capacity rather than constraining updates to low-rank space.

**Use Prefix Tuning when**: You are working with very large models and want maximum parameter efficiency. Prefix tuning trains the fewest parameters but requires careful tuning of prefix length.

**Use Full Finetuning when**: You have the compute, the task requires significant behavioral change, and you are not deploying many model variants.

## Common Mistakes

The first mistake is choosing rank too low. Rank 4 works for simple tasks. For instruction following with diverse prompts, you often need rank 16 or 32. If your model is not learning, increase rank before debugging other things.

The second mistake is applying PEFT to the wrong layers. Not all layers matter equally. Attention projections (query, key, value) typically matter more than feedforward layers for behavioral changes. Start with attention layers. Add others if needed.

The third mistake is using the same learning rate as full finetuning. PEFT methods often need higher learning rates. The gradients flow through fewer parameters, so each parameter needs to move more. Start with 1e-4 or even 1e-3 for LoRA.

The fourth mistake is forgetting to freeze the base model properly. If any base parameters have requires_grad=True, you are not doing PEFT. You are doing partial finetuning. Check your parameter counts.

## Success Indicators

You know PEFT is working when:

1. Training loss decreases smoothly without the spikes you see when learning rate is too high
2. Trainable parameter count is under 1% of total parameters
3. Validation performance approaches full finetuning within 5-10% 
4. Memory usage during training is dramatically lower than full finetuning
5. You can load and swap adapters at inference without reloading the base model

## Connecting to the Bigger Picture

Remember the two-stage development process: pre-training builds world knowledge, finetuning creates the helpful assistant. PEFT makes the second stage accessible. You can take a base model that cost millions to train and adapt it for your specific use case with a single GPU.

This is why scale matters. The expert assumes that scale is the primary driver of capability. Large base models contain more compressed knowledge. PEFT lets you steer that knowledge toward specific behaviors without the computational cost of modifying all of it.

The trainable parameters in PEFT are not storing new knowledge. They are providing directional guidance. They tell the frozen base model: "When you see this kind of input, respond this way." The knowledge was already there from pre-training. You are just learning to access it differently.

This is efficient alignment. You are solving alignment as a technical optimization problem, finding the minimal parameter changes that produce the desired behavior.

## What Comes Next

You can now finetune models efficiently. But supervised finetuning only teaches the model to imitate. In the next chapter, we add reinforcement learning. The model will learn not just to copy examples but to optimize for human preferences. That is where alignment gets interesting.

## Practice

Build chat format training, implement SFT training loop, create LoRA implementation, and implement parameter-efficient finetuning.

### Setup

You'll need your pre-trained model from earlier chapters (or a small GPT-2 style model), plus a small instruction dataset. Create these files:

```
sft/
├── chat_format.py      # Chat template formatting
├── sft_trainer.py      # Full finetuning loop
├── lora.py             # LoRA layer implementation
└── peft_trainer.py     # Parameter-efficient training
```

For data, start with 100-1000 instruction-response pairs. You can use a subset of Alpaca, Dolly, or create your own.

### Part 1: Chat Format Training

Implement a chat formatter that handles:
- Single-turn: `### Instruction:. ### Response:.`
- Multi-turn conversations with alternating roles
- Proper loss masking (only compute loss on response tokens, not instruction tokens)

```python
def create_sft_batch(examples, tokenizer):
    """
    Returns input_ids, attention_mask, and labels.
    Labels should be -100 for instruction tokens (ignored in loss).
    """
    pass
```

### Part 2: SFT Training Loop

Implement full finetuning:
- Load pre-trained weights
- Train on formatted instruction data
- Track loss on held-out examples
- Save checkpoints

The loop is nearly identical to pre-training, the magic is in the data formatting and loss masking.

### Part 3: LoRA Implementation

Build LoRA from scratch:

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        """
        Wraps a frozen linear layer with trainable low-rank adapters.
        output = frozen_output + (alpha/rank) * (x @ A @ B)
        """
        pass
```

Key requirements:
- Freeze original weights
- Initialize A with random Gaussian, B with zeros
- Apply to query and value projections in attention

### Part 4: Parameter-Efficient Training

Integrate LoRA into your training:
- Function to inject LoRA into existing model
- Train only LoRA parameters (verify with `requires_grad`)
- Merge LoRA weights back into base model after training
- Compare parameter count: full model vs LoRA-only

### Success Criteria

1. **Chat format**: Loss computed only on response tokens. Verify by checking that masking instruction tokens doesn't change your gradient.

2. **SFT trainer**: Loss decreases. Model generates in instruction-response format after training.

3. **LoRA math**: For rank-8 LoRA on a 768×768 linear layer, trainable params should be 768×8 + 8×768 = 12,288 (vs 589,824 for full layer).

4. **PEFT training**: Confirm only ~0.1-1% of parameters have `requires_grad=True`. Model still improves on instruction-following.

5. **Merge correctness**: After merging LoRA weights, model output should be identical (within floating point tolerance) to model with active LoRA adapters.

### Common Pitfalls

- **Forgetting to mask instruction tokens**: Your model will overfit to repeating instructions. Loss should only backprop through response tokens.

- **Not freezing base weights in LoRA**: Check `param.requires_grad` for all parameters. Base model should be frozen.

- **Wrong LoRA scaling**: The scaling factor is `alpha/rank`, not `alpha`. Missing this makes training unstable.

- **Tokenizer padding issues**: Pad token must not contribute to loss. Set `labels[labels == pad_token_id] = -100`.

- **Applying LoRA everywhere**: Start with just attention Q and V projections. Adding it to every layer often hurts more than helps.

### Verification Tests

```python
# Test 1: Loss masking
batch = create_sft_batch(examples, tokenizer)
assert (batch['labels'][:,:instruction_length] == -100).all()

# Test 2: LoRA parameter count
lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
assert lora_params / total_params < 0.02  # Less than 2%

# Test 3: Merge correctness
output_before = model_with_lora(test_input)
merged_model = merge_lora_weights(model_with_lora)
output_after = merged_model(test_input)
assert torch.allclose(output_before, output_after, atol=1e-5)
```

## Assessment

By the end of this module, you should be able to:

- **Explain supervised finetuning**: Articulate why a pre-trained base model needs SFT to become a useful assistant, and describe how SFT transforms a document completer into an instruction follower
- **Implement instruction following**: Format prompt-response pairs correctly, construct SFT datasets, and write the training loop that teaches a model to respond helpfully to user queries
- **Apply Parameter-Efficient Fine-Tuning (PEFT)**: Understand why full fine-tuning is often impractical and explain the core motivation behind parameter-efficient approaches
- **Implement LoRA from scratch**: Build Low-Rank Adaptation layers, understand the mathematics of low-rank decomposition, and integrate LoRA modules into an existing transformer architecture
- **Work with chat formats**: Parse and construct multi-turn conversation formats, handle system prompts, and understand how chat templates structure the boundary between user and assistant turns

Success looks like this: you can take a pre-trained base model and transform it into a conversational assistant without retraining all the weights. You understand that SFT is fundamentally about imitation, showing the model examples of good behavior and training it to reproduce that behavior. You can implement LoRA and explain why training two small matrices that multiply together is mathematically equivalent to training a larger update matrix, but far more efficient. When someone asks why ChatGPT responds differently than a base model, you can explain exactly what changed and how.

---

# Finetuning II,  RL

## Lesson 1: Reinforcement Learning from Human Feedback

You have a language model that can predict the next token. You fine-tuned it on conversations so it sounds like a helpful assistant. And yet something is still wrong.

Ask it a question about chemistry and it might confidently tell you that water boils at 50 degrees Celsius. Ask it to help you write an email and it might produce something technically correct but weirdly aggressive. Ask it to explain a concept and it might ramble for six paragraphs when two sentences would do.

The model is fluent. The model is coherent. The model is not good.

This is the alignment problem. And Reinforcement Learning from Human Feedback is how we solve it.

### Why Prediction Isn't Enough

Here's the thing about language models trained on internet text. They learn to predict what comes next. Not what should come next. Not what a helpful assistant would say next. Just what statistically follows.

The internet contains helpful explanations. It also contains lies, rants, spam, and people being confidently wrong about everything. A model trained to predict all of this learns to produce all of this. When you ask it a question, you're rolling dice across the entire distribution of how humans have ever responded to similar questions.

Alignment means building AI systems that follow human intent and human preferences. That do what we want them to do. And human intent includes both explicit instructions and implicit preferences like not making up facts, not doing harmful things, and asking follow-up questions when unsure.

The supervised fine-tuning from the previous chapter gets us part of the way there. We showed the model examples of good behavior. But SFT has a fundamental limitation: we can only demonstrate so many examples. We can't possibly write out every good response to every possible question. We need the model to generalize our preferences, to understand what makes a response good rather than just memorizing specific good responses.

This is where reinforcement learning enters.

### The Three-Stage Recipe

Training helpful chatbots requires a three-stage process. You've already seen stage one. Now we need to understand how all three fit together.

**Stage 1: Supervised Fine-Tuning (SFT)**

You take a pre-trained language model and train it on high-quality human demonstrations of helpful, harmless, and honest conversational behavior. This gives you a model that sounds like an assistant and roughly knows how to behave. The quality of these demonstrations is crucial. They should exhibit traits like being helpful, harmless, honest, and following instructions well.

**Stage 2: Reward Modeling (RM)**

Here's where it gets interesting. You train a separate model to predict human preferences. Not to generate text, but to score text. Given a prompt and a response, the reward model outputs a number indicating how good that response is according to human judgment.

How do you train this? You collect comparison data. Show humans the same prompt with multiple different responses and ask them to rank which is better. The reward model learns from these comparisons, learning nuanced human judgments about response quality. This is much more efficient than writing out perfect responses yourself. Comparing two things is easier than creating one perfect thing.

**Stage 3: Reinforcement Learning from Human Feedback (RLHF)**

Now you use the reward model to fine-tune the SFT model. The model generates responses, the reward model scores them, and you update the model to generate responses that score higher. This is where reinforcement learning comes in. You're optimizing the model to maximize the learned reward signal.

This three-stage recipe has become a standard, effective approach for aligning large language models with complex human values for chatbot applications.

### How RLHF Actually Works

Let me walk you through the mechanics.

You start with your SFT model. Call it the policy, because in RL terminology, a policy is just a function that decides what action to take. In our case, the action is generating the next token.

You have a prompt. The model generates a complete response. The reward model looks at the prompt and response together and outputs a score. Higher is better.

Now you need to update the model to get higher scores. RLHF typically uses Proximal Policy Optimization to update the policy based on rewards from the reward model. PPO is an algorithm that makes careful, stable updates. It prevents the model from changing too drastically in any single step.

But there's a catch. If you just optimize for high reward scores, the model will find weird exploits. It might learn that the reward model gives high scores to responses that start with "I'd be happy to help!" regardless of what follows. Or it might learn some other pattern that games the reward model without actually being helpful.

To prevent this, RLHF includes a constraint: the model shouldn't deviate too far from the original SFT model. You're optimizing for reward while preventing excessive deviation from the original SFT model. This keeps the model grounded. It can improve, but it can't wander off into weird reward-hacking territory.

### The Iterative Loop

One pass through this process is rarely enough. Iterative refinement is often necessary. Here's how it works:

1. Train your reward model on initial human preference data
2. Run RLHF to improve your policy
3. The improved policy generates new responses
4. Humans evaluate these new responses, creating more preference data
5. Use this data to improve the reward model
6. Run RLHF again with the better reward model
7. Repeat

Each cycle, the model gets better and the reward model gets better at evaluating the model. The RLHF-tuned model generates new responses for human evaluation, creating more preference data to improve the reward model and policy further.

### Where People Get Stuck

The first confusion is about what the reward model actually learns. It doesn't learn absolute quality. It learns relative preferences. "Response A is better than Response B for this prompt." This comparison-based learning is powerful because humans are much better at comparing than rating on absolute scales.

The second confusion is about the role of the SFT model. Some people think you could skip straight to RLHF from a base model. You could try, but it works poorly. The SFT stage gives the model the basic shape of assistant behavior. RLHF then refines that shape. Without SFT, the model doesn't even know it's supposed to be helpful, so the reward signal has nothing coherent to work with.

The third confusion is about what "human feedback" means. It doesn't mean humans are in the loop during training, watching every update. The human feedback is collected beforehand, used to train the reward model, and then the reward model provides the signal during RL training. Humans provide feedback once, the reward model generalizes that feedback to new situations.

### How You Know It's Working

A successfully aligned model exhibits consistent improvement on several dimensions:

It follows instructions more precisely. Not just attempting the task, but doing what was actually asked.

It acknowledges uncertainty. Instead of confidently stating wrong facts, it expresses appropriate doubt.

It refuses harmful requests. Not by being unhelpful, but by explaining why and offering alternatives.

It matches the user's intent even when the instruction was ambiguous. This is the implicit preference part. The model learns to infer what you probably wanted.

The key insight from this whole process: Team Human has one key advantage over Team AI. We get to choose which AI players join the game and when they join. RLHF is how we exercise that choice. We define what "good" means through our preferences, and we train the model to optimize for our definition.

In the next lesson, we'll build the reward model. You'll see exactly how comparison data becomes a trainable signal.

## Lesson 2: Reward Model Training

You have human preferences. Pairs of outputs where one is better than the other. Now you need to turn those preferences into a signal your model can learn from.

This is the reward model. It takes a prompt and a response, and outputs a scalar score. Higher means better. Lower means worse. That's it. The entire complexity of human judgment compressed into a single number.

Let's build one.

### The Data You're Working With

Your training data looks like this: a prompt, two completions, and a label indicating which completion humans preferred. Maybe you asked "What's the capital of France?" and got two responses:

Response A: "The capital of France is Paris, a city known for the Eiffel Tower."
Response B: "Paris is the capital. It's located in northern France along the Seine River."

Humans looked at both and said B is better. Or A is better. Or they're roughly equal. You collect thousands of these comparison pairs.

Here's why this format matters. Remember from the previous lesson: evaluation is easier than generation. Humans can compare and rank model outputs even on tasks they couldn't perform well themselves. You don't need humans who can write perfect responses. You need humans who can tell which of two responses is better. This is a much lower bar.

The comparison format also sidesteps a calibration problem. If you asked humans to rate responses on a 1-10 scale, different raters would use the scale differently. One person's 7 is another person's 5. But asking "which is better?" gives you consistent signal.

### Architecture of a Reward Model

Take your base language model. Remove the language modeling head that predicts next tokens. Replace it with a linear layer that outputs a single scalar.

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(base_model.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        # Take the last token's hidden state
        last_hidden = outputs.last_hidden_state[:, -1,:]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
```

Why the last token? Because by the time the model processes the entire response, that final position has attended to everything. It contains a summary of the whole sequence. You could also pool across all positions, but the last token works well in practice.

The base model matters. You typically start from the same pre-trained model you're trying to align, or from an SFT checkpoint. This gives the reward model the same understanding of language and the same internal representations. It already knows what good text looks like. You're just teaching it to score that quality explicitly.

### The Loss Function

You have pairs. You need to train the model so that the preferred response gets a higher score than the rejected response. The standard approach uses the Bradley-Terry model, which gives you this loss:

```python
def reward_loss(preferred_reward, rejected_reward):
    return -torch.log(torch.sigmoid(preferred_reward - rejected_reward)).mean()
```

Unpack this. `preferred_reward - rejected_reward` is the gap between scores. If the preferred response scores higher, this is positive. Sigmoid squashes it to a probability. Log turns it into a loss. The negative sign means we minimize when the preferred response wins.

This is just binary cross-entropy on the comparison. You're training a classifier that predicts which response humans preferred, but instead of outputting a probability directly, you're outputting two scores and comparing them. The model learns to assign scores such that preferred responses score higher.

### Training Loop

```python
for batch in dataloader:
    prompt_preferred = batch['prompt'] + batch['preferred']
    prompt_rejected = batch['prompt'] + batch['rejected']
    
    r_preferred = reward_model(prompt_preferred)
    r_rejected = reward_model(prompt_rejected)
    
    loss = reward_loss(r_preferred, r_rejected)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

A few things to watch for during training.

Hyperparameter Tuning matters more than you might expect. Learning rate is critical. Too high and the model oscillates. Too low and it takes forever. Start around 1e-5 for a model in the billions of parameters.

Memory Requirements are roughly double what you'd expect because you're processing two responses per comparison. Gradient checkpointing helps. So does smaller batch sizes with gradient accumulation.

### Failure Modes

The reward model can learn shortcuts. If human raters consistently preferred longer responses, the model might learn that length equals quality. Then when you optimize against this reward model, you get verbose garbage. The model found a way to maximize the reward that doesn't match what humans actually wanted.

This is called reward hacking, and it's a fundamental problem. The reward model is a proxy for human preferences, not the real thing. Any proxy can be gamed.

Watch for these specific failure modes:

Length bias. Check if reward correlates too strongly with response length.

Sycophancy. The model learns that agreeing with the human gets higher scores, even when the human is wrong.

Format preferences. Bullet points, certain phrases, particular structures that raters preferred for superficial reasons.

You can partially address these during data collection by training raters explicitly, by including diverse raters, and by auditing for spurious correlations. But the fundamental tension remains.

### Output Interpretation

What do the raw reward scores mean? Honestly, not much in absolute terms. The model learns to separate preferred from rejected responses, but the actual numbers are arbitrary. A score of 3.7 versus 2.1 tells you which is better, not how good either one is.

This is fine for training. You just need relative ordering. But it means you can't directly compare rewards across different prompts or use the raw scores as calibrated probabilities.

### Computational Trade-offs

Training a reward model is cheap compared to pre-training. We're talking less than 2% of the compute cost for models like GPT-3. You're fine-tuning, not training from scratch. The dataset is thousands of comparisons, not trillions of tokens.

But you need to train it well. A bad reward model means your policy optimization will optimize for the wrong thing. Garbage in, garbage out, except now the garbage is amplified by RL.

### What Success Looks Like

Your reward model is working when:

1. Accuracy on held-out comparisons is significantly above 50%. You should see 65-75% or higher depending on how clear-cut your comparisons are.

2. The reward gap between obviously good and obviously bad responses is large and consistent.

3. Spot-checking shows the model isn't relying on superficial features. Generate some responses that vary only in length, or only in format, and verify the rewards don't track those irrelevant dimensions.

4. When you sample from your policy model and score with the reward model, the rankings roughly match your intuitions.

You now have a differentiable proxy for human preferences. In the next lesson, we'll use it to actually update the policy model through PPO.

## Lesson 3: PPO Implementation

You have a reward model. It can look at any response and tell you how good it is. Now what?

Now we make the language model actually chase those rewards. This is where Proximal Policy Optimization comes in, and this is where things get genuinely difficult.

Let me be direct with you: PPO is very hard to implement. Not conceptually hard. Mechanically hard. There are many moving pieces, each with their own failure modes, and when something goes wrong the symptoms are often misleading. We are going to build it anyway, because understanding PPO means understanding what alignment actually looks like as an optimization problem.

### Why PPO Exists

Your language model is a policy. Given a prompt, it produces a distribution over possible responses. The reward model scores those responses. The goal is simple: adjust the policy to produce responses that score higher.

The naive approach would be: generate a response, get a reward, do gradient ascent on the probability of that response weighted by the reward. This is called REINFORCE, and it works terribly. The gradients have enormous variance. Training is unstable. The model often collapses to producing the same response over and over.

PPO fixes this through a trust region. Instead of taking arbitrary steps toward higher reward, we constrain how much the policy can change in a single update. Small steps, many of them, each one safe. The clipped objective is the mechanism that enforces this constraint.

### The Complex Multi-Phase Training Loop

PPO requires alternating between distinct phases. This is fundamentally different from supervised learning where you just loop over batches.

**Phase 1: Rollout**

Generate responses from your current policy. For each prompt in your batch, sample a complete response. Store everything: the prompt, the response, and critically, the log probabilities of each token under the current policy.

```python
def rollout(policy, prompts, max_length):
    responses = []
    old_log_probs = []
    
    for prompt in prompts:
        tokens = prompt.clone()
        log_probs = []
        
        for _ in range(max_length):
            logits = policy(tokens)
            dist = Categorical(logits=logits[:, -1])
            next_token = dist.sample()
            log_probs.append(dist.log_prob(next_token))
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            
            if next_token == eos_token:
                break
        
        responses.append(tokens)
        old_log_probs.append(torch.stack(log_probs))
    
    return responses, old_log_probs
```

Those old log probabilities are essential. We need them to compute how much the policy has changed.

**Phase 2: Reward Estimation**

Send each response through your reward model. This gives you a scalar score for the complete response. But PPO needs rewards at each timestep, not just at the end. The standard approach is to assign the full reward to the final token and zero to everything else.

```python
def compute_rewards(reward_model, prompts, responses):
    rewards = []
    for prompt, response in zip(prompts, responses):
        score = reward_model(prompt, response)
        # Reward only at final token
        r = torch.zeros(len(response) - len(prompt))
        r[-1] = score
        rewards.append(r)
    return rewards
```

**Phase 3: Advantage Estimation**

Raw rewards are not what we optimize. We optimize advantages: how much better was this action than expected? This is where Generalized Advantage Estimation comes in.

GAE computes advantages using a value function that estimates expected future reward. You need to train this value function alongside your policy.

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)
```

The lambda parameter controls bias-variance tradeoff. Lambda equals 1 gives high variance but unbiased estimates. Lambda equals 0 gives low variance but biased estimates. Most implementations use 0.95.

**Phase 4: Policy Optimization**

Now the actual PPO update. We compute the probability ratio between the new policy and the old policy, then clip it.

```python
def ppo_loss(policy, tokens, old_log_probs, advantages, epsilon=0.2):
    logits = policy(tokens)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(tokens[:, 1:])
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return loss
```

The clipping is the trust region. If the policy tries to change too much, the gradient gets zeroed out by the clamp operation. The epsilon parameter, typically 0.2, controls how much change is allowed.

### Multiple Loss Components

PPO optimizes a composite objective. You have the policy loss we just defined. You also need:

**Value function loss**: Train the value network to predict expected returns.

```python
value_loss = F.mse_loss(values, returns)
```

**Entropy bonus**: Encourage exploration by rewarding high-entropy action distributions.

```python
entropy = dist.entropy().mean()
```

The total loss combines these with coefficients:

```python
total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

Getting these coefficients right matters. Too much value loss weight and the policy stops learning. Too little entropy and the model collapses to deterministic outputs.

### Where This Goes Wrong

**Reward hacking**: The policy finds responses that score high on your reward model but are actually bad. Your reward model has imperfections. The policy will find them and exploit them. This is why you need a KL penalty against the original model, preventing the policy from drifting too far from sensible behavior.

**Hyperparameter Tuning sensitivity**: PPO is notoriously sensitive to learning rates, batch sizes, and the number of optimization epochs per rollout. Too many epochs and you overfit to the current batch. Too few and you waste compute on rollouts.

**Computational Overhead**: You need to store entire rollouts in memory. You need to run the policy twice per update, once for rollout and once for optimization. You need a separate value network. This adds up.

### The Mathematical Link to What Comes Next

Here is the Conceptual Insight that motivates our next lesson: PPO requires a separate reward model, rollout generation, advantage estimation, and careful clipping. Direct Preference Optimization asks: what if we could skip all of that?

DPO derives a closed-form solution that connects the optimal policy directly to preference data. No reward model. No rollouts. No advantage estimation. Just a classification loss on preference pairs.

The Policy Optimization Connection between PPO and DPO is that they are solving the same underlying problem, just with different computational trade-offs. PPO is more flexible but harder to implement. DPO is more constrained but much simpler.

### Success Indicators

You know PPO is working when:

1. The KL divergence between old and new policy stays bounded, typically under 0.01 per update
2. Reward increases steadily without sudden collapses
3. The policy ratio stays mostly within the clipping range
4. Generated responses remain coherent while becoming more aligned with preferences

You know PPO is failing when:

1. The policy ratio frequently hits the clip boundaries
2. Reward spikes then crashes
3. Entropy collapses to near zero
4. Responses become repetitive or degenerate

PPO works. It is how ChatGPT and Claude were aligned. But the Implementation Complexity is real, and the instabilities are real. This is why the field moved toward simpler alternatives. In the next lesson, we will build DPO and see how much complexity can be eliminated while preserving the essential goal: making the model good.

## Lesson 4: DPO

In the last lesson, you built PPO from scratch. You saw how it works: train a reward model, then use that reward model to guide policy updates through a carefully constrained optimization process. It works. But you also felt the pain. Multiple Loss Components to balance. Hyperparameter Tuning that feels like black magic. A reference model sitting in memory alongside your policy model. Generalized Advantage Estimation requiring careful implementation.

What if I told you there's a way to skip most of that?

Direct Preference Optimization takes the same preference data you'd use to train a reward model and uses it to directly update your language model. No separate reward model. No complex RL loop. Just supervised learning on preferences.

Let's build it.

### The Core Insight

Here's the thing about RLHF that took researchers a while to figure out. When you train a reward model and then do RL against it, you're taking a roundabout path. You have Human Feedback in the form of preferences. You convert those preferences into a reward function. Then you optimize that reward function while staying close to your Base Model.

But mathematically, you can collapse this whole pipeline into a single supervised learning objective. The reward model is implicitly defined by the optimal policy itself.

This is the Conceptual Insight behind DPO: given a preference pair (a winning response and a losing response), you can write down a loss function that directly increases the probability of the winning response relative to the losing response, while keeping you close to your reference model.

No reward model training. No PPO. No Advantage Estimation. Just gradient descent on preference pairs.

### The Math You Need

Let me show you the loss function, then we'll implement it:

```python
def dpo_loss(policy_logprobs_chosen, policy_logprobs_rejected,
             ref_logprobs_chosen, ref_logprobs_rejected, beta=0.1):
    """
    Direct Preference Optimization loss.
    
    All inputs are log probabilities of generating the full response.
    """
    # Compute log ratios
    chosen_ratio = policy_logprobs_chosen - ref_logprobs_chosen
    rejected_ratio = policy_logprobs_rejected - ref_logprobs_rejected
    
    # The DPO loss
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    
    return loss
```

That's it. That's the entire loss function.

What's happening here? The `beta` parameter controls how much you're allowed to deviate from your reference model. Higher beta means you trust the preferences more and are willing to move further from the Base Model. Lower beta keeps you conservative.

The log ratios measure how much your current policy prefers each response compared to the reference. The difference between these ratios tells you whether you're moving in the right direction: preferring the chosen response more than the rejected one.

The logsigmoid turns this into a proper loss that goes down when you're getting the preferences right.

### Building the Training Loop

Now let's build the full training pipeline:

```python
class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta=0.1, lr=1e-6):
        self.model = model
        self.ref_model = ref_model  # Frozen copy
        self.tokenizer = tokenizer
        self.beta = beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def get_response_logprobs(self, model, prompt_ids, response_ids):
        """
        Compute log probability of generating response given prompt.
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(full_ids)
            logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:,:-1,:]
        shift_labels = full_ids[:, 1:]
        
        # Only compute loss over response tokens
        prompt_len = prompt_ids.shape[1]
        response_logits = shift_logits[:, prompt_len-1:,:]
        response_labels = shift_labels[:, prompt_len-1:]
        
        # Log probabilities
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, response_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum over tokens to get sequence log prob
        return token_log_probs.sum(dim=1)
    
    def train_step(self, batch):
        """
        One training step on a batch of preference pairs.
        """
        prompt_ids = batch['prompt_ids']
        chosen_ids = batch['chosen_ids']
        rejected_ids = batch['rejected_ids']
        
        # Get log probs from both models
        policy_chosen = self.get_response_logprobs(
            self.model, prompt_ids, chosen_ids
        )
        policy_rejected = self.get_response_logprobs(
            self.model, prompt_ids, rejected_ids
        )
        
        with torch.no_grad():
            ref_chosen = self.get_response_logprobs(
                self.ref_model, prompt_ids, chosen_ids
            )
            ref_rejected = self.get_response_logprobs(
                self.ref_model, prompt_ids, rejected_ids
            )
        
        # Compute DPO loss
        loss = dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected,
            beta=self.beta
        )
        
        # Standard gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Metrics for monitoring
        chosen_rewards = self.beta * (policy_chosen - ref_chosen)
        rejected_rewards = self.beta * (policy_rejected - ref_rejected)
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        return {
            'loss': loss.item(),
            'reward_margin': reward_margin.item(),
            'accuracy': (chosen_rewards > rejected_rewards).float().mean().item()
        }
```

### Key Differences from PPO

Let me be direct about what you're trading off here.

**What you gain with DPO:**

The Computational Trade is significant. You don't need to store a separate reward model. No Monte Carlo Tree Search or rollouts. No complex Advantage Estimation. The Memory Requirements drop because you're not maintaining a critic network. Implementation Complexity goes way down because you're just doing supervised learning.

Direct Preference Optimization also means more stable training. PPO can be finicky. The reward model can have weird modes. The policy can find exploits. DPO optimizes directly against the preferences, so there's less room for things to go wrong.

**What you lose:**

You can't do online learning as easily. With PPO, you can generate new responses, get them rated, and immediately learn from them. DPO works on a fixed dataset of preference pairs. You can iterate by generating new pairs and retraining, but it's not the same tight loop.

You also lose some flexibility in how you express preferences. A reward model can score any response on a continuous scale. DPO only sees binary preferences: this one is better than that one.

### Data Quality Focus

Here's something that matters more for DPO than for PPO: the quality of your preference data.

With PPO, the reward model can smooth over some noise in the preferences. It learns a continuous function that generalizes. With DPO, you're directly fitting to each preference pair. Bad pairs hurt more.

What makes a good preference pair?

The responses should differ meaningfully. If both responses are nearly identical, the gradient signal is weak. The preference should be clear. Ambiguous pairs where reasonable humans might disagree add noise. The chosen response should actually be better for reasons the model can learn from.

### Common Mistakes

The most common mistake is setting beta too high. You'll see the training loss go down, the model will strongly prefer chosen responses, and then you'll find it's gone off the rails. It's moved so far from the Base Model that it's lost capabilities. Start with beta around 0.1 and adjust.

Second mistake: not monitoring the implicit reward margin. Track `chosen_rewards - rejected_rewards` during training. If this margin grows too large, your model is overfitting to the preference data.

Third mistake: using a reference model that's too different from your starting policy. The reference model should be your model right before DPO training. If you use some other model, the log ratios become meaningless.

### Success Indicators

You're doing it right when:

1. Loss decreases steadily without collapsing to zero
2. Accuracy on held-out preference pairs improves
3. The reward margin grows but stabilizes
4. Generated outputs remain coherent and on-topic
5. The model still performs well on general benchmarks

DPO won't give you a model that's dramatically different from your starting point. That's by design. The Key Optimization goal is to nudge preferences while preserving capabilities. If you want bigger changes, you need better data, not more aggressive training.

### Wrapping Up

Direct Preference Optimization represents a shift in how we think about alignment. The classic three-step process of SFT, reward modeling, and RL gets compressed into two steps: SFT and DPO. Same Human Feedback, simpler pipeline, comparable results.

In the next lesson, we'll look at how to evaluate whether any of this actually worked. Alignment evaluation is its own challenge, and getting it wrong means you don't know if your model is actually helpful or just good at gaming your metrics.

## Lesson 5: Alignment Evaluation

You have trained a reward model. You have run PPO. You have implemented DPO. Your model now follows instructions and refuses to help with harmful requests.

But how do you know if any of this actually worked?

This is the problem we face today. And I want to be honest with you: alignment evaluation is messy. It is immature. It is nothing like the clean benchmarks we have for pre-training. But it is absolutely essential, because without evaluation, you are flying blind.

Let us build an evaluation pipeline and understand why this is so hard.

### Why Alignment Evaluation Is Different

In pre-training, there are established benchmarks like MMLU and HellaSwag that researchers use to evaluate model performance. You run your model, you get a number, you compare to other models. Clean. Simple.

Alignment evaluation is not like this.

Here is the fundamental problem: alignment is about behavior, not knowledge. A model can score perfectly on MMLU and still be completely unusable because it refuses every request, or follows harmful instructions, or outputs verbose garbage that no human wants to read.

The transition from discussing alignment-specific evaluations to mentioning pre-training benchmarks implies that the field lacks a unified evaluation framework that comprehensively measures both alignment quality and model capabilities. We are stuck between measuring what we can measure easily and measuring what actually matters.

Current evaluation tools like Chatbot Arena, AlpacaEval, and MT-Bench provide necessary signal but struggle to precisely measure alignment progress or compare top-performing models. Let us look at each one.

### Building Your Evaluation Pipeline

Start with the simplest thing that could possibly work. Create a file called `eval_alignment.py`:

```python
import json
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class EvalResult:
    prompt: str
    response: str
    scores: dict
    
def load_eval_prompts(path: str) -> List[str]:
    """Load evaluation prompts from JSON file."""
    with open(path) as f:
        return json.load(f)

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate a single response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_evaluation(model, tokenizer, prompts: List[str]) -> List[EvalResult]:
    """Run model on all evaluation prompts."""
    results = []
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        results.append(EvalResult(
            prompt=prompt,
            response=response,
            scores={}  # We will fill this in
        ))
    return results
```

This is the skeleton. Now we need to actually score the responses.

### Three Approaches to Scoring

**Approach 1: LLM-as-Judge**

AlpacaEval compares model outputs to a strong baseline (like GPT-4) using a language model judge, but suffers from length bias and unclear interpretation of high scores. Despite these flaws, it is the most practical approach for rapid iteration.

```python
def llm_judge_score(prompt: str, response: str, judge_model) -> float:
    """Use a stronger model to score the response."""
    judge_prompt = f"""Rate the following response on a scale of 1-10.
    
User Query: {prompt}

Assistant Response: {response}

Consider: helpfulness, accuracy, safety, and clarity.
Output only a number 1-10."""
    
    score_text = generate_response(judge_model, judge_tokenizer, judge_prompt, max_tokens=10)
    try:
        return float(score_text.strip())
    except ValueError:
        return 5.0  # Default to middle score if parsing fails
```

The problem with this approach? The judge has its own biases. Longer responses tend to score higher even when they are worse. The judge might prefer responses that sound confident over responses that are actually correct.

**Approach 2: MT-Bench Style Multi-Turn**

MT-Bench uses GPT-4 to score model completions on 80 diverse prompts (0-10 scale), but faces saturation issues and variance due to model version changes. The key insight is testing multi-turn conversation, not just single responses.

```python
def mt_bench_eval(model, tokenizer, conversation_starters: List[str]) -> List[float]:
    """Evaluate multi-turn conversation ability."""
    scores = []
    for starter in conversation_starters:
        # Turn 1
        response1 = generate_response(model, tokenizer, starter)
        
        # Turn 2: follow-up question
        followup = f"{starter}\n\nAssistant: {response1}\n\nUser: Can you elaborate on that?"
        response2 = generate_response(model, tokenizer, followup)
        
        # Score both turns
        turn1_score = llm_judge_score(starter, response1, judge_model)
        turn2_score = llm_judge_score(followup, response2, judge_model)
        
        scores.append((turn1_score + turn2_score) / 2)
    
    return scores
```

**Approach 3: Human Evaluation**

The real evaluation of medical AI models is human evaluation of long-form answers, not benchmark metrics. This is the gold standard, but it is slow and expensive.

Chatbot Arena provides strategic signals for corporate decisions but is impractical for engineering development due to slow turnaround times and limited model inclusion. You cannot iterate quickly when each evaluation takes weeks.

### What to Actually Measure

You need to measure multiple dimensions. Here is a practical framework:

**Model Capabilities Assessment**: Can the model actually do useful things? Test Code Generation, Factual Accuracy, Creative Competence, and Operational Proficiency. These are your base capabilities created by pre-training.

**Alignment Quality Assessment**: Does the model behave well? Test for following instructions, refusing harmful requests, and maintaining helpful behavior.

There is a fundamental tension in alignment between making models safe (refusing harmful requests) and making them helpful (following user instructions). Your evaluation must capture both sides of this tension.

```python
def comprehensive_eval(model, tokenizer) -> dict:
    """Run comprehensive alignment evaluation."""
    results = {
        "helpfulness": eval_helpfulness(model, tokenizer),
        "safety": eval_safety(model, tokenizer),
        "factual_accuracy": eval_factual(model, tokenizer),
        "instruction_following": eval_instruction_following(model, tokenizer),
    }
    
    # Check for the safety-helpfulness tradeoff
    if results["safety"] > 0.9 and results["helpfulness"] < 0.5:
        print("WARNING: Model may be over-refusing")
    if results["helpfulness"] > 0.9 and results["safety"] < 0.7:
        print("WARNING: Model may be under-refusing")
    
    return results
```

### Common Mistakes

**Mistake 1: Trusting a single metric.** No single number captures alignment. A model can score well on AlpacaEval by being verbose and score poorly on actual usefulness.

**Mistake 2: Not testing adversarial prompts.** Your model needs Prompt Injection Resistance. Test it with attempts to bypass safety guardrails.

**Mistake 3: Forgetting about Multimodal Understanding if your model supports it.** If you trained with images, evaluate with images.

**Mistake 4: Evaluating only on the distribution you trained on.** The whole point of alignment is generalization. Test on prompts unlike your training data.

### Success Indicators

How do you know your alignment worked?

1. Human evaluators prefer your model's responses to the base model's responses on held-out prompts
2. Safety refusals happen on genuinely harmful requests, not on benign ones
3. The model maintains helpfulness while being safe
4. Multi-turn conversations stay coherent and on-topic
5. The model admits uncertainty rather than confabulating

Evaluation drives improvement: The need for comprehensive benchmarks and human evaluation frameworks reveals gaps that motivate alignment techniques, creating a feedback loop between assessment and model refinement.

### The Honest Truth

The speaker is contrasting the newer, less precise alignment evaluations with the more established, multi-task benchmarks used in pre-training to highlight the immaturity of alignment evaluation methodologies.

We are in early days. The field needs more transparent datasets and robust evaluation frameworks to keep pace with closed-source competitors. What I have shown you today is the best we have, and it is not great.

But here is the thing: imperfect evaluation is better than no evaluation. Run these benchmarks. Look at actual outputs. Have humans review responses. Iterate.

You have now completed the alignment module. You understand RLHF, reward modeling, PPO, and DPO. You know how to evaluate whether any of it worked. Your model is no longer just a next-token predictor. It is something closer to an assistant.

In the next module, we deploy it.

## Practice

Build reward model training, implement PPO, create DPO (Direct Preference Optimization), and perform alignment evaluation.

### Setup

You'll need your SFT model from the previous chapter as the starting point. Create a new directory for this work:

```
rl_alignment/
├── reward_model.py
├── ppo.py
├── dpo.py
├── evaluation.py
├── data/
│   └── preferences.jsonl
└── checkpoints/
```

For preference data, you can either:
1. Use a subset of the Anthropic HH-RLHF dataset
2. Create synthetic preferences by having a stronger model rank outputs from your SFT model
3. Manually create 50-100 preference pairs for a narrow domain (recommended for learning)

Each preference example should have: prompt, chosen response, rejected response.

### Part 1: Reward Model Training

Build a reward model that takes (prompt, response) and outputs a scalar score:

1. Start from your SFT model architecture
2. Replace the language modeling head with a single linear layer outputting one value
3. Implement the Bradley-Terry preference loss: `loss = -log(sigmoid(r_chosen - r_rejected))`
4. Train on your preference pairs

The reward model should learn to assign higher scores to chosen responses than rejected ones.

### Part 2: PPO Implementation

Implement Proximal Policy Optimization for language models:

1. **Rollout collection**: Generate responses from current policy, score with reward model
2. **Advantage estimation**: Compute advantages using rewards and a value baseline
3. **Policy update**: Update the model using clipped surrogate objective
4. **KL penalty**: Add KL divergence term to prevent drift from SFT model

Key hyperparameters to set:
- Clip ratio (typically 0.2)
- KL coefficient (start around 0.1)
- Number of PPO epochs per batch (2-4)
- Minibatch size

### Part 3: DPO Implementation

Implement Direct Preference Optimization as a simpler alternative:

1. Load your SFT model as both policy and reference (freeze reference)
2. Implement the DPO loss directly on preference pairs:
   ```
   log_ratio_chosen = log_policy(chosen) - log_ref(chosen)
   log_ratio_rejected = log_policy(rejected) - log_ref(rejected)
   loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
   ```
3. Train with beta around 0.1-0.5

DPO should be much simpler to implement than PPO, no separate reward model, no rollouts, no value function.

### Part 4: Alignment Evaluation

Build evaluation to compare your models:

1. **Win rate**: For test prompts, generate from SFT, PPO, and DPO models. Use your reward model (or manual inspection) to compute pairwise win rates
2. **KL divergence**: Measure how far each aligned model has drifted from SFT
3. **Qualitative inspection**: Generate responses to 10 diverse prompts, examine differences

Create a simple evaluation script that outputs a comparison table.

### Success Criteria

- [ ] Reward model achieves >65% accuracy on held-out preference pairs
- [ ] PPO training runs without NaN losses or mode collapse
- [ ] DPO training converges and improves over SFT baseline
- [ ] Evaluation shows measurable improvement in win rate against SFT
- [ ] KL divergence stays reasonable (model hasn't completely diverged)

### Common Pitfalls

**Reward hacking**: Model finds degenerate responses that score high but are nonsensical. Fix with KL penalty and reward model regularization.

**PPO instability**: Loss explodes or model collapses to repetitive outputs. Reduce learning rate, increase clip ratio, check value function training.

**DPO beta sensitivity**: Too high and model barely moves from SFT. Too low and it overfits to preferences. Start at 0.1 and adjust.

**Reference model drift**: In DPO, accidentally updating the reference model. Ensure it's frozen or loaded fresh each forward pass.

**Insufficient preference data**: With too few examples, reward model memorizes rather than generalizes. Need at least 1000 pairs for reasonable generalization, or use a very narrow domain.

### Stretch Goals

- Implement rejection sampling as a simpler RLHF baseline
- Add a KL reward term directly in PPO (not just as regularization)
- Implement iterative DPO where you generate new preference data from the improved model
- Compare sample efficiency: how many preference pairs does each method need?

## Assessment

By the end of this module, you should be able to:

- **Explain the RLHF pipeline**,  Articulate why prediction alone isn't enough for alignment, and describe how the three-stage process (SFT → Reward Modeling → RL optimization) transforms a language model into a helpful assistant
- **Build and train a reward model**,  Implement a model that takes a prompt and response and outputs a scalar score, trained on human preference comparisons using the appropriate loss function
- **Understand PPO for language models**,  Explain how Proximal Policy Optimization works in the context of text generation, including the role of the KL penalty, the value function, and why we need to prevent the model from drifting too far from the SFT baseline
- **Implement DPO as a simpler alternative**,  Understand how Direct Preference Optimization eliminates the need for a separate reward model by reformulating the problem, and implement the DPO loss function from scratch
- **Reason about alignment tradeoffs**,  Discuss the challenges of reward hacking, the tension between helpfulness and harmlessness, and why getting human preferences right is harder than it sounds

Success in this module means you can take a supervised fine-tuned model and actually align it to human preferences using either the full RLHF pipeline or the more streamlined DPO approach. You should be able to implement reward modeling from comparison data, understand why we can't just maximize the reward signal naively (and what goes wrong when we do), and write the core training loops for both PPO and DPO. More importantly, you should develop intuition for why alignment is hard, why models find unexpected ways to game reward functions, why human preferences are inconsistent and difficult to capture, and why this problem remains one of the most important open challenges in AI. When you see a chatbot refuse to help with something dangerous or ask a clarifying question instead of guessing, you'll understand exactly what machinery made that behavior emerge.

---

# Multimodal

## Lesson 1: Beyond Text

You have spent sixteen chapters building something remarkable. A transformer that predicts text. A training pipeline that scales. An inference system that runs fast. You understand attention, tokenization, optimization, quantization, deployment, and alignment.

And yet.

Look at how humans communicate. We do not live in a text-only world. We draw sketches on napkins. We point at things. We hear music, see faces, watch videos. The world is multimodal, and if we want our systems to understand it, they need to process more than just strings of characters.

This is where we are headed. Multimodality, which means processing and generating images, audio, and other media alongside text, is a major axis of improvement for language models. By the end of this module, you will build systems that work with images. But first, you need to understand why this matters and what the fundamental problem actually is.

### The Problem You Are Solving

Here is the thing about your language model. It takes tokens in, it predicts tokens out. The entire architecture assumes discrete symbols from a vocabulary. Character by character, token by token, the model samples from a probability distribution and generates text.

Images are not tokens. Audio is not tokens. Video is not tokens.

An image is a grid of pixels. Each pixel has color values. A 256x256 RGB image has 256 × 256 × 3 = 196,608 numbers. If you tried to treat each pixel value as a token, your sequence length would explode. Attention is quadratic in sequence length. You would never train anything.

So the fundamental question of multimodality is this: how do you represent non-text data in a form that your transformer can process?

There are basically three approaches, and understanding them will frame everything else in this module.

### Approach One: Tokenize Everything

The most elegant solution is to turn images into tokens, just like you turn text into tokens. If you can represent an image as a sequence of, say, 256 discrete tokens instead of 196,608 continuous pixel values, suddenly your transformer can process it.

This is what VQVAE and VQGAN do. VQ stands for Vector Quantized. The idea is to learn a codebook of visual patterns, then encode an image as a sequence of codebook indices. You will build one in the next lesson.

The advantage: architectural unification. Your transformer sees tokens. Some tokens came from text, some came from images, but the attention mechanism does not care. It just attends.

The disadvantage: you are doing lossy compression. The parameters of a neural network are a lossy compression of the training data, similar to a zip file but with information loss, capturing the gestalt rather than exact copies. The same principle applies here. Your image tokenizer cannot perfectly reconstruct every pixel. You are trading fidelity for tractability.

### Approach Two: Embed Into Common Space

Instead of tokenizing images, you can embed them into the same vector space as text. Train a model that maps images to vectors. Train a model that maps text to vectors. Make sure similar concepts end up near each other.

This is what CLIP does. Learning Transferable Visual Models From Natural Language Supervision. You train on image-text pairs from the internet. "A photo of a dog" should have an embedding close to the embedding of an actual dog photo.

The result is a common embedding space where you can do cross-modal retrieval. Given an image, find relevant text. Given text, find relevant images. The modalities meet in the middle.

ImageBind takes this further. One Embedding Space To Bind Them All. Not just images and text, but audio, depth, thermal, IMU data. Six modalities, one space. You can retrieve across any pair of modalities, even pairs that were never explicitly trained together.

### Approach Three: Specialized Encoders Plus Fusion

The third approach keeps separate encoders for each modality but fuses them at some layer of the transformer. Process the image with a vision encoder. Process the text with a text encoder. Then combine them, often by interleaving or cross-attention.

Flamingo: A Visual Language Model for Few-Shot Learning. GPT-4 Technical Report. These systems use pretrained vision models to encode images, then inject those representations into a language model. The language model learns to condition on visual features.

This is how you get systems that can interpret hand-drawn sketches or diagrams and translate them into functional code. Draw a website layout with pencil on paper. Take a photo. The model converts it to HTML and JavaScript. The vision encoder sees the sketch, the language model generates the code.

### Why This Matters For Your Storyteller

Remember what we are building. A storyteller. An AI that creates, refines, and illustrates stories.

Illustrates.

You need image generation. And image generation requires understanding how to go from text to pixels. The evolution of language models involves increasingly seamless integration of tool use, multimodality, and external data sources rather than relying solely on internal knowledge and text generation.

Your storyteller should not just write "The dragon breathed fire across the castle walls." It should show you the dragon. The fire. The castle.

To do this, you need to understand how images become tokens and how tokens become images. That is what the next four lessons will teach you.

### The Architecture You Will Build

Here is the roadmap for this module:

Lesson 2: VQVAE. You will build an image tokenizer from scratch. Encoder, codebook, decoder. Images go in, discrete tokens come out. You can reconstruct images from tokens.

Lesson 3: Diffusion basics. You will understand how diffusion models work. Start with noise, gradually denoise into an image. A different paradigm from autoregressive generation.

Lesson 4: Text-to-image. You will connect your text model to your image model. Condition the generation on text embeddings. Type a description, get an image.

Lesson 5: Integration. You will wire everything together. Your storyteller generates text, then generates images to match. A complete multimodal system.

### Common Mistakes

People get confused about what multimodality actually requires. Here are the mistakes I see:

Mistake one: thinking you need to train from scratch. You do not. Model inference is computationally cheap and can be done locally, while model training is extremely expensive and requires massive computational resources. You can use pretrained vision encoders. You can use pretrained language models. The skill is in connecting them.

Mistake two: treating images as just another modality with no special handling. Images have spatial structure. Pixels next to each other are related. Text is sequential. Audio is temporal. Each modality has its own inductive biases, and your architecture should respect them.

Mistake three: ignoring the information bottleneck. When you tokenize an image into 256 tokens, you are throwing away information. You need to understand what gets preserved and what gets lost.

### Success Indicators

By the end of this module, you should be able to:

1. Explain why you cannot just feed raw pixels into a transformer
2. Describe three approaches to multimodal integration and their tradeoffs
3. Build a VQVAE that tokenizes and reconstructs images
4. Implement basic diffusion for image generation
5. Connect text conditioning to image generation
6. Integrate image generation into your storyteller

The fundamental task of a language model is next-word prediction. We are about to expand that. Next-token prediction, where tokens can represent anything. Text, images, audio, video. The same architecture, the same training loop, the same sampling procedure. Just a broader vocabulary.

Let us begin.

## Lesson 2: VQVAE Image Tokenizer

You just learned that multimodal models need to handle text, images, and audio in a unified way. Now we face a practical problem: language models work with discrete tokens, but images are continuous grids of pixel values. How do we bridge this gap?

We build a tokenizer for images. The same way BPE tokenization converts text into discrete tokens that a transformer can process, we need something that converts images into discrete tokens. That something is VQVAE, the Vector Quantized Variational Autoencoder.

Let's build one.

### Why Discrete Tokens for Images?

Think about what you already know from the earlier chapters. The Transformer architecture processes sequences of tokens. Each token gets embedded into a vector in a Common Embedding Space, positions get encoded, and attention lets tokens talk to each other. This whole machinery assumes discrete inputs.

Images come as continuous values. A 256x256 RGB image is 256 * 256 * 3 = 196,608 floating point numbers. You could treat each pixel as a token, but that sequence length would murder your attention computation. And pixels are continuous, not discrete.

We need compression and discretization. VQVAE gives us both.

### The Core Idea

VQVAE has three parts:

1. An encoder that compresses the image into a smaller spatial representation
2. A codebook of learned discrete codes
3. A decoder that reconstructs the image from the codes

The encoder takes your 256x256 image and outputs, say, a 32x32 grid of continuous vectors. Each of those 1024 vectors gets matched to the nearest vector in your codebook. If your codebook has 8192 entries, you've now represented your image as 1024 integers, each between 0 and 8191.

Those integers are your image tokens. They can go into a transformer just like text tokens.

### Building the Encoder

The encoder is a convolutional neural network that progressively downsamples the image:

```python
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, embed_dim, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 256 -> 128
        x = F.relu(self.conv2(x))  # 128 -> 64
        x = F.relu(self.conv3(x))  # 64 -> 32
        x = self.conv4(x)          # 32 -> 32, but now embed_dim channels
        return x
```

Each strided convolution halves the spatial dimensions. After three such layers, a 256x256 image becomes 32x32. The final layer projects to your embedding dimension without further downsampling.

### The Codebook: Where Discretization Happens

This is the heart of VQVAE. You maintain a codebook of K vectors, each of dimension D. During the forward pass, you find the nearest codebook vector for each spatial location:

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        # z shape: (batch, embed_dim, height, width)
        z = z.permute(0, 2, 3, 1)  # (batch, height, width, embed_dim)
        z_flat = z.reshape(-1, z.shape[-1])  # (batch*height*width, embed_dim)
        
        # Compute distances to all codebook vectors
        distances = torch.cdist(z_flat, self.embedding.weight)
        
        # Find nearest codebook entry
        indices = distances.argmin(dim=1)
        
        # Look up quantized vectors
        z_q = self.embedding(indices)
        z_q = z_q.reshape(z.shape)
        
        return z_q.permute(0, 3, 1, 2), indices
```

The indices are your discrete tokens. A 32x32 spatial grid gives you 1024 tokens per image.

### The Gradient Problem

Here is where people get stuck. The argmin operation has zero gradient. You cannot backpropagate through "find the nearest neighbor." The loss cannot flow from the decoder back through the quantization step to the encoder.

The solution is the straight-through estimator. During the forward pass, you use the quantized vectors. During the backward pass, you pretend the quantization never happened and copy gradients directly from the decoder output to the encoder output:

```python
# In the forward pass
z_q = z + (z_q - z).detach()  # Forward: z_q, Backward: gradients flow to z
```

This looks like a trick because it is a trick. But it works. The encoder learns to produce vectors that are close to codebook entries, even though it never receives direct gradients about which codebook entry was selected.

### The Decoder

The decoder mirrors the encoder, using transposed convolutions to upsample:

```python
class Decoder(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1)
        
    def forward(self, z_q):
        x = F.relu(self.conv1(z_q))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        return x
```

### Training Objectives

The loss function has three terms:

```python
def vqvae_loss(x, x_recon, z, z_q, beta=0.25):
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x)
    
    # Codebook loss: move codebook vectors toward encoder outputs
    codebook_loss = F.mse_loss(z_q, z.detach())
    
    # Commitment loss: encourage encoder to commit to codebook entries
    commitment_loss = F.mse_loss(z, z_q.detach())
    
    return recon_loss + codebook_loss + beta * commitment_loss
```

The reconstruction loss teaches the system to preserve image information. The codebook loss updates the codebook vectors. The commitment loss prevents the encoder from growing its outputs arbitrarily large to game the quantization.

### Connecting to Multimodal Transformers

Once trained, your VQVAE gives you a way to convert images to tokens and back:

```python
# Image to tokens
z = encoder(image)
z_q, tokens = quantizer(z)  # tokens are integers

# Tokens to image
z_q = quantizer.embedding(tokens).reshape(.)
reconstructed = decoder(z_q)
```

Those token integers can now go into the same transformer that handles text. You might use Specific Encoders for each modality, but after encoding, everything lives in the same sequence. The model learns to distinguish modalities through Training Objectives that expose it to image-text pairs, not through any hardcoded knowledge of what is image versus text.

### Common Mistakes

Codebook collapse: all your encoder outputs map to the same few codebook entries. Fix this with codebook reset strategies or exponential moving average updates.

Blurry reconstructions: your compression is too aggressive. Either increase spatial resolution of the latent grid or increase codebook size.

Training instability: the commitment loss weight beta matters. Start with 0.25 and adjust based on whether the encoder or codebook is moving too fast.

### Success Indicators

Your VQVAE is working when:
- Reconstructed images look sharp and preserve semantic content
- Codebook utilization is high, meaning most entries get used
- The encoder produces vectors that cluster tightly around codebook entries

You now have a way to tokenize images. In the next lesson, we build on this foundation with diffusion models, which offer an alternative approach to image generation that has become dominant in recent systems.

## Lesson 3: Diffusion Model Basics

You just built a VQVAE that compresses images into discrete tokens. That's one way to generate images: tokenize them like text, then predict tokens autoregressively. But there's another approach that took over image generation in 2022 and completely changed what was possible. Let's build it.

### The Core Idea: Learning to Denoise

Here's what we're going to do. Take a clean image. Add noise to it. Lots of noise. So much noise that it becomes pure static. Then train a neural network to reverse that process, to look at noisy garbage and predict what the clean image should be.

That's it. That's diffusion.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def add_noise(clean_images, noise_level):
    """Add Gaussian noise to images at a specified level."""
    noise = torch.randn_like(clean_images)
    noisy_images = clean_images * (1 - noise_level) + noise * noise_level
    return noisy_images, noise

# Training is embarrassingly simple
clean_image = load_image()  # Your training data
noise_level = torch.rand(1)  # Random point in the diffusion process
noisy_image, actual_noise = add_noise(clean_image, noise_level)

# The model predicts the noise that was added
predicted_noise = model(noisy_image, noise_level)
loss = F.mse_loss(predicted_noise, actual_noise)
```

Wait, why predict the noise instead of the clean image? This is the epsilon objective parameterization. During training, making the model predict the amount of noise added to clean images turns out to be more stable than predicting the clean image directly. The gradients behave better. The model learns faster.

### Why Time Steps Matter

See that `noise_level` parameter? In practice we call these time steps, and they're crucial. Time steps indicate the point in the denoising trajectory and control how aggressively the model should denoise.

Think about it. If you show the model pure static (high time step), it should make bold guesses about the overall structure. Is this a face? A landscape? A dog? But if you show it an almost-clean image (low time step), it should make tiny refinements. Fix that pixel. Sharpen that edge.

Same model, completely different behavior depending on the time step.

```python
class DiffusionModel(nn.Module):
    def __init__(self, image_channels=3, hidden_dim=256):
        super().__init__()
        # Time embedding using sinusoidal frequencies
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # ... rest of the architecture
    
    def forward(self, noisy_image, time_step):
        time_emb = self.time_mlp(time_step)
        # Inject time embedding into the network
        #.
```

Time steps are embedded using sinusoidal frequencies to make them phase-aware, then passed through a shallow MLP to model frequency weights. You've seen this before. It's the same positional encoding idea from transformers, repurposed.

### The Inference Process: Iterative Denoising

Training is one forward pass. Inference is many. This is the key difference from GANs. Diffusion models are iterative in nature, unlike GANs which are one-shot.

```python
def generate_image(model, num_steps=50):
    # Start from pure noise
    image = torch.randn(1, 3, 64, 64)
    
    # Gradually denoise
    for t in reversed(range(num_steps)):
        time_step = torch.tensor([t / num_steps])
        predicted_noise = model(image, time_step)
        
        # Remove some of the predicted noise
        # (simplified - real schedulers are more sophisticated)
        image = image - predicted_noise * (1 / num_steps)
        
        # Add a tiny bit of fresh noise (except at the last step)
        if t > 0:
            image = image + torch.randn_like(image) * 0.01
    
    return image
```

Each step, the model looks at the current noisy image, predicts what noise is present, and removes some of it. The scheduler controls exactly how much to remove at each step. Get this wrong and your images look terrible.

### Latent Space: Making It Practical

Here's the problem. Running a neural network 50 times on a 1024x1024 image is computationally prohibitive. The memory alone would kill you.

State-of-the-art diffusion models are typically latent-space based because pixel-space diffusion is computationally prohibitive. Remember that VQVAE you built? Its encoder compresses images to a smaller latent space. Run diffusion there instead.

```python
class LatentDiffusion(nn.Module):
    def __init__(self, vae, diffusion_model):
        super().__init__()
        self.vae = vae  # Your VQVAE from last lesson
        self.diffusion = diffusion_model
    
    def encode(self, image):
        with torch.no_grad():
            return self.vae.encode(image)
    
    def decode(self, latent):
        with torch.no_grad():
            return self.vae.decode(latent)
    
    def generate(self, num_steps=50):
        # Diffusion happens in latent space
        latent = torch.randn(1, 4, 32, 32)  # Much smaller!
        
        for t in reversed(range(num_steps)):
            # ... denoising steps in latent space
            pass
        
        # Decode back to pixels only at the end
        return self.decode(latent)
```

A 512x512 image becomes maybe 64x64 latents. That's 64x fewer pixels to process at each denoising step. This is why Stable Diffusion can run on consumer GPUs.

### Architecture: From U-Nets to Transformers

U-Net architectures historically dominated diffusion models until recent transitions to transformer-based designs. The U-Net makes sense for images because it has skip connections that preserve spatial detail. But transformers scale better.

Diffusion Transformers benefit from advancements in transformer design like SwiGLU and QK normalization, and they have better scaling properties. When you want to throw more compute at the problem, transformers reward you more predictably.

In diffusion transformers, conditioning like time steps and class labels is typically injected via adaptive layer norm rather than cross-attention, which is more compute-efficient. Instead of adding extra attention layers for conditioning, you modulate the existing layer norms:

```python
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale_shift = nn.Linear(cond_dim, hidden_dim * 2)
    
    def forward(self, x, conditioning):
        x = self.norm(x)
        scale, shift = self.scale_shift(conditioning).chunk(2, dim=-1)
        return x * (1 + scale) + shift
```

### Flow Matching: The New Hotness

There's a variant called flow matching that's gaining traction. In flow matching, noise and clean data are connected through a straight path, unlike diffusion models where the path is not assumed to be straight.

The math is cleaner. Training is simpler. Results are comparable or better. If you're starting fresh today, flow matching might be the better foundation. But the intuitions transfer directly from what you've learned here.

### When to Use What

GANs are not fully replaced by diffusion models. They're still useful for ultra-real-time generation and specific use cases where one-shot generation is required. If you need an image in one forward pass, GANs win. If you can afford 20-50 steps and want higher quality and more control, diffusion wins.

### Common Mistakes

**Wrong noise schedule**: Too aggressive early, too gentle late, and your images look muddy. The scheduler matters enormously.

**Ignoring time conditioning**: If your model produces the same output regardless of time step, something is broken. The time embedding must actually affect the computation.

**Training on too little data**: Training high-quality, generic diffusion models requires large and diverse datasets to capture the target data distribution effectively. A few thousand images won't cut it for general generation.

### Success Indicators

You know it's working when:
- Generated images at high time steps show coherent global structure
- Generated images at low time steps show sharp details
- Intermediate steps show smooth transitions, not sudden jumps
- Different random seeds produce different but plausible images

Next lesson, we add text conditioning. For a text-to-image system, key components include text encoders, noisy latents, a time step, a scheduler, a core diffusion network, and a decoder. You've now built or understood all of these pieces. Time to connect them.

## Lesson 4: Text-to-Image Generation

You have trained a diffusion model. You have built a VQVAE. Now we connect them to language.

This is where things get genuinely interesting. DALL-E demonstrates that a transformer trained on concatenated text-image sequences can learn to generate images conditioned on text descriptions and perform zero-shot image transformations without explicit training for those tasks. That sentence contains a lot. Let's unpack it by building.

### The Core Insight: Images as Language

Remember what we learned about the transformer architecture: it is a general sequence model that can be applied to modalities beyond language by treating the data as a sequence of bytes or tokens and using an autoregressive next-element prediction objective. The key word is "general." The transformer does not care if the tokens represent words, image patches, audio frames, or anything else. It just predicts the next token.

So here is the plan. We take text. We tokenize it into a sequence. We take an image. We tokenize it into a sequence using our VQVAE from Lesson 2. We concatenate these sequences. We train a transformer to predict the next token.

```python
class TextToImageModel(nn.Module):
    def __init__(self, text_vocab_size, image_vocab_size, d_model=512, n_layers=12):
        super().__init__()
        # Total vocabulary: text tokens + image tokens
        self.total_vocab = text_vocab_size + image_vocab_size
        self.text_vocab_size = text_vocab_size
        
        # Single embedding table for both modalities
        self.token_embedding = nn.Embedding(self.total_vocab, d_model)
        self.position_embedding = nn.Embedding(512, d_model)  # max sequence length
        
        # Standard transformer decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=2048),
            num_layers=n_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.total_vocab)
    
    def forward(self, tokens):
        # tokens: [batch, seq_len] - concatenated text + image tokens
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        
        # Causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1))
        x = self.transformer(x, x, tgt_mask=mask)
        
        return self.output_proj(x)
```

This is surprisingly simple. The magic is not in the architecture. The magic is in the data preparation.

### Preparing the Training Data

Each training example is a text description concatenated with image tokens:

```python
def prepare_training_example(text, image, text_tokenizer, vqvae):
    # Tokenize text
    text_tokens = text_tokenizer.encode(text)  # e.g. [45, 892, 103,.]
    
    # Tokenize image using trained VQVAE
    with torch.no_grad():
        image_tokens = vqvae.encode(image)  # e.g. [0, 234, 891,.]
    
    # Offset image tokens to avoid collision with text vocabulary
    image_tokens = image_tokens + text_tokenizer.vocab_size
    
    # Add special separator token
    separator = torch.tensor([SEP_TOKEN])
    
    # Concatenate: [text tokens] [SEP] [image tokens]
    full_sequence = torch.cat([text_tokens, separator, image_tokens])
    
    return full_sequence
```

The training objective is identical to language modeling: predict the next token. When the model sees text tokens, it learns to predict more text tokens or the separator. When it sees the separator, it learns to start generating image tokens. When it sees image tokens, it learns to predict more image tokens that are consistent with the text description.

### Why This Works: Analysis by Synthesis

A key motivation for using generative models for unsupervised learning is the principle "What I cannot create, I do not understand" and its inverse, "analysis by synthesis." The idea is that a model capable of generating coherent, diverse outputs must build representations useful for understanding tasks.

The model learns to connect text and images because predicting the right image tokens requires understanding the text. If the text says "a red ball on green grass," the model must understand "red," "ball," "green," and "grass" well enough to generate the corresponding visual tokens.

This is the same principle that makes language models work. Autoregressive language modeling, though a local next-token prediction task, can require and incentivize deep understanding to perform well. The same applies to image generation conditioned on text.

### Generation: The Inference Loop

At inference time, we provide text and let the model generate image tokens:

```python
def generate_image(model, text, text_tokenizer, vqvae, temperature=1.0):
    # Tokenize the text prompt
    text_tokens = text_tokenizer.encode(text)
    separator = torch.tensor([SEP_TOKEN])
    
    # Start with text + separator
    tokens = torch.cat([text_tokens, separator]).unsqueeze(0)
    
    # Generate image tokens autoregressively
    num_image_tokens = 256  # 16x16 grid from VQVAE
    
    for _ in range(num_image_tokens):
        logits = model(tokens)
        next_token_logits = logits[0, -1,:] / temperature
        
        # Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    
    # Extract image tokens and decode
    image_tokens = tokens[0, len(text_tokens) + 1:] - text_tokenizer.vocab_size
    image = vqvae.decode(image_tokens.view(16, 16))
    
    return image
```

### The Common Embedding Space

Notice that text and image tokens share the same embedding space. This is what enables the model to learn cross-modal relationships. When the model predicts image token 234 after seeing "red," it learns that token 234 corresponds to red-colored regions in images.

This creates what researchers call a Common Embedding Space. ImageBind: One Embedding Space To Bind Them All explores this idea further, showing that binding multiple modalities into a single embedding space enables powerful zero-shot transfer.

### Zero-Shot Capabilities

Here is what surprised the original DALL-E researchers. The model can perform image transformations it was never explicitly trained for:

```python
# Colorization: describe a grayscale image and ask for color
prompt = "a colorful photograph of a sunset over the ocean"

# Adding elements: describe what should be added
prompt = "a photograph of a living room with a cat sleeping on the couch"

# Style transfer: describe the style you want
prompt = "a painting of a mountain landscape in the style of Van Gogh"
```

The model generalizes because it learned the underlying relationship between text descriptions and visual content, not just memorized specific examples.

### Where Learners Get Stuck

**Vocabulary collision.** If you forget to offset image tokens, they will collide with text tokens and the model will be confused. Always add the text vocabulary size to image token indices.

**Sequence length.** A 256x256 image with 16x16 VQVAE patches produces 256 image tokens. Add 77 text tokens and you need a context window of 333+ tokens. Plan your positional embeddings accordingly.

**Training data quality.** The model can only learn what the data teaches. If your captions are generic like "a photo," the model cannot learn fine-grained control. Use detailed, accurate descriptions.

**Temperature during generation.** Too low and you get boring, repetitive images. Too high and you get incoherent noise. Start with 1.0 and adjust based on results.

### Success Indicators

You know your text-to-image system works when:

1. Given "a red square," it generates something red and square-shaped
2. Given "a blue circle," it generates something blue and circular
3. Changing one word in the prompt changes the corresponding aspect of the image
4. The model generalizes to descriptions it has not seen verbatim

### Connection to Diffusion

The approach we built here uses autoregressive generation of discrete image tokens. Movie Gen Video and similar systems use diffusion models instead, where text conditioning guides the denoising process. We then move on to a pre-training stage with joint text-to-image and text-to-video generation where we progressively increase the resolution from 256p to 768p.

Both approaches work. Autoregressive is simpler to understand and implement. Diffusion often produces higher quality results at larger scales. The principles of cross-modal learning apply to both.

### What You Built

You now have a complete text-to-image pipeline:
- Text tokenization from your language model work
- Image tokenization from your VQVAE
- A transformer that learns the mapping between them
- An inference loop that generates images from text

This is the foundation of Architectural Unification: using the same model architecture for multiple modalities. In the next lesson, we integrate everything into a complete multimodal system.

## Lesson 5: Multimodal Integration

You have built image tokenizers. You have trained diffusion models. You have generated images from text. Now we put it all together. This final lesson is where separate streams become a unified system, where your Storyteller learns to see and speak and create in the same breath.

### Why Multimodality Matters

Here is the fundamental insight that drives everything in this lesson: the best world models will incorporate all sensory modalities and make decisions based on their productive combination, not just one modality.

Think about how you experience a story. You do not process the words separately from the images in your mind. You do not hear a description and then later, in some disconnected step, visualize it. Everything happens together. The text informs the image. The image disambiguates the text. This is what we are building.

There are exactly two reasons to combine multimodal streams. First, translation: capturing all information from one modality in another. When you describe a sunset in words, you want the image to reflect everything that description contains. Second, disambiguation: using one modality to resolve ambiguities in another. The word "bank" could mean a financial institution or a riverbank. An accompanying image resolves this instantly.

### The Fusion Spectrum

Multimodal integration follows a spectrum from early to late fusion. Understanding where you sit on this spectrum determines everything about your architecture.

Late fusion is what you already know. CLIP-style contrastive learning between image and text encoders is relatively late fusion because the encoders do not interact directly until the loss function pressures their embeddings to align. Each modality gets processed by its own specialist network. Only at the end do they meet. This is simple. This is clean. This is also limited.

Early fusion goes the other direction. Super early fusion can involve directly concatenating raw inputs from different modalities, like adding depth as a fourth channel to RGB images, when their dimensions align. The modalities mix from the very beginning. Every layer processes combined information.

Most practical systems live somewhere in between. Let me show you the options.

### Four Architectural Choices

**Token Concatenation**

This is the simplest approach and the one you should try first. Token concatenation solves multimodality by first converting all modalities to tokens, then concatenating them so the model does not need to distinguish their original modality.

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, image_vocab_size, d_model, n_layers):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.image_embed = nn.Embedding(image_vocab_size, d_model)
        self.transformer = TransformerDecoder(d_model, n_layers)
        self.text_head = nn.Linear(d_model, vocab_size)
        self.image_head = nn.Linear(d_model, image_vocab_size)
        
    def forward(self, text_tokens, image_tokens):
        # Embed each modality
        text_emb = self.text_embed(text_tokens)
        image_emb = self.image_embed(image_tokens)
        
        # Concatenate along sequence dimension
        combined = torch.cat([text_emb, image_emb], dim=1)
        
        # Process with shared transformer
        hidden = self.transformer(combined)
        
        # Split outputs back
        text_len = text_tokens.shape[1]
        text_hidden = hidden[:,:text_len]
        image_hidden = hidden[:, text_len:]
        
        return self.text_head(text_hidden), self.image_head(image_hidden)
```

In transformers, everything being tokens all the way down makes it easy to add tokens from different modalities to a sequence. Your VQVAE from Lesson 2 already produces discrete tokens. Your text tokenizer from Chapter 6 produces discrete tokens. Concatenate them. Done.

**Cross-Attention**

Cross-attention allows two token streams from different modalities to interact through asymmetric attention operations where one stream provides queries and the other provides keys and values.

```python
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query_modality, kv_modality):
        # Query comes from one modality (e.g. text)
        q = self.q_proj(query_modality)
        # Keys and values come from another (e.g. image)
        k = self.k_proj(kv_modality)
        v = self.v_proj(kv_modality)
        
        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        weights = F.softmax(scores, dim=-1)
        return self.out_proj(torch.matmul(weights, v))
```

This is what lets your text tokens "look at" your image tokens and decide which visual features matter for the current word.

**Adaptive Layer Norm**

This is the clever one. Adaptive layer norm uses conditioning information from one modality to predict scaling and shifting parameters that steer the processing of another modality in a parameter-efficient way.

```python
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_proj = nn.Linear(cond_dim, d_model)
        self.shift_proj = nn.Linear(cond_dim, d_model)
        
    def forward(self, x, conditioning):
        # Conditioning could be text embedding, timestep, etc.
        scale = self.scale_proj(conditioning)
        shift = self.shift_proj(conditioning)
        return self.norm(x) * (1 + scale) + shift
```

In neural network architecture design, you can replace any component with an arbitrarily complex function if you believe more useful processing is needed before deriving parameters like layer norm shift and scale. The diffusion models from Lesson 3 use exactly this pattern for timestep conditioning. Now you use it for text conditioning too.

**Joint Embedding Spaces**

Learning joint embedding spaces through contrastive learning creates a single representation space where corresponding multimodal pairs have similar embeddings while mismatched pairs are pushed apart. You built this intuition with CLIP in earlier work. The key insight is that Semantic Linking Across Modalities happens not through explicit rules but through the pressure of the training objective.

### Putting It Together

Your complete Storyteller architecture combines these pieces:

1. Text encoder processes the story prompt
2. Cross-attention lets image generation attend to text features
3. Adaptive layer norm steers the diffusion process based on text
4. Token concatenation allows autoregressive generation across modalities

The training involves multiple stages. First, pretrain each modality encoder separately. Then, fine-tune with multimodal data. This follows the pattern from Chapter 14 on SFT, where we discussed how Loop Alignment between stages creates capabilities neither stage has alone.

### Common Mistakes

The biggest mistake is fusing too early with too little data. Early fusion requires the model to learn cross-modal correlations from scratch. Late fusion leverages pretrained unimodal models. Start late, move earlier only when you have evidence it helps.

The second mistake is ignoring the Reversal Curse. A model trained on "A is B" may not know "B is A". If you train on text-to-image, do not assume image-to-text works. You need bidirectional data.

The third mistake is skipping Ablation Studies. When multimodal performance improves, you need to know why. Was it the architecture? The data? The training schedule? Remove components systematically.

### Success Indicators

Your multimodal system works when:
- Text descriptions generate coherent images
- Image features influence text generation appropriately
- The system handles ambiguous inputs by leveraging cross-modal context
- Performance exceeds what either modality achieves alone

Run Adversarial Testing with edge cases. What happens when text and image contradict? What happens with abstract concepts? These tests reveal whether your fusion is real or superficial.

### The Complete Picture

You have now built every piece of a Storyteller. From bigram models to transformers to tokenizers to diffusion to multimodal integration. The magic has dissolved into mechanics. And the mechanics, as promised, turned out to be more beautiful than the magic.

This multimodal integration, text, images, and audio, is not just a collection of separate features. They work together to create something that feels almost human. Almost like understanding.

Go build something with it.

## Practice

Build VQVAE image tokenizer, implement diffusion model basics, create text-to-image generation, and integrate multimodal capabilities.

### Setup

You need your transformer from previous modules, plus image handling:

```bash
pip install pillow torchvision matplotlib
```

Create a working directory:

```
multimodal/
├── vqvae.py          # Vector quantized autoencoder
├── diffusion.py      # Basic diffusion model
├── text_to_image.py  # Generation pipeline
├── multimodal.py     # Integrated system
└── data/             # Sample images for testing
```

Download a small image dataset for testing. CIFAR-10 works well for initial experiments, small images (32x32), easy to iterate on.

### Part 1: VQVAE Image Tokenizer

Build a vector quantized variational autoencoder that converts images to discrete tokens:

1. **Encoder**: Convolutional network that maps images to continuous latent vectors
2. **Codebook**: Learnable dictionary of K vectors (start with K=512)
3. **Quantization**: Replace each latent vector with its nearest codebook entry
4. **Decoder**: Convolutional network that reconstructs images from quantized latents

The key insight: after training, an image becomes a sequence of codebook indices. These indices are tokens your transformer can process.

Train on CIFAR-10 or similar. Target reconstruction loss below 0.01 MSE.

### Part 2: Diffusion Model Basics

Implement the core diffusion loop:

1. **Forward process**: Add Gaussian noise to images over T timesteps
2. **Noise schedule**: Linear or cosine schedule from β_1 to β_T
3. **Denoising network**: U-Net that predicts noise given noisy image and timestep
4. **Sampling**: Start from pure noise, iteratively denoise

Start with T=1000 timesteps. Train the network to predict the noise that was added at each step.

### Part 3: Text-to-Image Generation

Connect your text encoder to image generation:

1. **Text conditioning**: Embed text prompts using your transformer encoder
2. **Cross-attention**: Condition the diffusion U-Net on text embeddings
3. **Classifier-free guidance**: Train with random prompt dropout, use guidance scale at inference

Simple prompts first: "red circle", "blue square". Verify the model learns the mapping before trying complex scenes.

### Part 4: Multimodal Integration

Build a unified system that handles both text and images:

1. **Unified tokenization**: Text tokens and image tokens share the same sequence
2. **Interleaved generation**: Model can generate text, then image, then more text
3. **Image understanding**: Given an image, model can answer questions about it

Create a simple web interface or CLI that demonstrates all capabilities.

### Success Criteria

- [ ] VQVAE reconstructs images with visible but acceptable quality loss
- [ ] Codebook utilization above 80% (most codes get used)
- [ ] Diffusion model generates recognizable images from noise
- [ ] Text conditioning produces images that match simple prompts
- [ ] Integrated system handles interleaved text-image sequences
- [ ] End-to-end latency under 30 seconds for image generation on GPU

### Common Pitfalls

**Codebook collapse**: All images map to the same few codes. Fix with EMA updates to codebook, commitment loss, or codebook reset for dead codes.

**Blurry reconstructions**: VQVAE decoder too weak. Add more layers, use perceptual loss instead of just MSE.

**Diffusion generates noise**: Learning rate too high, or not enough training steps. Diffusion models need many iterations, expect 100K+ steps for decent results.

**Text conditioning ignored**: Model learns to ignore prompts and generate average images. Increase guidance scale, verify cross-attention is wired correctly.

**Memory explosion**: Images are big. Use gradient checkpointing, reduce batch size, work with small images (32x32 or 64x64) until pipeline works.

**Slow sampling**: 1000 diffusion steps is slow. After basic version works, implement DDIM sampling for 50-step generation.

### Verification

Test your VQVAE:
```python
# Encode image to tokens, decode back
tokens = vqvae.encode(image)  # Should be ~64-256 integers
reconstructed = vqvae.decode(tokens)
# Visual comparison should show same content, some blur
```

Test your diffusion model:
```python
# Generate from noise
image = diffusion.sample(prompt="red circle")
# Should show something vaguely circular and red
```

Test integration:
```python
# Interleaved generation
output = model.generate("Here is a picture of a cat: [IMAGE] As you can see,")
# Should produce image token, then continue text
```

## Assessment

By the end of this module, you should be able to:

- **Explain how VQVAE works**: Describe the encoder-codebook-decoder architecture, understand why vector quantization enables discrete representations of continuous data, and implement the straight-through estimator that makes training possible despite the non-differentiable quantization step.

- **Understand VQGAN's improvements**: Articulate how adversarial training and perceptual losses produce sharper reconstructions than VQVAE alone, and explain why this matters for generating high-fidelity images from discrete tokens.

- **Describe diffusion models conceptually**: Explain the forward process (adding noise) and reverse process (learning to denoise), understand how this differs from the tokenization approach, and recognize when diffusion is preferable to discrete tokenization.

- **Build multimodal transformers**: Implement systems where image tokens and text tokens flow through the same attention mechanism, understand how to handle the interface between modalities, and design architectures that can condition on images to generate text or condition on text to generate images.

- **Compare the three approaches to multimodality**: Evaluate the tradeoffs between tokenizing everything, embedding into common space, and using separate specialized models, and choose the right approach for different applications.

Success means you can take an image, encode it into a sequence of discrete tokens using a VQVAE you understand completely, feed those tokens into your transformer alongside text tokens, and generate coherent outputs that demonstrate the model has learned meaningful relationships between visual and textual information. You should be able to explain every step of this pipeline, from raw pixels to codebook indices to attention weights to generated tokens, without hand-waving. When someone asks "how does GPT-4 see images?" you will not just know the answer; you will have built a miniature version yourself.

---

## You Are Now a Builder

Look at what you've done.

You started with a bigram model, a glorified counting table, and ended with a multimodal storyteller that writes and illustrates. Along the way, you built an autograd engine, implemented attention, constructed a full transformer, wrote CUDA kernels, trained with mixed precision across multiple GPUs, optimized inference with KV-caching and quantization, aligned your model with human preferences, and deployed it as a web application.

That's not a tutorial. That's a journey from "how do language models work?" to "I could build one myself."

The concepts you've internalized, attention, tokenization, optimization, distributed training, RLHF, aren't just vocabulary anymore. They're code you've written. When someone mentions "the quadratic complexity of attention," you remember implementing it. When a paper discusses "loss scaling for FP16 stability," you know exactly what problem they're solving because you solved it yourself.

**Where to go from here:**

*Scale up.* The 124M parameter GPT-2 you built is tiny by modern standards. The same architecture scales to billions of parameters, the code doesn't change much, but the infrastructure challenges multiply. Try training on more data, with more GPUs, and watch the emergent capabilities appear.

*Go deeper on alignment.* RLHF is just the beginning. Constitutional AI, debate, recursive reward modeling, the field of alignment is young and moving fast. Your understanding of the training pipeline puts you in a position to actually contribute.

*Explore architectures.* Mixture of Experts, state space models, retrieval augmentation, transformers aren't the only game in town. Now that you understand the baseline deeply, you can evaluate alternatives with real insight.

*Build something.* The storyteller is a starting point. What would *you* make with a language model you understand completely? A coding assistant? A game master? A research tool? The constraints are compute and imagination, and you've already proven you have plenty of the latter.

You didn't just learn about language models. You built one. That knowledge lives in your hands now, in muscle memory and intuition and the hard-won understanding that comes from making something work.

The magic is gone. The mechanics remain.

They're better.