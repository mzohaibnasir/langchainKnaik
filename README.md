# Adaptation. Recall that a language model p

is a distribution over sequences of tokens x1:L
and thus can be used to score sequences:

` p(the,mouse,ate,the,cheese).`

It can also be used to perform conditional generation of a completion given a prompt:

` the mouse ate` ⇝ `the cheese.`

# task

A task is a mapping from inputs to outputs. For example, for question answering, we might have:
`          Input: What school did burne hogarth establish?
          Output: School of Visual Arts`

We use the term adaptation to refer to the process of taking a language model and turning it into a task model, given:

1. a natural language description of the task, and
2. a set of `training instances` (input-output pairs).

Adaptation refers to the process of fine-tuning a pre-trained language model to perform a specific task.
A language model is a distribution over sequences of tokens, allowing it to score sequences and generate completions.
For example:
` Scoring sequences: p(the,mouse,ate,the,cheese)`
`Generating completions: the mouse ate ⇝ the cheese`

# Tasks

A task is a mapping from inputs to outputs. For instance:
Input: What school did Burne Hogarth establish?
Output: School of Visual Arts
Adapting a Language Model to a Task
To adapt a language model to a task, we need:

1. Natural language description of the task: A brief description of the task in natural language.
2. Training instances: A set of input-output pairs to fine-tune the language model.

# By adapting a language model to a task, we can leverage its language understanding capabilities to perform specific tasks, such as question answering, text classification, and more.

#There are two primary ways to perform adaptation:

1. Training (standard supervised learning): train a new model that maps inputs to outputs, either by
   1. creating a new model that uses the language model as features (probing), or
   2. starting with the language model and updating it based on the training instances (fine-tuning), or
   3. something in between (lightweight fine-tuning).
2. Prompting (in-context learning): Construct a prompt (a string based on the description and training instances) or a set of prompts, feed those into a language model to obtain completions.
   1. Zero-shot learning: number of training examples is 0
   2. One-shot learning: number of training examples is 1
   3. Few-shot learning: number of training examples is few

# Which adaptation procedure should we go with?

1. Training can be challenging due to overfitting (just imagine fine-tuning a 175 billion parameter model based on 5 examples). How to do this effectively will be the topic of the adaptation lecture.
2. For now, we will be content with adaptation of GPT-3 using prompting. Note that the limitation of prompting is that we can only leverage a only small number of training instances (as many as can fit into a prompt). This is due to a limitation of Transformers, where the prompt and the completion must fit into 2048 tokens.

# Along the way, we will do ablations to see if model size and number of in-context training instances matters. Spoiler: `it does and more is better.`

# we can break down the the joint probability into the product of the conditional probabilities for each token by the chain rule.

# Perplexity: As explained earlier, it's a metric used to evaluate the performance of a language model. It essentially measures how well the model predicts the next word in a sequence. Perplexity doesn't deal with the training process itself, but rather how well the trained model performs.

#####################################################################################################3333

## langchain community

all third-party integrations will be available in langchain-community

## ollama - to run llms locally

- first do `ollama run <model name>` E pulling model
  -from langchain_community.llms import ( Ollama) ; llm = Ollama(model="llama2")
