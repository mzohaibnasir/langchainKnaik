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

# Perplexity, on the other hand, is a measure used to evaluate language models, particularly in the context of natural language processing. It quantifies how well a language model predicts a given sequence of tokens. Perplexity is calculated based on the probabilities assigned by the model to each token in the sequence. While perplexity is important for assessing the quality of language models, it does not directly address the vanishing gradient problem.

# Perplexity: As explained earlier, it's a metric used to evaluate the performance of a language model. It essentially measures how well the model predicts the next word in a sequence. Perplexity doesn't deal with the training process itself, but rather how well the trained model performs.

#####################################################################################################3333

## langchain community

all third-party integrations will be available in langchain-community

## ollama - to run llms locally

1. first do `ollama run <model name>` pulling model
2. from langchain_community.llms import ( Ollama) ; llm = Ollama(model="llama2")

################################################################

# API for deployment(langServe - fastAPI - SwaggerUI)

Langserve is for deployment

architecture: [APP] ------- [APIs]---------[Routes to LLMs]

###################################################################3333

# RAG: querying from data sources

Components:

1. load source data.(data ingestion)
2. You can load/transform/embed loaded source data.(conversion to chunks and then embeddings).
3. then data is stored in Vectorstore
4. Query vector store

###################################################################3

# Function calling

Learn how to connect large language models to external tools.

# Summarizer

A central question for building a summarizer is how to pass your documents into the LLM’s context window. Two common approaches for this are:

Stuff: Simply “stuff” all your documents into a single prompt. This is the simplest approach (see here for more on the create_stuff_documents_chain constructor, which is used for this method).

Map-reduce: Summarize each document on it’s own in a “map” step and then “reduce” the summaries into a final summary (see here for more on the MapReduceDocumentsChain, which is used for this method).

####################################################################333
In LangChain, a Runnable is a fundamental building block for creating data processing pipelines. It defines a standard interface for components that can be chained together to perform complex tasks. Here's a breakdown of what Runnables are and how they work:

What it is:

A Runnable is essentially a piece of code that can be invoked with data and produces an output.
Many LangChain components, like chat models, large language models (LLMs), data retrievers, and more, implement the Runnable interface.
This standardized interface ensures all these components can be seamlessly integrated and used together.
Key functionalities:

invoke(data): This is the core method that takes input data and processes it according to the Runnable's logic. The output can be anything, like text, data structures, or even calls to other Runnables.
stream(data) (optional): This method allows for processing data in chunks, which is useful for handling large datasets or real-time scenarios. It can return the results piece by piece as they become available.
batch(data_list) (optional): This method lets you process a list of inputs simultaneously, potentially improving efficiency for certain tasks.

pen_spark

#################################################################

# IN advaned RAG , we'll use LLMS using concept of chain-and-retrieval.

# create_stuff_documents_chain

is a function from a library called LangChain. It's designed to work with large language models (LLMs) like GPT-3 or Jurassic-1 Jumbo. Here's what it does:

Input: It takes a list of documents as input. These documents can be text passages, articles, emails, or anything you want the LLM to consider.

Formatting: It combines all the documents into a single prompt for the LLM. Each document is likely converted into text, and a separator (like a newline) might be added between them.

Feeding the LLM: The entire prompt, containing all the documents, is then fed to the LLM.

Important Note: LLMs have a limited context window. This means they can only effectively process a certain amount of information at once. "create_stuff_documents_chain" works best when the total length of all your documents fits within this context window. If it's too large, the LLM might get confused and generate inaccurate or nonsensical outputs.

Here's an analogy: Imagine you're showing a bunch of pictures to a friend and asking them a question. "create_stuff_documents_chain" is like putting all those pictures into a single collage before showing it to your friend. It works well if the collage isn't too overwhelming, but if there's too much information, your friend might have trouble understanding the big picture.

###################################################################

# RAG with agents - multiple sources(wiki, PDFs, texts , ARXiv)

# Tools are interface that an agent, chain or LLM can use to interact with the world.

suppose we have dependeny on folowing data sources ((wiki, PDFs, texts, ARXiv), we want tou integrate all these sources as wappers so we could implement this QA solution.

we'll use

1. Tools -> tooklits
2. Agents

we can use each source as different tool. and we can wrap all of these sources as toolkit. and, then, with helpp of agesmts we'll be able to every sort of QA search.
