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

Adaptation refers to the process of fine-tuning a pre-trained language model to perform a specific task. A language model is a distribution over sequences of tokens, allowing it to score sequences and generate completions. For example:
Scoring sequences: p(the,mouse,ate,the,cheese)
Generating completions: the mouse ate ⇝ the cheese
Tasks
A task is a mapping from inputs to outputs. For instance:
Input: What school did Burne Hogarth establish?
Output: School of Visual Arts
Adapting a Language Model to a Task
To adapt a language model to a task, we need:
Natural language description of the task: A brief description of the task in natural language.
Training instances: A set of input-output pairs to fine-tune the language model.
By adapting a language model to a task, we can leverage its language understanding capabilities to perform specific tasks, such as question answering, text classification, and more.

## langchain community

all third-party integrations will be available in langchain-community

## ollama - to run llms locally

- first do `ollama run <model name>` E pulling model
  -from langchain_community.llms import ( Ollama) ; llm = Ollama(model="llama2")
