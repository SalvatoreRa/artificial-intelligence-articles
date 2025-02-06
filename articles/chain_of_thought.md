# Chain-of-thought (CoT)

Chain-of-thought (CoT) Prompting is a technique that pushes the model to reason by intermediate steps. In other words, we are providing the model with intermediate steps to solve a problem so that the model understands how to approach a problem:

*"We explore the ability of language models to perform few-shot prompting for reasoning tasks, given a prompt that consists of triples: input, a chain of thought, and output. A chain of thought is a series of intermediate natural language reasoning steps that lead to the final output, and we refer to this approach as chain-of-thought prompting."--[source](https://arxiv.org/abs/2201.11903)*

![Cot Prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/cot.png?raw=true)
*from the [original article](https://arxiv.org/abs/2201.11903)*


**zero-shot Chain-of-thought (CoT)**

Instead of having to provide context, the authors of this study found that simply providing "Let's think step by step" was enough to suggest that the model reasons by intermediate steps:

*" Despite the simplicity, our Zero-shot-CoT successfully generates a plausible reasoning path in a zero-shot manner and reaches the correct answer in a problem where the standard zero-shot approach fails. Importantly, our Zero-shot-CoT is versatile and task-agnostic, unlike most prior task-specific prompt engineering in the forms of examples (few-shot) or templates (zero-shot)"--[source](https://arxiv.org/abs/2205.11916)*

![zero-shot Cot Prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/zeroshot-cot.png?raw=true)
*from the [original article](https://arxiv.org/abs/2205.11916)*

**Constrained-CoT**

is another variation in which you force the model to reduce the number of tokens in the output. According to the authors, today's LLMs are unnecessarily verbose and produce more tokens than necessary. In addition, larger models are generally more verbose and thus produce more tokens. This has a latency cost (as well as obviously computational) that can be problematic per service for users. Also, more verbiage means a lot of irrelevant unnecessary detail and a greater risk of hallucinations. In addition, CoT by requiring reasoning intermediates to be generated increases the number of tokens generated.

![verbosity of CoT prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/verbosity_cot.png?raw=true)
*CoT is increasing the number of generated tokens from  an LLM. from the [original article](https://arxiv.org/pdf/2407.19825)*

Therefore, the authors of [this study](https://arxiv.org/abs/2407.19825) suggest a new prompt that is a variation of the zero-shot prompt: *let's think step by step "and limit the length of the answer to n words* with n being the desired number of words.

![constrained CoT prompt](https://github.com/SalvatoreRa/tutorial/blob/main/images/constrained_cot.png?raw=true)
*Example of constrained CoT prompting, where n is 45. from the [original article](https://arxiv.org/pdf/2407.19825)*

For the authors, this prompt not only reduces the number of tokens generated but in several cases also leads to better reasoning (more exact answers on a reasoning database). This better reasoning is seen only with some models (LLaMA-2 70B and not with smaller models)

**Tree of Thoughts**

Tree of Thoughts (ToT) can be seen as a generalization of CoT, especially for more complex tasks. The idea is that the model maintains some degree of exploration to arrive at the solution. 

In this framework, the model produces several steps of thought as it moves toward solving the problem (similar to CoT or Self-consistency). In this case, though, the model not only generates these thoughts but evaluates them. In addition, there is a search algorithm (breadth-first search and depth-first search) that allows exploration but also backtracking

![TOT](https://github.com/SalvatoreRa/tutorial/blob/main/images/TOT.png?raw=true)
*from the [original article](https://arxiv.org/abs/2305.10601)*

An interesting development is to add an agent to conduct the search. In this article, they add a controller that precisely controls the search in the ToT. 

![TOT2](https://github.com/SalvatoreRa/tutorial/blob/main/images/TOT3.png?raw=true)
*from the [original article](https://arxiv.org/abs/2305.08291)*

Clarification these two methods are both complex and laborious. In one you have to generate the various steps, evaluate them, and conduct the research. In the second we have an external module trained with reinforcement learning that conducts the control and search. So instead of conducting multiple calls, [Hubert](https://github.com/dave1010/tree-of-thought-prompting) proposed a simple prompt to conduct ToT. This approach is simplified but seems to give superior results to simple CoT

```
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking,
then share it with the group.
Then all experts will go on to the next step, etc.
If any expert realises they're wrong at any point then they leave.
The question is...
```

**Multimodal CoT**

*Imagine reading a textbook with no figures or tables. Our ability to knowledge acquisition is greatly strengthened by jointly modeling diverse data modalities, such as vision, language, and audio. -[source](https://arxiv.org/abs/2302.00923)*

With the arrival of LLMs, there has been an increased interest in multimodal models. There are already several models that are multimodal, but most prompt engineering techniques are dedicated to traditional LLMs that do not take into account other modalities besides text.

![multimodal versus unimodal COT](https://github.com/SalvatoreRa/tutorial/blob/main/images/multimodal_vs_unimodal_cot.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2302.00923)*

Chain-of-thoughts (CoT) is a technique to improve reasoning skills but is not adapted to the presence of other modalities. In [this article](https://arxiv.org/pdf/2302.00923) they proposed a variant definite **Multimodal-CoT**. In the same manner, multimodal-CoT decomposes multi-step problems into intermediate reasoning steps (rationale) and then infers the answer. They used in this work a model that is 1B and purpose-built to consider both image and textual modalities. This approach is a two-stage framework, in the first step the rationale (the chain of thoughts) is created based on the multimodal information and then after that, the model generates the answer.

![multimodal COT](https://github.com/SalvatoreRa/tutorial/blob/main/images/multimodal_cot.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2302.00923)*

Here, is the pipeline in detail:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/multimodalcot.png)
*from the [original article](https://arxiv.org/pdf/2302.00923)*

# Is CoT really useful?l

*"CoT only helps substantially on problems requiring mathematical, logical, or algorithmic reasoning"* -[source](https://arxiv.org/pdf/2409.12183)

A recently published study suggests that CoT is useful only for some cases such as those problems requiring math or logical reasoning, while for problems requiring commonsense reasoning, there is not much advantage. So CoT would only help when you have a symbolic reasoning problem (*"We consider a problem to be symbolic if it can be grounded in a natural, well agreed-upon formal system”*) that is, it can be described in a formal system. 

![is Chain-of-thought really useful](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/cot_necessity.png)
*from the [original article](https://arxiv.org/abs/2409.12183)*

*CoT primarily helps with the execution step that performs computation and symbolic manipulation, but falls short of what LLMs with tool augmentation can do. We find that LMs prompted with CoT can generate executable formal solution plans and execute those plans better than direct answering. But using LMs to generate a solution plan and then using an external symbolic solver to solve the plan outperforms using CoT for both stages for these tasks.* -[source](https://arxiv.org/pdf/2409.12183)

In other words, CoT is needed when the problem requires multi-step solutions and you need to force the model to adhere to a plan. However, if we have the plan we can more efficiently use an external tool. for example, if we need to conduct calculations to solve a mathematical problem we can either use CoT or use LLM to generate an executable plan with another tool (such as Python code to run with the interpreter). The second case gives better results according to the authors

![is Chain-of-thought really useful](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/cot_necessity2.png)
*from the [original article](https://arxiv.org/abs/2409.12183)*

*"First, CoT is unnecessary for many problems where it is widely employed: there exist more efficient prompting strategies that yield similar performance for much lower inference cost. Second, we see a critical need to move beyond prompt-based CoT to more sophisticated approaches based on search, interacting agents, or models more heavily fine-tuned for CoT. "* -[source](https://arxiv.org/pdf/2409.12183)

CoT is expensive and should not be used for problems where its use does not bring benefit (common sense reasoning). Second for symbolic problems, we can leverage better approaches than simple CoT (such as agents and external tools).


  Articles describing in detail:
  * [To CoT or Not to CoT: Do LLMs Really Need Chain-of-Thought?](https://medium.com/gitconnected/to-cot-or-not-to-cot-do-llms-really-need-chain-of-thought-5a59698c90bb)
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)
  * [Multimodal Chain of Thoughts: Solving Problems in a Multimodal World](https://towardsdatascience.com/multimodal-chain-of-thoughts-solving-problems-in-a-multimodal-world-961a8ab9d0fa)
  * [Multimodal Chain of Thoughts: Solving Problems in a Multimodal World](https://towardsdatascience.com/multimodal-chain-of-thoughts-solving-problems-in-a-multimodal-world-961a8ab9d0fa)
 
  Suggested lecture:
  * [To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning](https://arxiv.org/abs/2409.12183)
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [original article CoT](https://arxiv.org/abs/2201.11903)
  * [original article zero-shot CoT](https://arxiv.org/abs/2205.11916)
