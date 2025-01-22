<summary><b>What does it mean LLM's hallucination?</b></summary>
Anyone who has interacted with ChatGPT will notice that the model generates responses that seem consistent and convincing but occasionally are completely wrong. 

![hallucination](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination.png?raw=true)
*from the [original article](https://arxiv.org/abs/2311.05232)*

Several solutions have obviously been proposed: 
* Provide some context in the prompt (an article, Wikipedia, and so on). Or see RAG below.
* If you have control over model parameters, you can play with temperature or other parameters.
* Provide instructions to the model to answer "I do not know" when it does not know the answer.
* Provide examples so it can better understand the task (for reasoning tasks).
* Adding an external memory

Investigating further, [this study](https://arxiv.org/pdf/2304.13734) provides three reasons why hallucinations emerge. Specifically discussing a model that generates a sequence of tokens in an autoregressive manner: 
* The LLM commits to a token at a time. Even when we choose a low temperature, during decoding we maximize the likelihood of each token given the previous tokens, but the probability of the entire correct sequence may be low. For example, once part of the sequence is generated the model will continue from there and will not correct (For a sequence to be completed "Pluto is the" if the model continues as "Pluto is the smallest," it will likely continue with "Pluto is the smallest dwarf planet in our solar system." and not with the correct completion).  The model completes a sentence that it does not know how to complete. For example, for the description of a city, the model predicts to describe its population, but having no information about its population in its parametric memory generates a hallucination.
* The second reason is that although there are multiple correct chances to complete the sequence, the incorrect one has a higher likelihood.
* Third, when we use an LLM we do not use the maximal probability for the next word, but we sample according to the distribution over the words. This makes it so that in some cases we sample words that result in false
information.

![hallucination LLM causes](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination_causes.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2304.13734)*

Other studies indicate among the causes as problems with the training dataset. Lack of relevant data to answer a question can lead to the emergence of hallucinations. The presence of duplicated data. In fact, duplicated data impacts the performance of the model.  With smaller models seeming more sensitive to repeated data. Repetitions seem to cause the model to store this data and lead to performance degradation (for more details you can read [here](https://arxiv.org/abs/2205.10487) and [here](https://aclanthology.org/2022.naacl-main.387/)). For some authors, hallucinations are also derived from inherent model limitations. In [this study](https://arxiv.org/abs/2305.14552) show that LLMs still rely on memorization at the sentence level and statistical patterns at the corpora level instead of robust reasoning, this is one of the reasons for the manifestation of hallucinations. This is also observed by the fact that LLMs are sensitive to reverse curse (lack of logical deduction, where an LLM trained on A implies B, hallucinates when questioned on B implies A, [here more details](https://arxiv.org/abs/2305.14552)). Other causes of hallucinations have been defined as the tendency of the model to be overconfident ([here](https://arxiv.org/abs/2307.11019)), favoring co-occurrences words over factual answers and thus generating spurious correlations ([here](https://arxiv.org/abs/2310.08256)) and the tendency to sycophancy to please the user ([here](https://arxiv.org/abs/2308.03958)). 

![an example of sycophancy](https://github.com/SalvatoreRa/tutorial/blob/main/images/sycophancy.png?raw=true)
*an example of sycophancy, from the [original article](https://arxiv.org/pdf/2308.03958)*

Recent studies show how fine-tuning and instruction tuning can increase an LLM's tendency to hallucinate (there is a correlation between examples unknown to the model and the tendency to hallucinate on prescient knowledge). Hallucinations then may also emerge due to a discrepancy between new knowledge and previously acquired knowledge (more details [here](https://arxiv.org/abs/2405.05904))

There is a dissonance between what is the meaning of the term "hallucination" in human psychology ("when you hear, see, smell, taste, or feel things that appear to be real but only exist in your mind") and what is meant by hallucination in machine learning. In a [recent article](https://arxiv.org/abs/2402.01769) they took care to align these two definitions. They divided hallucinations that are seen in the case of LLM into different types. This new classification is interesting because it is difficult to be able to resolve all causes of hallucinations in LLM with one approach. Instead, by having a classification of the various subtypes, one can think about acting on each subtype (which is perhaps the one most relevant to our task):

_By grounding our discussion in specific psychological constructs, we seek to shed light on these phenomena in language models, paving the way for the development of targeted solutions for different types of ”hallucinations.” - [source](https://arxiv.org/abs/2402.01769)_

![hallucination LLM causes](https://github.com/SalvatoreRa/tutorial/blob/main/images/llm_hallucination_psicology.png?raw=true)
*from the [original article](https://arxiv.org/abs/2402.01769)*

For example, **confabulation** is a hallucination that emerges from the LLM unpredictably, owing to internal factors that are unrelated to the prompt. In a sense, this type of hallucination is associated with the LLM's uncertainty in responding to the prompt. In [this paper](https://www.nature.com/articles/s41586-024-07421-0) they show that a high uncertainty in the response is an indication of confabulation (this uncertainty can be estimated with an entropy that is associated with the meaning of the response).

Another type of hallucination is **contextual hallucination**. In this case, although we provide the context (and thus the correct facts) in the prompt the model fails to generate the correct output. According to this [study](https://arxiv.org/pdf/2407.07071) contextual hallucinations are related to the extent to which an LLM attends to the provided contextual information. In other words, it depends on the relationship between the attention (attention weights) associated with the context and the attention devoted to the newly generated tokens. Therefore, one can classify when a model will generate these kinds of hallucinations by extracting the attention weights and constructing a linear classifier.

![hallucination RAG causes](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination_contextual.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2304.13734)*

One reason for the emergence of these contextual hallucinations is that the model gives too much attention to noise:

*we visualize the normalized attention scores assigned to different parts of the context by a Transformer. The task is to retrieve an answer embedded in the middle of a pile of documents. The visualization reveals that Transformer tends to allocate only a small proportion of attention scores to the correct answer, while disproportionately focusing on irrelevant context. --[source](https://arxiv.org/abs/2410.05258)*

![Transformer often over-attends to irrelevant context (i.e., attention noise)](https://github.com/SalvatoreRa/tutorial/blob/main/images/attention_noise.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2410.05258)*

So too much attention score is assigned to irrelevant context, this confuses the model and leads to generating a wrong response. This misallocation of attention scores is a major cause of contextual hallucinations. So one way to be able to reduce this type of hallucination is to reduce the noise in the attention pattern. [In this article](https://arxiv.org/abs/2410.05258) proposes the **differential transformer (DIFF Transformer)** for the very purpose of reducing this noise.

*The differential attention mechanism is proposed to cancel attention noise with differential denoising. Specifically, we partition the query and key vectors into two groups and compute two separate softmax attention maps. Then the result of subtracting these two maps is regarded as attention scores. The idea is analogous to differential amplifiers [19] proposed in electrical engineering, where the difference between two signals is used as output, so that we can null out the common-mode noise of the input. --[source](https://arxiv.org/abs/2410.05258)*

![The differential attention mechanism maps query, key, and value vectors to outputs](https://github.com/SalvatoreRa/tutorial/blob/main/images/diff_attention.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2410.05258)*

![Evaluation of contextual hallucination on text summarization and question answering](https://github.com/SalvatoreRa/tutorial/blob/main/images/diff_transformer.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2410.05258)*

In simple words, the model learns how to reduce the weight given to attention noise and to focus only on important information. This system is inspired by the idea that the difference between two signals cancels out the noise. On two retrieval tasks, the authors show that the DIFF transformer reduces contextual information (they focus on the cases where the input
context contains correct facts, but the model still fails to produce accurate outputs). They also show in the work that the model also has improvements in both in-context learning and long-context retrieval.

![The differential attention mechanism maps query, key, and value vectors to outputs](https://github.com/SalvatoreRa/tutorial/blob/main/images/contextual-hallucination-reduction.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2410.05258)*

Another interesting point is whether **one can identify the presence of the error from the internal state of the model**.  [In this paper](https://arxiv.org/abs/2410.02707) discuss how there are elements on the truthfulness of the response and how this can be derived from the internal state of the model:

*another line of work has explored the internal representations of LLMs, suggesting that LLMs encode signals of truthfulness. In this work, we reveal that the internal representations of LLMs encode much more information about truthfulness than previously recognized. [source](https://arxiv.org/abs/2410.02707)*

An interesting point is that they describe hallucinations as a function of model responses and model perception of error. They conduct sampling for each question and check whether the model responds correctly or incorrectly:

*More generally, we categorize the errors by logging three specific features for each example: (a) the number of different answers generated; (b) the frequency of the correct answer; and (c) the frequency of the most common incorrect answer. [source](https://arxiv.org/abs/2410.02707)*

This reveals some interesting patterns: 
* **Refuses to answer.** The model responds that it cannot answer the question
* **Consistently correct** Answers correctly in at least half of the cases. Interesting that the model either always responds correctly or sometimes responds incorrectly (despite responding correctly in most cases)
* **Consistently incorrect** Consistently generates the same incorrect response in at least half of the cases. Paradoxically in some cases, the model responds incorrectly for most of the sampling and occasionally responds correctly, this shows that the model has some understanding of the subject in cases even when it is wrong.
* **Two competing** Generates both correct and incorrect responses at similar rates
* **Many answers** Generates over 10 distinct answers

![taxonomy error generarate by the LLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/errors_llm_according_internal_state.png?raw=true)
*The figure illustrates three representative error type. from the [original article](https://arxiv.org/pdf/2410.02707)*

After that, they conduct an interesting experiment where by accessing the internal state of the model they train a classifier to see if they can predict the error. This shows that the model has a greater concept of truthfulness than thought (truthfulness refers to how an LLM internally represents the accuracy of its responses). Second, understanding how the model represents truthfulness allows us to develop diagnostic tools and then reduce hallucinations. Contrary to earlier thinking LLMs do not have a universal sense of truth (rather this is associated with specific skills). This makes it more complicated to develop subsequent diagnostic tools because a tool developed within a skill (such as factual accuracy) fails to generalize across different types of tasks, such as from fact-checking to sentiment analysis.

![taxonomy error generated by the LLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/failed_generalization_internal_llm_truthfulness.png?raw=true)
*This implies that the apparent generalization does not stem from a universal internal encoding of truthfulness but rather reflects information already accessible through external features like logits.
. from the [original article](https://arxiv.org/pdf/2410.02707)*

Also interesting is the fact that LLM can generate a wrong answer even when its internal state encodes the correct answer (discrepancy between the model's external behavior and internal states). According to the authors, this could result from the fact that the models have a bias to predict the most likely tokens (and this overrides internal mechanisms promoting truthfulness, after all, the correct answer may not be the most likely answer)

[This paper](https://arxiv.org/abs/2501.08292) presents a broad and inclusive benchmark of model-generated hallucinations (HALOGEN), consisting of principally two types of tasks: text summarization and open-domain text generation tasks. HALOGEN seeks to evaluate the factuality of models and not answers that are subjective or creative. Therefore, it does not introduce tasks where the evaluation can be subjective. HALOGEN consists of nine tasks: Code Packages (LLMs can hallucinate code libraries), Summarization, Simplification, Biographies, Rationalization (Binary), Rationalization (Numerical), Scientific Attribution, Historical Events, False Presuppositions.

![description of halogen and example of hallucinations ](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination.webp?raw=true)
*from the [original article](https://arxiv.org/abs/2501.08292)*



Articles describing in detail:
  * [A Requiem for the Transformer?](https://towardsdatascience.com/a-requiem-for-the-transformer-297e6f14e189)
  * [AI Hallucinations: Can Memory Hold the Answer?](https://towardsdatascience.com/ai-hallucinations-can-memory-hold-the-answer-5d19fd157356)
  * [Chat Quijote and the Windmills: Navigating AI Hallucinations on the Path to Accuracy](https://levelup.gitconnected.com/chat-quijote-and-the-windmills-navigating-ai-hallucinations-on-the-path-to-accuracy-0aaecf46354c)
  * [The Dream Machine: Decoding Why LLMs Hallucinate Reality](https://medium.com/gitconnected/the-dream-machine-decoding-why-llms-hallucinate-reality-fea8846a5bc5)
 
  Suggested lecture:
  * [Speak Only About What You Have Read: Can LLMs Generalize Beyond Their Pretraining Data?](https://pub.towardsai.net/speak-only-about-what-you-have-read-can-llms-generalize-beyond-their-pretraining-data-041704e96cd5)
  * [a good survey on the topic](https://arxiv.org/abs/2311.05232)
  * [discussing hallucination in psychological terms](https://arxiv.org/abs/2402.01769)
