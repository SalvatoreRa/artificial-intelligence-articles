# What is the role of small models in the LLM era?

![small model overview](https://github.com/SalvatoreRa/tutorial/blob/main/images/SLM_overview.png?raw=true) *from [here](https://arxiv.org/pdf/2409.06857)*
  

  LLMs have shown incredible ability and some reasoning properties. Motivated by the growth in performance, this has led to an increase in the number of parameters and larger and larger models. Obviously this scaling has a huge computational and energy cost. Therefore the question remains: is there still room for small models?


![small model versus large model comparison](https://github.com/SalvatoreRa/tutorial/blob/main/images/small_model_versus_large_model_comparison.png?raw=true) *from [here](https://arxiv.org/pdf/2409.06857)*

Building on these differences, there is a heated discussion about the future of small LLMs. We can imagine that there are two potential settings: 
* **Collaboration.** In this case large LLMs and small models collaborate, thus trying to balance efficiency and accuracy
* **Competition.** In certain niches it is preferable to leverage a small model than a large model, especially when we are interested in low cost, ease of deployment, and interoperability.

Speaking of collaboration we can see several possibilities:
* **Data curation.** It is critical for an LLM to have a good training dataset (and its performance depends largely on it). A small model can help in data curation by allowing faster filtering and deduplication of the dataset. Or a small LLM can be used to curate instruction-tuning datasets. Today there is *weak-to-strong generalization* in which a small LLM is used to improve the capabilities of a model. This is especially useful because today the capabilities of LLMs are getting stronger and stronger and you need experienced people to be able to cure alignment datasets.
* **Efficient inference.** As the number of users increases, the cost of inference is getting larger and larger. Since user queries are not always super complex, simpler models might be enough to answer them. so it has been proposed to make inference more efficient by using an ensemble of models of different sizes. Or to use model cascading where increasing models cascade: a smaller model tries to answer and transfers the query to a larger one if it has difficulty answering. Model routing is another alternative where a model decides which LLM responds. *Speculative decoding* is another technique in which a small LM helps the LLM by providing multiple token candidates in parallel that are then evaluated and refined by the LLM. This allows for faster inference.
* **Evaluating LLMs** Evaluating LLMs is a complex task, and the more they improve their skills, the more difficult it becomes for us humans to create benchmarks. So you can use a small LM to analyze the performance of a model or use it to predict the potential performance of an LLM we want to train (e.g., for fine-tuning, instruction tuning). 
* **Domain adaptation** Conducting fine-tuning of large models is quite expensive (especially when they are more than 100B). Conducting fine-tuning of a small model is much less expensive. You can still synergize the two types of models. In white-box adaptation, we use a specific domain fine-tuned small model to adjust the token distribution of a frozen LLM for an or a specific domain. Black-box adaptation we use a small model to provide knowledge to the LLM to answer a question or conduct a task.  Retrieval Augmented Generation (RAG) is an example of this approach in which we provide knowledge using a small model for retrieval
* **Prompt-based learning** In this case LLM prompts are crafted to facilitate the resolution of a task (e.g., in few-shot learning). A small model can be used to improve the prompt that is used with the large model
* **Deficiency repair** An LLM can generate outputs that are toxic, hallucinatory, or incorrect and one can use small LLM to correct these defects. An interesting approach is *Contrastive Decoding* where you use an LLM and a small LLM and exploit the contrast in the output, thus choosing tokens that maximize their log-likelihood difference

![How small LM model can enhance LLMs](https://github.com/SalvatoreRa/tutorial/blob/main/images/small_LM_enhance_LLM.png?raw=true) *from [here](https://arxiv.org/pdf/2409.06857)*

Similarly, LLMs can enable the enhancement of small models:
* **Knowledge Distillation** This is the most classic case where we use LLM knowledge to train a small LM. Depending on whether we have access to the LLM we can have two different approaches, white-box where we exploit the output distribution or black box where we use LLM to create a dataset for training
* **Data Synthesis** It is now efficient to create quality datasets using LLM to train small LMs. Also, you can use it to augment the dataset we have collected (data augmentation).

![How large LM model can enhance small LM](https://github.com/SalvatoreRa/tutorial/blob/main/images/LLM_enhance_small_LM.png?raw=true) *from [here](https://arxiv.org/pdf/2409.06857)*

There are three scenarios in which it is better to have a small LM than a large LLM:
* **Computation-constrained environment** LLMs are expensive both to train and in inference, plus this means more latency. You don't always have the resources for either training or deployment, in these conditions it is better to go for a small LM. Often you do not even need an LLm but a fine-tuned model on a specific dataset
* **Task-specific environment** For some domains the number of datasets is limited, and an LLM could not be trained. A small LLM fine-tuned on these data gives better results than a large LLM (conducting the fine-tuning of an LLM is an expensive task, and conducting multiple updates is not ideal). 
* **Interpretability-required Environment** As regulations grow there are now areas or tasks where an interpretable model is required.  For example, in healthcare, it is important to have interpretable models and to understand why a model makes a decision. Smaller models are more interpretable.

## Evolution of small LLMs

**Small LMs (SLMs)** are not simply LLMs with fewer parameters but they too have gone through an evolution. Recently [this survey](https://arxiv.org/abs/2409.15790) analyzed them:

*With such criteria, we select 59 SLMs as detailed in Table 1. Our selection encompasses a wide range of models from both industry and academia, based on factors such as model architecture, parameter size, and data availability. While all selected SLMs share similar architectures, they differ in specific hyperparameters and training datasets, with some datasets remaining closed-source. These variations lead to differing performance across tasks--[source](https://arxiv.org/pdf/2409.15790)*

The authors present in this survey the various modifications that have been conducted in respect to the original transformer.

![transformer modification in SLMs](https://github.com/SalvatoreRa/tutorial/blob/main/images/transformer_block.png?raw=true) *The architecture modification of the SLM, highlighting 6 configurations: attention type, FFN type, FFN ratio, FFN activation, vocabulary size, and normalization type. from [here](https://arxiv.org/pdf/2409.15790)*

The attention mechanism has been modified since its first appearance, and today we can see four types of attention mechanisms:
* **Multi-Head Attention (MHA)** or the original version, where we have different attention heads.
* **Multi-Query Attention (MQA)** is a simplified version of MHA where queries are shared across all heads (single key and value, but we have different key and value projections due to different queries). This reduces both complexity and time.
* **Group-Query Attention (GQA)** Similar to MQA we have queries that are shared across all heads but we have separate keys and values (heads are grouped, so one key/value for some of the heads). This system tries to reduce complexity while balancing the expressiveness of the model (effectiveness and diversity).
* **Multi-Head Latent Attention (MLA)** low-rank key-value joint compression to reduce the complexity of the attention mechanism. In other words, there is a projection step before the calculation of MHA.

![The type of self-attention in SLMs](https://github.com/SalvatoreRa/tutorial/blob/main/images/attention_mechanism_SLM_modification.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

Two variants exist today with regard to feed-forward networks:
* **Standard FFN** Two layers of neurons with nonlinear activation (e.g., classical MLP).
* **Gated FFN** There is an additional gate layer that allows the network to control and regulate the flow of information.

![The type of feed-forward neural network in SLMs](https://github.com/SalvatoreRa/tutorial/blob/main/images/FFNN_SLM_modification.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

Another parameter that is evaluated is the intermediate ratio of the feed-forward network (size of the middle layer in relation to the whole network). For example, if the intermediate ratio is 4, it means that the hidden layer in the middle of the feed-forward network is 4 times larger than the input/output layer of the network. This is also called temporal expansion and allows more complex transformations thus learning a better representation of the data.

Size is also an important parameter, and there have been variations in vocabulary size.

![The type of feed-forward neural network in SLMs](https://github.com/SalvatoreRa/tutorial/blob/main/images/ratio_vocabulary_size_slm.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

There have also been changes in the activation function of feed-forward neural networks (FFN), in the transformers these functions are found:
* **ReLU (Rectified Linear Unit)**.
* **GELU (Gaussian Error Linear Unit)** allows smoothing between zero and positive values.
* **SiLU (Sigmoid Linear Unit)** a combination of the properties of the sigmoid and ReLU
* **GeLU TanH**

![ The activation function of the feed-forward neural network](https://github.com/SalvatoreRa/tutorial/blob/main/images/activation_slm.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

Finally, we have several types of layered normalization: 
**LayerNorm** Normalizes all features in a layer. t takes the mean and variance of all features in a layer (or the entire set of input features) and scales them to have zero mean and unit variance. After normalization, the learned scaling and shift parameters are applied.
* **RMSNorm** focuses only on normalizing the variance (i.e., the “scaling” of the input), without centering the data by subtracting the mean. It calculates the root mean square (RMS) of the input and scales the data according to this.

![The type of layer normalization in SLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/layer_norm_SLM.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

The authors analyze the different trends by showing:
* MHA appears to be slowly being replaced by GQA.
* Standard FFN is being replaced by Gated FFN. Today almost all models use the latter.
* The intermediate ratio of the feed-forward neural network is set at 4 for Standard FFN while there is a different range for Gated FFNN (2 to 8).
* The vocabulary size is the number of unique tokens an SLM can recognize and this value has grown over the years to be greater than 50K
* The type of layer normalization in the original transformer was LayerNorm, today most new models use RMS normalization

![architecture trends in SLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/architecture_trend.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

*As of August 2024, a typical SLM architecture tends to use group-query attention, gated FFN with SiLU activation, an intermediate ratio of FFN between 2 and 8, RMS normalization, and a vocabulary size larger than 50K. However, the choice of such settings is mostly empirical, without strict and public validation on the superiority of such model’s capacity --[source](https://arxiv.org/pdf/2409.15790)*

Another intriguing point is that there are some innovations in the model architecture for SLM and that are present in modern SLMs today: Layer-wise parameter scaling and Nonlinearity compensation.

**[Layer-wise parameter scaling](https://arxiv.org/pdf/2306.09380)** is a technique that allows the same set of weights to be reused in different parts of the model (thus reducing the number of parameters but maintaining performance). We can typically see two approaches: 
* **Connect input and output embeddings**: Input embeddings (used to convert words or tokens into vectors) and output embeddings (used to predict the next token) can share the same parameters. This reduces the number of parameters learned without significant loss of accuracy.
* **Level Sharing (Weight Tying)**: Some models share parameters among different levels of the network. Instead of learning different sets of weights for each layer, a single set of weights is reused across multiple layers. For example, the weights of the attention mechanism or feed-forward layers can be the same in different layers, helping to reduce the model footprint.

![parameter sharing in small language model](https://github.com/SalvatoreRa/tutorial/blob/main/images/parameter_sharing.png?raw=true) *from [here](https://arxiv.org/pdf/2306.09380)*

Nonlinearity compensation was introduced by [this article](https://arxiv.org/abs/2312.17276) where they note the so-called feature collapse phenomenon. In the deeper layers of an LLM, the variety in the features (or their representation) is reduced. So in deeper layers, features are more similar to each other, which impacts quality and creativity of the model. In this paper, they solved the problem with increasing the nonlinearity of the model with augmented shortcuts into the Multi-Head Attention (MHA) and a series activation function to the FFNN layer.

![Nonlinearity compensatio in small language model](https://github.com/SalvatoreRa/tutorial/blob/main/images/nonlinear_compensation.png?raw=true) *from [here](https://arxiv.org/pdf/2312.17276)*

They note in the article that data quality is critical to the quality of the model. For example, SLMs that are state-of-the-art take special care with datasets. Also, today t SLMs are trained over large amounts of tokens (typically >1.5T, but even up to 12T). These models would theoretically be over-trained (at least according to [Chinchilla's scaling law](https://arxiv.org/abs/2203.15556)). Being over-trained though would have utility for SLMs that are deployed on resource-constrained devices. 
The performance of SLMs grew from 2022 to 2024, showing future potential for SLMs on devices. it is true that larger models usually perform better, but smaller models can still excel at specific tasks. SLMs show good in-context learning ability, although they remain inferior to the large closed-source LLMs.  These SLMs are closing the gap with closed-source models and commonsense tasks, though there remains a significant in tasks requiring complex reasoning or logic.

![SLM capabilities over time](https://github.com/SalvatoreRa/tutorial/blob/main/images/SLM_capabilities.png?raw=true) *from [here](https://arxiv.org/pdf/2409.15790)*

Other results shown in the survey are:
* Apart from the model size, the model architecture also impacts latency. Factors such as the number of layers, the width of the FFNN, the vocabulary size, and whether parameters are shared play significant roles. The impact on memory is related to the number of parameters and the size of the vocabulary.
* Greater quantization accuracy, better performance. 4-Bit seems to be a sweet spot.
* Prefill phase is the most important phase for using GPU in inference (feed the prompt to the model) while decoding is conducted one token at a time. matrix by vector multiplication seems to be the most time-consuming operation (70% end-to-end inference time). Increasing context length seriously impacts memory utilization.


suggested lectures:
* [Small Language Models: Survey, Measurements, and Insights](https://arxiv.org/abs/2409.15790)
* [What is the Role of Small Models in the LLM Era: A Survey](https://arxiv.org/abs/2409.06857)
