# Grokking: Learning Is Generalization and Not Memorization
### Understanding how a neural network learns helps us prevent the model from forgetting what it learns

This article is divided into different sections where we will answer these questions:

* What is grokking and why can help us to learn more about how neural networks learn
* How does this elusive phenomenon originate?

Check the list of references at the end of the article, I provide also some suggestions to deepen the topics.

*Updated: November 24*

## Grokking and the inside of the network

**What is it grokking?**
One of the most repeated concepts in [neural networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) is: that when the [training loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss?hl=it) converges to a low value, the neural network will no longer learn much. [Yet in a 2021 study](https://arxiv.org/abs/2201.02177), they observed a strange phenomenon, which the authors called “grokking.” The model seems to reach plateaux, where low and stable training loss with poor generalization is observed, and then with further training, the model is capable of perfect generalization.

![example of grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking1.webp)

*image source: [here](https://arxiv.org/pdf/2201.02177)*

_In some situations we show that neural networks learn through a process of “grokking” a pattern in the data, improving generalization performance from random chance level to perfect generalization, and that this improvement in generalization can happen well past the point of overfitting. ([source](https://arxiv.org/pdf/2201.02177.pdf))_

So in this study, they show that although the model seems to have reached [overfitting](https://en.wikipedia.org/wiki/Overfitting), validation accuracy sometimes suddenly increases to near-perfect [generalization](https://en.wikipedia.org/wiki/Generalization_error).

**Why does performance in the test set improve when performance in training has already peaked?**

**In fact, this behavior is counterintuitive.** Typically, it is assumed that when the model reaches a plateau with the training set, it begins to learn patterns that exist only in the training set and loses the ability to generalize. **So why does grokking occur?**

### A recipe for grokking
Meanwhile, [the authors noted](https://arxiv.org/pdf/2201.02177.pdf) that there was a relationship between grokking and dataset size. In addition, they noticed two interesting results:

* Optimization time for generalization increases when the number of examples in a dataset decreases.
* [Weight decay](https://fr.wikipedia.org/wiki/Weight_decay) is particularly effective at improving generalization

![grokking is dependent on the dataset size](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking2.webp)

*grokking is dependent on the dataset size. image source: [here](https://arxiv.org/pdf/2201.02177.pdf)*

