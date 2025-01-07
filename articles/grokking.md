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

[As suggested in this study](https://arxiv.org/abs/2205.10343), grokking can be described as a kind of “delayed generalization” where neural networks begin to generalize (on certain problems) after overfitting. Since this seems to violate the basic principles of machine learning. Understanding how this happens also means studying: how models generalize, the critical training size, and the conditions necessary for this delayed generalization.

[For this study](https://arxiv.org/abs/2205.10343), generalization means that the model learns a good representation of the input embeddings. The critical training set size corresponds to the least amount of training data that can allow the model to learn the data representation. Grokking for the authors would appear at the stage between understanding the data and memorizing it (the model memorizes examples more than underlying patterns). So it would be solvable with the right choice of [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)).

![grokking is dependent on the dataset size](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking3.webp)

*image source: [here](https://arxiv.org/abs/2205.10343)*

[Another study](https://arxiv.org/abs/2210.01117), states that the issue is parameter initialization, and therefore grokking could be observed with any type of data (even images and text). In fact, the authors showed that even using [MNIST](https://en.wikipedia.org/wiki/MNIST_database) with the right combination of parameters could observe delayed generalization.

![grokking is dependent on the dataset size](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking4.webp)

*image source: [here](https://arxiv.org/pdf/2210.01117.pdf)*

[Another interesting hypothesis](https://arxiv.org/pdf/2210.01117.pdf) links grokking to [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). During its descent to the minimum, the model explores loss regions that are primarily overfitting. [Stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) continuing its exploration toward the loss minimum, will eventually drift into ‘generalizing regions’ in the loss manifold, and then grokking begins to appear.

![Generalizing solutions (green stars) are concentrated around a sphere in the weight space](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking5.webp)

_Generalizing solutions (green stars) are concentrated around a sphere in the weight space. image source: [here](https://arxiv.org/pdf/2210.01117.pdf)_

Simply put, there is a connection between minimum loss (the right combination of weights for the network) and grokking. This is because these regions of minimum, dispersed in wide regions of overfitting, are regions of generalization. Thus, the SGD algorithm gradually manages to reach them.

![the SGD algorithm gradually manages to reach generalization regions (simplified scheme)](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking6.webp)

_the SGD algorithm gradually manages to reach generalization regions (simplified scheme). Image by the author_

### Inspect the inside of the neural network

**Can this grokking behavior be better understood if we take a mechanistic look at neural networks? **It has recently been suggested that within neural networks [there are “circuits”](https://distill.pub/2020/circuits/zoom-in/) consisting of the features and weights that connect them. Moreover, analogous features and circuits form across all models and tasks (universality of the circuits). Therefore, we can not only study them but they could provide us with information about behaviors of a model.

In [computer vision](https://en.wikipedia.org/wiki/Computer_vision), one can clearly understand the formations of these circuits, where they serve a precise function (e.g., identifying curves). Combining these circuits at a higher level allows the model an understanding of more complex structures (e.g. identifying faces). These circuits have been identified in all computer vision models.

![the SGD algorithm gradually manages to reach generalization regions (simplified scheme)](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking7.webp)

_image source: [here](https://distill.pub/2020/circuits/zoom-in/)_

Similarly, [in large language models](https://en.wikipedia.org/wiki/Large_language_model), induction heads can be defined as circuits that play an important role in [in-context learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610).
This is important because it increases the interpretability of the model. In fact, a mechanistically interpretable algorithm could be obtained by identifying its subnetworks (circuits).

As the authors of this study note, while these circuits are not actively searched by the SGD, when it actually finds them the loss decreases (during training). Moreover, these circuits are composable (one circuit improves the loss in the presence of other circuits), so it leads the model to develop more sophisticated circuits. So since these circuits are useful for prediction, SGD reinforces them during training and they emerge.

_A similar mystery arises from how organisms develop sophisticated machinery, like the human eye. Each part is only useful in the context of other parts. A compelling explanation is a component first developed that was somewhat useful in its own right, like a light-detecting membrane. It was reinforced as a useful component. Then, later components developed depending on the first, like the lens of the eye. ([source](https://arxiv.org/abs/2301.05217))_

According to the [lottery ticket hypothesis](https://arxiv.org/abs/1803.03635), these circuits are already initially present in the network (or at least partially), then an evolutionary advantage and SGD push to complete them, create more complex structures, and then emerge.

### Grokking & neural circuits

At this point, we can try to combine the elements together. Grokking emerges during training and seems to be related to SGD and its eventual drift in the “_generalization islands_” in the loss. Neural circuits improve the generalization capabilities of the model and together help reduce the loss. **Could there therefore be a link?**

So, [according to one study](https://arxiv.org/abs/2301.05217), the training dynamics underlying grokking can be divided into three stages:

* **Memorization** of the training examples.
* **Circuit formation**, where the model learns to generalize (because of these circuits).
* **Cleanup**, where the model removes memorization components (via weight decays).

According to the authors, this occurs during cleanup. In fact, during these stages, the model learns different tasks:

_These results show that grokking, rather than being a sudden shift, arises from the gradual amplification of structured mechanisms encoded in the weights, followed by the later removal of memorizing components. ([source](https://arxiv.org/pdf/2301.05217.pdf))_

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking8.webp)

_image source: [here](https://arxiv.org/pdf/2303.11873.pdf)_

Although these studies have highlighted the importance of dataset size, choice of hyperparameters, and circuits, a number of open questions remain:

* what leads to this transition between memorization and generalization?
* In what tasks can it happen?
* disentangle the statistical (dataset size) from the computational ([computation time](https://lunalux.io/computational-complexity-of-neural-networks/), [size of the network](https://www.quora.com/How-do-you-calculate-the-size-of-a-neural-network-in-memory))?


## A tale of sparsity

As we have above, they highlighted three stages in this study. In one of these phases (cleanup) memorization circuits are removed. In other words, some weights are removed (their value approaches zero).

On the one hand, this is reminiscent of [pruning](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)), a technique in which weights that are redundant are removed (to reduce the computational cost in inference) without altering the performance of the model. In a sense, we could say that during cleanup the model performs pruning of circuits that do not serve performance. Obviously, in contrast to pruning the circuits involved in generalization see their weights increase and emerge.

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking9.webp)

_image source: [here](https://arxiv.org/pdf/2003.03033.pdf)_

In any case, this highlights, an interesting relationship with [network sparsity](https://en.wikipedia.org/wiki/Sparse_network). Confirmed by the fact that grokkin-like behaviors seem to emerge from models that force sparsity into the architecture during training. In fact, [in this study](https://arxiv.org/abs/2305.08746) where they use a cost function that forces sparsity (in an almost extreme way), they observe behaviors similar to those described [in one of the studies above](https://arxiv.org/abs/2205.10343) (the emergence of similar representations of the data).

[In this study](https://arxiv.org/pdf/2303.11873.pdf), the authors investigate the relationship between sparsity and grokking. The authors clearly show that there is a relationship between grokking and the emergence of sparsity during training. If the network is initially dense ([fully connected](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/ch04.html)), after grokking the network seems to be controlled. **How does this sparsity develop?**

Through increasing weights of a specific set of neurons and decay of weights elsewhere.

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking10.webp)

_image source: [here](https://arxiv.org/pdf/2303.11873.pdf)_

_[The figure] shows that neurons active during the memorization phase slightly grow in norm before grokking but then “die out”, while the the neurons of sparse subnetwork are inactive during memorization and then explode in norm. ([source](https://arxiv.org/abs/2303.11873))_

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking11.webp)

_image source: [here](https://arxiv.org/pdf/2303.11873.pdf)_

Thus, we have two phases: one in which there is a memorization phase of the examples and a second in which the model learns to generalize. During the first phase, the memorization circuits increase their weights, but in the second phase their weights are reduced and instead, the generalization circuits explode. The memorization circuits in the second phase decrease their weight so much that we can speak of an actual [introduction of sparsity](https://stackoverflow.com/questions/41325673/what-does-sparse-mean-in-the-context-of-neural-nets) into the model.

Since the model is trained with SGD, this could be connected that during the first phase, the model explores the loss surface where there is overfitting, until it finds these islands of generalization (or [as it is called by this article](https://arxiv.org/pdf/2210.01117.pdf), the “_Goldilocks zone_”). These studies also show us another interesting aspect: the model first memorizes (and this happens quickly) and then learns to generalize. However, it remains to better understand the role of both memorization and generalization circuits and how they emerge.

## A dance of memorization and generalization
