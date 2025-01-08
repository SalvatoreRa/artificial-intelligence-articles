# Grokking: Learning Is Generalization and Not Memorization
### Understanding how a neural network learns helps us prevent the model from forgetting what it learns

This article is divided into different sections where we will answer these questions:

* What is grokking and why can help us to learn more about how neural networks learn
* How does this elusive phenomenon originate?

Check the list of references at the end of the article, I also provide some suggestions to deepen the topics.

*Updated: January 24*

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

At this point, we can try to combine the elements together. Grokking emerges during training and seems to be related to SGD and its eventual drift in the “_generalization islands_” in the loss. Neural circuits improve the generalization capabilities of the model and help reduce the loss together. **Could there therefore be a link?**

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

When we train a model we clearly prefer it to be able to generalize rather than memorize. Why, though, should training favor circuits dedicated to generalization and not memorization? After all, the loss is calculated on the examples in the training set and not on the generalization in the test set.

One element we have left out so far is [weight decay](https://paperswithcode.com/method/weight-decay). Weight decay is a [regularization technique](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/) in which we add a penalty proportional to the magnitude of the weights of the neurons. This means that when there are multiple circuits that achieve strong training performance, those that are more efficient (fewer parameters) are favored. And fewer parameters also mean [sparsity](https://arxiv.org/abs/2102.00554), as we saw earlier.

So far we have three properties for grokking:

1. Generalization circuits (C-gen) are more important for generalization than memorization circuits (C-mem)
2. C-gen is more efficient
3. C-gen is learned more slowly than C-men

**This is the basic recipe for achieving grokking**. Now we need to understand how the combination of these elements leads to grokking. **The first force acting on these circuits is the SGD**, which pushes to increase the value of the weights that decrease the loss. Conversely, **weight decay pushes in the opposite direction**, leading to decreasing parameters. These two forces must be balanced to achieve minimum loss.

So in the first stage, C-mem is learned quickly, and the model memorizes the training set, thus it has good train performance but poor test performance. In the second phase, C-gen is learned, SGD for the growth of C-gen parameters while weight decay pushes the shrinkage of inefficient circuits. So you strengthen C-gen and weaken C-mem, thus leading to increased performance in the test set. In addition, the strength of the weight decay affects the [number of epochs](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) required for grokking to appear

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking12.webp)

_image by the author adapting from: [here](https://arxiv.org/pdf/2301.05217.pdf) and [here](https://arxiv.org/pdf/2303.11873.pdf)_

We have seen how the interplay of SGD and weight decay leads to the emergence of these three phases. However, some questions remain open: Why is dataset size important? Why are generalizing circuits more efficient?

The answers to these questions are actually interrelated. In fact, whether a circuit generalizes well is independent of dataset size and it should perform well each time a new data point is added. In contrast, a memorization circuit must store additional data points to the training dataset, so its efficiency decreases as the training dataset increases. Because neural networks have many parameters, they are quite capable of memorizing small datasets but over a threshold this process becomes inefficient.

If this is true, there must be a critical D-crit point where C-gen circuits are more efficient than C-mem circuits. This would lead to three cases:

* If the dataset is larger than D-crit we have grokking because C-gen is more efficient than C-mem circuits.
* If the dataset is smaller than D-crit we have no grokking (or ungrokking if we re-train the network on a smaller dataset). This is because C-gen is less efficient than C-mem and the network uses memorization.
* There may also be situations where the dataset size is similar to D-crit, in which both circuits are present (semi-grokking).

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking13.webp)

_image source: [here](https://arxiv.org/abs/2309.02390)_

**But why are these generalization circuits more efficient?**

Recapitulating, if the generalization circuits are not more efficient the loss would remain high (the model relies on memorization). If, on the other hand, memorization and generalization are equally efficient, the training loss and test loss behave similarly and we would not have grokking.

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking14.webp)

_image source: [here](https://arxiv.org/abs/2309.02390)_

Intuitively, if we have a dataset of size D and the network memorizes it, it will not be able to predict a newly added data point. The network must also memorize this new point in order to predict it correctly. The performance of a generalization circuit is independent of a new point added (after all, the circuit learns a pattern, not a data point). So generalization circuits are independent of the size of the dataset.

The efficiency of a generalization circuit remains constant as the dataset increases. In contrast, memorization circuits increase in parameters to accommodate new data points and thus beyond a critical dataset size become less efficient.

This can be demonstrated experimentally: as the size of the dataset increases, the weight of the memorization circuits increases (decreasing their efficiency).

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking15.webp)

_image source: [here](https://arxiv.org/abs/2309.02390)_

**In conclusion, memorization circuits are efficient as long as the dataset reaches a certain size.** At this critical threshold, generalization circuits are more efficient. Beyond the threshold, the combined effect of SGD and [weight decay](https://www.linkedin.com/advice/3/how-do-you-compare-weight-decay-other) causes the generalization circuits to grow in weight. In contrast, the same forces reduce the weights associated with the memorization circuits. This is the recipe for grokking.

**One question remains: why are we so interested in understanding this behavior?**

## Looking at the bigger picture

The first reason we are interested in grokking is because it is related to how neural networks learn. Understanding this interplay between memorization and generalization allows us to investigate the learning mechanism of neural networks.

[An article in 2016](https://arxiv.org/abs/1611.03530), asked **if neural networks have enough parameters to store the dataset why should they generalize?**

_What is it then that distinguishes neural networks that generalize well from those that don’t? A satisfying answer to this question would not only help to make neural networks more interpretable, but it might also lead to more principled and reliable model architecture design ([source](https://arxiv.org/pdf/1611.03530.pdf))_

The authors noted that technically these models have enough parameters to learn the entire dataset. **Nevertheless, they learn to generalize**. This could be explained by the fact that generalization circuits are more efficient than memorization circuits.

Moreover, what we learn about grokking on small neural networks can be used to understand much more complex and larger neural networks.

_We thus might view grokking as a controlled test bed for emergence in large language models, and hope that understanding the dynamics of grokking could lead to hypotheses for analyzing such emergent capabilities. ([source](https://arxiv.org/pdf/2303.11873.pdf))_

There is some interest in grokking because this behavior resembles the so-called emergent properties of large language models. Indeed, in these emergent properties, a given skill of the model appears abruptly. Since recently, emergent properties have been questioned, it is important to have a much cheaper test bed to study this phenomenon.

Ungrokking, on the other hand, recapitulates another problematic but not understood phenomenon of neural networks: catastrophic forgetting. Catastrophic forgetting is the inability of a neural network to retain old information in the presence of a new one. This is especially problematic during continual learning when a model is deployed and receives new data from users. Although there are differences, ungrokking is considered a special case of catastrophic forgetting and can help us understand it.

![memorization versus generalization grokking](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking16.webp)

_image source: [here](https://arxiv.org/pdf/1903.06070.pdf)_

In addition, the study of generalization circuits may be the key to making neural networks human readable:

* These generalization circuits are subgraphs that can be extracted and recognize specific patterns.
* In transformers, these circuits are represented by induction heads and connected to in-context learning

## The link between complexity and grokking

While it is true a model must explain the data, a good explanation should be simple ([Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor)). The [Minimum Description Length (MDL)](https://en.wikipedia.org/wiki/Minimum_description_length) states that the best model is the one that provides the shortest total description of the model itself and the data when encoded using the model. MDL in other words states: that a model that is too simple will not describe the data well (thus a long description of bits of the data), while a too complex one will require a large model to describe the data well. MDL then looks for sweet spots, where the model uses few rules to describe the data well. 

In [information theory](https://en.wikipedia.org/wiki/Information_theory), we can describe a distribution (the data) without losing information if the lower bound is the entropy of the distribution. Optimal encoding is practically the same as entropy. This means that data that is totally random will require many bits for encoding, whereas if we find patterns we can use fewer bits. A model then tries to find these patterns to learn a compact and compressed representation of the data.

So if a model fully explains the data the prediction of the data becomes deterministic and entropy is zero. If a model stores all the data, it reduces the entropy of the data to zero, but the complexity of the models is exactly equal to the original entropy, there is no compression (and therefore we have gained nothing).

The idea of adding a regularization term is to reduce the complexity of the model and thus avoid memorization, thus respecting the MDL principle. 

![grokking complexity](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking18.png)

_image source: [here](https://arxiv.org/pdf/2412.09810)_

Measuring the complexity of a model is not an easy task (and model capacity is only a good approximation). Another proxy for defining the complexity of a model is the compression capacity of a model. If a model compresses data well, it means that it has learned generic patterns that allow us to represent the data distribution well (and reduce entropy).

Initially, the model has no understanding of the data, but during training, it learns patterns and incorporates them into its representation, increasing complexity. This increased complexity does not immediately improve compression, as the learned patterns may not generalize. If the model merely memorizes the data, complexity remains high, and compression does not improve. However, if the model generalizes, it learns compact representations of the data, reducing complexity and increasing compression capacity after an initial rise. Generalization reflects low test error, as the model transitions from memorizing to encoding general patterns, achieving better data compression and general performance.

![grokking complexity link](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking19.png)

_image source: [here](https://arxiv.org/pdf/2412.09810)_

Grokking refers to delayed generalization, where a model initially overfits during early training (low training loss but poor test accuracy) but eventually achieves strong generalization with extended training. This phenomenon is analyzed using loss, accuracy curves, and the complexity of the model's learned function. Regularization, such as weight decay, plays a key role by encouraging simpler solutions, which are linked to generalization.

**Compression is used as a proxy for generalization**. Good compression is lossy but retains essential information while discarding noise or irrelevant details. In machine learning, lossy compression helps models focus on meaningful patterns and avoid overfitting spurious details. Grokking can be seen as the model's delayed discovery of a compressed representation, transitioning from memorizing irrelevant details to learning essential features that generalize well across training and test data. Regularization facilitates this by promoting simpler, more effective representations aligned with the information bottleneck principle, which emphasizes retaining only essential information for the output.

![grokking complexity link](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking20.png)

_image source: [here](https://arxiv.org/pdf/2412.09810)_

Using regularization as weight decay promotes a reduction in complexity but does not guarantee it. Therefore [in this paper](https://arxiv.org/pdf/2412.09810), they create a training algorithm to promote compression, reduction in model complexity and thus also grokking

## Application of Grokking

Grokking seems more like a theoretical case without practical applications, especially since it needs many iterations to emerge. A [paper was recently presented](https://arxiv.org/pdf/2405.20233) that discusses the possibility of creating an algorithm called Grokfast, to accelerate model convergence toward generalization.

The system decomposes the gradient of a parameter into two components: a fast-varying component and a slow-varying component. The former is responsible for overfitting, and the latter is responsible for generating (inspired by circuits described in other articles). By exploiting this you can then speed up convergence, and simply strengthen the influence of the slow-varying component. [Here](https://github.com/ironjr/grokfast) is the code.

## Open questions

Although all these recent studies shed light on what is required for grokking and the interplay between memorization and generalization, several open questions remain. For example, [in this article](https://arxiv.org/abs/2303.06173), they hypothesize a link with [inductive bias](https://en.wikipedia.org/wiki/Inductive_bias). according to the authors, generalization is slowly learned but is favored by [inductive bias](https://towardsdatascience.com/a-fairy-tale-of-the-inductive-bias-d418fc61726c).

As was mentioned earlier, neural networks have little inductive bias (and this is especially true for transformers). Since generalization and grokking appear only beyond a critical dimension, this inductive bias could be learned only with enough examples. There is definitely a link between efficiency and inductive bias, but this should be investigated further.

An intriguing hypothesis is that learned network capabilities are quantized into discrete chunks (quanta). This hypothesis seeks to provide an explanation for the emergent properties. According to the authors, there is a “universal and discrete set of computations which are instrumental for reducing loss.” This set would be none other than the generalization circuits.

For the authors, we have a universal set of building blocks (the circuits) that discretely control performance. The emergence of properties at scale would simply be learning new circuits (or quanta for the authors). Grokking and efficiency of generalization circuits would explain why they appear. Much remains to be discovered about this suggestive hypothesis.

![Simplified version of the quanta hypothesis](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/grokking17.webp)

_Simplified version of the quanta hypothesis. image by the author_

In addition, numerous others remain:

* Why is it that if the dataset is reduced, the time required for grokking increases exponentially?
* What is the effect of random initialization?
* Why do generalization circuits develop more slowly?
* How does gradient descent uncover these circuits?
* Why is weight decay necessary? Why do some studies obtain grokking even without weight decay?

## Conclusions

Grokking or delayed generalization is an elusive phenomenon that has intrigued researchers in recent years. Recent studies have shown its importance in explaining how a neural network learns.

Grokking is not only a theoretical challenge but also has important practical implications. Understanding the phenomenon helps us explain the behavior of neural networks and design neural networks that can generalize more efficiently. Analyzing these generalization circuits would help us design more interpretable and more secure neural networks. Ungrokking, on the other hand, allows us to analyze catastrophic forgetting and thus obtain better networks for continual learning. Finally, there remain exciting new questions that forthcoming studies will try to solve.

## Reference

Here is the list of the principal references I consulted to write this article (only the first author name of an article is cited).

1. Power, 2022, Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets, [link](https://arxiv.org/abs/2201.02177)
2. Liu, 2022, Towards Understanding Grokking: An Effective Theory of Representation Learning, [link](https://arxiv.org/abs/2205.10343)
3. Liu, 2022, Omnigrok: Grokking Beyond Algorithmic Data, [link](https://arxiv.org/abs/2210.01117)
4. Distill, Zoom In: An Introduction to Circuits, [link](https://distill.pub/2020/circuits/zoom-in/)
5. Nanda, 2023, Progress measures for grokking via mechanistic interpretability, [link](https://arxiv.org/abs/2301.05217)
6. Merrill, A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks. [link](https://arxiv.org/abs/2303.11873)
7. Blalock, 2020, What is the State of Neural Network Pruning? [link](https://arxiv.org/abs/2003.03033)
8. Liu, 2023, Seeing is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability, [link](https://arxiv.org/abs/2305.08746)
9. Kolouri, 2019, Attention-Based Selective Plasticity, [link](https://arxiv.org/pdf/1903.06070.pdf)
10. Zhang, 2016, Understanding deep learning requires rethinking generalization, [link](https://arxiv.org/abs/1611.03530)
11. Zhang, 2021, Understanding deep learning (still) requires rethinking generalization, link
12. Davies, 2023, Unifying Grokking and Double Descent, [link](https://dl.acm.org/doi/10.1145/3446776)
13. Michaud, 2023, The Quantization Model of Neural Scaling, [link](https://arxiv.org/abs/2303.13506)
14. Frankle, 2018, The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, [link](https://arxiv.org/abs/1803.03635)
15. Hoefler, 2021, Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks, [link](https://arxiv.org/abs/2102.00554)
16. DeMoss, 2024, The Complexity Dynamics of Grokking, [link](https://arxiv.org/abs/2412.09810)

