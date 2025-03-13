# Tabular deep learning

* [Why Do Tree Based Algorithms Outperform Neural Networks](#Why-Do-Tree-Based-Algorithms-Outperform-Neural-Networks)
  * [Challenges with tabular data](#Challenges-with-tabular-data)
  * [Tabular Learning and the Forest](#Tabular-Learning-and-the-Forest)
  * [Why decision tree perform better than a neural network](#Why-decision-tree-perform-better-than-a-neural-network)
  * [Why we are interested in neural networks for tabular datasets](#Why-we-are-interested-in-neural-networks-for-tabular-datasets)
  * [What do we want for a model in tabular learning](#What-do-we-want-for-a-model-in-tabular-learning)
 * [Why categorical data are problematic for ML and DL algorithms](#Why-categorical-data-are problematic-for-ML-and-DL-algorithms)
 * [Suggested lectures](#Suggested-lectures)

## Why Do Tree Based Algorithms Outperform Neural Networks

Artificial intelligence has made great strides in recent years. [ChatGPT](https://medium.com/data-driven-fiction/everything-but-everything-you-need-to-know-about-chatgpt-546af7153ee2) has stunned the world. Yet although we have seen incredible applications for both images and text, tabular data is still a problem. (So much so that [Kadra in 2021](https://arxiv.org/abs/2106.11189) called them the “unconquered castle” for neural networks). But why do they remain a problem?

### Tabular data

[Tabular data](https://www.statology.org/tabular-data/) can be defined as a sub-branch of structured data. Simply put, tabular data can be defined as any data that can be described as a table (like an [Excel sheet](https://en.wikipedia.org/wiki/Microsoft_Excel)) in which by convention, the rows represent examples and the columns represent features.
Paradoxically, despite their simplicity, in real-world applications, most data are in tabular format: finance, medicine, climate science, and manufacturing.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data.webp)
*image source: [here](https://arxiv.org/abs/2110.01889)*

In contrast to images, text, or audio (called homogeneous data because they have only one type of feature), tabular data  are heterogeneous, since they can contain multiple types of features:

* **[Categorical values:](https://en.wikipedia.org/wiki/Categorical_variable)** a particular type of feature that contains qualitative (or discrete values, a limited set of terms) and there is not an intrinsic order. For example, gender, class, product class, and so on.
* **[Binary values:](https://en.wikipedia.org/wiki/Binary_data)** Binary feature is a special case of categorical features, where the feature can have only two values (generally 0/1 or false/true). An example is the vital status of a patient which can be alive or dead.
* **[Ordinal values:](https://en.wikipedia.org/wiki/Ordinal_data)** a special type of categorical feature, where while the values are discrete there is an intrinsic order. For example, frequency of gym activity (always > sometimes > rarely) or height if expressed in discrete value (tall > average > short)
* **[Continuous values:](https://en.wikipedia.org/wiki/Continuous_or_discrete_variable)** a feature that contains continuous values.

Real-world datasets are a mix of continuous and discrete values, making it hard to design an algorithm that can work with both at the same time.

Despite this format seeming to limit, you can use these data for many different tasks, such as:

* **[Classification](https://www.sciencedirect.com/topics/computer-science/classification-task)**, where you want to predict a categorical value (e.g., whether a patient with a condition survives or not).
* **[Regression](https://www.sciencedirect.com/topics/computer-science/regression-task)**, where the target is a continuous value (such as the price of an apartment).
* **[Clustering](https://www.sciencedirect.com/topics/computer-science/clustering-task)**, exploiting patterns in the data we want to cluster examples (divide data points into a series of homogeneous groups)

### Challenges with tabular data

Tabular data, as simple and rudimentary as they may seem compared to other modalities, actually have some unexpected complexities. Here, I will briefly describe the various possible challenges, but each of them deserves to be treated separately in an article.

**Low-Quality Data**
Tabular data generally require extensive [pre-processing](https://towardsdatascience.com/data-preprocessing-e2b0bed4c7fb) since they often cannot be directly fit into a model. They can often include missing data. [Missing data](https://en.wikipedia.org/wiki/Missing_data) may originate randomly (and be found randomly distributed in the dataset). In other cases, they might be derived from some [data collection bias](https://towardsdatascience.com/types-of-biases-in-data-cafc4f2634fb). For example, in a survey, some people may forget to fill out some of the questions, or a group of individuals (e.g., young people under 25) may be inclined not to answer specific questions. So although there are different methods of imputation, being the origins different, deciding how to treat them is not an easy task

For this reason, the input of missing data often comes from the data domain, and specific solutions for each type of data have been studied.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data2.webp)
*image source: [here](https://arxiv.org/abs/2210.08258)*

Another plague of tabular data is [outliers](https://en.wikipedia.org/wiki/Outlier); An **outlier** is basically a value that is different from all the others in the dataset. These can originate from errors in data entry, a poorly calibrated sensor, or a very clear reason (for example, a temperature sensor near a fire will show a much higher value that day). Not all models are resistant to outliers, and [evaluation metrics](https://www.geeksforgeeks.org/metrics-for-machine-learning-model/) can also be altered by outliers (e.g., a large error can blow up metrics such as [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)). Sometimes, though, outliers are exactly what we are looking for (adversary attacks, network intrusion, and credit fraud can be outliers in the dataset), and a sub-branch of tabular learning is exactly that: **[anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection)**.

A good percentage of datasets suffer from the so-called **[curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)**. This occurs when a dataset has many more features than examples, which makes it complex for the model to learn.

Because it is often expensive to acquire data, many datasets are **[class unbalanced](https://medium.com/mlearning-ai/handling-class-imbalance-in-machine-learning-cb1473e825ce)**. Or it can result from the nature of the data itself: most emails are spam, there is less credit fraud than legitimate transactions, and so on. Many models with unbalanced classes may predict only the majority class, so some corrections are required.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data3.webp)
*image source: [here](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)*

**Complex or missing spatial dependencies**
It was attempted to fit it through algorithms that would transform the dataset into images (and then be able to use a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)). However, there is no spatial correlation among the features of the dataset. This lack of [spatial correlation](https://www.sciencedirect.com/topics/computer-science/spatial-correlation) is an issue in designing a good algorithm for tabular data.

In any case, tabular features have some relationships among them. In fact, tabular datasets have complex or irregular dependencies among the features (which are, therefore, difficult for an algorithm to learn from scratch). For example, [neural networks learn simple features first](https://dspace.mit.edu/handle/1721.1/121680) (simpler rules first), and so it takes many examples or epochs to learn these complex relationships

For example, in a popular dataset such as the [Adult Income dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset), there are variables that are contextually interconnected. The features _age_, _marital status_, and _education_ have their own consistency but it is difficult to code. In fact, you can’t get married before a certain age, and it’s purportedly impossible to get a PhD before a certain age. These complex dependencies are difficult for the model to understand. This can be seen, for example, when other examples are generated from a dataset.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data4.webp)
*image source: [here](https://arxiv.org/abs/2210.06280)*

**Need for preprocessing**
[Preprocessing](https://en.wikipedia.org/wiki/Data_Preprocessing) is required to use tabular data. In general, the choice of preprocessing can lead to variations in algorithm performance and should be considered with caution. An example is the handling of [categorical variables](https://www.kaggle.com/code/alexisbcook/categorical-variables): when there are too many values it could lead either to too [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) or the introduction of an order that does not exist (increasing features or introducing bias, making learning more difficult).

[Dropping features](https://stats.stackexchange.com/questions/443457/when-to-drop-correlated-features) to reduce computational cost is also an often-used strategy. At the same time, this can lead to a loss of information and reduce the algorithm’s performance. Or also create some sort of artifacts in some cases (i.e. introducing numerical order during [categorical encoding](https://medium.com/aiskunks/categorical-data-encoding-techniques-d6296697a40f)).

**Single feature impact**
To transform an image from one class to another, it is not enough to change the value of one pixel, instead, the value of a group of pixels must be changed. Instead, changing a value for one feature (for example, a zero becoming one in a binary feature) can cause the model to assign that example to another class.

### Tabular Learning and the Forest

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data5.webp)
*image source: [here](https://www.kaggle.com/code/shivamb/data-science-trends-on-kaggle/notebook), license: [here](https://www.apache.org/licenses/LICENSE-2.0)*

In general, we can say that in both Kaggle’s challenges and the day-to-day work of a data scientist, [decision trees](https://en.wikipedia.org/wiki/Decision_tree) are the first choice.

In addition, [a recently published study](https://arxiv.org/abs/2207.08815) shows that tree-based models appear to perform better than neural networks for both types of datasets:

* Datasets containing numeric-only features
* Datasets that also contain categorical variables

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data6.webp)
*image source: [here](https://arxiv.org/pdf/2207.08815.pdf)*

These results have been confirmed [by this study](https://arxiv.org/abs/2407.00956).

If they perform better why do they do?

It has been speculated theoretically that the [inductive bias](https://towardsdatascience.com/a-fairy-tale-of-the-inductive-bias-d418fc61726c) of decision trees is efficient for tabular datasets.

In fact, in the dimensional manifold of a tabular dataset, there are often [decision boundaries](https://en.wikipedia.org/wiki/Decision_boundary) between classes and a decision tree can efficiently approximate them. After all, the idea behind a [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is that one can solve any task using binary divisions.

Another benefit of [decision trees](https://en.wikipedia.org/wiki/Decision_tree) is that going back up the tree (at least in the basic form) one can reconstruct the decision path and thus they are highly interpretable. Although this is not true for [ensembles](https://en.wikipedia.org/wiki/Ensemble_learning), there are post-hoc mechanisms for being able to explain them ([LIME](https://github.com/marcotcr/lime), [SHAP](https://shap.readthedocs.io/en/latest/), and so on).

In contrast to neural networks, they are much faster to train. This is because for a dataset normally an ensemble has far fewer parameters (by increasing the number of layers in a [neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) the parameters grow rapidly). So much so that it is often said that neural networks for tabular data are often over-parametrized.

### Why decision tree perform better than a neural network

[As seen on Kaggle](https://www.kaggle.com/code/shivamb/data-science-trends-on-kaggle/notebook), for tabular datasets, the first choice for a model for a tabular dataset is to use a decision tree ensemble ([Random Forest](https://en.wikipedia.org/wiki/Random_forest) or [XGBoost](https://en.wikipedia.org/wiki/XGBoost)). But why?

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data7.webp)
*image source: [here](https://www.kaggle.com/code/shivamb/data-science-trends-on-kaggle/notebook), license: [here](https://www.apache.org/licenses/LICENSE-2.0)*

**Why the inductive bias is important?**

Tabular data present different challenges and models that have optimal performance in one dataset might have poor performance in another type of dataset.

Considering a tabular dataset, it has been speculated that the lack of inductive bias does not allow [multi-layer perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) and other neural networks to find optimal solutions for tabular datasets. For example, convolutional neural networks exploit the assumption that a pattern is constructed by a number of pixels, whereas in a tabular dataset, this assumption is of no help:

_One key difference is that in image classification, many pixels need to change in order for the image to depict a different object [25].1 In contrast, the relative contribution of the input features in the electronic health records example can vary greatly: Changing a single input such as the age of the patient can profoundly impact the life expectancy of the patient, while changes in other input features, such as the time that passed since the last test was taken, may have smaller effects. ([source](https://arxiv.org/abs/1805.06440))_

[A recent study sought to investigate more formally](https://arxiv.org/abs/2207.08815) why the inductive bias of tree-based models is most suited for tabular datasets. Smoothing with a Gaussian kernel in the training set made sure that the model could not learn irregular patterns in the target function (in simple words they used a [Gaussian kernel](https://pages.stat.wisc.edu/~mchung/teaching/MIA/reading/diffusion.gaussian.kernel.pdf.pdf) on the training dataset). Smoothing impacts the performance of tree-based models but much less neural networks (NN). The authors suggest that NNs struggle to fit an irregular function.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data8.webp)
*image source: [here](https://arxiv.org/pdf/2207.08815.pdf)*

Anyway, it is expected that neural networks are biased toward smooth solutions, since they are trained with [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) (which by definition is smooth since it is based on differentiable search spaces).

This is best understood by looking at the decision boundaries for the two most important features in a dataset. The [Random Forest](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/) does better at learning irregular patterns (this is because the tree-based models approximate a piece-wise function). According to the authors, [regularization](https://www.geeksforgeeks.org/regularization-in-machine-learning/) and other additions allow the model to succeed better at learning irregular patterns.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data9.webp)
*image source: [here](https://arxiv.org/pdf/2207.08815.pdf)*

Last note, [as we saw earlier](https://medium.com/towards-data-science/a-fairy-tale-of-the-inductive-bias-d418fc61726c), the lack of inductive bias (or a weak inductive bias) can be compensated by enough examples. On the other hand, tabular datasets are often small or medium-sized, so weak bias cannot be compensated for.

**Neural networks are less robust to non-informative features**
One of the problems with tabular datasets is that they contain many features that are not informative to the task. The authors [of this study](https://arxiv.org/abs/2207.08815) decided for each dataset to drop an increasing number of features (according to their importance, using Random Forest to determine it). Thus, they showed that the tree-based models are resistant to removing these noninformative features.

Surprisingly, for the authors, removing non-informative features decreases the gap between NNs and tree-based models, while adding redundant features increases the gap. Therefore, NNs are not resistant to non-informative and redundant features.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data10.webp)
*image source: [here](https://arxiv.org/pdf/2207.08815.pdf)*

In addition, decision-tree algorithms are robust to the effect of a single feature. In fact, they can find an exact threshold for the split that is appropriate for a feature while ignoring the other features (at least for that split).

In other words, [Information Gain](https://en.wikipedia.org/wiki/Information_gain_(decision_tree)) and [Entropy](https://www.geeksforgeeks.org/gini-impurity-and-entropy-in-decision-tree-ml/) make decision trees more robust with regard to noninformative features. After all, they are designed to choose the best feature for the split.

**Other challenges for neural networks in tabular learning**
Neural networks generally need missing values to be imputed (even deep learning libraries have problems in [handling missing values](https://stats.stackexchange.com/questions/216219/can-neural-network-handle-data-samples-with-incomplete-attributes-bp-network), and it is one of the causes of when one finds NA as a loss).

Also, neural networks when datasets are too unbalanced easily predict only the majority class (a [heuristic trick](https://en.wikipedia.org/wiki/Heuristic) to reduce loss). Although it is then possible to exploit some tricks to reduce the problem, several tests must be done.

Although neural networks generally do not need pre-processing, if the matrix is too sparse after categorical encoding it could impact the training.

Moreover, neural networks perform worse in the small data regime (n<10³ or even n < 300 examples). In tabular data, deep learning cannot rely on [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) to tackle this issue.

Lastly, unlike tree-based models, neural networks do not have an innate way of defining the importance of features, and their [black-box nature](https://hdsr.mitpress.mit.edu/pub/f9kuryi8/release/8) makes it difficult to interpret how much importance they assign to a feature.

### Why we are interested in neural networks for tabular datasets

Since decision tree-based models work well with tabular datasets, why should we use deep learning models?

There are actually several reasons why we are interested in deep learning-based models.

* The performance of deep learning increases with the size of the dataset
* Another benefit is that deep learning reduces the need for [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering) (which is required for tree-based models). Moreover, can be used for data generation and used for solving class imbalance.
* Deep learning is compatible with [data streaming](https://en.wikipedia.org/wiki/Streaming_data), and iterative training, which can be highly interesting for AutoML.
* Deep learning models are [multi-modal](https://en.wikipedia.org/wiki/Multimodal_learning) and could use both tabular data and associated images (or other modalities).
* Deep learning models are capable of building an internal representation of the data, and this representation can be used for exciting applications such as [domain adaptation](https://en.wikipedia.org/wiki/Domain_adaptation), [generative modeling](https://en.wikipedia.org/wiki/Generative_artificial_intelligence), and [semi-supervised learning](https://en.wikipedia.org/wiki/Weak_supervision).
* Decision tree learners do not produce a reliable probability estimation.
* Decision tree models do not allow [end-to-end optimization](https://arxiv.org/abs/2103.16378) and instead employ greedy and local optimization.

**Scaling up the dataset**
You can expect a deep learning-based model to perform better than tree-based models when the dataset is very large. This is expected since recent studies show that weak inductive bias can be compensated for by a large number of examples.

In general, [a 2017 paper introduced a power law for deep learning](https://arxiv.org/abs/1712.00409), in which a deep learning model struggles to learn when there are few examples in the training set, subsequently, the model improves as the number of examples increases (power law region). The model then reaches a plateau (or region of irreducible error).

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data11.webp)
*image source: [here](https://arxiv.org/abs/1712.00409)*

However, this curve cannot be predicted in advance and it depends on the problem domain and data distribution. Also, it has an interesting property: the slope (steepness of the curve) indicates the difficulty of a model in finding appropriate solutions to the problem.

The authors conclude:

_Finally, for most real world applications, there is likely to be a non-zero lower-bound error past which models will be unable to improve. This lower bound includes Bayes error — the information theoretic lower bound based on the data generating function — and a combination of other factors that cause imperfect generalization. For instance, mislabeled samples in the training or validation data sets are likely to cause irreducible error. ([source](https://arxiv.org/pdf/1712.00409.pdf))_

In other words, some of the errors cannot be corrected (it is implicit in the problem and the model itself). The other part, however, depends on the quality of the dataset. As was seen for [LLM](https://en.wikipedia.org/wiki/Large_language_model), the quality of the dataset also impacts the so-called power law.

As seen, while the so-called [emergent properties](https://towardsdatascience.com/emergent-abilities-in-ai-are-we-chasing-a-myth-fead754a1bf9) are questionable, as model parameters and the number of examples in the dataset increase, the model increases its performance. So if a tabular dataset is very large a deep learning model might have superior performance (and its large number of parameters might be justified).

**Do NNs need feature engineering**

_Tabular data has welldefined features, but data scientists still spend a lot of time curating and creating new features. This shows the need for representation learning that can automate feature selection and feature interactions. ([source](https://arxiv.org/abs/2207.08548))_

It can generally be said that NN network models do feature engineering and feature selection internally. For example, vanilla NN does variable selection because the weights associated with some features go to zero and no longer contribute to prediction.

In addition, [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) automatically build new features during their training. Each layer creates a new representation from that of the previous layer, and in this process, new features are created. In addition, along the model, this process of abstraction takes place, from up to the bottom we have the emergence of a representation (and thus more and more abstract figures).

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data12.webp)
*image source: [here](https://arxiv.org/abs/1702.08835)*

Obviously, this cannot be achieved with decision trees or ensembles because the growth of the model is horizontal, not vertical. So much so, that a single layer of NN even growing infinitely horizontally (adding neurons to the layer) cannot be as successful as an NN with fewer neurons but more layers.

So it can be said that ensembles work on the original features by requiring feature engineering and tree-based models do not create new features in the process (there is no in-model transformation).

So we no longer need feature engineering?

Actually, although NNs do some internal feature engineering this also has a cost:

* [interpretability](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), the more the number of connections grows a deep learning model is harder to explain. Also, removing non-predictive features can be helpful for the model to learn.
* [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering) helps to understand the quality of the data or if there are problems with the data.
* [Deep learning](https://en.wikipedia.org/wiki/Deep_learning) models are expensive; removing unnecessary features reduces the [computational cost](https://en.wikipedia.org/wiki/Computational_complexity) of the model.
* Nonlinear feature transformations can help model convergence (reducing the number of epochs) and reduce the impact of outliers.
* Considering the vastness of the [hypothesis space](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_373), building features that represent feature interdependence helps the model learn better (instead of having to learn these interactions on its own). In some cases, a few well-constructed features can have a dramatic impact.

In short, although one of the advantages of NNs is that they create new features (which tree-based models cannot do), feature engineering improves model performance.

**Fine-tuning, pretraining, and other stories**

_An important advantage of deep models over GBDT is that they can potentially achieve higher performance via pretraining their parameters with a properly designed objective. ([source](https://arxiv.org/abs/2207.03208))_

In [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) or in [computer vision](https://en.wikipedia.org/wiki/Computer_vision), the ultimate advantage of deep learning models is the ability to train a model with a large amount of data so that the model learns a representation of the domain. After that, [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) can be done on the dataset of interest.

This is convenient because you only have to train one model, and then just fit it to the dataset of interest. For example, [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) has been used for many articles and applications. The authors trained the model on [1 million images](https://en.wikipedia.org/wiki/ImageNet), after which the model can be used (frozen or slightly updating parameters) for any other image dataset. This, however, has not happened for tabular learning.

These broad models are also called [foundation models](https://www.adalovelaceinstitute.org/resource/foundation-models-explainer/). The idea is to train these models in an [unsupervised manner](https://en.wikipedia.org/wiki/Unsupervised_learning) with a large amount of unlabeled data and then use the data representation that the models have learned to fit the tasks we are interested in. This has allowed in NLP and computer vision some extremely interesting applications ([commonsense reasoning](https://en.wikipedia.org/wiki/Commonsense_reasoning), q[uestion answering](https://en.wikipedia.org/wiki/Question_answering), and so on). An example of that for computer vision:

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data13.webp)
*image source: [here](https://arxiv.org/abs/2108.07258)*

In addition, efforts have also been made to unify some of these modalities such as images and text. This is to the advantage of not only being able to get applications such as answering questions on pictures but the model benefits from what it learns from one modality for another.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data14.webp)
*image source: [here](https://arxiv.org/abs/2108.07258)*

Tabular data has been left out of this revolution because it is difficult to design a foundation model for tabular learning. Unfortunately, algorithms for self and [semi-supervised learning](https://www.ibm.com/think/topics/semi-supervised-learning) were designed for data that have a spatial or semantic structure (images or text) and therefore are not effective on tabular data. Also, regularization methods and other ideas that work well in other domains are designed having in mind [prior knowledge](https://en.wikipedia.org/wiki/Prior_knowledge_for_pattern_recognition) about the data we use, which we do not have in tabular data.

For example, we thought of [data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) in computer vision because we had both knowledge about the inductive biases and the structures of the images. Although it works very well in computer vision, rotating or cropping an image makes sense but has no equivalent for tabular data (the notion of rotation does not exist for tabular data).

### What do we want for a model in tabular learning

The first requirement is its performance in [inference](https://cloud.google.com/bigquery/docs/inference-overview). In general, we expect a model to be able to produce correct predictions based on past observations. A neural network model for tabular data should be able to compete with tree-based models if not outperform them.

A second important point is the need for [interpretability](https://arxiv.org/abs/2103.11251). Indeed, regulations such as the  California Consumer Privacy Act (CCPA) and the [European General Data Protection Regulation (EU GDPR)](https://gdpr-info.eu/) require algorithms to be explainable. This is a crucial requirement when it comes to the deployment of an algorithm for sensitive applications (medical, insurance, financial, criminal justice, and so on).

On the other hand, neural networks especially when composed of different layers can be computationally expensive. Also, the models become increasingly complicated and theoretically complex to achieve interpretability. Therefore, one would need to design an algorithm that is both computationally cheap (especially with few parameters) and at the same time interpretable.

![description of tabular data](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/tabular_data15.webp)
*image source: [here](https://arxiv.org/abs/2206.10610)*

In general, there is a threshold between accuracy and explainability. We need both, especially when it comes to sensitive applications:

_This resulted in emergence of many government and regulatory initiatives to ensure that high risk and impact AI applications become trustworthy and ethical by providing the following common components: accuracy and robustness, transparency and explainability, human control and oversight, fairness and elimination of bias, and mitigation of risk and safety. ([source](https://arxiv.org/abs/2212.11136))_

## Why categorical data are problematic for ML and DL algorithms

## Suggested lectures

1. Historical Data Science Trends on Kaggle. [link](https://levelup.gitconnected.com/tabula-rasa-why-do-tree-based-algorithms-outperform-neural-networks-db641862859b#:~:text=Historical%20Data%20Science%20Trends%20on%20Kaggle.%20link)
2. Vasyl Harasymiv, 2015 Lessons from 2 million machine learning models on Kaggle, [link](https://arxiv.org/abs/1805.06440)
3. Shavitt, 2018, Regularization Learning Networks: Deep Learning for Tabular Datasets, [link](https://arxiv.org/abs/1805.06440)
4. Hestness, 2017, Deep Learning Scaling is Predictable, Empirically, [link](https://arxiv.org/abs/1712.00409)
5. Grinsztajn, 2022, Why do tree-based models still outperform deep learning on tabular data?, [link](https://arxiv.org/abs/2207.08815)
6. Borisov, 2021, Deep Neural Networks and Tabular Data: A Survey, [link](https://arxiv.org/abs/2110.01889)
7. Zhou, 2017, deep forest, [link](https://arxiv.org/abs/1702.08835)
8. Liu, 2022, Handling missing values in healthcare data: A systematic review of deep learning-based imputation techniques, [link](https://arxiv.org/abs/2210.08258)
9. Pang, 2020, Deep Learning for Anomaly Detection: A Review, [link](https://arxiv.org/abs/2007.02500)
10. Mansour, 2019, Deep neural networks are lazy: on the inductive bias of deep learning, [link](https://dspace.mit.edu/handle/1721.1/121680)
11. Huang, 2020, TabTransformer: Tabular Data Modeling Using Contextual Embeddings, [link](https://arxiv.org/abs/2012.06678v1)
12. Goodman, 2016, European Union regulations on algorithmic decision-making and a “right to explanation”, [link](https://arxiv.org/abs/1606.08813)
13. Petkovic, 2022, It is not “accuracy vs. explainability” — we need both for trustworthy AI systems, [link](https://arxiv.org/abs/2212.11136)
14. Borisov, 2022, Language Models are Realistic Tabular Data Generators, [link](https://arxiv.org/abs/2210.06280)
15. Joseph, 2022, GANDALF: Gated Adaptive Network for Deep Automated Learning of Features, [link](https://arxiv.org/abs/2207.08548)
16. Rubachev, 2022, Revisiting Pretraining Objectives for Tabular Deep Learning, [link](https://arxiv.org/abs/2207.03208)
17. Bommasani, 2021, On the Opportunities and Risks of Foundation Models, [link](https://arxiv.org/abs/2108.07258)
18. Yoon, 2020, VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain, [link](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)
19. Ye, 2024, A Closer Look at Deep Learning on Tabular Data, [link](https://arxiv.org/abs/2407.00956)
