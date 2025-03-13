# Tabular deep learning

* [Why Do Tree Based Algorithms Outperform Neural Networks](#Why-Do-Tree-Based-Algorithms-Outperform-Neural-Networks)
* [Challenges with tabular data](#Challenges-with-tabular-data)
* [Tabular Learning and the Forest](#Tabular-Learning-and-the-Forest)
* [Why decision tree perform better than a neural network](#Why-decision-tree-perform-better-than-a-neural-network)

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
