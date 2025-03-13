# Tabular deep learning

* [Why Do Tree Based Algorithms Outperform Neural Networks](#Why-Do-Tree-Based-Algorithms-Outperform-Neural-Networks)
* [Challenges with tabular data](#Challenges-with-tabular-data)

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
