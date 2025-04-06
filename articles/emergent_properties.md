# A focus on emergent properties in artificial intelligence

## Index
* [The scaling law](#The-scaling-law)
* [What are emergent properties](#What-are-emergent-properties)
* [Appendix](#Appendix)
* [Other resources](#Other-resources)


## The scaling law

OpenAI proposed in 2020 a _power law for the performance of LLMs_: according to this [scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law), there is a relationship with three main factors (model size (N), dataset size (D), and the amount of training compute (C)) and the model loss $L$. Given these factors we can derive the performance of the models:

![scaling law](https://github.com/SalvatoreRa/tutorial/blob/main/images/scaling_law.png?raw=true)
*from the [original article](https://arxiv.org/abs/2001.08361)*

A [later work](https://arxiv.org/abs/2407.13623) suggests that vocabulary size also follows a scaling law (model performance is also impacted by vocabulary size). A larger vocabulary allows more concepts and nuances to be represented. According to the authors, vocabulary size in today's models is not optimal but is underestimated.

![Vocabulary scaling law](https://github.com/SalvatoreRa/tutorial/blob/main/images/vocabulary_scaling_law.png?raw=true)
*from the [original article](https://arxiv.org/abs/2407.13623)*

In addition, the scaling law might be different depending on the skills in which we are interested. [This work](https://arxiv.org/abs/2503.10061) suggests the existence of skill-dependent scaling laws. While traditional scaling laws tend to optimize the overall loss on a generic validation set, this paper suggests that knowledge tasks are capacity-hungry (i.e., prefer more parameters), while coding tasks (qua used as a proxy for reasoning) prefer more tokens in training (data-hungry). These differences would remain even if you change the percentage of tokens in the mix, and for the authors stem from the fact that knowledge is harder to compress than code, requiring more capacity to memorize facts. In practice, the training strategy should derive from the skill we are interested in.

![Compute Optimal Scaling of Skills](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/skill_specific_scaling_law.png?raw=true)
*from the [original article](https://arxiv.org/abs/2503.10061)*


After the publication of [ChatGPT4-o1](https://openai.com/index/introducing-openai-o1-preview/), extending the scaling law to inference time was also discussed. ChatGPT4-o1 was trained on a large amount of chain-of-thoughts (i.e., with a series of reasoning intermediates) to improve its reasoning ability. The model is then trained to conduct a whole series of reasoning steps to improve its ability on complex problems requiring reasoning. 

*We have found that the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute).--[source](https://openai.com/index/learning-to-reason-with-llms/)*

![inference scaling law](https://github.com/SalvatoreRa/tutorial/blob/main/images/inference_time_scaling_law.png?raw=true)
*from [here](https://openai.com/index/learning-to-reason-with-llms/)*

This is why some researchers suggest that we can also talk about a kind of inference scaling law (thus relating inference time to accuracy), the topic is still debated

## What are emergent properties

**Emergent properties** are properties that appear only with scale (as the number of parameters increases)

*"In the literature, emergent abilities of LLMs are formally defined as “the abilities that are not present in small models but arise in large models”, which is one of the most prominent features that distinguish LLMs from previous PLMs."*-[source](https://arxiv.org/pdf/2303.18223.pdf)

![emergent_properties](https://github.com/SalvatoreRa/tutorial/blob/main/images/emergent_properties.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2206.07682.pdf)*

![emergent_properties](https://github.com/SalvatoreRa/tutorial/blob/main/images/emergent_properties2.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2206.07682.pdf)*

More formally, four definitions have been proposed for an emergent property in LLM:
1. A property is emergent if it is present in the large models but not in the small models. A property then emerges with scale (the definition we saw above)
2. A property that a model exhibits without being trained specifically for. For example, the emergence of in-context learning (fitting a model to a downstream task without the need for training but only providing examples) is a property that emerges that was neither anticipated nor the model was trained for (more info [here](https://arxiv.org/abs/2108.07258) and [here](https://arxiv.org/abs/2005.14165))
3. A property that emerges from what the model learns from the pre-training data. This definition takes the magic out of the model since it means that the model learns successfully from the data and so the emergent property is just a learned property. In addition, it means that the emergent property is derived only from the training dataset and thus from its quality (more info [here](https://arxiv.org/abs/2305.17266))
4. Similar to the first but requires two conditions: sharpness (transitioning seemingly instantaneously from not present to present) and unpredictability (transitioning at seemingly unforeseeable model scales). The first means that by plotting the model scale the property emerges instantaneously at some point and that this transition is discontinuous (flat and then a sharp increase).

These definitions are themselves controversial. For example, definition 3 is simply the definition of learning (in unsupervised learning we expect the model to learn patterns and properties from the data). The former is confusing since larger models are expected to be better than smaller ones (e.g., lower loss, better capabilities) if not why spend all these resources? In agreement with definition 3, LLMs are better than small language models (SLMs), which means that LLMs just have more resources to learn from the data. Consequently, we do not have a clear definition of emergence.

Moreover, not everyone agrees on the real existence of these emerging properties

*" There are also extensive debates on the rationality of emergent abilities. A popular speculation is that emergent abilities might be partially attributed to the evaluation setting for special tasks (e.g., the discontinuous evaluation metrics)."*-[source](https://arxiv.org/pdf/2303.18223.pdf)

So using discontinuous metrics favors the appearance of discontinuous properties (aka emergent properties). [In addition](https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities), we considered only a few data points (1B, 7B, 13B, 70B, 150B) and have no results for intermediate points, this favors discontinuous behavior (if we had more data points many properties would be less discontinuous). The unpredictability itself is also controversial. In fact, many behaviors of an LLM can be predicted with smaller models (although a difficult task both at the technical level and because there are still things we are not clear about the scaling of LLMs).

At present only definition 2 stands (the property emerges without the model being trained for it). This definition is a bit vague and could refer to one of two cases:
* The property emerges even if the model is not exposed to the data for that property.
* The model is exposed to the data for the property but the developers do not know about it.

The problem is that we don't know that we don't know what is in the training set of many LLMs (especially the closed-source ones). For example, [ChatGPT is not trained to play chess](https://www.reddit.com/r/AnarchyChess/comments/10ydnbb/i_placed_stockfish_white_against_chatgpt_black/) but because there is a huge amount of discussion on the internet about chess the model has been exposed about the game of chess, rules, moves used by users and so on. This then makes it complicated to decide whether a property is emergent or not (especially if we don't know what the model has seen).

There are also many others for which many researchers remain skeptical of emergent properties. The first reason is **prompt sensitivity** or the phenomenon that LLMs respond differently depending on the prompt (even though the semantic meaning is equivalent). This is a phenomenon observed in all LLMs and would seem to indicate that LLMs are more of statistical pattern machines than capable of reasoning. In this study, they show how [in-context learning depends on the structure of the prompt](https://aclanthology.org/2022.acl-long.556/). In [other paper](https://arxiv.org/pdf/2406.11050) they show how perturbations of the prompt cause reduced performance. If the pattern were capable of reasoning or if it were an emergent property this phenomenon should not happen. On the contrary, the appearance of this phenomenon indicates that the model is just matching the request in the prompt in the data. This phenomenon is also called **Token bias** and indicates that the model is biased toward data that it saw in the training:

*A strong token bias suggests that the model is relying on superficial patterns in the input rather than truly understanding the underlying reasoning task--[source](https://arxiv.org/pdf/2406.11050)*

![example of token bias in LLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/LLM_unable_reasoning.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2406.11050)*

Another indication is that the model seems to regurgitate data more than it understands (the model then only performs correctly if it has seen the data). [This article](https://arxiv.org/abs/2211.09110) shows that there is a correlation between model accuracy and copyright issue risk. This agrees with the results of [this study](https://arxiv.org/pdf/2309.13638) where the performance of the model depends on the frequency of the words in the given training

![Effects on the performance of large language models that are attributable to the fact that they are statistical next-word prediction systems.](https://github.com/SalvatoreRa/tutorial/blob/main/images/sensibility_probability.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2309.13638)*

[This other study](https://arxiv.org/abs/2309.01809) suggests that emergent properties could be reduced to in-context learning and that this is favored by instruction tuning. In-context learning (ICL) itself [appears to appear](https://proceedings.neurips.cc/paper_files/paper/2022/hash/77c6ccacfd9962e2307fc64680fc5ace-Abstract-Conference.html) only when transformers are trained on sequences that are similar to which ICL is tested. [This study](https://arxiv.org/abs/2304.03439) suggests that there is some contamination in the datasets, since ChatGPT performs better on old benchmarks than in new ones.

In contrast, subsequent articles renew interest in [emerging properties](https://en.wikipedia.org/wiki/Emergence). [This article](https://arxiv.org/abs/2408.12578) agree we need another definition of emergent property. Therefore, the authors define three properties that an emergent property must have (this is inspired by physics where emergent properties are well characterized):

*Specifically, we argue three characteristics should be observed to claim a capability is emergent (see Def. 1): beyond (i) sudden performance improvement for a specific task, we claim emergence is more likely to represent a meaningful concept if (ii) performance on several tasks improves simultaneously and (iii) there are precise structural changes in the model at the point of emergence. The intuition, borrowed from the study of emergence in other fields -[source](https://arxiv.org/pdf/2408.12578)*

![definition of emergent properties in LLMs](https://github.com/SalvatoreRa/tutorial/blob/main/images/emergent_properties_llm_definition.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2408.12578)*

The authors then state that an emergent property appears because something has changed in the structure of the model. For example, in their experiments, some properties appear because the model has learned the grammar of the system and the constraints they had defined. They talk extensively in the paper about the relationship between memorization and generalization, and it would probably be interesting to discuss in terms of grokking. The authors use a toy model and dataset, so the question about emergent properties remains open.

## Appendix

### Power law

[Power laws](https://en.wikipedia.org/wiki/Power_law) are laws that describe the relationship between two quantities. Typically, for LLMs, we try to tie test loss (which would be a proxy for model performance or accuracy in downstream tasks) and another quantity that we are interested in scaling (e.g., model parameters or the number of tokens to train it). The power law informs us what the relationship is between these two quantities; for example, by increasing the number of parameters, we should expect the test loss to decrease by a constant factor. A general formula to represent the power law, given two quantities x and y, the lore relationship is defined as in the formula below (where a and p are two constants):

$$
y = ax^p
$$

We can generate similar plots in Python for a similar function, in this a is 1 and p is 2:

```python 
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.01, 1, 100)
y = np.sqrt(x)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(x, y)
axs[0].set_title("Power law: Normal Scale")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].loglog(x, y)
axs[1].set_title("Power law: Log-Log Scale")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

plt.tight_layout()
plt.show()
```

![Example of power law](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/power_law.png?raw=true)

The inverse is usually used to better show the decrease in loss: 

$$
y = a \left( \frac{1}{x} \right)^p
$$

We can plot in this case as well:

```python 
x = np.linspace(0.01, 1, 100)
y =  np.sqrt(1/ x)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(x, y)
axs[0].set_title("Inverse power law: Normal Scale")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].loglog(x, y)
axs[1].set_title("Inverse power law: Log-Log Scale")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

plt.tight_layout()
plt.show()
```

![Example of inverse power law](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/inverse_power_law.png?raw=true)

We can try a small experiment in which we gradually increase the number of parameters in a neural network (in this case, simply the hidden size) and plot the results as if we were looking for a scaling law. Note that all parameters remain the same (epochs, loss, optimizer)

```python 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model definition with variable hidden size
class SimpleNet(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Settings
hidden_sizes = [2, 8, 16, 32, 64, 128, 256, 512, 1024]
final_losses = []
param_counts = []

# Training loop
for h in hidden_sizes:
    model = SimpleNet(h).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_counts.append(total_params)

    # Training
    model.train()
    for epoch in range(3):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward + backward
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    final_losses.append(loss.item())  # Last batch loss

# Plotting
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(param_counts, final_losses, marker='o')
plt.title("Normal Scale")
plt.xlabel("Number of Parameters")
plt.ylabel("Final Training Loss")

plt.subplot(1, 2, 2)
plt.loglog(param_counts, final_losses, marker='o')
plt.title("Log-Log Scale")
plt.xlabel("Number of Parameters")
plt.ylabel("Final Training Loss")

plt.tight_layout()
plt.show()
```

![Example of scaling law with MNIST](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/MNIST_power_law.png?raw=true)

What we observe is that for small/medium models, the [training loss](https://www.baeldung.com/cs/training-validation-loss-deep-learning) decreases, and in the log-log plot, it looks like a straight-ish line (a sign of a power-law scaling between parameters and performance). This is in line with deep learning scaling laws and has been described in the literature. In addition, when the model becomes too large, the training loss starts to flatten and then grow, showing overfitting signs (remember this is a simple experiment where we are disproportionately increasing the hidden layer).

These curves can help guide how we choose to design our models. One common approach is to analyze model behavior under fixed compute budgets. Compute is often measured in FLOPs, and an approximate relation is _B≈6pt_, where _p_ is the number of parameters and _t_ is the number of training examples. For large language models, _t_ typically refers to the number of tokens used during training.

In this experiment, we adopt a similar mindset but focus on a toy setup. We fix the total number of neurons in the hidden layers to 128 — a stand-in for fixed model "capacity" — and explore how distributing these neurons across different depths affects learning. That is, we vary depth (e.g., 1×128, 2×64, 3×42, 4×32), while keeping the total number of hidden units constant.

The results show that with limited data, shallower models (like 1×128) tend to perform better. This is likely because they are easier to optimize and less prone to underfitting small datasets. In contrast, deeper models like 4×32 struggle in the low-data regime — indicating they are harder to train without sufficient data.

Overall, even though this is a simplified setup, it demonstrates how IsoFLOP-style curves can reveal useful insights into the interplay between model architecture and data — insights that scale to much larger models and real-world tasks.


![Example of scaling law with MNIST](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/MNIST_power_law_same_number_neurons.png?raw=true)

## Other resources

Articles describing in detail:
  * [A Requiem for the Transformer?](https://towardsdatascience.com/a-requiem-for-the-transformer-297e6f14e189)
  * [Emergent Abilities in AI: Are We Chasing a Myth?](https://towardsdatascience.com/emergent-abilities-in-ai-are-we-chasing-a-myth-fead754a1bf9)

 
Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [About emergent properties](https://arxiv.org/pdf/2206.07682.pdf)
  * [a good survey on LLMs, scaling law, and so on](https://arxiv.org/abs/2303.18223)
