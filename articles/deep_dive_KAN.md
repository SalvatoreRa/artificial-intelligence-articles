# Kolmogorov-Arnold Networks (KANs)

## What are the Kolmogorov-Arnold Networks (KANs)?

  **Kolmogorov-Arnold Networks (KANs)** are a new type of neural network that is based on the *[Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)* (while classical neural networks are based universal approximation theorem, according to which a neural network could approximate any function). 

According to the Kolmogorov-Arnold representation theorem, any multivariate function can be expressed as a finite composition of continuous functions (combined by addition). To make a simpler example, we can imagine a cake as the result of a series of ingredients combined together in some way. In short, a complex object can be seen as the sum of individual elements that are combined in a specific way. In a recipe, we add only one ingredient at a time to make the process simpler. 

$$f(x_1, \ldots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

Observing this equation, we have a multivariate function (our cake) and univariate functions (our ingredients and $\Phi_q$ explaining how they are combined (the recipe steps). In short, from a finished product, we want to reconstruct the recipe.

Why is this theorem of interest to us? Because in machine learning we need systems that allow us to approximate complex functions efficiently and accurately. Especially when there are so many dimensions, neural networks are in danger of falling into what is called the curse of dimensionality. 

The second theoretical element we need is the concept of **spline.** Spline is a piecewise polynomial function that defines a smooth curve through a series of points. **B-splines**, on the other hand, is the mode of fit. For example, let's imagine that we have collected temperature data throughout the day at varying intervals and we want at this point a curve that shows us the trend. We can use a polynomial curve. The problem is that we would like the best one, only this doesn't happen and these curves tend to fluctuate quite a bit ([Runge's phenomenon](https://en.wikipedia.org/wiki/Runge%27s_phenomenon) for friends). Spline allows us to fit better because it divides the data into segments and fits an individual polynomial curve for each segment (before they had one curve for all the data). **B-splines** are an improvement that allows for better-fit curves. B-spline in short provides better accuracy. It achieves this by using control points to guide the fitting.

 ![comparison spline and polynomial function](https://github.com/SalvatoreRa/tutorial/blob/main/images/spline.png?raw=true) 

Mathematically this is the equation of the b-spline:

$$C(t) = \sum_{i=0}^{n} P_i N_{i,k}(t)$$

where $P_i$ are the control points, $N_{i,k}$ are called the basis fucntion and $t$ is called knot vector.

Now we have the theoretical elements, what we need to keep in mind is: 
* given a complex function we have a theorem that tells us we can reconstruct it from single unitary elements and a series of steps.
* we can fit a curve with great accuracy for a series of points and thus highlight trends and other analyses.

<center>
  
## Why do we care about it? How can we use it for a neural network?

</center>

The classical neural network has some limitations: 
* Fixed activation functions on the node. Each neuron has a predetermined activation function (like ReLU or Sigmoid). This is fine in many of the cases though it reduces the flexibility and adaptability of the network. In some cases, it is difficult for a neural network to optimize a certain function or adapt to certain data.
* Interpretability. Neural networks are poorly interpretable, the more parameters the worse it becomes. Understanding the internal decision-making process becomes difficult and therefore it is harder to trust the predictions.

At this point, KANs have recently been proposed to solve these two problems.

 ![KAN introduction](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_introduction.png?raw=true) * from [the original papers](https://arxiv.org/pdf/2404.19756)*

KANs are based in joining the Kolmogorov-Arnold Representation (KAR) theorem with B-splines. At each edge of the neural network, we use B-splines at each edge of each neuron so as to porter learn B-spline activation function. In other words, the model learns the decomposition of the data (our pie) into a series of b-splines (our ingredients).

![KAN versus MLP](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_vs_MLP.png?raw=true) * from [the original papers](https://arxiv.org/pdf/2404.19756)*

Now let us go into a little more detail. In KAN the matrix of weights is replaced by a set of univariate function parameters at the edges of the network. Each node then can be seen as the sum of these functions (which are nonlinear). In contrast in MLPs, we have a linear transformation (the multiplication with the matrix of weights) and a nonlinear function. In formulas we can clearly see the difference:

$$\text{KAN}(\mathbf{x}) = \left( \Phi_{L-1} \circ \Phi_{L-2}  \circ \ \cdots \circ \Phi_1 \circ \Phi_0 \right) \mathbf{x}$$

$$\text{MLP}(\mathbf{x}) = \left( \mathbf{W}_{L-1} \circ \sigma \circ \cdots \circ \mathbf{W}_1 \circ \sigma \circ \mathbf{W}_0  \right) \mathbf{x}$$

In a more compact version, we can rewrite it like this: 

$$f(x_1, x_2, \ldots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

Where $ϕ_q,p$ are univariate functions (b-splines) and $ϕ_q$ is the final function that assembles everything.

Now each layer of a KAN network can be seen like this:

$$\mathbf{x}^{(l+1)} = \sum_{i=1}^{n_l} \phi_{i,j} \left(x_i^{(l)} \right)$$

where $x(l)$ is the transformation of the input to layer $l$ (basically the cooked dish after a number of steps and added ingredients) and $ϕ_l,i,j$ are the functions at the edges between layer $l$ and $l+1$.

Notice that the authors want to maintain this parallelism with the MLP. In MLPs you can stack different layers and then each layer learns a different representation of the data. The authors use the same principle, where they have a KAN layer and more layers can be added.

![KAN B-splines representation](https://github.com/SalvatoreRa/tutorial/blob/main/images/kan_splines.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2404.19756)*

B-splines allow us to learn complex relationships in the data, simply that they adjust their shape to minimize the approximation error. These flexibilities allow us to learn complex yet subtle patterns.

The beauty of B-splines is that they are controlled by a set of control points (called grid points), the greater these points the greater the accuracy a spline can use to represent the feature. the greater the grid points, the more detail a splines can capture in the data. The authors therefore decided to use a technique to optimize this process, and learn more detailed patterns (add grid points) without conducting retraining, however. 

As you can see the model starts with a coarse grid (fewer intervals). the idea is to start with learning the basic structure of the data without focusing on the details. As learning progresses you start adding points (refine the grid) and this di allows you to capture more details in the data. This is achieved by using least squares optimization to try to minimize the difference between the refined spline and the original one (the one with fewer points, in short learning more detail without losing the overall knowledge about the data previously learned). We could define it as starting with a sketch of a drawing, to which we gradually add details, trying not to transform the original basic idea

![KAN Grid extension](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_grid_extension.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2404.19756)*

*In principle, a spline can be made arbitrarily accurate to a target function as the grid can be made arbitrarily fine-grained. This good feature is inherited by KANs. By contrast, MLPs do not have the notion of “fine-graining”. Admittedly, increasing the width and depth of MLPs can lead to improvement in performance (“neural scaling laws”). However, these neural scaling laws are slow -[source](https://arxiv.org/pdf/2404.19756)*

In other words, grid extension allows us to make the KAN more accurate without having to increase the number of parameters (add functions, for example). For the authors, this allows even small KANs to be accurate (sometimes even more so than larger ones (with more layers, and more functions) probably because the latter capture more noise).

To summarize:

*The architecture of Kolmogorov-Arnold Networks (KAN) is unique in that its core idea is to replace traditional fixed linear weights with learnable univariate functions, achieving greater flexibility and adaptability. The architecture of KAN consists of multiple layers, each containing several nodes and edges. Each node is responsible for receiving input signals from the previous layer and applying nonlinear transformations to these signals via learnable univariate functions on the edges. These univariate functions are typically parameterized by spline functions to ensure smoothness and stability in the data processing process-[source](https://arxiv.org/pdf/2407.11075)*

The use of splines makes it possible to capture complex patterns in the data as nonlinear relationships. The advantage of this architecture is that it is adaptable to various patterns in the data, and these functions can be adapted dynamically and also be refined in the process. This then allows the model to be not only adaptable but also very expressive.

One of the strengths of KANs is precisely the interpretability. To improve this, the authors use two techniques:
* Sparsification and Pruning
* Symbolification

**Sparsification** is used to sparsify the network, so we can eliminate connections that are not needed. to do this the authors use L1-regularization. Usually in neural networks, L1-norm reduces the magnitude of the weights by inducing sparsification (making them zero or close to zero). In KAN there are no “weights” proper so we should define the L1 norm of these activation functions. In this case, we therefore act on the function:

$$\left\lvert \phi \right\rvert_1 = \frac{1}{N_p} \sum_{s=1}^{N_p} \left\lvert \phi(x_s) \right\rvert$$.

with $\phi(x_s)$ representing the value of the function $N_p$ the number of input samples. With a $L_1$ in short we evaluate the value of the function and try to reduce it as a function of the absolute value of their mean. 

**Pruning** is another technique used in neural networks in which we eliminate connections (neurons or edges) that are below a certain threshold value. This is because weights that are too small do not have an impact and can be eliminated. So by pruning, we can get a smaller network (a subnetwork). In general, this subnetwork is less heavy and more efficient, but it maintains the performance of the original network (like when pruning dead branches in a tree, the tree usually grows better). After pruning and sparsification, we then have a network that is less complex and potentially more interpretable.

**Symbolification** is an interesting approach because the goal is to replace learned univariate functions with known symbolic functions (such as cosine, sine, or log).  So given a univariate function we want to identify potential symbolic functions. In practice, the univariate function can practically be approximated by another function that is better known and humanly readable. This task may not seem easy:

*However, we cannot simply set the activation function to be the exact symbolic formula, since its inputs and outputs may have shifts and scalings. -[source](https://arxiv.org/pdf/2404.19756)*

So taking the input $x$ and the output $y$ we want to replace with a function $f$, but we learn parameters (a,b,c,d) to try to approximate the original univariate function with: $y \approx c f(ax + b) + d$. This is then done with grid search and linear regression.

![KAN Grid extension](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_sparsification_and_symbolification.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2404.19756)*

At this point:
* we know how to train a KAN network. 
* we have eliminated unnecessary connections by sparsification and pruning.
* we have made it more interpretable because now our network is no longer composed of univariate functions but of symbolic functions

The authors provide some examples in this paper where the elements we have seen are useful. For example, KANs are most efficient at tasks such as fitting in inputs of various sizes. KANs for the authors are more expressive but more importantly more efficient than MLPs (they require fewer parameters and scale better). It is also easy to interpret when the relationship between X and y is a symbolic function.

![KAN scaling in comparison to MLP](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_scaling.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2404.19756)*

Another interesting point is that for the authors, KANs work best for **continual learning**. According to them, these networks are better able to retain learned information and adapt to learn new information (for more details on continual learning we discuss it in more detail [in this section](https://github.com/SalvatoreRa/tutorial/blob/main/artificial%20intelligence/FAQ.md#:~:text=What%20is%20continual%20learning%3F%20Why%20do%20neural%20networks%20struggle%20with%20continual%20learning%3F) later). 

*When a neural network is trained on task 1 and then shifted to being trained on task 2, the network will soon forget about how to perform task 1. A key difference between artificial neural networks and human brains is that human brains have function ally distinct modules placed locally in space. When a new task is learned, structure re-organization only occurs in local regions responsible for relevant skills , leaving other regions intact. -[source](https://arxiv.org/pdf/2404.19756)*

For the authors, this favors KANs and the local nature of splines. This is because MLP rely on global activations that impact the entire model, while KANs have a more local optimization for each new example (as a new example arrives they only change limited sets of spline coefficients). In MLPs, on the other hand, any local changes are propagated throughout the system likely damaging learned knowledge (catastrophic forgetting).

*As expected, KAN only remodels regions where data is present on in the current phase, leaving previous regions unchanged. By contrast, MLPs remodels the whole region after seeing new data samples, leading to catastrophic forgetting. -[source](https://arxiv.org/pdf/2404.19756)*

As can be seen in this case, KANs do not forget the information learned up to that point:

![KAN continual learning](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_continual_learning.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2404.19756)*

*Currently, the biggest bottleneck of KANs lies in its slow training. KANs are usually 10x slower than MLPs, given the same number of parameters. We should be honest that we did not try hard to optimize KANs’ efficiency though, so we deem KANs’ slow training more as an engineering
problem to be improved in the future rather than a fundamental limitation. If one wants to train a model fast, one should use MLPs. In other cases, however, KANs should be comparable or better than MLPs, which makes them worth trying. -[source](https://arxiv.org/pdf/2404.19756)*

![KAN guide](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_guide.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2404.19756)*

The question is: are these KANs better than MLPs?

Not everyone agrees with the supposed superiority of KANs. For example, [in this report](https://vikasdhiman.info/reviews/KAN_a_review.pdf) four criticisms of the original article are offered:
1. **MLPs have learnable activation functions as well.** Indeed, learnable functions can be used as activation functions. For example, something similar has already been explored [here](https://arxiv.org/abs/1906.09529). For example, if we consider a very simple MLP with two layers and rewrite it with a learnable activation function it is very reminiscent of kan $f(\mathbf{x}) = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x}) = \mathbf{W}_2 \phi_1(\mathbf{x}).$
2. **The content of the paper does not justify the name, Kolmogorov-Arnold Networks (KANs).** An MLP can be rewritten as an addition. The difference between the KAT theorem and the Universal Approximation Theorem (UAT), is that the latter requires that to approximate any two-layer function with infinite neurons, while KAT would reduce to (2n + 1) function in hidden layers. The authors do not consistently use (2n + 1) in the hidden layer.
3 **KANs are MLPs with spline-basis as the activation function.** Rather than new neural networks, they would be MLPs in which the activation function is a spline
4. **KANs do not beat the curse of dimensionality.** For the author, the claim is unwarranted by the evidence.

## How do they compare with MLP?

[In this paper](https://arxiv.org/abs/2407.16674), instead they try to conduct an in-depth comparison between MLP and KAN for different domains (controlling the number of parameters and evaluating different tasks). The authors comment: *"Under these fair settings, we observe that KAN outperforms MLP only in symbolic formula representation tasks, while MLP typically excels over KAN in other tasks.“* In the ablation studies they show that its B-spline activation function gives an advantage in symbolic formula representation.

![KAN guide](https://github.com/SalvatoreRa/tutorial/blob/main/images/kan_mlp_fair_comparison.png?raw=true) *from [the original papers](https://arxiv.org/pdf/2407.16674)*

Overall, the article has the positive effect of making the use of B-splines in neural networks more tractable and proposing a system that is more interpretable (via sparsification, pruning, and symbolification). Also, the system is not yet optimized so in computational terms it is not exactly competitive with an MLP. It can be an interesting alternative though and still develop.

In [this other article](https://arxiv.org/abs/2410.01803) they compared the ability of KAN and MLP to represent and approximate functions.

*In this article, we study the approximation theory and spectral bias of KANs and compare them with MLPs. Specifically, we show that any MLP with the ReLUk activation function can be reparameterized as a KAN with a comparable number of parameters. This establishes that the representation and approximation power of KANs is at least as great as that of MLPs. -[source](https://arxiv.org/pdf/2410.01803)*

Thus, KANs can represent MLPs that have a similar size (in other words, they have the same approximation power). 

*On the other hand, we also show that any KAN (without SiLU non-linearity) can be represented using an MLP. However, the number of parameters in the MLP representation is larger by a factor proportional to the grid size of the KAN-[source](https://arxiv.org/pdf/2410.01803)*

MLPs can represent KANs, however, at the cost of increasing the parameter number. The number of parameters of an MLP increases significantly with the grid size of the KAN. So if we have a task that requires KANs with a large grids, it is not efficient to use an MLP.

Another interesting result is the spectral bias analysis. MLPs are known to have a spectral bias for low-frequency components first (like smooth, gradual changes in the function that the model wants to learn). Gradient descent favors low frequencies  are learned earlier in training, in contrast, high-frequency details require finer adjustments, which typically come later in training as the model fits more of the detailed features of the data. This spectral bias serves as a regularizer and seems to be useful for many machine-learning applications. Sometimes high-frequencies (like sharp, rapid changes or very detailed variations, where the function's value changes rapidly over a short interval) are useful to learn. In these cases often high-frequency information has to be encoded using methods like Fourier feature mapping or use different nonlinear functionals. For the authors, KANs theoretically have reduced spectral bias.

![MLPs manifest strong spectral biases (top), while KANs do not (bottom).](https://github.com/SalvatoreRa/tutorial/blob/main/images/spectral_bias_kan.png?raw=true) *MLPs manifest strong spectral biases (top), while KANs do not (bottom). from [the original papers](https://arxiv.org/pdf/2410.01803)*

*Precisely since KANs are not susceptible to spectral biases, they are likely to overfit to noises. As a consequence, we notice that KANs are more subject to overfitting on the training data regression when the task is very complicated. On the other hand, we can increase the number of training points to alleviate the overfitting-[source](https://arxiv.org/pdf/2410.01803)*

So being less susceptible to spectral bias comes at the cost of being more at risk of overfitting. For smooth function, one can use a KAN with a small grid size (fewer points in the B-splines), while for high frequencies function better a KAN with a large grid. For large-scale and smooth function, however, an MLP is recommended.

## Working with KAN

It is actually very easy to train KANs with the official Python library: [PyKAN](https://kindxiaoming.github.io/pykan/). For example, you just need to define the model and to train it:

```Python
model = KAN(width=[4, 5, 3], grid=5, k=3, seed=0, device=device)
results = model.fit(dataset, opt="Adam", steps=100, metrics=(train_acc, test_acc), 
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    lamb=0.01, lamb_entropy=10.);
```

As you can see from the image below, we can see the progressive sparsification effect that happens with KANs. This increases the interpretability of the system:

![KAN guide](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_trained.png?raw=true) 

## Optimization of KAN

Compared with the first version of the KANs, several new implementations and various optimizations have been conducted, especially with respect to the complexity of the high-dimensional function approximations they perform. The power of the KANs comes from their ability to decompose a multivariate function into univariate spline functions, but the efficiency of the process depends on the optimization of the splines. Optimization means adjusting these splines' control points and knots to minimize the error between predicted and actual outputs. Optimizing these splines can be complicated because the landscape is nonlinear and harder to navigate, there is a risk of overfitting as the size increases, and it can become computationally expensive. As noted by this article, there are still some sore points: 

* Sensitivity to Initialization. Poor initialization of spline parameters can lead to suboptimal convergence or the model can get stuck in a local minima. 
* Slow Convergence. This stems from the high number of learnable parameters in KAN.
* Regularization and Dropout. Overfitting can be problematic, especially for high-dimensional data. The flexible nature of splines can lead to overfitting.
* Optimization Instabilities. Some tend to converge to local minima rather than global ones, especially with splines (some variants are less sensitive).

Several techniques have been used or different alternatives have been proposed to limit these problems. For example, the use of batch normalization can lead to better training convergence and stability, regularization techniques can reduce overfitting, residual activation functions (adapted for KANs), specific initializations, and so on can be added. Then there are variants of KANs that focus on speeding up training (FastKAN replaces B-splines with more efficient Gaussian RBFs)

![Cox-KAN pipeline](https://github.com/SalvatoreRa/tutorial/blob/main/images/optimization-of-the-KAN-comparison-with-other-models.png?raw=true) *from [here](https://arxiv.org/pdf/2411.06078)*

## Applications of KAN

![KAN guide](https://github.com/SalvatoreRa/tutorial/blob/main/images/KAN_progression.png?raw=true) *Fast development of KAN in one year. from [the original papers](https://arxiv.org/pdf/2411.06078)*

KANs have been efficiently adapted to deep learning. This was done through a series of modifications and new architectures. This has meant that they can also be used for a variety of applications outside those thought of.

Several articles have come out today that present the application of KANs, below are some examples of applications:
* KAN-based models for medical image segmentation ([U-net based](https://arxiv.org/abs/2406.02918)) have been proposed in computer vision.
* There are also KAN proposed for time series as [here](https://arxiv.org/abs/2405.07344), [here](https://arxiv.org/abs/2406.02486) or [here](https://arxiv.org/pdf/2405.08790). They show they can have similar results to RNNs with less parameters.
* For graph analysis, like graph collaborative filtering ([here](https://arxiv.org/abs/2406.01034)) or molecular representation ([here](https://arxiv.org/abs/2408.01018))
* [Wav-KAN](https://arxiv.org/abs/2405.12832) showcasing KANs’ broad applicability in modern fields like signal
processing.

Seeing some applications in detail, [this study](https://arxiv.org/abs/2409.04290) proposes an extension of KANs for survival modeling (called time-to-event analysis, where they model time until an event happens). In this task, deep learning-based models usually perform better than traditional machine learning models but at the cost of loss of interpretability. Precisely because of the need for interpretability, KANs have been considered a good alternative:

*The key contributions of this paper are in demonstrating that (a) CoxKAN finds interpretable symbolic formulas for the hazard function, (b) CoxKAN identifies biomarkers and complex variable interactions, and (c) CoxKAN achieves performance that is superior to CoxPH and consistent with or better than DeepSurv (the equivalent MLP-based model). --[source](https://arxiv.org/pdf/2409.04290)*

CoxKAN can be seen as an extension of KANs in which censored regression is conducted. Everything is pretty much the same except the loss (you use Cox loss plus a regularization coefficient to sparsify). Since KANs are slower to train than MLPs regularization helps speed up training. Unlike MLP we have both pruning and symbolic fitting. These two steps help make the model more interpretable.

![Cox-KAN pipeline](https://github.com/SalvatoreRa/tutorial/blob/main/images/coxKAN_pipeline.png?raw=true) *from the original article*

once the [specific library](https://github.com/knottwill/CoxKAN/) is installed:

```Python
# Install coxkan
! pip install coxkan scikit-survival
```
We are also installing [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/index.html) to make a comparison with a trained model based on decision trees (XGBoost based).

At this point, we can train our neural network:

```Python
from coxkan import CoxKAN
from sklearn.model_selection import train_test_split
import numpy as np
from coxkan.datasets import gbsg

# load dataset
df_train, df_test = gbsg.load(split=True)
name, duration_col, event_col, covariates = gbsg.metadata()

# init CoxKAN
ckan = CoxKAN(width=[len(covariates), 1], seed=42)

# pre-process and register data
df_train, df_test = ckan.process_data(df_train, df_test, duration_col, event_col, normalization='standard')

# train CoxKAN
_ = ckan.train(
    df_train, 
    df_test, 
    duration_col=duration_col, 
    event_col=event_col,
    opt='Adam',
    lr=0.01,
    steps=100)

print("\nCoxKAN C-Index: ", ckan.cindex(df_test))

# Auto symbolic fitting
fit_success = ckan.auto_symbolic(verbose=False)
display(ckan.symbolic_formula(floating_digit=2)[0][0])

# Plot coxkan
fig = ckan.plot(beta=20)
```
Results for coxKAN:

![KAN network for survival](https://github.com/SalvatoreRa/tutorial/blob/main/images/cox_kan.png?raw=true)

```Python
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# Prepare the target variables for survival analysis
y_train = Surv.from_arrays(event=df_train['event'].astype(bool), time=df_train['duration'])
y_test = Surv.from_arrays(event=df_test['event'].astype(bool), time=df_test['duration'])

# Prepare the feature matrices
X_train = df_train.drop(['duration', 'event'], axis=1)
X_test = df_test.drop(['duration', 'event'], axis=1)

# Initialize and train the model
model = GradientBoostingSurvivalAnalysis()
model.fit(X_train, y_train)

# Predict risk scores on the test set
pred_test = model.predict(X_test)

# Compute the concordance index
cindex = concordance_index_censored(y_test['event'], y_test['time'], pred_test)

print("C-index on the test set:", cindex[0])
```

results for the gradient boosting method:
![scikit survival comparison with coxKAN](https://github.com/SalvatoreRa/tutorial/blob/main/images/scikit_survival.png?raw=true)

We can notice three things: 
* the result is similar to that obtained with traditional machine learning.
* We can get the symbolic formula for our KAN that allows us to interpret the relationship between the various features.
* We can also visualize these features.

## Parting thoughts

KANs have attracted attention as alternatives to traditional deep learning architectures, because of their ability to be able to decompose a multivariate function into simple univariate components. The implementation of KANs in high-dimensional spaces still presents significant computational challenges. This stems from the non-convex nature of the optimization problem thus making training slower and longer. This also impacts hardware consumption according to this [study](https://arxiv.org/abs/2407.17790). In addition, [KANs appear to be sensitive to noise](https://arxiv.org/abs/2407.14882), and its presence reduces the performance of the model. If for low-dimensional tasks they seem more efficient than MLPs for high dimensionality they seem less efficient. Although there are faster implementations of KANs these are not yet optimized enough.

*Empirical studies demonstrate that KANs often outperform traditional architectures, particularly in low-dimensional settings, graph learning, and scientific discovery, where their ability to model compositional and univariate structures proves advantageous --[source](https://arxiv.org/pdf/2411.06078)*

Speaking of generalization, there is still no consensus on their superiority in comparison with traditional architectures. According to some authors, KANs may be advantageous in some tasks but are still inferior to traditional MLPs in other tasks. In addition, KANs would seem prone to overfitting, and so techniques are recommended to reduce this risk

*The interpretability challenges for KANs primarily stem from the complexity of their function compositions, which can obscure the underlying relationships in high-dimensional or nonlinear data. [...] While KANs possess intrinsic interpretability features such as symbolic regression in time series forecasting and scientific discovery SHAP and LIME can augment these by pinpointing feature contributions at a more granular level, potentially bridging the gap between KANs and simpler, interpretable models --[source](https://arxiv.org/pdf/2411.06078)*

In other words, KANs are more interpretable than traditional neural networks, however, they are not as interpretable as inherently interpretable models, such as decision trees and linear regression. So for sensitive applications (such as healthcare) it is better to use methods that are transparent or to use KANs with other interpretability approaches such as LIME or SHAP, especially when we are interested in feature importance. As suggested in [this article](https://arxiv.org/abs/2007.15884), KANs are expressive, but their functions are not intuitive and therefore less interpretable than simpler models

So KANs are one of the most interesting models in recent years, but they are an under-mature model, especially in terms of scalability, lack of benchmarks, and clear use cases.

Suggested lectures:
* [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
* [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205)
* [A Comprehensive Survey on Kolmogorov Arnold Networks (KAN)](https://arxiv.org/abs/2407.11075)

Other resources:
* [official code](https://github.com/KindXiaoming/pykan)
* [Awesome KAN](https://github.com/mintisan/awesome-kan) - a list of resources about KAN
* [The Annotated Kolmogorov-Arnold Network (KAN)](https://alexzhang13.github.io/blog/2024/annotated-kan/) - a great explanation and implementation from scratch of the KAN
* [KANvas](https://kanvas.deepverse.tech/#/kan) - a tool to play and understand KAN
* [Why is the (KAN) Kolmogorov-Arnold Networks so promising](https://engyasin.github.io/posts/why-the-new-kolmogorov-arnold-networks-so-promising/)
* [Implementation on how to use Kolmogorov-Arnold Networks (KANs) for classification and regression tasks.](https://github.com/mintisan/awesome-kan?tab=readme-ov-file)
* [A Survey on Kolmogorov-Arnold Network](https://arxiv.org/abs/2411.06078)
