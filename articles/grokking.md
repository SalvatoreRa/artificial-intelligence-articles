# Grokking: Learning Is Generalization and Not Memorization
### Understanding how a neural network learns helps us prevent the model from forgetting what it learns

This article is divided into different sections where we will answer these questions:

* What is grokking and why can help us to learn more about how neural networks learn
* How does this elusive phenomenon originate?

Check the list of references at the end of the article, I provide also some suggestions to deepen the topics.

*Updated: November 24*

## Grokking and the inside of the network

What is it grokking?
One of the most repeated concepts in neural networks is: that when the training loss converges to a low value, the neural network will no longer learn much. Yet in a 2021 study, they observed a strange phenomenon, which the authors called “grokking.” The model seems to reach plateaux, where low and stable training loss with poor generalization is observed, and then with further training, the model is capable of perfect generalization.
