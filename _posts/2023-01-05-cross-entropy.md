---
layout: post
title: "The Cross Entropy Loss: a log-likelihood in disguise 🥷"
description: "What's the connection between the cross entropy loss and the log-likelihood?"
date: 2023-01-05 8:00:00+0100
tags: machine-learning likelihood
categories: ml-projects blogposts
---

The **cross-entropy loss** that is so much used in classification tasks is nothing but... the **negative log-likelihood** in disguise! 🥸 In this short post, we will clarify this point and make the connection explicit.

Let's consider a dataset of input-label pairs $$\{x^i, y^i\}_{i=1}^N$$ similar to what we considered in the [last post]({% post_url 2022-12-18-likelihood %}). For Italian names, $$x^i$$ consisted of a set of characters that are used to predict $$y^i$$, the next character.
The model we used, `n`-grams, consists of look-up-tables that, given $$x^i$$, return an array of normalized probabilities that the model assigns to the next character $$y^i$$ being the first, second,... `L`-th element in the vocabulary.
With the use of some notation, our model $$\mathcal{M}$$ that takes as input the $$x^i$$'s and returns a length-`L` vector of probabilities $$p^i = \mathcal{M}(x^i)$$. Neural networks, as we will see, also fit in this setting.

We recall that the log-likelihood of a given sample corresponds to the negative log probability that the model assigns to the correct class $$y^i$$:

$$\mathcal{l}^i = -\log [\mathcal{M}(x^i)]_{y^i} = -\log [p^i]_{y^i}$$

where we used the notation $$[\mathbf{x}]_{i}$$ to indicate the $$i$$-th element of vector $$\mathbf{x}$$ and lowercase $$\mathcal{l}$$ to indicate the loss of one single data point.

At this stage, the $$y^i$$'s take values in $$\{0, 1,\dots, L\}$$. If we instead represent them in [one-hot-encoding](https://en.wikipedia.org/wiki/One-hot), i.e.

$$
\begin{aligned}
\cdot &\rightarrow 0 \rightarrow [1,0,0,..., 0] \\
a &\rightarrow 1 \rightarrow [0,1,0,..., 0]\\
&\qquad\quad\vdots\\
ù &\rightarrow L \rightarrow [0, ...,0,0,1]\\
\end{aligned}
$$

the negative log-likelihood of a sample simply reads

$$\mathcal{l}^i = -\sum_{c=1}^L y^i_c\log (p^i_c),$$

that is **the definition of cross-entropy loss**!!

I hope that this helped shed some light 🔦 on the widespread use of this seemingly obscure loss function, and its justification.
