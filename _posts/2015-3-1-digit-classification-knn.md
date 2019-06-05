---
layout: post
title: "Digit Classification using KNN"
category: clatern
---

This is a tutorial on classifying handwritten digits with KNN algorithm using Clatern. [Clatern](https://github.com/rinuboney/clatern) is a machine learning library for Clojure, in the works.

{% highlight  clojure %}
(use 'clojure.core.matrix)
(use '(clatern io knn))
{% endhighlight %}

### Dataset

This tutorial uses a stripped down version of handwritten digits dataset available [here](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits). The stripped down version(taken from the sklearn library) is available [here](https://github.com/scikit-learn/scikit-learn/raw/master/sklearn/datasets/data/digits.csv.gz). 

The dataset consists of 1797 samples of 8x8 pixels and the target labels. The first 64 columns are the 8x8 pixels and the 65th column is the label target. Let's have a look at a sample,

![digit](/img/plot_digit.png) 

Let's load the data,
{% highlight  clojure %}
(def digits (load-data "digits.csv"))
{% endhighlight %}

Splitting the data into training and test sets,
{% highlight  clojure %}
(def digits' (order digits 0 (shuffle (range 1797))))
(def train-mat (take 1400 digits'))
(def test-mat (drop 1400 digits'))
{% endhighlight %}

Splitting the training and test set into features and labels,
{% highlight  clojure %}
(def X-train (select train-mat :all (range 64)))
(def y-train (get-column train-mat 64))
(def X-test (select test-mat :all (range 64)))
(def y-test (get-column test-mat 64))
{% endhighlight %}
### Training

Here, we use the KNN model for classifying the digits. The syntax for KNN is,
{% highlight  clojure %}
(knn X y v :k k)
{% endhighlight %}
where,  
*X* is input data,  
*y* is target data,  
*v* is new input to be classified, and  
*k* is the number of neighbours(optional, default = 3)

Let's define a function to perform kNN on our dataset.

{% highlight  clojure %}
(def h #(knn X-train y-train %))
{% endhighlight %}

Now, **h** can be used to classify the training set.

### Testing

Let's test the KNN model. Classifying the data in the testing set, 
{% highlight  clojure %}
(def preds (map h (rows X-test)))
{% endhighlight %}

Now let's check the accuracy of the model.
{% highlight  clojure %}
(* (/ (apply + (map #(if (= %1 %2) 1 0) y-test preds)) 
      (row-count y-test))
   100.0)
; 99.74
{% endhighlight %}

The model has a 99.74% accuracy on the test set! The accuracy of the model could vary highly depending on the shuffling of the dataset.

### Conclusion

The KNN model has a really good accuracy for the digit classification dataset used here. The problem with KNN is it's inefficiency. It requires  computation involving all samples in the dataset to classify a new sample. The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is a large dataset of handwritten digits - 50,000 training set and 10,000 test set samples. A more complex model such as SVM or MLP(Multi Layer Perceptron) may be used for better efficiency and classification accuracy for such datasets. That's it! more work on Clatern to follow soon. So, keep an eye out :-)
