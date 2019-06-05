---
layout: post
title: "Multiclass Classification using Clatern"
category: clatern 
---

[Clatern](https://github.com/rinuboney/clatern) is a machine learning library for Clojure, in the works. This is a short tutorial on performing multiclass classification using Clatern.
  
#### Importing the libraries

The libraries required for the tutorial are core.matrix, Incanter and Clatern. Importing them, 

{% highlight  clojure %}
(require '[clojure.core.matrix :as m])
(use '(incanter core datasets charts))
(use '(clatern logistic-regression knn))
{% endhighlight %}

*NOTE: This tutorial requires Incanter 2.0 (aka Incanter 1.9.0). This is because both Incanter 2.0 and Clatern are integrated with core.matrix.*
  
### Dataset
This tutorial uses the popular Iris flower dataset. The dataset is available here: https://archive.ics.uci.edu/ml/datasets/Iris. For this tutorial, we'll use Incanter to load the dataset.

{% highlight  clojure %}
(def iris (get-dataset :iris))
(view iris)
{% endhighlight %}

![iris](https://camo.githubusercontent.com/6e7e613199cfb729b52792639c7b24ace67585e8/687474703a2f2f696e63616e7465722e6f72672f696d616765732f6578616d706c65732f697269735f646174612e6a7067)

Now converting the dataset into a matrix, where non-numeric columns are converted to either numeric codes or dummy variables, using the to-matrix function.

{% highlight  clojure %}
(def iris-mat (to-matrix iris))
(view iris-mat)
{% endhighlight %}

![iris-mat](https://camo.githubusercontent.com/1fa4972cc40ded5570931f7f567d1c595f010a47/687474703a2f2f696e63616e7465722e6f72672f696d616765732f6578616d706c65732f697269735f6d61742e6a7067)

Now let's split the dataset into a training set and a test set,

{% highlight  clojure %}
(def iris' (m/order iris-mat 0 (shuffle (range 150))))
(def train-mat (take 120 iris'))
(def test-mat (drop 120 iris'))
{% endhighlight %}

Splitting the training and test set into features and labels,

{% highlight  clojure %}
(def X-train (m/select train-mat :all [0 1 2 3]))
(def y-train (m/get-column train-mat 4))
(def X-test (m/select test-mat :all [0 1 2 3]))
(def y-test (m/get-column test-mat 4))
{% endhighlight %}  

### Logistic Regression

Here comes the interesting part - training a classifier using the data. First, let's try the logistic regression model. Gradient descent is a learning algorithm for the logistic regression model. The syntax of gradient descent is,
 
{% highlight  clojure %}
(gradient-descent X y :alpha alpha :lambda lambda :num-iters num-iters)
{% endhighlight %}
where,  
*X* is input data,  
*y* is target data,  
*alpha* is the learning rate,  
*lambda* is the regularization parameter, and  
*num-iters* is the number of iterations.

alpha(default = 0.1), lambda(default = 1) and num-iters(default = 100) are optional.

{% highlight  clojure %}
(def lr-h (gradient-descent X-train y-train :lambda 1e-4 :num-iters 200))
{% endhighlight %}

That's it. Here, gradient-descent is a function in the clojure.logistic-regression namespace. It trains on the provided data and returns a hypothesis in the logistic regression model. Now, **lr-h** is a function that can classify an input vector. 
  
  
### K Nearest Neighbors

Next, let's try the k nearest neighbors model. There is actually no training phase for this model. It can be directly used. The syntax for knn is,

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
(def knn-h #(knn X-train y-train %))
{% endhighlight %}

Similar to the logistic regression hypothesis, now **knn-h** can be used to classify an input vector. 
  
  
### Classification

Both **lr-h** and **knn-h** are functions that take input feature vectors and classify them. So to classify a whole dataset, the function is mapped to all rows of the dataset.

{% highlight  clojure %}
(def lr-preds (map lr-h (m/rows X-test)))
(def knn-preds (map lr-h (m/rows X-test)))
{% endhighlight %}

Now **lr-preds** and **knn-preds** contains the classifications made by logistic regression and knn on the orignal dataset, respectively.
  
  
### Conclusion

So which model performs better here? Let's write a function to assess the classification accuracy

{% highlight  clojure %}
(defn accuracy [h X y]
  (* (/ (apply + (map #(if (= %1 %2) 1 0) y (map h (m/rows X)))) 
        (m/row-count y))
     100.0))
{% endhighlight %}

Now let's evaluate both the classifiers:

{% highlight  clojure %}
(accuracy lr-h X-test y-test)
; 92.53
(accuracy knn-h X-test y-test)
; 96.13
{% endhighlight %}

The accuracy of the models could vary highly depending on the shuffling of the dataset. These are values I averaged over 100 runs. Both models perform well on this datatset. So, that's it for multiclass classification using Clatern. More work on Clatern to follow soon. So, keep an eye out :-)
