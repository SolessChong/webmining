Webmining
=========
webmining


Framework
=========
We use NLTK, scikit

Methods
=======
We used some basic routines in this project. 

Preprocess
----------
Regex tokenizer
  Cut the tweeters into words.

Feature
-------
TF-IDF feature extractor
  We use basic TF-IDF for this.
  
Additional feature
  We added some words that we thought might be useful, like "rainy", "windy"...

Machine learning
----------------
It's a regression problem so we used several regression methods, including:
  Ridge regression
  SVM regression (RBF kernel, linear)
  Random forest
  
Performance
===========
See `submission/`
