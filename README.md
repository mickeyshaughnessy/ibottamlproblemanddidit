**Inner Median**

The inner median function as written doesn't properly construct the intersection set.
There is a builtin `set` function, which implements the correct logic; I use it instead.

To see the function perform, run the `inner_median.py` function (Python 2.7)


**ML Problem**
   I chose Option A because I'm curious to see how a simple heuristic feature encoding can perform.

   If I had more time on this, I'd work to normalize the words more / generate better features. This could include a word2vec vectorization, as well as richer features like count of numbers, count of characters, and lookup of words in a catalog.


