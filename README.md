**Inner Median**

The inner median function as written doesn't properly construct the intersection set.
There is a builtin `set` function, which implements the correct logic; I use it instead.

To see the function perform, run the `inner_median.py` function (Python 2.7)

-----------------------------


**ML Problem**

   *Methodology and approach*

   I was curious to see how the naive one-hot feature encoding (`simple_one_hot_encoding` in the code) with a logistic regression (`ftrl.py`) would perform. So, I completely ignored the numeric feature columns and broke the product name feature into a list of strings, based on the space character. The reason I did this is because I am familiar with this type of encoding, and I suspected it would perform well. I suspected it would perform well because the product category should be well-correlated with the presence of certain words in the product name (things like "cheese" <--> dairy, for instance). Its true that a Watermelon IPA might be confused as produce, but in this case the FTRL logistic regression model should also be helpful, since presumably the "IPA" word is more tightly correlated with beer than the "Watermelon" word is correlated with produce and the ftrl should get this eventually. Anyway, the individual product AUCs and a manual inspection of the labeled results suggests this approach is decent.

   I checked 2**13 and 2**17 for the dimension of the hash modulus space (`_D` in the code). Smaller `_D` performed a bit worse, and larger `_D` seemed to not be any better. 2**15 is ~32k, and with ~100k training rows, this seemed like a fine number of features. 

   I've written the ftrl.py class/model myself several times from scratch. Note that its an online-model. In fact, it would be interesting to see how the running AUC improves during training, since a prediction is available after each `_train_one` call - note this function _both_ updates the model state and returns a prediction.

   In re: performance for dairy, the AUC on a 25k row test set is about ~0.97. This means the probability of the trained model producing a greater probability-of-dairy for a random non-dairy product than a random dairy product is about 3%. The AUCs for other labels are similar - beer then baking are the lowest, presumably becasue they have the least training data.   
  
   If I had more time on this, I'd work to normalize the words more / generate better features.
   This could include a word2vec vectorization, as well as richer features like count of numbers, count of characters, and lookup of words in a catalog.
   I would also clean up the script more, break things out into classes, try a variety of models and features and tune hyperparameters.
