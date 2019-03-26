**Inner Median**

The inner median function as written doesn't properly construct the intersection set.
There is a builtin `set` function, which implements the correct logic; I use it instead.

Code:
```
def inner_median(x,y):
    # returns the median of the intersection of the lists x and y
    inter = sorted(list(set([a for a in x if a in y])))
    _len = len(inter)
    if _len == 0: raise ValueError("There were no common elements in x, y")
    if _len % 2 == 1:
        return inter[(_len - 1)/2] # return the middle element
    return 0.5*(inter[(_len)/2] + inter[(_len)/2 - 1]) # average of the two middle-most elements

if __name__ == "__main__":
    print inner_median([1], [1]), ' should be 1'
    print inner_median([1,1,3,2,1,3], [1,2,3,4]), ' should be 2'
    print inner_median([3,1,2], [1,2,3,4]), ' should be 2'
    print inner_median([1,3,2,1,3], [1,2,3,4]), ' should be 2'
    print 'should be an error next...'
    print inner_median([],[])
```


To see the function perform, run the `inner_median.py` function (Python 2.7)

```
>> python inner_median.py
1  should be 1
2  should be 2
2  should be 2
2  should be 2
should be an error next...
Traceback (most recent call last):
  File "inner_median.py", line 16, in <module>
    print inner_median([],[])
  File "inner_median.py", line 5, in inner_median
    if _len == 0: raise ValueError("There were no common elements in x, y")
ValueError: There were no common elements in x, y
```
-----------------------------


**ML Problem**

   Run the script with python3 (requires numpy) as: `python3 products.py`


   *Methodology and approach*

   I was curious to see how the naive one-hot feature encoding (`simple_one_hot_encoding` in the code) with a logistic regression (`ftrl.py`) would perform. So, I completely ignored the numeric feature columns and broke the product name feature into a list of strings, based on the space character. The reason I did this is because I am familiar with this type of encoding, and because the product category should be well-correlated with the presence of certain words in the product name (things like "cheese" <--> dairy, for instance). Its true that a Watermelon IPA might be confused as produce, but in this case the FTRL logistic regression model should also be helpful, since presumably the "IPA" word is more tightly correlated with beer than the "Watermelon" word is correlated with produce and the ftrl should get this eventually. Anyway, the individual product AUCs and a manual inspection of the labeled results suggests this approach is decent.

   I checked 2^13 and 2^17 for the dimension of the hash modulus space (`_D` in the code). Smaller `_D` performed a bit worse, and larger `_D` seemed to not be any better. 2^15 is ~32k, and with ~100k training rows, this seemed like a fine number of features. I didn't alter any other model hyperparameters - the defaults tend to work very well and I don't have a good enough intuition to confidently change them, anyway 

   I've written the ftrl.py class/model myself several times from scratch. Note that its an online model - it trains one event at a time. In fact, it would be interesting to see how the running AUC improves during training, since a prediction is available after each `_train_one` call - note this function _both_ updates the model state and returns a prediction.

   In re: performance for dairy, the AUC on a 25k row test set is about ~0.97. This means the probability of the trained model producing a greater probability-of-dairy for a random non-dairy product than a random dairy product is about 3%. The AUCs for other labels are similar - beer then baking are the lowest, presumably becasue they have the least training data.   
  
   If I had more time on this, I'd work to normalize the words more / generate better features.
   This could include a word2vec vectorization, as well as richer features like count of numbers, count of characters, and lookup of words in a catalog.
   I would also clean up the script more, break things out into classes, try a variety of models and features and tune hyperparameters.


-------------------------------------

*Production*

The online FTRL-logistic regression model lends itself well to massive training data sets; the model should be able to consume millions of labeled training data rows per hour on a single instance. If the volume of training data was greater, multiple models could be trained in parallel on subsets of the available data, and a periodic model reconcilliation process could be defined to condense most of the learning into a single model. I suspect diminishing returns from additional data for this particular problem would start to occur well before the need for any complex training system, anyway.

To serve the model execution / inference, it would depend on the format of the requests. If the requests came as a stream of unlabeled product data (as in `unlabeled.csv`), and the response was the class probabilities for each class or the predicted label, a simple approach would be to instantiate the trained model in a python HTTP web server, and provide a `label` endpoint, to which requests could be POSTed. This approach scales well, and provides a clean, standard interface to one or more possible clients. The HTTP servers would be stateless and could be horizontally scaled.

If the cost or latency of such an HTTP-endpoint type solution became prohibitive, it would also be feasible to instantiate trained models directly on the client (say an Android app or javascript on a webpage). The ftrl-logistic regression model and one-hot encoding are simple enough to write in something besides python, and its straightforward to test a translation - run both versions on the same data and the outputs should be identical. Because python is a C-based language and Android is generally a Java-like environment, big vs little endian issues may arise when translating the python `hash` function, but its not hard to do correctly.

Finally, a cache of requests and responses might also be helpful, to quickly serve to commonly seen requests.

If the production service returns data tables (more than one row), I'd use a standard database, like Cassandra, redshift, or postgres/mysql. These systems are designed to return multiple rows of data of this type.

To maintain quality over time, I'd monitor the running AUC during training - remember the `_train_one` method both updates the model state and returns a class prediction. If the AUC dropped, I'd check first for any mechanical problems with the system, then if the problem persisted I'd take a look at the data and see what changed. I'd also monitor the mechanical perform of the server / inference system (latency and request volume, basically) to ensure the service was available.

If the category taxonomy changed over time, say the dairy category was broken into cheese, milk and yogurt, it would naively require an additional model for each new category. This would be fine, and the models could be back trained on historical data. If the number of categories became cumbersome, it may be feasible to do a multi-step prediction process, in which parent categories are defined (say "dairy") and in serving/inference we only evaluate the child category ("yogurt", "milk", "cheese") probabilities if the parent category is the most likely or among the most likely parent labels. With a very large number of categories, a different feature encoding / model approach might be better, something like a k-nearest neighbor model would probably be very good. In fact, k-nearest neighbor could probably also handle changing or rare categories very easily.


------------------------------

*Results*
```
['label : AUC']
['dairy : 0.9783967382633719', 'beverages : 0.9706098060561152', 'beer : 0.8894595016508056', 'produce : 0.9784887921275736', 'baking : 0.938535648269121']
```
