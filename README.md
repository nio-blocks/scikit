TextClassifier
==============

Take input text and classify it.

When the block first is configuring, it loads in some training data to build a classifier.

Properties
----------

* **Training Data**: Map text to category

Dependencies
------------
* [scikit-learn](http://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [NumPy](http://www.numpy.org/)

Commands
--------
None

Output
------

Each input will output a signal with a new attribute `target` that is the predicted target value, based on the training data.


BatchTextClassifier
==============

Similar to text classifier but with the added benefit that data can be loaded from elasticsearch or another database and sent in through a list of signals. All signals must be passed through the first list that goes through.

Properties
----------

Dependencies
------------
* [scikit-learn](http://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [NumPy](http://www.numpy.org/)

Commands
--------
None

Input
------
* **Training**: Takes in a single list of signals such that each signal has a `data` and a `target` attribute.
* **Classify**: Takes in signals that have at least a `data` attribute.

Output
------
* **Result**: Outputs the signal from the classify input but adds a `label` attribute with the correct classification of the `data`.
* **Ready**: Outputs a signal that can be used to communicate with other services or parts of the same service that training for the block is completed.
