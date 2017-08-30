BatchTextClassifier
===================
Take input text and classify it.  Data can be loaded from elasticsearch or another database and sent in through a list of signals. All signals must be passed through the first list that goes through.

Properties
----------
None

Inputs
------
- **classify**: Takes in signals that have at least a `data` attribute.
- **training**: Takes in a single list of signals such that each signal has a `data` and a `target` attribute.

Outputs
-------
- **Ready**: Outputs a signal that can be used to communicate with other services or parts of the same service that training for the block is completed.
- **Result**: Outputs the signal from the classify input but adds a `label` attribute with the correct classification of the `data`.

Commands
--------
None

Dependencies
------------
* [scikit-learn](http://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [NumPy](http://www.numpy.org/)

TextClassifier
==============
Take input text and classify it.  When the block first is configuring, it loads in some `training_data` to build a classifier.

Properties
----------
- **training_set**: Training text to map to categories.

Inputs
------
- **default**: Any list of signals.

Outputs
-------
- **default**: Each input will output a signal with a new attribute `target` that is the predicted target value, based on the `training_set`.

Commands
--------
None

Dependencies
------------
* [scikit-learn](http://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [NumPy](http://www.numpy.org/)
