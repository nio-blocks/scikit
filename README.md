TextClassifier
==============

Take input text and classify it.

When the block first is configuring, it loads in some training data to build a classifier.

Properties
----------

* Training Data - Map text to category

Dependencies
------------
scikit-learn
scipy
numpy

Commands
--------
None

Output
------

Each input will output a signal with a new attribute `target` that is the predicted target value, based on the trainging data.


BatchTextClassifier
==============

Properties
----------

Dependencies
------------
scikit-learn
scipy
numpy

Commands
--------
None

Input
------
* Training - Takes in a single list of signals such that each signal has a "data" and a target attribute.
* Classify - Takes in signals that have at least a "data" attribute.

Output
------
* Result - Outputs the signal from the classify input but adds a "label" attribute with the correct classification of the "data".
* Ready - Notifies a signal that can be used to communicate with other services or parts of the same service that training for the block is completed.