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
