TextClassifier
==============

Take input text and classify it.

Properties
----------
None

Dependencies
------------
scikit-learn

Commands
--------
None

Input
-----

Train
    Training data is used to define a fit, mapping text to a category (i.e. target value).

Sample
    Sample data is matched againt the training data to predict a target value.

Output
------

Each `Sample` input will output a signal with a new attribute `target` that is the predicted target value, based on the trainging data.
