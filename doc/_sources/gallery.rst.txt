Gallery
=======

Model Abstraction
#################
.. image:: images/ml_workflow.png

InMemory Model
**************
1. https://github.com/datascienceinc/Skater/blob/master/examples/ensemble_model.ipynb
2. https://github.com/datascienceinc/Skater/blob/master/examples/image_interpretation_example.ipynb
3. https://github.com/datascienceinc/Skater/blob/master/examples/credit_analysis/Credit_Analysis.ipynb


Deployed Model
**************

1. `python` deployed model: https://github.com/datascienceinc/Skater/tree/master/examples/python-deployed-model
2. `r` deployed model: https://github.com/datascienceinc/Skater/tree/master/examples/r/deployed_model
3. pre-trained/canned model: https://github.com/datascienceinc/Skater/tree/master/examples/third_party_model


Interpretation Examples
#######################

Global Interpretation
*********************
.. image:: images/pdp.png
   :width: 500
1. Model Agnostic Partial Dependence Plot(PDP)

    + https://github.com/datascienceinc/Skater/blob/master/examples/ensemble_model.ipynb
    + https://github.com/datascienceinc/Skater/blob/master/examples/sklearn-classifiers.ipynb
    + https://github.com/datascienceinc/Skater/blob/master/examples/sklearn_regression_models.ipynb

.. image:: images/feature_importance.png
   :width: 500
2. Model Agnostic Feature Importance

    + https://github.com/datascienceinc/Skater/blob/master/examples/ensemble_model.ipynb
    + https://github.com/datascienceinc/Skater/blob/master/examples/sklearn-classifiers.ipynb
    + https://github.com/datascienceinc/Skater/blob/master/examples/sklearn_regression_models.ipynb

.. Adding an extra blank line for readability ease
|

Local Interpretation
********************
.. image:: images/lime.png
   :width: 800
1. Local Interpretable Model Explanations(LIME)

   + https://github.com/datascienceinc/Skater/blob/master/examples/image_interpretation_example.ipynb
   + https://github.com/datascienceinc/Skater/blob/master/examples/NLP.ipynb
   + https://github.com/datascienceinc/Skater/blob/master/examples/third_party_model/algorithmia_indico.ipynb

.. Adding an extra line
|

2. DeepInterpreter for interpreting DNNs
 * epsilon-Layer-wise Relevance Propagation(e-LRP): only for image currently
 * Integrated Gradient(IG): image and text

.. image:: images/example_lrp_ig.png
       :width: 49%
       :alt: Interpreting pre-trained Inception-V3 model
.. image:: images/interpreting_inverted_panda_ig_elrp.png
       :width: 49%
       :alt: some more examples on image interpretability

Image Interpretability
   * Image Classification:
      + https://github.com/datascienceinc/Skater/blob/master/examples/image_interpretability/image_interpretation_example_cats_dogs.ipynb
      + https://github.com/datascienceinc/Skater/blob/master/examples/image_interpretability/imagenet_adv_inceptionv3_tensorflow.ipynb
      + https://github.com/datascienceinc/Skater/blob/master/examples/image_interpretability/mnist_cnn_keras.ipynb
      + https://github.com/datascienceinc/Skater/blob/master/examples/image_interpretability/mnist_mlp_tensorflow.ipynb


.. image:: images/text_ig.png
       :width: 500
Text Interpretability with Integrated Gradient
   * Sentiment Analysis:
      + https://github.com/datascienceinc/Skater/blob/master/examples/text_interpretability/LSTM_sentiment_imdb.ipynb
      + https://github.com/datascienceinc/Skater/blob/master/examples/text_interpretability/cnn_sentiment_imdb.ipynb

.. Adding an extra blank line for readability ease
|

Global And Local Interpretation
*******************************
.. image:: images/sbrl.png
       :width: 600
1. Rule Based Models(Transparent Design)

   + https://github.com/datascienceinc/Skater/blob/master/examples/rule_list_notebooks/rule_lists_continuous_features.ipynb
   + https://github.com/datascienceinc/Skater/blob/master/examples/rule_list_notebooks/rule_lists_titanic_dataset.ipynb

