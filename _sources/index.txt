
Confidence Intervals for Scikit Learn Random Forests
=====================================================

Random forest algorithms are useful for both classification and regression
problems. This package adds to scikit-learn the ability to calculate confidence
intervals of the predictions generated from scikit-learn
:class:`sklearn.ensemble.RandomForestRegressor` and :class:`sklearn.ensemble.RandomForestClassifier` objects.

This is an implementation of an algorithm developed by Wager et al. [Wager2014]_
and previously implemented in R (`here <https://github.com/swager/randomForestCI>`_).

To examine and download the source code, visit our `github repo <https://github.com/uwescience/sklearn-forest-ci#readme>`_.

.. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
       Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
       of Machine Learning Research vol. 15, pp. 1625-1651, 2014.

    .. toctree::
       :maxdepth: 2

       installation_guide
       auto_examples/index
       reference/index


.. figure:: _static/eScience_Logo_HR.png
   :align: center
   :figclass: align-center
   :target: http://escience.washington.edu

   Acknowledgements: this work was supported by a grant from the
   `Gordon & Betty Moore Foundation <https://www.moore.org/>`_,  and from the
   `Alfred P. Sloan Foundation <http://www.sloan.org/>`_ to the
   `University of Washington eScience Institute <http://escience.washington.edu/>`_ , and through a grant from the `Bill & Melinda Gates Foundation <http://www.gatesfoundation.org/>`_.
