"""
======================================
Plotting Classification Forest Error Bars
======================================

Explanation explanation explanation [Reference2001]_

.. [Reference2001] Author, A., Author, B. (2001). Title of the paper.
   Journal of important results 1: 1

"""

# Classification example
spam_names = pd.read_csv(("http://archive.ics.uci.edu/ml/machine-learning-"
                          "databases/spambase/spambase.names"), skiprows=30)
spam_names = spam_names['1']
spam_names = [n.split(':')[0] for n in spam_names] + ['spam']
spam_df = pd.read_csv(("http://archive.ics.uci.edu/ml/machine-learning-"
                       "databases/spambase/spambase.data"), names=spam_names)

spam_y = spam_df["spam"].as_matrix()
spam_df["spam"].as_matrix()
spam_X = spam_df.drop(["spam"], axis=1).as_matrix()

spam_X_train, spam_X_test, spam_y_train, spam_y_test = xval.train_test_split(
                                                       spam_X, spam_y,
                                                       test_size=0.2)


spam_RFC = RandomForestClassifier(max_features=5, n_estimators=n_trees)
spam_RFC.fit(spam_X_train, spam_y_train)
spam_inbag = fci.calc_inbag(spam_X_train.shape[0], spam_RFC)

spam_V_IJ_unbiased = fci.random_forest_error(spam_RFC, spam_inbag,
                                             spam_X_train, spam_X_test)


spam_y_hat = spam_RFC.predict_proba(spam_X_test)

idx = np.where(spam_y_test == 1)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75)

idx = np.where(spam_y_test == 0)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75)
