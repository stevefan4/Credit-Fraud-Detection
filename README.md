# Credit-Fraud-Detection
Anonymized credit card transactions labeled as fraudulent or genuine

Dataset: 
- 284,807 Transactions made by credit cards in September 2013 by European cardholders.
- As we might expect, we have a dramatically unbalanced dataset where only .172% of recorded transactions involve fraud.
- Due to the sensitive nature of the original data features issues, PCA transformation was used to return only a numerical representation of each datapoint
- As not every fraudulent transaction is caught and reported, misclassified data can be another major issue, 
- Given the dataset contains the transaction amount feature, we can visualize this as a cost sensitive learning problem where our goal is to minimize the total transaction amount of incorrectly classified data points. 
- Given the nature of the problem, a false-positive (fraud detected no occurance) is significantly less detrimental than a false-negative (fraud occured but not detected) 
- Therefore, instead of purely evaluating our model from the returned confusion matrix, we will look at the cost matrix where the costs for a false positive is the monetary cost of follow-up with the customer to the company and the cost of a false negative is the cost of the insurance claim. (This isn't exactly true because there are reputational costs involved. Ex: It would annoy our customers if we assumed 99% of transactions were fraud and chose to follow up after every purchase) 
- A good starting point for imbalanced classification tasks is to assign costs based on the inverse class distribution. In this case, the cost of a false-negative will be 1/.00172. Therefore, the cost of a false-positive is 1 and the cost of a false-negative is 581.4 (This assumes that the class distribution observed in the training data is representative of the broader underlying distribution). 





