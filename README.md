#                                                         Bug_Triage
## Bug Triage using Graph Convolutional Neural Networks


Bug triaging is the task of assigning reported bugs to the potential resources who can resolve the problem associated, which is a pivotal task in product development and maintenance services. Many industries address this issue manually which is laborious, time-consuming and highly expensive. To circumvent these drawbacks, researchers are implementing automated bug triage using machine learning techniques. One can employ text classification models by considering features from description, title, and keywords in the bug report as input and the potential resource as the class label. Bulk of the literature considered Bag of Words as useful feature extractor and developed algorithms to classify those features. However, Bag of Words does not consider the context of the text and semantic similarity which is pivotal to avoid ambiguity in long sequences.

In contrast to Bag of Words, in this work, we present feature extraction from bug reports using graphs by which semantic similarity is restored and the context of the text in nonconsecutive long distant sequences is preserved through the edges and nodes present in the graph. We further use Graph Convolution Neural Networks for the classification of the extracted feature graphs. Our extensive experiments over 3 publicly available datasets attest that the proposed methodology improves the rank-10 F-Measure over 11 state of the art algorithms by 50%.
The Folder Baselines contains of the all the implementation of state of the art algorithms.
 
* Baseline
   * Term Frequency Inverse Document Frequency with MNB
   * Term Frequency Inverse Document Frequency with SVM
   * Term Frequency with MNB
   * Term Frequency with SVM
   * Word2Vec with MNB
   * Word2Vec with SVM
   * DBRNNA 
   * BOW with MNB
   * BOW with SVM
   * BOW with Cosine Similarity
   * BOW with Softmax classifier
* Graph SAGE
   * Graph SAGE with GCN aggregator
   * Graph SAGE with Maxpool aggregator
* Text GCN
   * Text GCN 

Datasets that were used for the experiments are
  * Google Chromium(http://bugtriage.mybluemix.net/#chrome)
  * Mozilla Firefox(http://bugtriage.mybluemix.net/#firefox)
  * Mozilla Core(http://bugtriage.mybluemix.net/#core)
