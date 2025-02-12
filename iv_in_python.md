---
title: "Implementation of independent validation in Python."
shorttitle: "IV in Python"
author:
  - name: "Thede von Oertzen"
    affiliations:
      - name: "Thomas Bayes Institute, Berlin"
  - name: "Hannes Diemerling"
    affiliations:
      - name: "Thomas Bayes Institute, Berlin"
  - name: "Timo von Oertzen"
    email: "timovoe@gmx.de"
    affiliations:
      - name: "Thomas Bayes Institute, Berlin"

abstract: "To statistically test whether two groups are different or whether two models differ, the accuracy of classifiers can be estimated and compared statistically. However, the distribution of common accuracy estimates like cross validation (CV) is unknown and depends on the classifier used, which makes it inapplicable for statistical inference. Independent validation (IV) has been suggested as an alternative, as the distribution of the IV estimate is always a binomial distribution, allowing both for conventional tests on the accuracy and the Bayesian analysis of classifier performance. Although Python is the most frequently used tool for machine learning methods, so far an implementation of IV in Python is missing. This article introduces a new Python package that implements IV. In addition to the core IV algorithm, the package includes the option to (1) plot the accuracy in dependence of the training set size, (2) to estimate the asymptotical classifier accuracy including its posterior distribution, and (3) to query the posterior distribution for confidence intervals. The package also allows to compare different accuracy posterior distributions of each class of the dataset, different classifiers on the same dataset, or of a single classifier on different datasets. An introduction is provided how to use the package for some examples, and a short simulation that shows how the package works in practice. Using this new Python package will allow empirical scientists to easily use IV in their research, for example to analyze group differences or compare group differences between different data sets. "
bibliography: bibliography.bib
keywords: [APA Paper, Machine Learning, Classification, Validation, Independent Validation, Cross Validation, Accuracy, Python, Implementation]
floatsintext: true
numbered-lines: false
draft-date: false
format:
  # Can be jou (journal), man (manuscript), stu (student), or doc (document)
  apaquarto-pdf:
    documentmode: man
    keep-tex: true

---

# Introduction

Research in behavioral and social science freuqently uses group comparisons, as evidenced by a plethora of research papers [e.g., @wee_comparison_2000; @weisberg_gender_2011; @zhao_comparison_2020]. 
A variety of statistical tools are available for univariate comparisons, including parametric tests such as z-tests, t-tests, ANOVAs and Levene's test [@pearson1900x; @student1908probable; @fisher1970statistical; @levene1960robust]. 
In fact, in a systematic review of major Canadian psychology journals, Counsell and Harlow [-@counsell_reporting_2017] found that 40% of analyses used simple mean comparisons like ANOVA.
A variety of methods is available for group comparisons for multidimensional purposes. 
However, all of those are predicated on assumptions regarding the model that the data is assumed to follow [@kim_classifiers_2018]. 
In instances where these assumptions are not well-founded, an alternative approach for comparing groups is necessary.

In this context, classifiers [@boucheron_theory_2005; @bay_detecting_2001] have been proposed as a universal non-parametric alternative for group comparisons [@kim_classifiers_2018].
In the domain of machine learning, classifiers are a prevalent instrument for distinguishing between groups.
A variety of classifiers exist, including for example Decision Trees [@breiman_classification_2017], Random Forests [@ho_random_1995], Support Vector Machines [@cortes_support-vector_1995], or K-nearest neighbor [@cover_nearest_1967].
Groups comparisons by classifier is done by training the classifier on the data set to distinguish the groups. 
If the groups are identical, the classifier will be unable to classify better than by guessing. 
Therefore, if the classifier predicts better than guessing, it can be concluded that there is a group difference.
This procedure makes it necessary to estiamted a classifier's prediction accuracy.
That can be achieved in various ways; typical examples include a train-test split, cross-validation, or bootstrap [@kohavi_study_1995].

Train-test split works by splitting the data into a train set and a test set. 
The training set is used to train the classifier, which then predicts the elements of the test set [@kohavi_study_1995].
The accuracy is then the number of correct classifications over the total number of classifications, which is binomially distributed and hence allows for all kinds of statistical tests. For instance, a binomial test against a null hypothesis of a guessing classifier can be used to determine if the groups differ significantly in a frequentistic analysis.
However, since the amount of data is limited [@sahiner_classifier_2008], splitting the data reduces the size of both the training set, giving less learning opportunities for the classifier, and the test set, adding more standard error in the binomial test. 
In consequence, this leads to a decline in the accuracy of the model [@santafe_dealing_2015] and a less precise estimate of the accuracy. This also limits the statistical power, which is undesirable [@rossi_statistical_2013].
Consequently, the train-test split is an uncommon method; usually, variants of cross validation (CV) is used instead [@kohavi_study_1995; @devroye_distribution-free_1979; @geisser_predictive_1975; @stone_cross-validatory_1974].

In the process of cross-validation, the dataset is split into $k$ subsets called "folds." Then, a procedure analogous to the train-test split is repeated for each fold, in which the classifier is trained on $k-1$ folds and subsequently predicts the items of the remaining fold. In total, every element is tested, so that the testset size equals the size of the full dataset, and the size of the training set is $\frac{k-1}{k}$ times the size of the full dataset. This approach addresses the problem posed by the reduced dataset sizes, thereby eliminating the diminished statistical power inherent to the train-test split method. As the value of $k$ increases, the training set size concomitantly increaes. In the particular case where $k$ is equal to the dataset size, this method is referred to as Leave-One-Out (LOO) [@misc{hastie2009elements], maximizing training- and testset size. 

It is often believed that the accuracy of CV predictions also follows a binomial distribution [@salzberg_comparing_1997].
However, this is incorrect due to the dependency inherent in the repeated training and testing procedure, which results in an increased variance, as conjectured by Bouckaert [-@bouckaert_choosing_2003] and proven by Kim & von Oertzen [-@independent_validation].
This results in alpha inflation of frequentistic tests, as has already been observed by Dietterich [@dietterich_approximate_1998]. It also makes Bayesian estimation of the accuracy impossible. This effect should not be confused with the alpha inflation caused by actual dependency between the samples of the data set themselves [@kohavi_study_1995]. The alpha inflation in this case is caused by the repeated training and testing progress, which creates dependencies of the probabilities that samples are classified correctly, even if the data samples themselves are independent.
The accuracy distribution of CV is an unknown distribution that depends on the choice of classifier.

One approach to conducting a hypothesis test against the null hypothesis of no difference between the groups is to utilize permutation tests [@pesarin_permutation_2010]. Permutation tests repeatedly estimate the cross validation accuracy on the dataset while shuffling the group labels. This allows to approximate the distribution of the accuracy under the assumption of no difference between the groups. Comparing the CV accuracy with the original labels to this distribution allows to generate a significance test against the null hypothesis of no group difference. However, permutation tests are computationally expensive, as they have to repeat the CV procedure multiple times. In addition, they can only be used for frequentistic tests against the null hypothesis of no group differences, not for any other significance test (e.g., whether one classifier outperforms another) or any Bayesian analysis. 

An alternative is to repeatedly sample a training set by bootstrapping and then testing those samples that have not been sampled in the bootstrap [@kohavi_study_1995]. This emulates a Bayesian approach, as it generates a distribution around the point estimate on training set sizes identical to the full data set size. However, the method introduces a dependency between the training test data samples, as some of those are repeated, which compromises the resulting distribution. In general, the distribution is not the posterior distribution, so that it cannot correctly be used for freuqentistic or Bayesian testing. 

An alternative validation called Independent Validation (IV) method was proposed by Kim and von Oertzen. This method uses samples for training exclusively after they have been predicted, ensuring independence between the predictions. In this way, the training set size increase during the validation procedure, and tests on almost all samples of the data set. This consequently leads to a binomial distribution of the accuracy [@kim_classifiers_2018]. With this known distribution of the accuracy, it is possible to perform frequentistic tests against different null hypothesis and, more importantly, Bayesian estimation and testing of the accuracy. 

However, for small data sets IV shows a tendency to underestimated the accuracy, since testing in the first few steps has small training set size. The issue was addressed by Braun, Eckert, and von Oertzen [@braun_independent_2023] by using a estimate of the asymptotic accuracy as the training set size approaches infinity. This utilizes the fact that IV provides correctness of items on differing training set sizes, which allows to estimate the function mapping training set size to the accuracy distribution.

The implementation of this method was done in R [@kim_classifiers_2018; @braun_independent_2023; @r_development_core_team._r_2010]. R is a useful tool for statistical computing, with a wide range of statistical libraries and packages. However, R is not explicitly build for machine learning applications, and consequently this community does not use R frequently. Instead, the by far most common language for machine learning applications is Pyhthon [@kaggle-survey-2022]. To make IV commonly available for machine learning community, an implementation in Python is required. In addition, the R implementation of IV does not estimates the asymptotical accuracy in a Bayesian way, which is one of the most important advantages of IV.   

In this article, an implementation of IV in Phython is introduced. A similar approach has been previously reported by [@diemerling2024implementing], who implemented IV for the evaluation of neural networks in Python. Building on this earlier work, the current study focuses on developing a flexible and comprehensive package designed to bring IV to machine learning classification tasks, focusing on classifiers from the sklearn library. It applies a Bayesian algorithm that can compute the accuracy of the classifier within each class (e.g., the specificity and sensitivity for detecting a depression), on the whole data set, and the Balanced Accuracy (BAC) for a weighted comparison of the class. For each of these accuracies, the posterior distribution can be obtained for the asymptotic accuarcy or for every training set size. A small simulation is provided to explain the usage of the package and to demonstrate the results. The article closes by a discussion of the package for the research field. 


---
# IV Background and Package Implementation

To reiterate, Kim and von Oertzen showe¬d that independence of the tests is guaranteed if every tested sample has not been used for training before. To this end, IV starts by using a small starting set for training, tests a single sample or a batch of samples, and then adds this sample to the training set. The classifier is then trained on this extended training set, and the procedure is repeated until the full dataset has been used. 

In theory, the initial size for the training set can be zero.  In this case, the classifier starts by guessing the class of the first sample before adding them to the training set.. In practice, the training set is typically initialized with a number of initial samples, since no information can be gained from the first classifications, but noise is nevertheless added. Also, some classifiers require a certain minimum amount of samples to operate. For example, a $k$-Nearest-Neighbor classifier needs at least $k$ elements in the training set. 
In the current implementation, the default value for the initial size of the training set is two. For larger data sets, it is recommended to increase this size to 10 or beyond. 

In IV, the classifier is trained on the initial training set and then predicts a single sample or a batch of samples. The number of correct classifications in this batch is stored for each class along with the training set size.. The batch size is set to one by default. Again, with larger data sets it is advised to increase that number since information gain is then minimal, and otherwise the procedure is too time consuming. 

The probability that a sample is classified correctly increases with as the training set size increases. Braun, Eckert and von Oertzen [-@braun_independent_2023] showed that the accuracy within each class can be approximated by
$$
p_n(outcome=1) = a - \frac{\b}{n}
$$ {#eq-likelihood}
where $a$ is the asymptotic accuracy, that is, the theoretical accuracy as \( n \to \infty \), and $b$ is an offset factor that  controls the reduction from the asymptote for finite training sample size \( n \).
Both parameters depend on the classifier. They can be estimated by a Bayesian parameter estimation. We use a uniform prior between 0 and 1 for $a$ and a flat prior on the positive numbers for $b$. The likelihood for a correct classification at any value of $(a,b)$ is then given by Equation (@eq-likelihood) , and one minus that for a failed classification. The likelihood of the complete set of classification results (containing the training sample size $n$ and whether the classification was correct) is then given by the product of the likelihoods for each classification result.   
In the current study, the log likelihood is used instead to avoid numerical problems. With that, a Monte Carlo Markov Chain (MCMC; XXX citation is missing) is applied to sample from the posterior distribution of $a$ and $b$. The MCMC used in this implementation is a Metropolis Hastings algorithm, since it is computationally more efficient and more robust (XXX needs a citation). The MCMC can be started with different burn-in size (default 100), thinning (default XXX), target number of samples (default XXX), and step size for the next candidate choice (default $(XXX,XXX)$). Larger burn-in sizes, number of samples and thinning will improve the sample quality, but at the cost of higher running time. However, since the MCMC operates on the classification results that will not be recomputed, adding more samples is less problematic. 
The MCMC is done separately for the accuracy of each class. For each class, it provides a distribution of $a$ and $b$, where  $a$ represent the asymptotical accuracy in that class, i.e., the accuracy the classifier would reach for an infinite amount of training data. The user of the package can access the distribution of $a$  for each class directly as a list of samples, but they can also request specific information like the MAP accuracy in that class, the posterior mean, the standard deviation, the probability to exceed a certain threshold (that is, the cumulative distribution function), or the probability that the asymptotic accuracy exceeds the asymptotic accuracy of a second distribution (e.g., a different classifier that the user wishes to compare against). In addition to the accuracy for each class, the package provides the accuracy and the BAC for the whole dataset, as well as any other weighting of the class accuracies. These metrics are a weighted um of the individual class accuracies. in the balanced accuracy, all classes are weighted equally, which provides an accuracy index which is independent of the class sizes in the data set. The combination of class accuracies are computed by first multiplying the random variable for each class with its weight, and then convolving the distributions into a distribution for the weighted sum.

The package also combines the posteriors of $a$ and $b$ to provide a distribution of the expected accuracy for any finite sample size.  As for the asymptotic accuracy, the class accuracies for all classes can be accessed as well as the total accuracy, balanced accuracy, or any other weighting of class accuracies. Using the same methods as above, again the MAP, mean, standard deviation, or any cumulative probability of the probability distributions are provided by the package. 

The package also provides an option to generate plots for each of the above distributions, or for multiple of the distributions in a single plot to compare the posterior distributions. 

---
## XXX

### Is there a group difference

Assume somebody, lets call her Annabelle, want to find out if there is a difference in some chemical features between three types of wines, lets say between Barolo, Lugana, Primitivo
. Annabelle uses a Support Vector Machine classifier and IV to get the accuracy of the classifier. To find if there is at all any difference between the groups the question she asks is: How high is the probability that the classifiers asymptotic accuracy is higher or equal to guessing? She uses the global balanced accuracy which would be at 1/3 for a random guessing classifer. As Annabelle is interested in finding if there is a group difference at all, she can use the asymptotic accuracy the SVM would have for infinite data from the groups. The distribution for the global asymptotic accuracy is a result of the IV. Then the hypothesis of no group difference can be rejected with a significance niveau of the integral of 0 to 1/3 of the accuracy distribution. 

NOTE: The dataset is with real data, but we didn't find which types of wine were used.

### Sensitivity and Specificity 
Annabelle does not want to drink white wine, so she uses her classifier as a test if a wine is red (Barolo or Primitivo) or white (Lugana). Now she is interested not in the global accuracy or balanced accuracy but in the sensitivity of her test. The sensitivity is how often a wine that is actually Lugana is classified as Lugana. This is the accuracy for all samples that are actually Lugana and is also an output of the IV process. 

### Comparing classifers
Lets say Annabelle found that there is a strong difference between the three wines. Now she wants to build a classifier for inference but she is unsure what kind of classifer she should use. 
Her final choice will be trained on a dataset of 25 samples, this is the trainset size for which she wants to optimize the accuracy. Annabelle runs iv with multiple classifers like K-Nearest Neighbor, Random Forest and Linear Regression. As an output she generates the distribution for the global accuracy for a trainset size of 25. For the SVM she does not need to rerun the IV but can simply compute the distribution for global accuracy of 25. Now she has 4 different distributions for the accuracy and can see not only which has the highest MAP or mean but also how much the areas overlap giving her the information that most of these classifers are very similarly good. 

### Development
Annabelle considers getting more than 25 samples for her final classifier, but getting more data by chemically analyzing more bottles of wine would be expensive. So before she does that, she looks at how much better the classifier would get if she had some more samples. She is also unsure on how many more samples she should get. So she looks wants to look at the development of how the balanced accuracy increases over increasing training set size. With IV she can get the balanced accuracy for every n from 1 to 100 and look at when it is good enough to satisfy her. In this implementation this proccess is directly accessable as a service. The result is a list of means (XXX or MAP's?) and some range around them that can be specified in differnt ways (XXX: Elaborate on different ways?). There are 100 means, one for each training size ranging from 1 to 100. Plotting these on a graph makes it easy to see how much further any amount of samples would enhance the quality of the classifier.

# Results


## Synthetic data

To estimate the quality of a KNN classifier run on some synthetic data. For the data the features are normally distributed with different means and standard deviations for the different groups. The data consists of three different groups that are not balanced. 

TODO: Add Details about data and classifier. 

![Estimated Accuracy Distribution of a Classifier Trained on 50 samples of the synthetic dataset.](demo/synthetic/acc50_1.png "Figure 1. Accuracy Distribution - 50 Samples")

![Estimated Balanced Accuracy Distribution of a Classifier Trained on 50 samples of the synthetic dataset.](demo/synthetic/bacc50_1.png "Balanced Accuracy Distribution - 50 Samples")

And now the improved variant. Though this needs to be rechecked whether this is realistic.

![Estimated Accuracy Distribution of a Classifier Trained on 100 samples of the synthetic dataset.](demo/synthetic/acc100_1.png "Figure 1. Accuracy Distribution - 100 Samples")

![Estimated Balanced Accuracy Distribution of a Classifier Trained on 100 samples of the synthetic dataset.](demo/synthetic/bacc50_1.png "Balanced Accuracy Distribution - 100 Samples")

If you are really interested in the development these metrics over increasing sample size you can plot this as well.

![Development of Balanced Accuracy Distribution of a Classifier on the synthetic dataset.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

## The Titanic

The tragic sinking of the Titanic in 1912, which claimed the lives of over 1,500 passengers and crew, remains one of the most infamous maritime disasters in history; through the lens of modern data analysis, this historical event provides a unique opportunity to classify and predict survival outcomes based on passenger demographics and circumstances.

Is it just random chance who survived and who didn't? Or is it dependent on some of the features recorded in this dataset?
If the classifier is capable of predicting the outcomes of the passengers significantly better than chance there must be a difference between the group that the classifier detects.

For this it is interesting if the classifier would be capable of finding a difference when trained on infinite amounts of data. Therefore we observe the asymptotical accuracy. 

![Asymnptotical Accuracy Distribution of a Classifier on the Titanic Dataset. TODO: Does not exist yet and needs to be generated.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

Alternatively if we want to use it for inference to predict if somebody on board the titanic is going to drown and we want to know how accurate our own predictions are, we can observe the accuracy for a training set of the size of our actual training set.

![Accuracy Distribution of a Classifier trained on ??? Samples from the Titanic Dataset. TODO: Does not exist yet and needs to be generated.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

Or lets assume we have different classification algorithms and want to find out which is best for our scenario. In that case it would be interesting to look at the development of the accuracy over sample size because some classifiers perform better for small sample sizes but scale less than other classifiers which then outperform the first ones as soon as a certain threshold is reached.

TODO: Can this work in accordance with the assumed formular? Or does the formular imply that all classifiers scale similarly? I fear its the latter. 

![Accuracy Development of a Classifier on the Titanic Dataset. TODO: Does not exist yet and needs to be generated.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")



# Next chapter

1. **Initialization:**  
    A portion of the available data (the amount is user-definable) is used to train a classifier instance. This initial training set must be smaller than the complete dataset.

2. **Prediction & Recording:**  
    - The remaining dataset is split up in batches, smallest batch size must be one.  
    - For each batch, the classifier makes predictions on the unseen samples.
    - For every sample in the batch, the outcome (1 for a correct prediction or 0 for an incorrect one) is recorded along with the current training set size.

3. **IV Iteration:**  
    Step 2 is repeated until all samples are processed.

4. **Posterior Distribution Computation:**
    The probability to classify a new sample correctly for a given training set size \( n \) is modeled as:
    
    $$
    p_n(outcome=1) = \text{asymptote} - \frac{\text{offset\_factor}}{n}
    $$
     
    Here, the *asymptote* represents the classifier’s theoretical accuracy as \( n \to \infty \), and the *offset factor* controls the decline from the asymptote for finite \( n \).  
    Using the Metropolis-Hastings algorithm (an MCMC sampler), we compute the posterior distribution for these model parameters. This posterior is computed separately for each class (label) in the dataset, enabling both class-specific accuracy assessments and aggregated metrics.

5. **Outputs:**
    The IV process produces several types of outputs:
    - **Accuracy Distribution for an Infinite Training Set**  
      For an infinitely large dataset, the classifier’s expected accuracy is represented by the posterior distribution of the *asymptote* parameter, which is provided for each label by IV.
    - **Finite Training Set Accuracy Distribution:**  
    For a finite training set of size 𝑛, IV can compute the corresponding accuracy distribution. For each MCMC sample, the accuracy at size 𝑛 is determined using:
      $$
      p_n = \text{asymptote} - \frac{\text{offset\_factor}}{n},
      $$  
      All MCMC values together form a distribution for the accuracy at this particular n.
    - **Overall Accuracy Distributions:**  
      Instead of assessing accuracy Distributions for a single label, IV also allows evaluation of the classifier’s performance over the full dataset. Two metrics are available:
      - **Balanced Accuracy (bacc):**  
        Using the IV, it is also possible to get a distribution for the balanced accuracy by convolving the distributions of accuracy per label with equal weights. 
      - **Standard Accuracy (acc):**  
        The accuracy is computed by convolving the per-label distributions with weights proportional to the frequency of each label in the dataset.
    - **Development over Trainingset size:**
      Another alternative is to observe the development of the accuracy while the trainingset increases and therefore the classifier improves. To do so, one of the prior functions is run multiple times with n values from 1 to 100. TODO: Elaborate

---

## Demo

The implementation is demonstrated through a collection of example scripts (e.g., `demo.py` and `services.py`) which exercise different facets of the IV process:

1. **Artificial Data Validation:**
   - **Binary Data Demo:**  
     A synthetic dataset is generated with 2D features where the binary target is defined by the condition \( X_0 + X_1 > 1 \). A k-Nearest Neighbors (kNN) classifier is used to run the IV process.  
     *Output:*  
     - Mean balanced accuracy is computed via repeated IV calls.
     - The overall accuracy distribution is assessed.
     - A comparison is made by computing the probability that the balanced accuracy exceeds the overall accuracy.
     
   - **Multi-class Data Demo:**  
     An artificial dataset with three classes is processed. The IV class (`IV` in `iv8.py`) is used to log outcomes, compute the posterior using MCMC, and generate distributions for per-label and aggregated accuracies.

   - **Graphical Outputs:**  
     The demo routines create several plots using Matplotlib:
     - **Histogram Plots:**  
       For example, a histogram of MCMC samples (generated in `demo_mcmc()`) is overlaid with the analytic standard normal probability density function.
     - **Development Curves:**  
       Line plots showcase the evolution of accuracy (mean and interquartile ranges) as a function of training set size. These plots help visualize how performance converges to its asymptotic value.
     - **Distribution Plots:**  
       Density curves for balanced accuracy (e.g., saved as `demo/synthetic/bacc50_1`) and overall accuracy are generated.
     
2. **Real Data Validation (Titanic Demo):**  
   The IV process is also demonstrated on a real-world dataset (the Titanic dataset). After necessary pre-processing and encoding, a Logistic Regression classifier is evaluated:
   - The IV process records predictions over increasing training set sizes.
   - Posterior distributions for the classifier’s asymptotic and finite-sample performance are computed.
   - Plots are generated for balanced and overall accuracy distributions as well as for development trends.

*Figure placeholders in the demo include:*
- **Figure 1:** Histogram of MCMC samples vs. the analytical standard normal.
- **Figure 2:** Density plots for balanced accuracy with varying training sizes.
- **Figure 3:** Development curves showing mean accuracy and quartile bounds over increasing training set sizes.

---

## Testing Procedures

Robustness and correctness of the IV implementation are ensured via a comprehensive battery of unit tests that cover:

1. **MCMC Sampling (`mcmc.py`):**  
   - **1D and Multi-dimensional Sampling:** Tests confirm that the Metropolis-Hastings algorithm produces samples with approximately the correct mean and variance (e.g., resembling a standard normal distribution).
   - **Acceptance Rate:** The proportion of accepted proposals is verified to be within sensible bounds.

2. **Weighted Sum Distribution (`weighted_sum_distribution.py`):**  
   - **Normalization:** Tests ensure that the combination of individual probability distributions yields a final density that integrates to one.
   - **Weighting Options:** Both default (uniform) and user-specified weights are tested for consistency.

3. **IV Core Functionalities (`iv8.py` and `services.py`):**  
   - **IV Record Logging:** The `run_iv` method correctly logs training set sizes and prediction outcomes.
   - **Posterior Computation:** The `compute_posterior` method populates the posterior dictionary with valid MCMC samples for each label.
   - **Distribution Retrieval and Caching:** Methods such as `get_label_accuracy`, `get_bacc_dist`, and `get_acc_dist` are tested to return valid frozen distribution objects (e.g., instances of `rv_histogram`), with caching ensuring repeated calls yield consistent results.
   - **Service Interface:** The high-level `independent_validation` function is tested for correct behavior under different parameter settings (e.g., when returning only the mean or the full distribution).

These tests are implemented using Python's `unittest` framework and verify the statistical properties and consistency of the simulation outcomes.

---

## Example

To illustrate the practical application of IV, consider a fictional scenario in biomedical diagnostics.

### Fictional Scenario: Diagnostic Accuracy of a Biomarker

**Objective:**  
Assess the performance of a classifier that predicts the presence of a disease based on a single biomarker.

**Fictional Dataset:**

- **Features (Biomarker Level):**  
  \[
  X = \{0.5,\; 1.2,\; 1.8,\; 2.1,\; 2.5,\; 3.0,\; 3.3,\; 3.8,\; 4.2,\; 4.8\}
  \]
- **Labels (Disease Status):**  
  \[
  y = \{0,\; 0,\; 0,\; 1,\; 0,\; 1,\; 1,\; 1,\; 1,\; 1\}
  \]
  Here, a label of 1 indicates disease presence and 0 indicates absence.

### IV Application

1. **Initialization:**  
   A simple k-Nearest Neighbors classifier is initialized. The first three samples, for instance, \((0.5, 1.2, 1.8)\) (all with label 0), are used to train the classifier initially.

2. **Incremental Prediction & Training:**
   - **Batch Processing:**  
     The next sample, \(2.1\) (with true label 1), is evaluated. Assume the classifier incorrectly predicts 0.  
     The outcome \( (0 \text{, with training set size } 3) \) is recorded.
   - The sample \(2.1\) is then added to the training set and the classifier is retrained.
   - This process continues for the remaining samples, with each prediction’s correctness recorded along with the training size.

3. **Posterior Computation:**  
   After the entire dataset is processed, the IV records (comprising prediction outcomes and the respective training set sizes) are used to compute the posterior distribution of the model parameters via MCMC.  
   - For example, the posterior for label 1 (disease present) might indicate an asymptotic accuracy mean of 0.87 with a credible interval of [0.82, 0.91].

4. **Aggregated Metrics:**  
   - **Balanced Accuracy (bacc):**  
     By convolving the separate per-label accuracy distributions (with equal weight for each class), a balanced accuracy distribution is obtained. Suppose this yields a mean balanced accuracy of 0.84 at a training set size of 50.
   - **Overall Accuracy (acc):**  
     The overall accuracy is determined via a weighted average of the per-label accuracies. This metric can be used to compare the diagnostic performance against a baseline (e.g., chance-level performance).

### Interpretation

- The **posterior distributions** derived from the IV process offer a transparent picture of uncertainty in classifier performance, facilitating rigorous statistical tests (e.g., testing against a null hypothesis of no predictive power).
- **Development curves** (accuracy as a function of training set size) help determine how quickly the classifier's performance converges and whether additional data could significantly improve accuracy.
- The derived plots (similar to those generated in our demos) serve as both a validation tool and a means for comparative analysis in applied research.

---

By combining a rigorous, incremental validation process with robust posterior inference, the described methods enable the practical application of Independent Validation in varied research settings—from artificial simulations to real data scenarios such as biomedical diagnostics.


# Results


## Synthetic data

To estimate the quality of a KNN classifier run on some synthetic data. For the data the features are normally distributed with different means and standard deviations for the different groups. The data consists of three different groups that are not balanced. 

TODO: Add Details about data and classifier. 

![Estimated Accuracy Distribution of a Classifier Trained on 50 samples of the synthetic dataset.](demo/synthetic/acc50_1.png "Figure 1. Accuracy Distribution - 50 Samples")

![Estimated Balanced Accuracy Distribution of a Classifier Trained on 50 samples of the synthetic dataset.](demo/synthetic/bacc50_1.png "Balanced Accuracy Distribution - 50 Samples")

And now the improved variant. Though this needs to be rechecked whether this is realistic.

![Estimated Accuracy Distribution of a Classifier Trained on 100 samples of the synthetic dataset.](demo/synthetic/acc100_1.png "Figure 1. Accuracy Distribution - 100 Samples")

![Estimated Balanced Accuracy Distribution of a Classifier Trained on 100 samples of the synthetic dataset.](demo/synthetic/bacc50_1.png "Balanced Accuracy Distribution - 100 Samples")

If you are really interested in the development these metrics over increasing sample size you can plot this as well.

![Development of Balanced Accuracy Distribution of a Classifier on the synthetic dataset.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

## The Titanic

The tragic sinking of the Titanic in 1912, which claimed the lives of over 1,500 passengers and crew, remains one of the most infamous maritime disasters in history; through the lens of modern data analysis, this historical event provides a unique opportunity to classify and predict survival outcomes based on passenger demographics and circumstances.

Is it just random chance who survived and who didn't? Or is it dependent on some of the features recorded in this dataset?
If the classifier is capable of predicting the outcomes of the passengers significantly better than chance there must be a difference between the group that the classifier detects.

For this it is interesting if the classifier would be capable of finding a difference when trained on infinite amounts of data. Therefore we observe the asymptotical accuracy. 

![Asymnptotical Accuracy Distribution of a Classifier on the Titanic Dataset. TODO: Does not exist yet and needs to be generated.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

Alternatively if we want to use it for inference to predict if somebody on board the titanic is going to drown and we want to know how accurate our own predictions are, we can observe the accuracy for a training set of the size of our actual training set.

![Accuracy Distribution of a Classifier trained on ??? Samples from the Titanic Dataset. TODO: Does not exist yet and needs to be generated.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

Or lets assume we have different classification algorithms and want to find out which is best for our scenario. In that case it would be interesting to look at the development of the accuracy over sample size because some classifiers perform better for small sample sizes but scale less than other classifiers which then outperform the first ones as soon as a certain threshold is reached.

TODO: Can this work in accordance with the assumed formular? Or does the formular imply that all classifiers scale similarly? I fear its the latter. 

![Accuracy Development of a Classifier on the Titanic Dataset. TODO: Does not exist yet and needs to be generated.](demo/synthetic/baccDevelopment_1.png "Development off Balanced Accurracy")

## Random Guesses

If all groups have the same distribution for the features, the results of IV look like this:


# Discussion

## The last paragraph
Independent validation is the method of choice to optimize for statistically accurate results. 
With this Python implementation it can now easily be used in combination with classifiers from the sklearn library. 



# References


::: {#refs}
:::
