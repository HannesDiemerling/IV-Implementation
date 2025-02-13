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

abstract: "To statistically test whether two groups are different or whether two models differ, the accuracy of classifiers can be estimated and compared statistically. However, the distribution of common accuracy estimates like Cross Validation (CV) is unknown and depends on the classifier used, which makes it inapplicable for statistical inference. Independent Validation (IV) has been suggested as an alternative, as the distribution of the IV estimate is always a binomial distribution, allowing both for conventional tests on the accuracy and the Bayesian analysis of classifier performance. Although Python is the most frequently used tool for machine learning methods, so far an implementation of IV in Python is missing. This article introduces a new Python package that implements IV. In addition to the core IV algorithm, the package includes the option to (1) plot the accuracy in dependence of the training set size, (2) to estimate the asymptotical classifier accuracy including its posterior distribution, and (3) to query the posterior distribution for confidence intervals. The package also allows to compare different accuracy posterior distributions of each class of the dataset, different classifiers on the same dataset, or of a single classifier on different datasets. An introduction is provided how to use the package for some examples, and a short simulation that shows how the package works in practice. Using this new Python package will allow empirical scientists to easily use IV in their research, for example to analyze group differences or compare group differences between different data sets. "
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
That can be achieved in various ways; typical examples include a train-test split, Cross Validation (CV), or bootstrap [@kohavi_study_1995].

Train-test split works by splitting the data into a train set and a test set. 
The training set is used to train the classifier, which then predicts the elements of the test set [@kohavi_study_1995].
The accuracy is then the number of correct classifications over the total number of classifications, which is binomially distributed and hence allows for all kinds of statistical tests. For instance, a binomial test against a null hypothesis of a guessing classifier can be used to determine if the groups differ significantly in a frequentistic analysis.
However, since the amount of data is limited [@sahiner_classifier_2008], splitting the data reduces the size of both the training set, giving less learning opportunities for the classifier, and the test set, adding more standard error in the binomial test. 
In consequence, this leads to a decline in the accuracy of the model [@santafe_dealing_2015] and a less precise estimate of the accuracy. This also limits the statistical power, which is undesirable [@rossi_statistical_2013].
Consequently, the train-test split is an uncommon method; usually, variants of CV are used instead [@kohavi_study_1995; @devroye_distribution-free_1979; @geisser_predictive_1975; @stone_cross-validatory_1974].

In the process of CV, the dataset is split into $k$ subsets called "folds." Then, a procedure analogous to the train-test split is repeated for each fold, in which the classifier is trained on $k-1$ folds and subsequently predicts the items of the remaining fold. In total, every element is tested, so that the testset size equals the size of the full dataset, and the size of the training set is $\frac{k-1}{k}$ times the size of the full dataset. This approach addresses the problem posed by the reduced dataset sizes, thereby eliminating the diminished statistical power inherent to the train-test split method. As the value of $k$ increases, the training set size concomitantly increaes. In the particular case where $k$ is equal to the dataset size, this method is referred to as Leave-One-Out (LOO) [@misc{hastie2009elements], maximizing training- and testset size. 

It is often believed that the accuracy of CV predictions also follows a binomial distribution [@salzberg_comparing_1997].
However, this is incorrect due to the dependency inherent in the repeated training and testing procedure, which results in an increased variance, as conjectured by Bouckaert [-@bouckaert_choosing_2003] and proven by Kim & von Oertzen [-@independent_validation].
This results in alpha inflation of frequentistic tests, as has already been observed by Dietterich [@dietterich_approximate_1998]. It also makes Bayesian estimation of the accuracy impossible. This effect should not be confused with the alpha inflation caused by actual dependency between the samples of the data set themselves [@kohavi_study_1995]. The alpha inflation in this case is caused by the repeated training and testing progress, which creates dependencies of the probabilities that samples are classified correctly, even if the data samples themselves are independent.
The accuracy distribution of CV is an unknown distribution that depends on the choice of classifier.

One approach to conducting a hypothesis test against the null hypothesis of no difference between the groups is to utilize permutation tests [@pesarin_permutation_2010]. Permutation tests repeatedly estimate the CV accuracy on the dataset while shuffling the group labels. This allows to approximate the distribution of the accuracy under the assumption of no difference between the groups. Comparing the CV accuracy with the original labels to this distribution allows to generate a significance test against the null hypothesis of no group difference. However, permutation tests are computationally expensive, as they have to repeat the CV procedure multiple times. In addition, they can only be used for frequentistic tests against the null hypothesis of no group differences, not for any other significance test (e.g., whether one classifier outperforms another) or any Bayesian analysis. 

An alternative is to repeatedly sample a training set by bootstrapping and then testing those samples that have not been sampled in the bootstrap [@kohavi_study_1995]. This emulates a Bayesian approach, as it generates a distribution around the point estimate on training set sizes identical to the full data set size. However, the method introduces a dependency between the training test data samples, as some of those are repeated, which compromises the resulting distribution. In general, the distribution is not the posterior distribution, so that it cannot correctly be used for freuqentistic or Bayesian testing. 

An alternative validation called Independent Validation (IV) method was proposed by Kim and von Oertzen. This method uses samples for training exclusively after they have been predicted, ensuring independence between the predictions. In this way, the training set size increase during the validation procedure, and tests on almost all samples of the data set. This consequently leads to a binomial distribution of the accuracy [@kim_classifiers_2018]. With this known distribution of the accuracy, it is possible to perform frequentistic tests against different null hypothesis and, more importantly, Bayesian estimation and testing of the accuracy. 

However, for small data sets IV shows a tendency to underestimated the accuracy, since testing in the first few steps has small training set size. The issue was addressed by Braun, Eckert, and von Oertzen [@braun_independent_2023] by using a estimate of the asymptotic accuracy as the training set size approaches infinity. This utilizes the fact that IV provides correctness of items on differing training set sizes, which allows to estimate the function mapping training set size to the accuracy distribution.

The implementation of this method was done in R [@kim_classifiers_2018; @braun_independent_2023; @r_development_core_team._r_2010]. R is a useful tool for statistical computing, with a wide range of statistical libraries and packages. However, R is not explicitly build for machine learning applications, and consequently this community does not use R frequently. Instead, the by far most common language for machine learning applications is Pyhthon [@kaggle-survey-2022]. To make IV commonly available for machine learning community, an implementation in Python is required. In addition, the R implementation of IV does not estimates the asymptotical accuracy in a Bayesian way, which is one of the most important advantages of IV.   

In this article, an implementation of IV in Phython is introduced. It applies a Bayesian algorithm that can compute the accuracy of the classifier within each class (e.g., the specificity and sensitivity for detecting a depression), on the whole data set, and the Balanced Accuracy (BAC) for a weighted comparison of the class. For each of these accuracies, the posterior distribution can be obtained for the asymptotic accuarcy or for every training set size. A small simulation is provided to explain the usage of the package and to demonstrate the results. The article closes by a discussion of the package for the research field. 

___

# Background and Implementation

To reiterate, Kim and von Oertzen showed that independence of the results is guaranteed if every tested point hasn't been used for training before. To achieve this IV starts by using a small starting set for training, tests a batch of points and records the result and then adds this batch to the trainset. The classifier is retrained on the new trainset and this process is repeated until the full dataset is used. 

In IV, the classifier is trained on the initial training set and then predicts a single sample or a batch of samples. The number of correct classifications in this batch is stored for each class along with the training set size. The batch size is set to one by default. Again, with larger data sets it is advised to increase that number since information gain is then minimal, and otherwise the procedure is too time consuming. 

The probability that a sample is classified correctly increases with as the training set size increases. Braun, Eckert and von Oertzen [-@braun_independent_2023] showed that the accuracy within each class can be approximated by:

$$
P(\text{outcome} = 1) = a - \frac{b}{n}
$$ {#eq-likelihood}

Where $a$ is the asymptotic accuracy, that is, the theoretical accuracy as \( n \to \infty \), and $b$ is an offset factor that  controls the reduction from the asymptote for finite training sample size \( n \).
Both parameters depend on the classifier. They can be estimated by a Bayesian parameter estimation. 
We use a uniform prior between 0 and 1 for $a$ and a flat prior on the positive numbers for $b$. 
The likelihood for a correct classification at any value of $(a,b)$ is then given by Equation (@eq-likelihood), and one minus that for a failed classification. 
The likelihood of the complete set of classification results (containing the training sample size $n$ and whether the classification was correct) is then given by the product of the likelihoods for each classification result.   

In the current study, the log likelihood is used to avoid numerical problems. With that, a Monte Carlo Markov Chain [@metropolis1953equation] is applied to sample from the posterior distribution of $a$ and $b$. The MCMC used in this implementation is a Metropolis Hastings algorithm, since it is computationally more efficient and more robust [@hastings1970monte]. 
The MCMC can be started with different burn-in size (default 100), thinning (default XXX), target number of samples (default XXX), and step size for the next candidate choice (default $XXX$ in both directions). Larger burn-in sizes, number of samples and thinning will improve the sample quality, but at the cost of higher running time. However, since the MCMC operates on the classification results that will not be recomputed, the computational costs are limited even with high number of total samples. 

The MCMC is done separately for the accuracy of each class. 
For each class, it provides a distribution of $a$ and $b$, where  $a$ represent the asymptotical accuracy in that class, i.e., the accuracy the classifier would reach for an infinite amount of training data. 
The user of the package can access the distribution of $a$  for each class directly as a list of samples, but they can also request specific information like the MAP accuracy in that class, the posterior mean, the standard deviation, the probability to exceed a certain threshold (that is, the cumulative distribution function), or the probability that the asymptotic accuracy exceeds the asymptotic accuracy of a second distribution (e.g., a different classifier that the user wishes to compare against). 
In addition to the accuracy for each class, the package provides the accuracy and the BAC for the whole dataset, as well as any other weighting of the class accuracies. 
These metrics are a weighted um of the individual class accuracies. in the balanced accuracy, all classes are weighted equally, which provides an accuracy index which is independent of the class sizes in the data set. 
The combination of class accuracies are computed by first multiplying the random variable for each class with its weight, and then convolving the distributions into a distribution for the weighted sum.

The package also combines the posteriors of $a$ and $b$ to provide a distribution of the expected accuracy for any finite sample size. 
As for the asymptotic accuracy, the class accuracies for all classes can be accessed as well as the total accuracy, balanced accuracy, or any other weighting of class accuracies. 
Using the same methods as above, again the MAP, mean, standard deviation, or any cumulative probability of the probability distributions are provided by the package. 

## predicting samples

The startsize for the trainset can theoretically be zero, in this case the classifier starts by guessing the first sample. 
Then it would train on the very small trainset of a single sample and logically choose the same label for the second sample. 
In Practice it is not useful to start with a startset size of 0 for multiple reasons. 
The information gained on the first sample is zero and on the second sample only an information about the amounts of the different labels. 
This is because the chance to classify the first sample correctly is 1/amount of labels and the probability to predict the second is the probability for both samples to have the same label. 
Therefore the information whether the first two samples are predicted correctly or not is not dependent on the realtion between features and labels and therefore not that interesting. 

Another reason why starting with a trainset size of zero is the practical problem that some classifiers need a certain minimum amount of samples to work. 
An example is the K-nearest Neighbor classifier that predicts a new sample by taking a majority voting of the k nearest neighbors. 
Logically a trainset would have to have at least k samples (and if it had exactly k samples a new sample would have exactly these k samples as nearest neighbors, so the classification would be independent of the features of the new sample). 
<!-- Also the formular doesn't work for n=0 wegen zero division. -->
In this implementation the default value for the startsize of the trainset is 2, this is also a good value for most cases, if data is plenty, increasing the start trainset size to 10 makes sense, as early values are often less stable. 

As a first step in IV the classifier is trained on the starting trainset being a subset of the full datset. Then this classifier predicts a batch of samples. 
For each sample in that batch it is recorded whether that sample was classified correctly and on which trainset size the classifier was currently trained (which would be the same value for all samples of the same batch). 
Then this batch is added to the trainset and the classifier is retrained. 
Therfore the classifier gets retrained batch size / dataset size - trainset start size times. 
For a small batch size and a big dataset this can be computationally intensive. 
For big datasets a batch size of about a fifth of the dataset size works fine. 
For small datasets the default and minimum value of 1 leads to the best estimation of the asymptotical accuracy, though the difference is minimal. 
A batch size higher than a third of the dataset is not recommended.

## computing posterior

The accuracy with which a sample is classified correctly increases with the increasing training set. Braun, Eckert and von Oertzen [@braun_independent_2023] found that the accuracy in dependence of the trainset size can be modeled witht his equation (@eq-asymptote):

$$
p_n(outcome=1) = \text{asymptote} - \frac{\text{offset\_factor}}{n}
$$ {#eq-asymptote}

Where the *asymptote* represents the classifier’s theoretical accuracy as \( n \to \infty \), and the *offset factor* controls the decline from the asymptote for finite \( n \).
Both are parameters dependend on the dataset and classifier and can be estimated with bayesian parameter estimation. The Prior for the bayesian estimation is uniform between 0 and 1 for the asymptote and uniform positive for the offset factor. Then the likelihood for any possible combination of asymptote and offset_factor can be computed by using the above formular for every prediction recorded in the prior step. For a successfully predicted sample the likelihood for asymptote and offset_factor is defined by the equation (@eq-likelihood2):

$$
\text{Likelihood} = p_n(outcome=1) = \text{asymptote} - \frac{\text{offset\_factor}}{n}
$$ {#eq-likelihood2}

and for an unsuccessfully predicted sample it is 1 minus that. Therefore the full likelihood formular (@eq-likelihood3) comes out to be:

$$
\text{Likelihood} = outcome*p_n(outcome=1) + (1 - outcome) * (1 - p_n(outcome=1))
$$ {#eq-likelihood3}

Multiplying this for every recorded prediction gives the likelihood for the pair of asymptote and likelihood. 
Practically in the implementation instead the logarithm for each is taken and then these are summed up to be the logarithmic likelihood. This improves numerical stability and speeds up the process.

Prior times likelihood gives a distribution that looks like the posterior distribution but has an area that is not equal to one. To get the actual posterior distribution one could compute the integral of this distribution and divide by it. This would normalize the area to one. An alternative option is to use a marcov chain monte carlo (MCMC) which makes it possible to sample from the actual posterior distribution based on the prior and likelihood. The MCMC implemented in this paper uses the metropolis hastings algorithm. This has the advantage of being computationally more efficient and more robust. Also having the posterior distribution as a set of samples allows for manipulations that would require the extra step of sampling if the numerical variant was used instead of MCMC. MCMC takes some parameters that can be specified when called.

## Output

This whole process of computing the posterior is done seperately for each class. For each class the distribution of the asymptote parameter can be returned. This represents the asymptotical accuracy that the classfier would reach for an infinite amount of data. In addition to the accuracy for each class also the accuracy for the whole dataset can be generated. In this case a distinction needs to be made between the accuracy and the balanced accuracy. For the normal accuracy the classes are (usually implicitly) weighted by their frequency in the dataset. The balanced accuracy weights the classes all equally independent from their frequency. Either way all weights add up to one. Both, balanced and normal accuracy are achieved by first multiplying the random variable for each class with its weight. Then the new distributions are convolved giving the final distribution for balanced or normal accuracy.

Alternatively to using the asymptotical accuracy, it is also possible to obtain a posterior distribution for any trainingset size. For a trainingset size n this is acheived by going through the samples of asymptote and offset factor of the posterior distribution and computing p_n for each. This gives then many samples for p_n making together the posterior distribution. Doing this for each label gives multiple accuracy distributions that can be combined the same way as in the asymptotical case to achieve normal or balanced global accuracy. 

# Illustrative Example

<!-- Text von Hannes und Thede -->

# Discussion
IV is a method for evaluating classifier accuracy based on known data likelihood. Existing implementations are limited to R and are not widely available as packages (e.g., on CRAN; @XXX). This article introduces a Python package, integrated with the scikit-learn library, which is currently the most widely used programming language in machine learning. The package employs a Metropolis-Hastings MCMC algorithm to estimate the posterior probability of classifier accuracy within each class and aggregates these estimates into multiple relevant metrics and distributions. An empirical example using the XXX dataset demonstrates the package’s functionality and output. The package code and examples presented here are available at XXX. 

To the best of the authors’ knowledge, no other implementation exists that computes the Bayesian posterior of a classifier’s asymptotic within-class accuracy. Existing methods that approximate at least frequentist baseline distributions for a purely guessing classifier have significant drawbacks: some are computationally expensive (e.g., permutation tests), other reduce available data (e.g., training-testset-separation), and some do not work correctly to begin with (e.g., CV using a binomial null distribution). Moreover, none of these methods allow a comparisons beyond anything but chance performance or support a Bayesian approach. 

The Bayesian distributions of the asymptotic within-class accuracy serve as the foundation for computing the posterior of key indices describing classifier performance. They also enable Bayesian inference on empirical hypotheses, such as whether two groups differ or one classifier outperforms another. 

These indices include the classifier’s specificity and sensitive (i.e., within-class accuracy for two classes), the asymptotic total classifier accuracy, the asymptotic balanced accuracy, and any other weighted sum of class accuracies. For all of these performance measures, the research can also use the package to compute the expectation for a final training set size (again represented as a Bayesian posterior). Each posterior distribution can be accessed either through the full set of samples or via summary descriptors, including the Maximum A-Posteriori (MAP), mean, standard deviation, credible intervals and percentile intervals of any size, and specific probabilities for defined thresholds. For example, researchers can determine the probability that the classifier performs better than chance or better than a chosen performance threshold. 

Using these capabilities, the package enables researchers in the behavioral sciences to address various common research questions. For example, if a clinical research wants to use a classifier – such as a threshold on a sum score - as screening tool for a psychological disorder, the package provides Bayesian estimates of the sensitivity and specificity or the test. This allows to test hypotheses about these, such as whether the sensitivity exceeds 90 percent. The test can be done by accessing the posterior probability for this event from the within-class accuracy of the classifier among patients with the disorder. 

Similarly, a sociologist may want to examine whether individuals with low socio-economic status (SES) perform differently from those with high SES on a diverse set of cognitive tasks that cannot be easily combined into a single score. By training a classifier on the two groups, they can use the Bayesian posterior of balanced accuracy (BAC) to determine the probability that the classifier performs better than chance, providing evidence of a meaningful group difference. 

Analogously, a neurologist might investigate whether brain activities differs between two experimental conditions. Again, by training a classifier on imaging data, they can use the package to obtain a Bayesian posterior probability that the classification performance exceeds guessing. 

Finally, an expert in machine learning may be interested in comparing a newly developed classifier to an existing one on a benchmark dataset. The package allows them to obtain the posterior distributions of both classifiers and visualize the posterior probability that one outperforms the other in terms of accuracy, BAC, or any other weighted accuracy measure. 	

A limitation of IV is that the assumed functional form mapping training set size to classifier performance within a group is a good approximation for training set sizes around ten samples but less accurate on very low sample sizes. Additionally, testing is not entirely independent when sample sizes are very low, as missclassified items added to the training set can bias subsequent classifications. Overly small starting sets may lead to Haywood cases in the Bayesian estimation. 

However, for moderate samples sizes, this issue largely disappears, as the initial training set size can be made sufficiently large. For small data sets, though, this approach results in some data loss, albeit less than in methods like train-test-splitting. To mitigate this effect for small N, the Python package introduced constraints the minimal expected success rate probability to the guessing probability. While this improves the approximation, it does not fully resolve the issue. Therefore, whenever possible, users are advised to use sufficiently large initial test sets. 

The batch size used to add items to the training set in IV involves a tradeoff. From an estimation perspective, a batch size of one is optimal. However, for large data sets with thousands of samples, this approach is computationally expensive. However, for such large data sets using larger batch sizes - in the order of magnitude of one tenth of the total data set - results only in negligible loss of information, while significantly accelerating processing. 

A promising future direction in this line of research is to mathematically investigate whether an improved model for expected accuracy at very low training set sizes can be developed, and possibly integrated with the existing model, which performs well for moderate sample sizes. 

Another open question is to what extent IV can be applied to evaluate machine learning regression results or in the context of Large Language Models \cite{XXX}. Similar to categorical classification, CV is commonly employed to assess regression performance, with various suggestions for loss functions \cite{XXX}. Accurately quantifying uncertainty in loss may require an independent validation technique analogous to the one proposed here for categorical classification. 

Independent validation is the preferred method for optimizing statistically accurate accuracy estimation. With the Python implementation introduced here, it can now be seamlessly integrated with classifiers in empirical research. This will help researchers in mostly all empirical research fields to improve their research and strengthen progress of science. 

# References

::: {#refs}
:::

