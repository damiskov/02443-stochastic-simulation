## Summary
----
- Theoretical tests/properties
	- Tests of *global behavior* (entire cycles)
	- Mathematical theorems
	- Typically investigates multidimensional uniformity
- Tests for uniformity
- Tests for independence
- Relevant statistical properties: [[Random Number Generation]]

## Testing
---
### Tests for distribution type:
- Visual plots
- $\chi^2$ test
- *Kolmogorov Smirnov* test
### Tests for independence
- Plots
- Run test up/down
- Run test length of runs (?)
- Correlation coefficients
### Significance tests
- Assume model - *Hypothesis*
- Identify certain characterizing random variable - *Test statistic*
- Reject hypothesis if the test statistic is an abnormal observation under the hypothesis

## Multinomial distribution
-----
- $n$ items
- $k$ classes
- Each item falls into class $j$ with probability $p_j$
- $X_j$ is the (random) number of items in class $j$
- Write $X = (X_1, ..., X_k) \sim Mul(n, p_1, ..., p_k)$

$\therefore X_j \sim Bin(n, p_j), \ \ E(X_j) = n \cdot p_j, \ \ Var(X_j) = n \cdot p_j (1-p_j)$ 

and $r_{X_j} =\frac{X_j - n p_j}{\sqrt{n p_j (1-p_j)}}$  (standardized residual)

and $E(r_{X_j}) = 0 \ \ \ Var(r_{X_j})= 1$

Thus $\lim_{n \to \infty} r_{X_j} \sim N(0,1)$
## Test statistic for $k=2$

As $\lim_{n \to \infty} r_{X_j} \sim N(0,1)$

Thus, $r_{X_j}^2 \sim \chi^2 (1)$

Consider the case $k = 2$:

$r_{X_1}^2 = ... = \frac{X_1 - n p_1}{n p_2} + \frac{X_2 - n p_2}{n p_2}$ 

Which is the same as the [chi-squared test statistic](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test). *"[normalized](https://en.wikipedia.org/wiki/Normalization_(statistics) "Normalization (statistics)") sum of squared deviations between observed and theoretical [frequencies](https://en.wikipedia.org/wiki/Frequency_(statistics))"*

## Test for distribution type
----
### $\chi^2$ test
----
General form of test statistic is:

$T = \sum^{n_{\text{classes}}}_{i=1} \frac{(n_{\text{observed}, i} - n_{\text{expected},i})^2}{n_{\text{expected}, i}}}$

- Evaluated with a $\chi^2$ distribution with $df$ degrees of freedom.
- Generally $df = n_{\text{classes}} - 1 - m$ 
	- $m$ is the estimated number of parameters
- Recommended to only consider groups such that $n_{\text{expected}, i} \geq 5$
### [Kolmogorov Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
----
- Compare empirical distribution function $F_n(x)$ with hypothesized distribution $F(x)$.
- For known parameters the test statistic does not depend on $F(x)$
- Better [power](https://en.wikipedia.org/wiki/Power_of_a_test) than the $\chi^2$ test
- No grouping considerations  needed
- Works only for completely specified distributions

