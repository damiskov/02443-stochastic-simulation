## Practical Overview
----------
Pass/Not passed based on:
- Lab reports
- Final project
	- Individual contributions considered

#### What is Stochastic Simulation?

*"To have a computer simulate a system affected by randomness"*

I.e., Generating (pseudo)random numbers from a prescribed distribution.

## Course goals 
-----
- Specialized techniques
	- Random number generation
	- Random variable generation
	- Event-by-event principle
	- Variance reduction methods
- Simulation based statistical techniques
	- Markov chains
	- Monte Carlo
	- Bootstrap
- Validation and Verification of Models
- Model building

## Probability basics
----------
- $0 \leq P(A) \leq 1 \ , \ P(\Omega) = 1 \ , \ P(\emptyset) = 0$ 
- $A \cup B = \emptyset \implies P(A \cap B) = P(A) + P(B)$

##### Complement rule:
- $P(\bar{A})=1 - P(A)$

##### Difference rule for $A \subset B$:
- $P(B \cap \bar{A}) = P(B) - P(A)$

##### Inclusion/exclusion for two events:
- $P(A \cup B) = P(A) + P(B) - P(A \cup B)$

##### Conditional probability for $A$ given $B$ (partial information):
- $P(A|B) = \frac{P(A \cap B)}{P(B)}$

##### Multiplication rule:
- $P(A \cap B) = P(B) \cdot P(A|B)$
- $\implies P(A|B) = \frac{P(A \cap B)}{P(B)}$

##### Law of total probability ($B_i$ is a partitioning):
- *Suppose $B_1, ..., B_k$ are mutually exclusive and exhaustive events in a sample space.
- *Then, for any event $A$ in that sample space:
	- $P(A) = \sum_i P(B_i) \cdot P(A|B_i) = \sum_i P(B_i) \cdot \frac{P(A \cap B_i)}{P(B_i)} = \sum_i P(A \cap B_i)$ 

##### Bayes theorem:
- *Relation between two (or more) conditional probabilities*
- $P(B|A) = \frac{P(A|B) \cdot P(B)}{P(A)}$
- When  $B_i$ is a partitioning:
	- $P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{\sum_j P(A|B_j) \cdot P(B_j)}$

##### Independence:
- $P(A|B) = P(A|B^c) \ \ \ (P(A \cup B) = P(A) \cdot P(B))$



## Random Variables
--------------------
Map *outcomes* to *real values*.

# Finish notes on distributions etc later !


##### Distribution
- $P(X = x)$
- $\sum_x P(X = x) = 1$
	- Cumulative sum of probabilities is $1$.
##### Joint distribution
