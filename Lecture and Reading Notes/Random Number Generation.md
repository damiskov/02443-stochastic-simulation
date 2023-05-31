## Summary
----------
- Generating **pseudo**random numbers
- Large number of RNGs available
- **Don't** implement your own
- Built-in RNGs should be checked before use
- Any RNG will fail - given sufficiently extreme circumstances

## Definition
----------
- Uniform distribution $[0; 1]$
- Randomness (independence)
- **Random numbers:** A sequence of independent random variables, $U_i$, uniformly distributed on $]0, 1[$
- Generating a sequence of independently and identically distributed $U(0,1)$ numbers.
- Problem:
	- **Computers do not work in $\mathbb{R}$

### Definition of an RNG
----------
*An RNG is a computer algorithm that outputs a sequence of reals or integers, which appear to be:*
- Uniformly distributed on $[0; 1]$ or $\{0, ..., N-1\}$
- Statistically independent

### Caveats:
----------
*Appear to be:* The sequence must have the same **relevant** statistical properties as I.I.D uniformly distributed random variables.

With any finite precision format, such as *double* or *long*, uniform on $[0; 1]$ can never be achieved.

## Fibonacci
----------
Integer sequence defined by:

$x_i = x_{i-1} + x_{i-2}, \ \ i \geq 2, \ \ x_0 = 1, \ \ x_1 = 1$

### Fibonacci Generator
----------

Also known as additive congruential method.

$x_i = mod(x_{i-1} + x_{i-2}, \ M) \ , \ U_i = \frac{x_i}{M}$

where $x = mod(y, \ M)$ is the modulus after division.
Notice $x_i \in [0, \ M-1]$. 

## Congruential Generator 
----------
The generator:

$U_i = mod(aU_{i-1}, \ 1) \ \ U_i \in [0, 1]$

Illustrates the principle provided $a$ is large, the last digits are retained.
Can be implemented as ($x_i$ is an integer):

$x_i = mod(ax_{i-1}, M), \ \ U_i = \frac{x_i}{M}$

## Mid conclusion
----------
- Initial state determines the whole sequence
- Potentially many different cycles
- Length of each cycle
If $x_i$ can take $N$ values then the maximum length of a cycle is $N$.

### Properties for a Random generator
----------
- Cycle length
- Randomness
- Speed
- Reproducible
- Portable

## Linear Congruential Generator
----------

LCG are defined as 

$x_i = mod(ax_{i-1}+c, M), \ \ U_i = \frac{x_i}{M}$

- *Multiplier* $a$ 
- *Shift* $c$
- *Modulus* $M$


