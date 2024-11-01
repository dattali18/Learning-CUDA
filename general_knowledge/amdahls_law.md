# Amdahl's Law

Amdahl's law is a formula to calculate the theoretical speedup of a program when using multiple processors. It is named after Gene Amdahl, a computer architect who was the chief architect of the IBM System/360.

The idea behind the law is as follows: when trying to compute some data we would think that by doubling our computing cores would we double our computing speed? or when continuing to add resource could we expect to see a linear speedup?

The answer is no, and Amdahl's law gives us a way to calculate the theoretical speedup of a program when using multiple processors.

The formula is as follows:
$$S_{\text{latency}}(N) = \frac{1}{(1 - P) + \frac{P}{N}}$$

Where:

- $S_{\text{latency}}$ is the theoretical speedup of the program.
- $P$ is the percentage of the program that can be parallelized.
- $N$ is the number of processors.

Here is a graph to understand the was:

![Amdahl's Law](/images/12_image.png)

In the graph above we can see the decreasing effects that adding more resources has on the final speedup of the program. The more processors we add the less speedup we get.


Let's give an example:

Let's say we have a program that can be parallelized by 90% and we have 10 processors. What would be the theoretical speedup of the program?

$$S_{\text{latency}}(10) = \frac{1}{(1 - 0.9) + \frac{0.9}{10}} = 5.26$$

So, in this case, the theoretical speedup of the program would be 5.26.

This means that by using 10 processors we would get a speedup of 5.26 times. And not by a factor of 10 like we could have thought.

The idea behind Amdahl's law is if it take a women 9 months to birth 1 baby, it doesn't mean that it would take 9 women 1 month to birth 9 babies. The same goes for computing, adding more resources doesn't always mean that the program would run faster.

## Example

If 30% of the execution time may be the subject of a speedup, _p_ will be 0.3; if the improvement makes the affected part twice as fast, _s_ will be 2. Amdahl's law states that the overall speedup of applying the improvement will be:

$$S = \frac{1}{(1 - 0.3) + \frac{0.3}{2}} = 1.17$$

For example, assume that we are given a serial task which is split into four consecutive parts, whose percentages of execution time are $p_1 = 0.11$, $p_2 = 0.18$, $p_3 = 0.23$, and $p_4 = 0.48$ respectively. Then we are told that the 1st part is not sped up, so $s_1 = 1$, while the 2nd part is sped up 5 times, so $s_2 = 5$, the 3rd part is sped up 20 times, so $s_3 = 20$, and the 4th part is sped up 1.6 times, so $s_4 = 1.6$. By using Amdahl's law, the overall speedup is

$$S = \frac{1}{(1 - \sum_{k}^{k}{p_k}) + \sum_{k=1}^{n}{\frac{p_k}{s_k}}}$$
$$ S =  \frac{1}{(1 - 0.11 - 0.18 -0.23 - 0.48) +\frac{0.11}{1} + \frac{0.18}{5} + \frac{0.23}{20} + \frac{0.48}{1.6}} = 1.56$$