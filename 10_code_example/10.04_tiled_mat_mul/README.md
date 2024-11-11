# Tiled Matrix Multiplication

In this section we will try and optimized the matrix multiplication kernel we wrote in the last section to better use the GPU by using shared memory and using tiling.

The operation of data tiling is a very important concept to understand in general for GPU programming and understanding it to a deeper level will give us the tool to use it in many different scenarios.
