# 9.0 `ncu` Tools

In this unit we will take a brief look at the `ncu` tools that are provided by NVIDIA to help you analyze the performance of your CUDA code.

## 9.1 What is `ncu`

`ncu` is a command-line tool that is provided by NVIDIA to help you analyze the performance of your CUDA code. It can be used to profile your code and identify performance bottlenecks.

## 9.2 How to use `ncu`

To use `ncu` you can run the following command:

```bash
ncu -o output_file input_file
```

This command will profile the `input_file` and create a report in the `output_file`.

## 9.3 `ncu` Options

`ncu` has many options that you can use to control the profiling process. Here are some of the most common options:

- `-o` : Set the output file name.
- `-k` : Set the kernel to profile.
- `-s` : Set the number of profiling sessions.
- `-t` : Set the number of profiling iterations.
- `-f` : Set the output format.
- `-m` : Set the metrics to collect.
- `-w` : Set the working directory.

For a full list of options you can use the following command:

```bash
ncu --help
```

## 9.4 Profiling Process

The profiling process with `ncu` is as follows:

1. The `ncu` tool will run your CUDA code and collect performance data.
2. The performance data will be analyzed and a report will be generated.
3. The report will show you the performance bottlenecks in your code and suggest ways to optimize it.

![ncu](/images/18_image.png)

As we can see in the diagram above we can see how `ncu` is profiling a kernel. It will setup the environment and run the kernel multiple times to collect performance data. Then it will analyze the data and generate a report that shows the performance bottlenecks in the code.

## 9.5 Reading the Report

The `ncu` report will show you the following information:

The report will show you the following information:

- The device vector add time.
- The host vector add time.
- The DRAM frequency.
- The SM frequency.
- The memory throughput.
- The DRAM throughput.
- The L1/TEX cache throughput.
- The L2 cache throughput.
- The SM active cycles.
- The compute (SM) throughput.

The report will also show you the following information:

- The block size.
- The function cache configuration.
- The grid size.
- The registers per thread.
- The shared memory configuration.
- The driver shared memory per block.
- The dynamic shared memory per block.
- The static shared memory per block.
- The threads.
- The waves per SM.

The report will also show you the following information:

- The block limit SM block.
- The block limit registers block.
- The block limit shared mem block.
- The block limit warps block.
- The theoretical active warps per SM.
- The theoretical occupancy.
- The achieved occupancy.
- The achieved active warps per SM.

