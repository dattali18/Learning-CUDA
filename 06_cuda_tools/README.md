# CUDA Tools

In this units, we will explore tools provided by NVIDIA for profiling and optimization such as Nsight Compute (for profiling kernels).

Since we are not able to run Nsight Compute on the cloud, we will provide you with a brief overview of the tool and how to use it on your local machine.

## Nsight Compute

Nsight Compute is a profiler for CUDA applications. It allows you to analyze the performance of your CUDA kernels and identify bottlenecks in your code.

### Compiling the Code

To use Nsight Compute, you need to compile your CUDA code with the `-lineinfo` flag to include line number information in the executable. This will allow Nsight Compute to map the performance data to the source code.

```bash
nvcc -lineinfo my_kernel.cu -o my_kernel
```

### Profiling Kernels

To profile your CUDA kernels using Nsight Compute, you can use the following command:

```bash
nsys profile --stats=true --force-overwrite true -o my_kernel_report ./my_kernel
```

This command will generate a report file `my_kernel_report.qdrep` that contains the profiling data for your CUDA kernels.


### Analyzing the Report

You can open the report file `my_kernel_report.qdrep` using Nsight Compute to analyze the performance data. Nsight Compute provides a graphical interface to visualize the performance metrics of your CUDA kernels and identify bottlenecks in your code.

We will put images of the interface here. But due to our limited resources we are not able to show the details of how to use and understand the tool fully.

![Nsight Compute](/images/10_image.jpg)

Example for wrap occupancy:

![Nsight Compute](/images/11_image.webp)


### Conclusion

In this unit, we explored the Nsight Compute profiler for CUDA applications. We learned how to compile our CUDA code with line number information and profile our kernels using Nsight Compute. We also learned how to analyze the profiling data to identify performance bottlenecks in our code.

There many more to explore for this tool that is very useful for optimization of you CUDA code.

## Nsight System

We will not dive into the Nsight System in this unit, but it is also a very useful tool to profile the whole system and not only the CUDA kernels.
