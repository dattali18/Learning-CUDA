# SIMD

SIMD (Single Instruction, Multiple Data) is a type of parallelism that allows a single instruction to perform the same operation on multiple data points simultaneously. SIMD is a way to increase the performance of a program by executing multiple operations in parallel.

SIMD is widely used in modern processors to increase the performance of applications that require parallel processing. SIMD instructions are used in multimedia applications, scientific computing, and other applications that require parallel processing.

SIMD instructions are used in modern processors to perform operations on multiple data points simultaneously. SIMD instructions are used in multimedia applications, scientific computing, and other applications that require parallel processing.

![diagram](/images/13_image.png)

In the diagram we saw comparing SISD (Single Instruction, Single Data) and SIMD (Single Instruction, Multiple Data) we can see that SIMD can process multiple data points simultaneously, while SISD can only process one data point at a time.

SIMD differ from SIMT (Single Instruction, Multiple Threads) in that SIMD is a type of parallelism that allows a single instruction to perform the same operation on multiple data points simultaneously, while SIMT is a type of parallelism that allows multiple threads to execute the same instruction simultaneously.

## Usage

SIMD is widely used in modern processors to increase the performance of applications that require parallel processing. SIMD instructions are used in multimedia applications, scientific computing, and other applications that require parallel processing.

## Example

```c
#include <stdio.h>
#include <immintrin.h>

int main() {
    __m256 a = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    __m256 b = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    __m256 c = _mm256_add_ps(a, b);
    float result[8];
    _mm256_storeu_ps(result, c);
    for (int i = 0; i < 8; i++) {
        printf("%f\n", result[i]);
    }
    return 0;
}
```

In the example above we are using SIMD to add two vectors of 8 elements each. We are using the `_mm256_set_ps` function to create the vectors, the `_mm256_add_ps` function to add the vectors, and the `_mm256_storeu_ps` function to store the result in an array.

## References

- [Wiki SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)

