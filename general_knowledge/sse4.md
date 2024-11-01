# SSE4

SSE4 is (Streaming SIMD Extensions 4) is a set of SIMD instructions that are used to perform operations on multiple data points simultaneously. SSE4 is an extension to the x86 instruction set architecture that was introduced by Intel in 2006.

SSE4 is used in modern processors to increase the performance of applications that require parallel processing. SSE4 instructions are used in multimedia applications, scientific computing, and other applications that require parallel processing.

## SSE4.2

SSE4.2 is an extension to the SSE4 instruction set that was introduced by Intel in 2008. SSE4.2 adds new instructions to the SSE4 instruction set that are used to perform operations on multiple data points simultaneously.

SSE4.2 is used in modern processors to increase the performance of applications that require parallel processing. SSE4.2 instructions are used in multimedia applications, scientific computing, and other applications that require parallel processing.

## Usage

SSE4 is widely used in modern processors to increase the performance of applications that require parallel processing. SSE4 instructions are used in multimedia applications, scientific computing, and other applications that require parallel processing.

## Example

```c
#include <stdio.h>
#include <immintrin.h>

int main() {
    __m128i a = _mm_set_epi32(1, 2, 3, 4);
    __m128i b = _mm_set_epi32(1, 2, 3, 4);
    __m128i c = _mm_add_epi32(a, b);
    int result[4];
    _mm_storeu_si128((__m128i*)result, c);
    for (int i = 0; i < 4; i++) {
        printf("%d\n", result[i]);
    }
    return 0;
}
```

In the example above we are using SSE4 to add two vectors of 4 elements each. We are using the `_mm_set_epi32` function to create the vectors, the `_mm_add_epi32` function to add the vectors, and the `_mm_storeu_si128` function to store the result in an array.

## References

- [Wiki SSE4](https://en.wikipedia.org/wiki/SSE4)
