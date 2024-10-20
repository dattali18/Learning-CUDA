# 3. CUDA Memory Model

In this unit, we will explore the CUDA Memory Model and it's hierarchy. we will try and understand the different types of memory available in CUDA and how to use them best to optimize our code.

## 3.1 Introduction

In the CUDA programming model there are a few type of accessible memory and each has it's pros and cons. scope and rules. Here are the main types of memory in CUDA:

1. **Global Memory**: Memory that is accessible by all threads in all blocks. It is the slowest memory in terms of access speed. Usually used for large data structures.

2. **Shared Memory**: Memory that is shared by all threads in a **block**. It is the fastest memory in terms of access speed. Usually used for temporary data.

3. **Local Memory**: Memory that is private to each thread. It is the slowest memory in terms of access speed. Usually used for local variables.

4. **Constant Memory**: Memory that is read-only and is accessible by all threads in all blocks. It is cached and has a very high access speed. Usually used for constants.

5. **Local Memory**: Memory that is private to each thread. It is the slowest memory in terms of access speed. Usually used for local variables.

6. **Texture Memory**: Memory that is optimized for 2D spatial locality. It is cached and has a high access speed. Usually used for image processing. (not usually used in DL).

7. **Registers**: Memory that is private to each thread. It is the fastest memory in terms of access speed. Usually used for local variables.

## 3.2 Memory Hierarchy

Here is a diagram of the memory hierarchy in CUDA:

![CUDA Memory Hierarchy](/images/03_image.png)

The same can be represented in a table:


|           | On/Off Chip | Access | Scope  | Liftime         |
| --------- | ----------- | ------ | ------ | --------------- |
| Registers | on          | R/W    | Thread | Thread          |
| Shared    | on          | R//W   | Block  | Block           |
| Local     | off         | R//W   | Thread | Thread          |
| Global    | off         | R//W   | Global | Host Controlled |
| Constant  | off         | R      | Gloabl | Host Controlled |

