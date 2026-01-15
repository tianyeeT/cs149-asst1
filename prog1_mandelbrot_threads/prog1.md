# Program 1: Parallel Fractal Generation Using Threads

## Overview

This program implements a multi-threaded version of the Mandelbrot set fractal generation using C++ std::thread. It parallelizes the computation across multiple threads by dividing the image into horizontal strips, where each thread computes a portion of the image.

## Implementation Details

- **Threading Model**: Uses std::thread to create worker threads. The main thread also participates as a worker.
- **Work Distribution**: The image is divided into `numThreads` equal horizontal strips. The last thread handles any remaining rows if the height is not evenly divisible.
- **Synchronization**: Threads are joined after completion to ensure all work is done before proceeding.
- **Performance Measurement**: Each thread's execution time is measured using CycleTimer.

## How to Build and Run

### Prerequisites
- C++11 compatible compiler (g++)
- POSIX threads support

### Building
```bash
make
```

### Running
```bash
./mandelbrot [options]
```

### Command Line Options
- `-t, --threads <N>`: Number of threads to use (default: 2)
- `-v, --view <INT>`: View setting (1 for default view, 2 for zoomed view)
- `-?, --help`: Display help message

### Example
```bash
./mandelbrot -t 4 -v 1
```

## Performance Results (View 1)

The following table shows the speedup achieved with different numbers of threads compared to the serial implementation:

| Threads | Serial Time (ms) | Thread Time (ms) | Speedup |
|---------|------------------|------------------|---------|
| 1       | ~245             | ~245             | 1.00x   |
| 2       | ~245             | ~123             | 1.98x   |
| 3       | ~246             | ~151             | 1.63x   |
| 4       | ~244             | ~101             | 2.42x   |
| 5       | ~243             | ~102             | 2.39x   |
| 6       | ~249             | ~77              | 3.23x   |
| 7       | ~249             | ~73              | 3.38x   |
| 8       | ~247             | ~62              | 3.99x   |

### Analysis

- **Non-linear Speedup**: Speedup is not perfectly linear due to overhead from thread creation, synchronization, and resource contention.
- **3-Thread Anomaly**: With 3 threads on a 4-core processor, speedup drops to 1.63x due to inefficient core utilization.
- **Hyper-threading**: Beyond 4 threads, hyper-threading provides additional speedup up to 8 threads.
- **Load Imbalance**: Thread execution times vary because different image regions have different computational complexity in the Mandelbrot set.

### Per-Thread Timing Example (4 threads)
```
Per-thread times:
  Thread 0: 24.781 ms
  Thread 1: 102.918 ms
  Thread 2: 104.929 ms
  Thread 3: 24.602 ms
```

## Output Files

- `mandelbrot-serial.ppm`: Serial implementation result
- `mandelbrot-thread.ppm`: Multi-threaded implementation result

Both are PPM image files that can be viewed with image viewers supporting PPM format.

## Key Code Changes

1. Added timing measurement in `workerThreadStart()` function
2. Implemented work distribution logic for dividing image rows among threads
3. Added per-thread timing output for analysis

## Conclusion

This implementation demonstrates effective parallelization of a computationally intensive task using multi-threading, achieving up to 4x speedup on an 8-thread capable system while highlighting the complexities of parallel performance optimization.</content>
<parameter name="filePath">/home/eric/cudaRepo/stanford-cs149/cs149-asst1/prog1_mandelbrot_threads/prog1.readme