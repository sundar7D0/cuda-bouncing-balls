# cuda-bouncing-balls
Accelerating a 2D bouncing balls game using CUDA DSL

Play the [Game](https://covidchaos.github.io/). [Code](https://github.com/sundar7D0/covid-chaos)

![ball_bounce_game](./images/Ball Bounce.png)

1.5× speed-up of 2D collision of balls by splitting the rendering into multiple windows, each handled by a CUDA thread
◦ Used OpenGL for graphics; Utilized pinned memory, texture buffer, read/write coalescing, ternary operator to reduce
thread divergence, stream kernels, atomically updated list for ball-states, cache-efficient nested for-loop, nvprof profiler
