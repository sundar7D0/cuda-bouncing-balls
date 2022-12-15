# Accelerating 2D bouncing-balls game using CUDA DSL
To accelerate a 2D balls-collision game (designed initially in JavaScript, hosted online and was played 1000+ times), used CUDA (a design specific language to use Nvidia's GPU's as General Purpose GPUs) to accelerate it by 1.5×. This speedup is achieved by leveraging:
* Pinned memory, texture buffer for a fast, dedicated buffer memory-space
* Read/write coalescing for fast data access
* Ternary operator to reduce thread divergence
* Stream kernels to exploit the parallelism in game-construct
* Atomically updated list for ball-states in each tile
* Cache-efficient nested for-loop
* nvprof profiler
* OpenGL for graphics

![ball_bounce_game](images/Ball Bounce.png)

After reading, implementing few other techniques for accelerating this game-construct, "atomically updated list for ball-states in each tile" approach seemed to be the effective, optimal one.

## Dependencies
1. [Installing RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
2. python 3.6+
3. tensorflow 2.1+
4. numba v0.52

## Using the code
A single Jupyter notebook `Modified_Transformer.ipynb`, downloads the dataset, evaluation toolkit (`RDKit`), builds, trains and evaluates the transformer model. It's parameters can be easily modified and the whole setup can be easily ported to run with public-cloud like GCP, AWS, etc. or `google-colab`.

## Additional resources
1. Play the JS [game](https://covidchaos.github.io/), [Code](https://github.com/sundar7D0/covid-chaos)
2. [Slides](https://docs.google.com/presentation/d/1SjBRra2Wo6VOd1nS5jwRlU2aCEokyeq-LyCCo21CvDQ/edit?usp=sharing)
