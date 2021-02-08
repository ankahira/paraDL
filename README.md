# paraDL: An Oracle for Guiding Large-Scale Model/Hybrid Parallel Training of Convolutional Neural Networks


We analyze the compute, communication, and memory requirements of Convolutional Neural Networks (CNNs) to understand the trade-offs of different parallelism approaches on performance and scalability. We leverage our model-driven analysis to be the basis for an oracle utility which can help in detecting the limitations and bottlenecks of different parallelism approaches at scale. We evaluate the oracle on six parallelization strategies, with four CNN models and multiple datasets (2D and 3D), on up to 1024 GPUs. 
