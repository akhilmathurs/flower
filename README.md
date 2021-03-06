# Flower - A Friendly Federated Learning Research Framework

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/master/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/workflows/Build/badge.svg)

Flower is a research framework for building federated learning systems. The design of Flower is based on a few guiding principles:

* **Customizable**: Federated learning systems vary wildly from one use case to another. Flower allows for a wide range of different configurations depending on the needs of each individual use case.

* **Extendable**: Flower originated from a research project at the Univerity of Oxford, so it was build with AI research in mind. Many components can be extended and overridden to build new state-of-the-art systems. 

* **Framework-agnostic**: Different machine learning frameworks have different strengths. Flower can be used with any machine learning framework, for example, [PyTorch](https://pytorch.org), [TensorFlow](https://tensorflow.org), or even raw [NumPy](https://numpy.org/) if you enjoy computing gradients by hand :)

* **Understandable**: Flower is written with maintainability in mind. The community is encouraged to both read and contribute to the codebase.

> Note: This is pre-release software. Incompatible API changes are likely.

## Installation

Flower can be installed directly from our GitHub repository using `pip`:

```bash
$ pip install git+https://github.com/adap/flower.git
```

Official [PyPI](https://pypi.org/) releases will follow once the API matures.

## Run Examples

We built a number of examples showcasing different usage scenarios in `src/flower_example`. To run an example, first install the necessary extras (available extras: `examples-tensorflow`):

```bash
pip install git+https://github.com/adap/flower.git#egg=flower[examples-tensorflow]
```

Once the necessary extras (e.g., TensorFlow) are installed, you might want to run the Fashion-MNIST example by starting
a single server and client in two terminals using the following commands:

```bash
# Terminal one
$ python -m flower_example.tf_fashion_mnist.server

# Terminal two
$ python -m flower_example.tf_fashion_mnist.client --cid=0 --partition=0 --clients=1
```

### Docker

If you have Docker on your machine you might want to skip most of the setup and try out the example using the following commands:

```bash
# Build docker container and create docker network
$ ./dev/docker_build.sh
$ docker network create flower

# Terminal one
$ docker run --rm --network flower -p 8080:8080 --name server flower:latest flower_example_fashion_mnist_server

# Terminal two
$ docker run --rm --network flower flower:latest flower_example_fashion_mnist_client --cid=0 --partition=0 --clients=1 --grpc_server_address=server
```

## Documentation

* [Documentation](https://flower.adap.com/docs/)

Documentation is still WIP, please consider contributing.

## Contributing to Flower

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) to get started!
