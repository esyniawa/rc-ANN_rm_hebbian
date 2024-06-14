# RC-ANN RM Hebbian

This project implements a reservoir in the Neurosimulator [ANNarchy](https://github.com/ANNarchy/ANNarchy), using the "Reward-Modulated Hebbian" learning rule of [Legenstein et al. (2010)](https://www.jneurosci.org/content/30/25/8400.short) and [Hoerzer et al. (2014)](https://academic.oup.com/cercor/article/24/3/677/392266?login=false). 
The network is designed to perform dynamic and memory tasks in both closed-loop and open-loop configurations.
## Installation 

To run this project, you need to have Python installed. You can install the required packages using the provided `.yml` file. First, ensure you have `conda` installed, then run:

```sh
conda env create -n [name] --file env.yml
conda activate [name]
python main.py
```

## Project Structure

```
project
│   README.md
│   env.yml
│   main.py
│
└───network/
      model.py
      utils.py
      definitions.py  
```

* ```main.py```: This is the main script to run the project. It includes functions to set up and execute the neural network in both open-loop and closed-loop configurations
* ```network/```: This directory contains the core components of the neural network.
  * ```model.py```: Defines the RCNetwork class, which includes methods for building, compiling, and running the reservoir computing network.
  * ```utils.py```: Contains utility functions used throughout the project.
  * ```definitions.py```: Defines the neuron models and learning rules used in the network.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

