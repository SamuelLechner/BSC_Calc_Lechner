## About

#### General Info

Hi! Thanks for showing interest in my project. This repository contains the code developed during my [bachelor's thesis](https://www.youtube.com/watch?v=dQw4w9WgXcQ) at [TU Wien](https://www.tuwien.at/en/) in collaboration with the [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/en). It encompasses a tool designed for manipulating density profiles and gaining insights into their influence on the bootstrap current.

#### How To Use

The most effective method for gaining insight into its functionality is to examine the [main file](main.py), where you can explore the routines that I have implemented in my bachelor's thesis. If you are only interested in the resulting plots, jump into the [Plots](Plots/) directory.

## Contribute

While I don't have immediate plans to implement additional enhancements or extensions, I'm open to collaboration. If you're interested in contributing and making improvements, please feel free to reach out to me, and I'd be happy to add you as a collaborator!

#### Suggestions for Enhancing
* Enhance the [collisionality_calculator](collisionality_calculator.py) module by modifying it to utilize data provided by the [data_reader](data_reader.py) module, instead of independently reading data from auf_sfutils and other sources.
* Improve the interpolating functionality within the [data_reader](data_reader.py) module by allowing users to specify a particular time range for interpolation, rather than interpolating data across the entire time axis. Currently, the module interpolates the entire dataset, followed by the user manually selecting a specific time, which can be inefficient.
* Enhance the precision of the Bootstrap Computation Method beyond the current approach, which relies on the Peeters' Approximation. This enhancement promises a substantial increase in accuracy.

## Disclaimer

I recognize that there is room for enhancement in certain logical aspects of this code. However, it's important to note that my bachelor thesis primarily focuses on achieving the project's goals within the given scope, rather than striving for absolute perfection in the software product. Consequently, I've had to make strategic compromises in specific areas.
