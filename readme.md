## General Info

This repository contains the code developed during my bachelor's thesis at TU Wien in collaboration with the Max Planck Institute of Plasma Physics. It encompasses a tool designed for manipulating density profiles and gaining insights into their influence on the bootstrap current.

## How To Use

The best way to understand how it works is by having a look at the [main file](main.py) where you will find the routines I've used in my bachelor's thesis.

## Further Improvements

Although I am currently not planning to implement further improvements and extensions, of course YOU can do it. Contact me and I can add you as a collaborator ;)

#### Possible improvements
* Make [collisionality_calculator](collisionality_calculator.py) use the data provided from [data_reader](data_reader.py) and not reading on its own from auf_sfutils etc.
* Make [data_reader](data_reader.py)'s interpolating feature not to interpolate over all time axis, but only over a specified time. At the moment, it first interpolates and then one can pick a certain time which is quite inefficient.

#### Disclaimer

I am aware that some logical pieces of this code can be further improved. It's just that the scope of my bachelor thesis is not to provide a 100% perfect software product.
