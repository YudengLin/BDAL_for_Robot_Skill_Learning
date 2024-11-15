###### **Overview**

This code repository is partly to support robot skill learning experiment in the submitted manuscript 'Deep Bayesian Active Learning Using In-memory Computing Hardware'.

The segment of code (https://github.com/caelan/LTAMP) associated with LTAMP incorporates and references the code from the subsequent scholarly investigations:

Zi Wang*, Caelan Reed Garrett*, Leslie Pack Kaelbling, Tomás Lozano-Pérez. Learning compositional models of robot skills for task and motion planning, The International Journal of Robotics Research (IJRR), 2020.

Zi Wang, Caelan R. Garrett, Leslie P. Kaelbling, Tomás Lozano-Pérez. Active model learning and diverse action sampling for task and motion planning, International Conference on Intelligent Robots and Systems (IROS), 2018.

###### **System Requirements**

The code requires a standard computer with GPU and enough RAM to support the operations defined by a user.The developmental version of the package has been tested on the `Linux: Ubuntu 16.04` system. And the code mainly depends on the following `Python3.8` scientific packages.

```
numpy
scipy
sklearn
pybullet
GPy
```

###### **Main Files description**

1. To train a memristor Bayesian deep neural network (BDNN) through deep Bayesian active learning framework, run:
   >python ./learn_tools/run_pr2_bnnactive.py 

  
2. To visualize the process of the robot pouring the beads from the cup into the bowl, run:
    >python ./learn_tools/analysis/visualize_pours.py
    >python ./learn_tools/analysis/visualize_diverse.py


###### **License**

This code repository is covered under the GPL-3.0 license.

