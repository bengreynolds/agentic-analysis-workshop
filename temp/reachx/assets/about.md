# ReachX 0.7.1 (07 May 2026)

A Python/Qt application used to review and analyze reaching behavior experiments in mice conducted in the 
Jason Christie lab at CU Anschutz.

## License
***ReachX*** was created by [Scott Ruffner](mailto:sruffner@srscicomp.com). It is licensed under the terms of the MIT license.

## Credits
***ReachX*** is based on a Python [application suite](https://github.com/Cerebellum-Lab/reach-training) originally developed by W. Ryan Williamson, Ben Reynolds, and
others. It is being developed with funding provided by the [Jason Christie laboratory](https://www.cerebellumlab.org/) at the University of
Colorado Anschutz Medical Campus.

In addition to the Python standard library, the ***ReachX*** user interface is built upon the Qt for Python framework, 
[PySide6](https://doc.qt.io/qtforpython-6/index.html), with images and graphics drawn using [PyQtGraph](https://pyqtgraph.readthedocs.io/en/latest/index.html). It relies on the Python build of 
[OpenCV](https://opencv.org/) for reading video files, scientific computing packages like [Numpy](https://numpy.org/) and 
[SciPy](https://scipy.org/), and [PyYAML](https://pyyaml.org/) to parse the YAML files on which apps in the existing application
suite depend. Code to train a body part detection model and use that model to analyze ***ReachX*** session videos was 
adapted from the [TensorFlow](https://www.tensorflow.org/)-based implementation in the 
[DeepLabCut library](https://gitlab.uni-marburg.de/polgari/deeplabcut_project).
