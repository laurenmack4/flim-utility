# FLIM-Utility

This repository is designed to handle files created by the Flimera time-correlated single-photon counting camera. The
core functionality is found in the `FlimeraData` module, which allows the .H5 files to be opened, parsed and processed.
In addition, the repository offers functionality to perform:

* Registration between images. While not limited to this use-case, this functionality is intented to be used to register
images recorded on different cameras, such as the Flimera and a conventional CCD/sCMOS camera.
* Processing of raw photon streams into intensity and/or TCSPC images.
* Lifetime calculation from TCSPC images. This uses FlimJ as a backend [1], making calls to Fiji plugin fitting routines via
pyimagej.


### Prerequisites
To use this repository, there are a number of prerequisites that must be available on the local machine:

* Python 3 install
* A suitable Java installation: to bridge from python to imagej via pyimagej the Java SDK must be installed. Recommended
version is `11.0.15`.
* Maven must be installed. Installation instructions can be found [here](https://maven.apache.org/download.cgi). On Windows, the `bin` directory must be added to `PATH`. 
* On Windows, the `JAVA_HOME` environment variable must be set correctly.
* Fiji must be installed, with both the FlimJ and FlimOps plugins installed. These can both be installed within Fiji via
the update-sites manager. 
  * Details for installing FlimJ can be found [here](https://github.com/flimlib/flimj-ui?tab=readme-ov-file)
  * Details for installing FlimOps can be found [here](https://github.com/flimlib/flimj-ops/tree/master)

### Usage
Example uses are provided in the `Notebooks` directory.


### References
[1] Gao, Dasong, et al. "FLIMJ: an open-source ImageJ toolkit for fluorescence lifetime image data analysis." PloS one 15.12 (2020): e0238327.