.. _installation:

##############################
Installation
##############################

Some dependencies are rather usual (numpy, scipy, matplotlib) while others are more uncommon but very useful (fabio, networkx).

GUIs are based on wxpython graphical libraries which can be tricky to install (Sorry. We are working on it). 


How to Get LaueTools code
********************************

- Download the very last version of the code at **gitlab.esrf.fr** (but you are also welcome to fork this project):

	https://gitlab.esrf.fr/micha/lauetools

- or Download last (or older releases) on **pypi** by means of pip

	https://pypi.org/project/LaueTools/

	if pip is installed:

	.. code-block:: python

	   pip install lauetools 


Build LaueTools Documentation
****************************

Documentation can be generated, by installing sphinx and a cool html theme:

.. code-block:: python

	   pip install sphinx

           pip install sphinx-rtd-theme

You may need rinohtype:

.. code-block:: python

	   pip install RinohType

Then from /LaueTools/Documentation folder which contains `Makefile` and 2 folders `build` and `source`, build the documentation

.. code-block:: shell

	   make html

Files in html format will be browsed in /build/html folder with any web navigator. You can start with index.html.

