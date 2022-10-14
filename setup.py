from setuptools import setup, find_packages

import sys
if sys.version_info.major == 3:
    import LaueTools
    python3entry_points = {
    'console_scripts': [
        'lauetools = LaueTools.LaueToolsGUI:start',
	'peaksearch = LaueTools.FileSeries.Peak_Search:start',
	'indexrefine = LaueTools.FileSeries.Index_Refine:start',
	'plotmeshGUI = LaueTools.plotmeshspecGUI:start',
	'buildsummary = LaueTools.FileSeries.Build_Summary:start',
	'plotmap = LaueTools.FileSeries.Plot_Maps2:start',
	'mapanalyzer = LaueTools.FileSeries.mainFileSeriesGUI:start',
	'daxmgui = LaueTools.daxmgui:start']}
else:
    python3entry_points= {}

# this will be displayed on pypi at the front page of the project
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="LaueTools",
    version="3.0.46",
    packages=find_packages(),
    python_requires='>=2.6 , <3.9',

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3',
                      'numpy>=1.11.3',
                      'scipy>=0.19.0',
                      'matplotlib>=2.0.0',
                      'wxpython>=3.0',
                      'networkx>=2.1',
			'tqdm>=4.60.0',
			'h5py>=3.1'],  # add fabio libtiff?

    include_package_data=True,


    # metadata for upload to PyPI
    author="J S Micha",
    author_email="micha@esrf.fr",
    description="Distribution of LaueTools Package from gitlab.esrf.fr repository for pip",
    long_description=long_description,
    long_description_content_type="text/markdown",
	
    license="MIT",
    keywords="Lauetools x-ray scattering data analysis GUI Laue",
    url="https://sourceforge.net/projects/lauetools/",  # project home page, if any
    classifiers=["Programming Language :: Python :: 3.8",
"Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
	"Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 2.7",
	"Programming Language :: Python :: 2.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	entry_points = python3entry_points,
    # could also include long_description, download_url, classifiers, etc.
)
