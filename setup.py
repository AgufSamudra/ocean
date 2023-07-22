import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	# Here is the module name.
	name="ocean-ml",

	# version of the module
	version="1.0.1",

	# Name of Author
	author="Gufranaka Samudra",

	# your Email address
	author_email="gufranakasamudra348@gmail.com",

	# #Small Description about module
	# description="adding number",

	# long_description=long_description,

	# Specifying that we are using markdown file for description
	long_description=long_description,
	long_description_content_type="text/markdown",

	# Any link to reach this module, ***if*** you have any webpage or github profile
	url="https://github.com/AgufSamudra/ocean",
	packages=setuptools.find_packages(),


	# if module has dependencies i.e. if your package rely on other package at pypi.org
	# then you must add there, in order to download every requirement of package

	install_requires=[
		"numpy==1.25.1",
		"scikit-learn==1.3.0",
		"scipy==1.11.1",
		"tabulate==0.9.0",
		"xgboost==1.7.6"
     ],
 
	keywords=['python', 'ocean', 'machine', 'learning', 'machinelearning', 'automl'],


	license="MIT",

	# classifiers like program is suitable for python3, just leave as it is.
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
