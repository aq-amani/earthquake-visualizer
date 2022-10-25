# earthquake-visualizer

A repo to visualize and animate earthquake data on map.

:broom: Needs some code clean up and use of argparse to make passing of data file easier.

:warning: Code to scrape data from the source is not included to avoid misuse.

## Example output

<p align="center">
  <img src="./readme_images/quake.gif" width="400" />
</p>

## :hammer_and_wrench:Setup/ Preparation
1) Make sure you have
```bash
sudo apt-get install python3.9-tk # for matplotlib GUI backend
```
2) Setup the pipenv as follows
```bash
pipenv install --ignore-pipfile --skip-lock --python 3.9
pipenv shell
```

## :rocket: Usage
Data file to use is specified within the main script.
```bash
python earthquake-visualizer.py
```
