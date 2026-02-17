#SELECT WHICH CONFIG FILE TO LOAD FOR THE WHOLE PROJECT
'''
configuration file to be loaded for the whole project
'''

from pathlib import Path

configuration_file = (Path(__file__).resolve().parent / "../config/configOMG.ini").resolve()

def load(conf = configuration_file):
    return str(conf)
