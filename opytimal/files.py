'''
Module for files proccessment methods
'''

__all__ = ['createFolder', 'dirname']

import os


def createFolder(pathFolder: str) -> (None):
    # Disconsider the filename
    pathFolder = os.path.dirname(pathFolder)

    # Adjust the input data
    pathFolder = pathFolder.rstrip(' /\\')

    if '/' in pathFolder or '\\' in pathFolder:
        # Split path in subfolders
        folders = pathFolder.split('/')\
            if '/' in pathFolder\
            else pathFolder.split('\\')
    else:
        # Put input data in list
        folders = [pathFolder]

    # String to acumulate the folders
    totalPath = ''

    if not any(folders[0]):
        # Access the root
        totalPath += "/"

        folders = folders[1:]

    # Looping in folders
    for folder in folders:
        # Update the total path
        totalPath += f'{folder}/'

        # Create resepctive folder if not exist
        if not os.path.exists(totalPath):
            try:
                os.mkdir(totalPath)
            except FileExistsError:
                pass

    return None


def dirname(path: str) -> (str):
    return os.path.dirname(path)


def basename(path: str) -> (str):
    return os.path.basename(path)
