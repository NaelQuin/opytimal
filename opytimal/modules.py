'''
Module of the modules proccessment methods
'''

__all__ = ['isInstalled', 'install', 'update']


def isInstalled(
    module: str
        ) -> (bool):

    try:
        # It trying import the module
        exec(f"import {module}")

        # Set the answer as True
        answer = True

    except ModuleNotFoundError:
        # Set the answer as False
        answer = False

    return answer


def install(
    module: str,
        ) -> (None):

    # Set the install command
    installCmd = "python3 -m pip install {module}"

    if sys.version_info().minor >= 11:
        installCmd += " --break-system-packages"

    # Install the module
    info = subprocess.check_output(
        installCmd.split(' ')
        )

    # Print installation information
    print(info)

    return None
