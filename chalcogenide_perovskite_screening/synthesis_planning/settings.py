'''
Created on Feb 1, 2023

@author: jiadongc@umich.edu
'''

import os

from dotenv import load_dotenv

load_dotenv()

"""
MPI_KEY is the Materials Project API key.

Set it via the MP_API_KEY environment variable, or in a .env file at the
project root. See README.md for instructions.

Legacy Materials Project API key for pymatgen.ext.matproj.MPRester can be obtained:
https://legacy.materialsproject.org/open

New Materials Project API key for mp_api.MPRester can be obtained:
https://materialsproject.org/api
"""

MPI_KEY = os.environ.get("MP_API_KEY", "")

if not MPI_KEY:
    import warnings
    warnings.warn(
        "MP_API_KEY environment variable not set. "
        "Materials Project API calls will fail. "
        "Set it in a .env file or export MP_API_KEY=your_key",
        stacklevel=2,
    )