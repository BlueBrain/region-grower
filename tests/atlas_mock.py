from os import devnull
from subprocess import call


def small_O1(folder_path):
    """Dump a small O1 atlas in folder path"""
    with open(devnull, "w") as f:
        call(["brainbuilder", "atlases",
              "-n", "6,5,4,3,2,1",
              "-t", "200,100,100,100,100,200",
              "-d", "100",
              "-o", str(folder_path),
              "column",
              "-a", "1000",
              ], stdout=f, stderr=f)
