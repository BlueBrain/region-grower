from subprocess import call


def small_O1(folder_path):
    """Dump a small O1 atlas in folder path"""
    call(["brainbuilder", "atlases",
          "-n", "1,2,3,4,5,6",
          "-t", "200,100,100,100,100,200",
          "-d", "100",
          "-o", str(folder_path),
          "column",
          "-a", "1000",
          ])
