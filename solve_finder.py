import subprocess as subp
from recognizer import cube_colors, cube_done

if cube_done:
  subp.Popen(["cube_explorer\\cube514htm.exe", "-new-tab"])  #use .call without the 2nd argument if new window isn't wanted