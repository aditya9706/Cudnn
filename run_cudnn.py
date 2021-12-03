import glob
import os
import subprocess

list_executable = glob.glob("Executables/*")

f = open("cudnn.test", "r")

cmd_args = {}
for l in f :
  line = l.strip().split(" ")
  key = line[0]
  value = ""
  for l in range(1, len(line)):
    value = value + line[l] + " "
  cmd_args[key] = value

print(cmd_args)
print(list_executable)

for exe in list_executable :
  print(exe + " " + cmd_args[exe.split("/")[1]])
  os.system(exe + " " + cmd_args[exe.split("/")[1]])
  # subprocess.check_output("./" + exe + " " + cmd_args[exe.split("/")[1]])
