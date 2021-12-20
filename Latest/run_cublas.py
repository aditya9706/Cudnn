import glob
import os 
import argparse
import subprocess
import tabulate
from datetime import datetime

# creating object files 
list_cpp = glob.glob("*.cpp")
print(list_cpp)
for cpp in list_cpp :
  file_name = cpp.split(".")[0]
  os.system("g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -o {file}.o -c {cpp_file}".format(file = file_name, cpp_file = cpp))

# creating executables
list_obj = glob.glob("*.o")
print(list_obj)
os.system("mkdir Executables")
for obj in list_obj :
  file_name = obj.split(".")[0]
  os.system("/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -o Executables/{executable} {object} -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include-L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/ppc64le-linux/lib -lcublas -lcudnn -lstdc++ -lm".format(executable = file_name, object = obj))

# Reading config file     
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--configfile", required=True, help="config file name")
args = vars(ap.parse_args())
cfg_path = args['configfile']
config = open(cfg_path, "r")

Table = []
total_cases = 0
passed_cases = 0

# Running executables            
for cmd in config:
  print(cmd)  
  cmd = "./Executables/" + cmd
  os.system("cd Executables")
  total_cases += 1
  
  output = []
  status = ""
  lines = []
  if (os.system(cmd + " >> output.txt") == 0):
    status = "PASSED"
    passed_cases += 1
    proc = subprocess.Popen([cmd], stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = str(out).split("\\n")


  else :
    status = "FAILED"

  summary = {"API": "", "Latency": "", "Throughput": "", "Status": status}

  arguments = cmd.split(" ")
  for line in arguments :
    if ("./" in line) :
      executable = line.split("/")[2]
      summary["API"] = executable.split("_")[1]

  for line in output :
    if ("Latency" in line) :
      summary["Latency"] = line.split(": ")[1]
    elif ("Throughput" in line) :
      summary["Throughput"] = line.split(": ")[1]
  Table.append(summary)

# Printing table
print("Test Result Summary of cuBLAS")
print("=============================")
print("Executed on: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

header = Table[0].keys()
rows =  [x.values() for x in Table]
print("\n")
print(tabulate.tabulate(rows, header))

failed_cases = total_cases - passed_cases
passed_percentage = (passed_cases * 100) / total_cases

print("\n\n[{passed}/{total} PASSED]".format(passed = passed_cases, total = total_cases))
print("{percent}% tests passed, {failed} tests failed out of {total}".format(percent = passed_percentage, failed = failed_cases, total = total_cases))

