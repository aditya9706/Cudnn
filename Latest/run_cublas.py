import glob
import os 
import argparse
import subprocess
import tabulate
from datetime import datetime
import sqlite3
import json
import csv
import sys

# Creating object files 
list_cpp = glob.glob("*.cpp")
print(list_cpp)
for cpp in list_cpp :
  file_name = cpp.split(".")[0]
  os.system("g++ -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include -o {file}.o -c {cpp_file}".format(file = file_name, cpp_file = cpp))

# Creating executables
list_obj = glob.glob("*.o")
print(list_obj)

for obj in list_obj :
  file_name = obj.split(".")[0]
  os.system("/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -o {executable} {object} -I/usr/local/cuda/include -I/usr/local/cuda/targets/ppc64le-linux/include-L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/ppc64le-linux/lib -lcublas -lcudnn -lstdc++ -lm".format(executable = file_name, object = obj))

# Reading config file     
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--configfile", required=True, help="config file name")
args = vars(ap.parse_args())
cfg_path = args['configfile']
config = open(cfg_path, "r")

Table = []
summary_table = []
commands = []
total_cases = 0
passed_cases = 0
table_name = "SUMMARY3"

# Setting up conection to connect to database
connection = sqlite3.connect('summary.db')
if (connection) :
  print("database connected")
else :
  print("Failed to connect to database")
  
# Table creation using cursor command
cursor = connection.cursor()
cursor.execute("CREATE TABLE {table} (COMMAND TEXT, LATENCY TEXT, THROUGHPUT TEXT, TEST_LEVEL TEXT, STATUS TEXT)".format(table = table_name));

json_summary = {}

# Running executables            
for cmd in config:
  commands.append(cmd)
  cmd = "./" + cmd
  total_cases += 1
  
  output = []
  status = ""
  lines = []
  
  summary = {"Command": "", "Latency": "", "Throughput": "", "Test_Level": "", "Status": ""}
  summary["Command"] = cmd.split("\n")[0]
  summary["Test_Level"] = "L" + summary["Command"].split("-L")[1].split("\n")[0]
  
  API = summary["Command"].split("_")[1]

  if (os.system(cmd + " >> output.txt") == 0):
    summary["Status"] = "PASSED"
    passed_cases += 1
    proc = subprocess.Popen([cmd], stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = str(out).split("\\n")
  else :
    summary["Status"] = "FAILED"
  
  for line in output :
    if ("Latency" in line) :
      summary["Latency"] = line.split(": ")[1]
    elif ("Throughput" in line) :
      summary["Throughput"] = line.split(": ")[1]
  
  if API not in json_summary:
      json_summary[API] = [summary]
  else:
      json_summary[API].append(summary)
  
  Table.append(summary)

  # Values insertion in table
  cursor.execute("INSERT INTO {table} (COMMAND, LATENCY, THROUGHPUT, TEST_LEVEL, STATUS) VALUES(?, ?, ?, ?, ?)".format(table = table_name), \
              (summary["Command"], summary["Latency"], summary["Throughput"], summary["Test_Level"], summary["Status"]))

# Saving changes in table  
connection.commit()
print("\n==================================")
print("Data Uploaded on SQLite Database")
print("==================================")
connection.close()

with open("Summary.json", 'w') as json_file :
    json.dump(json_summary,json_file)
print("\n============================")
print("Printing Data in JSON format")
print("============================")
formatted_json = json.dumps(json_summary, indent=2)
print(formatted_json)

print("\n\n=============================")
print("Test Result Summary of cuBLAS")
print("=============================")
print("Executed on: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Printing Summary in table format
header_flag = 0
header = []
rows = []

for key in json_summary : 
  if(header_flag == 0) :
    header = json_summary[key][0].keys()
    header_flag = 1
    break

values = [json_summary[key] for key in json_summary]
complete_list = []

for cmds in values :
  for cmd in cmds :  
    complete_list.append(cmd)

rows = [x.values() for x in complete_list]
print("\n")
print(tabulate.tabulate(rows, header))

failed_cases = total_cases - passed_cases
passed_percentage = (passed_cases * 100) / total_cases

print("\n\n[{passed}/{total} PASSED]".format(passed = passed_cases, total = total_cases))
print("{percent}% tests passed, {failed} tests failed out of {total}".format(percent = passed_percentage, failed = failed_cases, total = total_cases))
