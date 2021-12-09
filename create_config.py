import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--configfile", required=True, help="config file name")
ap.add_argument("-n", "--name", required=True,  help="executable name")
ap.add_argument("-b", "--batch_size", required=True)
ap.add_argument("-d", "--channel", required=True)
ap.add_argument("-hh", "--height", required=True)
ap.add_argument("-w", "--width", required=True)
ap.add_argument("-a", "--activation", required=False)
args = vars(ap.parse_args())

cfg_path = args['configfile']
name= args['name']
batch_size = int(args['batch_size'])
channels = int(args['channel'])
height = int(args['height'])
width = int(args['width'])
activation = args['activation']

f  = open(cfg_path+".test", "a")

if (activation == None) :
  command = name + " -n{batch} -c{channel} -h{height} -w{width} \n".format(batch = batch_size, 
                                                                           channel = channels, height = height, 
                                                                           width = width)
else : 
  command = name + " -n{batch} -c{channel} -h{height} -w{width} -a{activation}\n".format(batch = batch_size, 
                                                                                    channel = channels, height = height, 
                                                                                    width = width, activation = activation)
f.write(command)
f.close()

#  Creating data file
# f = open(cfg_path+"_data.txt", "w")
# rand_arr = np.random.randint(0,255,size=(batch_size * channels * height * width))
# print(rand_arr)

# for i in rand_arr:
#     f.write(str(i)+" ")
# f.close()
