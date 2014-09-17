import subprocess
import h5py
from scipy.io.netcdf import NetCDFFile

def cmd(cmd):
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, close_fds=True)
  result = [ tag.strip() for tag in p.communicate()[0].split('\n')[:-1]]
  p.stdout.close()
  return result

def hdf5_dimension(filename, dimension):
  fin = h5py.File(filename, "r")
  res = fin.attrs[dimension]
  fin.close()
  return res

def hdf5_strings(handle, name, data):
  dset = handle.create_dataset(name, (len(data),), dtype="S10")
  dset[...] = data #numpy.string_(data)
