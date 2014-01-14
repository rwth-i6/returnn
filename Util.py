import subprocess
from scipy.io.netcdf import NetCDFFile

def cmd(cmd):
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, close_fds=True)
  result = [ tag.strip() for tag in p.communicate()[0].split('\n')[:-1]]
  p.stdout.close()
  return result

def netcdf_dimension(filename, dimension):
  nc = NetCDFFile(filename, 'r')
  res = nc.dimensions[dimension]
  nc.close()
  return res
