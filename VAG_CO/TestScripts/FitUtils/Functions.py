from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np

def decay_func_offset(xs, E_o, k, c):
  y = (E_o)*1/(xs**k) + c
  return y

def decay_func(xs, E_o, k):
  y = (E_o)*1/(xs**k)
  return y


def rentable_func(E_res, E_VNA, E_HiVNA, k_VNA, k_HiVNA, c):
  ys = (E_HiVNA)**(1/k_HiVNA)/(E_VNA**(1/k_VNA))*E_res**(1/k_VNA)/((E_res-c)**(1/k_HiVNA))

  return ys


def fit_with_offset(xdata, ydata, err):
  popt, pcov = curve_fit(decay_func_offset, xdata, ydata, p0 = (ydata[0],ydata[-1], 0.2))
  print(popt, pcov)

  plt.figure()
  plt.plot(xdata, decay_func_offset(xdata, *popt), label = "fit")
  plt.plot(xdata, ydata, label = 'fit: E=%5.3f, c=%5.3f, k=%5.3f' % tuple(popt))
  plt.xscale("log")
  plt.yscale("log")
  plt.legend()
  plt.show()


def fit(xdata, ydata, y_err):
  xdata = np.array(xdata)
  ydata = np.array(ydata)
  y_err = np.array(y_err)

  popt, pcov = curve_fit(decay_func, xdata, ydata, sigma = y_err)
  print("fit")
  print(popt, pcov)

  plt.figure()
  plt.plot(xdata, decay_func(xdata, *popt), label = 'fit: E=%5.3f, k=%5.3f' % tuple(popt))
  plt.errorbar(xdata, ydata, yerr = y_err, fmt = "x")
  plt.xscale("log")
  plt.yscale("log")
  plt.legend()

  perr = np.sqrt(np.diag(pcov))
  print(perr)
  plt.plot(xdata, decay_func(xdata, popt[0]+perr[0], popt[1] + perr[1]), label = 'fit: E=%5.3f, k=%5.3f' % tuple(popt))
  plt.plot(xdata, decay_func(xdata, popt[0] - perr[0], popt[1] - perr[1]), label='fit: E=%5.3f, k=%5.3f' % tuple(popt))

  plt.show()


  return popt, pcov


