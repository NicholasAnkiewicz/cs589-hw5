import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def prior(m):
  dic = {
    0: 0.25,
    1: 0.2,
    2: 0.2,
    3: 0.1,
    4: 0.1,
    5: 0.05,
    6: 0.025,
    7: 0.025,
    8: 0.025,
    9: 0.025,
  }
  return dic[m]

def likelihood_single(x,y,m):
    #prob_density = (np.pi*(0.1**2)) * np.exp(-0.5*((x-prior(m))/(0.1**2))**2)
    pdf = sp.stats.norm(loc=prior(m), scale=0.1**2).pdf(y)
    return pdf

def make_bar(x_axis, y_axis):
  fig = plt.figure(figsize = (8, 5))
  plt.bar(x_axis, y_axis)
  plt.xlabel("m")
  plt.ylabel("p(m)")
  plt.title("p(m) vs. m")
  plt.show()

def main():
  m_ = []
  pm = []
  for m in range(10):
      m_.append(m)
      pm.append(prior(m))
      #print("m", m, "prior(m)", prior(m))
  #make_bar(m_, pm)
  print(likelihood_single(1.366797720937803495e-01, 4.395447963207432113e-02, 0))

if __name__ == "__main__":
  main()