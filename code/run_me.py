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
  if m > 0:
    f_mx = x**m
  else:
    f_mx = 0
  return sp.stats.norm.pdf(y, f_mx, 0.1)

def likelihood(X,Y,m):
  p = likelihood_single(X[0], Y[0], m)
  for n in range(1, len(X)):
    p = p * likelihood_single(X[n], Y[n], m)
  return p

def posterior(X,Y,m):
  p_data = 0
  for i in range(10):
    p_data += likelihood(X, Y, i) * prior(i)
  return (prior(m)*likelihood(X, Y, m))/p_data

def MAP(X,Y):
  ap = [posterior(X, Y, i) for i in range(10)]
  return ap.index(max(ap))

def predict_MAP(x,X,Y):
  if MAP(X, Y) == 0:
    return 0
  return x**(MAP(X, Y))

def mse(x_t, y_t, x_train, y_train, type):
  if type == "MAP":
    '''
    x_t = np.power(x_t)
    print(MAP(x_train, y_train))
    x_t = np.power(x_t, MAP(x_train, y_train))
    pred = y_t - x_t
    pred = np.power(pred, 2)
    pred = np.sum(pred)
    pred = pred / 100
    return pred
    '''
    return np.sum(np.power((y_t - np.power(x_t, MAP(x_train, y_train))), 2))/100
  if type == "Bayes":
    for i in range(len(x_t)):
      x_t[i] = predict_Bayes(x_t[i], x_train, y_train)
    return np.sum(np.power((y_t - x_t), 2))/100

def predict_Bayes(x,X,Y):
  p_bayes = 0.0
  for m in range(10):
    if m == 0:
      p_bayes += posterior(X, Y, m)*0
    else:
      p_bayes += posterior(X, Y, m)*(x**m)
  return p_bayes

def make_scat_MAP(x_axis, y_axis, x_train, y_train, title, ylabel, xlabel):
  x_map = []
  for i in range(len(x_axis)):
    x_map.append(predict_MAP(x_axis[i], x_train, y_train)) 
  plt.figure(figsize=(10,7))
  p = plt.scatter(x_axis, x_map, marker="o")
  l = plt.scatter(x_axis, y_axis, marker="x")
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend((p, l), ('f_map(x)', 'True Value'))
  plt.grid(True)
  plt.show()

def make_scat_Bayes(x_axis, y_axis, x_train, y_train, title, ylabel, xlabel):
  x_bayes = []
  for i in range(len(x_axis)):
    x_bayes.append(predict_Bayes(x_axis[i], x_train, y_train)) 
  plt.figure(figsize=(10,7))
  p = plt.scatter(x_axis, x_bayes, marker="o")
  l = plt.scatter(x_axis, y_axis, marker="x")
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend((p, l), ('Bayes', 'True Value'))
  plt.grid(True)
  plt.show()


def make_bar(x_axis, y_axis, title, ylabel, xlabel):
  fig = plt.figure(figsize = (8, 5))
  plt.bar(x_axis, y_axis)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()

def main():
  m_ = []
  pm = []
  likelihoods = []
  posteriors = []
  file_x = np.loadtxt('./code/x.csv')
  file_y = np.loadtxt('./code/y.csv')
  for m in range(10):
      m_.append(m)
      pm.append(prior(m))
      likelihoods.append(likelihood(file_x, file_y, m))
      posteriors.append(posterior(file_x, file_y, m))
      #print("m", m, "prior(m)", prior(m))
  #make_bar(m_, pm, "p(m) vs. m", ("p(m)", "m")
  #make_bar(m_, likelihoods, "Likelihood vs. m", "Likelihood", "m")
  #print(MAP(file_x, file_y))
  x_test = np.loadtxt('./code/x_test.csv')
  y_test = np.loadtxt('./code/y_test.csv')
  #print(mse(x_test, y_test, file_x, file_y, "MAP"))
  #print(mse(x_test, y_test, file_x, file_y, "Bayes"))
  #make_scat_MAP(x_test, y_test, file_x, file_y, "f_map(x) vs True value", "y", "x")
  #make_scat_Bayes(x_test, y_test, file_x, file_y, "Bayes vs True value", "y", "x")
  #make_bar(m_, posteriors, "Posterior vs. m", "Probability", "m")
  
  
  

if __name__ == "__main__":
  main()