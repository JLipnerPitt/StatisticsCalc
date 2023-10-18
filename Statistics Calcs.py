# Write your code here :-)
import numpy as np
import statistics as stats
import scipy
import re
from statistics import NormalDist
from tkinter import filedialog

class Statistics:
    def __init__(self):
        #  self.directory = filedialog.askopenfilename()
        self.data = None
        self.n = None
        self.mean = None
        self.var = None
        self.dev = None

    def readFile(self):
        directory = r"C:\Users\iamth\Desktop\Written Programs\Python Programs\ENGR201\input.txt"
        #with open(filedialog.askopenfilename()) as inputFile:
        with open(directory) as inputFile:
            lines = inputFile.readlines()
            #print(lines)
            if (len(lines)>len(lines[0])):
                lines = [line.rstrip() for line in lines]
                #print(lines)
                lines = [line.replace(',', '') for line in lines]
                lines = [float(line) for line in lines]
                lines.sort()
                #print(lines)
                inputFile.close()
        print(lines)
        self.data = lines
        # print("{}:".format(dataName), arr)

    def sample_statistics(self):
        X = np.array(self.data)
        self.n = len(X)
        self.mean = (np.sum(X) / self.n)
        sample_mean = self.mean

        if self.n % 2 == 1:
            sample_median = X[int((self.n)/2)]
        else:
            sample_median = (X[int(self.n/2) - 1] + X[int(self.n/2)]) / 2
        sample_range = X[-1] - X[0]
        self.var = (1/(len(X)-1))*np.sum((X-sample_mean)**2)
        S_var = self.var
        self.dev = np.sqrt(S_var)
        S_std_dev = self.dev
        # print("x̄ = {}, S² = {}, S = {}, n = {}".format(sample_mean, S_var, S_std_dev, len(X)))
        print("mean = {}, median = {}, range = {}, variance = {}, std_deviation = {}, n = {}"
              .format(sample_mean, sample_median, sample_range, S_var, S_std_dev, self.n))

    def prediction_interval(self, alpha):
        if (self.n>=30):
            Z_alpha = abs(NormalDist().inv_cdf((1+alpha)/2.))
            theta_L = self.mean - (Z_alpha*self.dev)*np.sqrt(1+(1/self.n))
            theta_H = self.mean + (Z_alpha*self.dev)*np.sqrt(1+(1/self.n))
            print("The prediction interval is: {}<x₀<{}".format(round(theta_L, 4), round(theta_H, 4)))
        else:
            t_alpha = abs(scipy.stats.t.ppf(q=(1-alpha)/.2, df=self.n-1))
            theta_L = self.mean - (t_alpha*self.dev)*np.sqrt(1+(1/self.n))
            theta_H = self.mean + (t_alpha*self.dev)*np.sqrt(1+(1/self.n))
            print("The prediction interval is: {}<x₀<{}".format(round(theta_L, 4), round(theta_H, 4)))

    def confidence_interval(self, alpha):
        if (self.n>=30):
            Z_alpha = abs(NormalDist().inv_cdf((1+alpha)/2.))
            theta_L = self.mean - (Z_alpha*self.dev)/(np.sqrt(self.n))
            theta_H = self.mean + (Z_alpha*self.dev)/(np.sqrt(self.n))
            print("The confidence interval is: {}<μ<{}".format(round(theta_L, 4), round(theta_H, 4)))
        else:
            t_alpha = abs(scipy.stats.t.ppf(q=(1-alpha)/2., df=self.n-1))
            theta_L = self.mean - (t_alpha*self.dev)/(np.sqrt(self.n))
            theta_H = self.mean + (t_alpha*self.dev)/(np.sqrt(self.n))
            print("The confidence interval is: {}<μ<{}".format(round(theta_L, 4), round(theta_H, 4)))

    def norm(self, x, mu, sigma=1.0, flag=None):

        if mu == 0 and sigma == 1:
            probability = scipy.stats.norm.cdf(x)
            print("P(Z = {}) = {}".format(x[0], probability))
            return

        if flag == 'g' or flag == 'G':
            probability = 1 - scipy.stats.norm.cdf(x[0], mu, sigma)
            print("P(X ≥ {}) = {}".format(x, probability))

        elif flag == 'l' or flag == 'L':
            probability = scipy.stats.norm.cdf(x[0], mu, sigma)
            print("P(X ≤ {}) = {}".format(x[0], probability))

        else:
            probability = scipy.stats.norm.cdf(x[1], mu, sigma) - scipy.stats.norm.cdf(x[0], mu, sigma)
            print("P({} ≤ X ≤ {}) = {}".format(x[0], x[1], probability))

    def singleztest(self, z, choice=None):
        if choice is None:
            probability = 1 - scipy.stats.norm.cdf(z,0,1)
            print("P(Z < {}) = {})".format(z,probability))
        else:
            probability = 1 - scipy.stats.norm.cdf(z,0,1)
            print("P(Z > {}) = {}".format(z,probability))


    def doubleztest(self, z1, z2):
        probability = scipy.stats.norm.cdf(z2,0,1) - scipy.stats.norm.cdf(z1,0,1)
        print("P({} < Z < {}) = {}".format(z1, z2, probability))

obj = Statistics()
#obj.readFile()
#obj.sample_statistics()
obj.doubleztest()
#obj.readFile()

# print(abs(scipy.stats.f.sf(.8663, dfn = 8, dfd = 8)))
# print(abs(scipy.stats.t.sf(-1.92, 31)))
# print(abs(scipy.stats.t.ppf(.025, 31)))
