import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
from matplotlib import style

style.use('ggplot')

Data = np.array([[2.9929, -2.9068],
                 [-0.17631, 2.8043],
                 [2.9158, -2.9107],
                 [-3.013, 5.434],
                 [-4.4922, -4.046],
                 [1.7213, 3.7995],
                 [3.5338, 6.7335],
                 [1.6801, 4.8254],
                 [2.5616, -2.1414],
                 [2.0188, 3.574],
                 [-2.8542, -3.8188],
                 [5.4075, 5.1469],
                 [-4.4657, -3.9241],
                 [4.0821, -2.0176],
                 [-3.0518, 5.8671],
                 [-5.0127, -4.2385],
                 [-3.5869, -3.4983],
                 [-5.3239, -4.5748],
                 [-3.983, -3.7808],
                 [2.181, 4.6575],
                 [-3.0685, -3.1747],
                 [-4.3636, -3.8504],
                 [1.7952, 3.0222],
                 [-4.9632, -2.8861],
                 [-0.2529, 2.8316],
                 [3.4508, -4.565],
                 [3.3518, -2.1925],
                 [2.9492, -3.8127],
                 [-4.7407, -5.185],
                 [-0.1838, 4.2035],
                 [1.738, 2.9046],
                 [-4.9188, -5.9195],
                 [-3.2052, -3.9286],
                 [2.7443, -2.5954],
                 [-2.4087, 5.2406],
                 [3.7909, 4.7973],
                 [-3.7574, -4.1006],
                 [-1.2865, 4.2298],
                 [-1.6379, 5.6199],
                 [1.6313, 3.1618],
                 [-3.9331, -5.6394],
                 [3.2897, 3.5051],
                 [-1.5056, 6.1072],
                 [-3.5464, -4.6035],
                 [1.6283, 5.1929],
                 [2.0151, 2.8355],
                 [2.3471, -2.7175],
                 [3.8918, 2.7797],
                 [0.81563, 4.2062],
                 [-1.8478, 5.994],
                 [2.9536, 2.585],
                 [-4.4203, -4.1539],
                 [1.5073, 4.4411],
                 [-4.8499, -4.7964],
                 [0.7432, 2.964],
                 [2.2648, 3.6064],
                 [0.59179, 4.1412],
                 [-2.332, 5.4979],
                 [-4.5279, -3.2769],
                 [2.9623, 3.746],
                 [3.5908, 4.5265],
                 [2.8485, -1.8373],
                 [1.458, 4.1223],
                 [-2.6966, 4.7577],
                 [2.2918, 3.3854],
                 [-1.3312, 3.4562],
                 [-1.9217, -6.222],
                 [-4.939, -5.7389],
                 [-5.413, -4.516],
                 [2.1674, 3.5211],
                 [3.362, -3.2631],
                 [2.9212, -3.9418],
                 [-2.9738, -4.7581],
                 [-1.3646, 5.4178],
                 [2.6744, 4.6784],
                 [-3.5919, -2.9276],
                 [-4.7737, -3.2258],
                 [3.1391, 4.4274],
                 [2.3653, 6.4124],
                 [-5.3225, -4.8595],
                 [-2.4276, -3.4395],
                 [-0.68012, 4.1456],
                 [2.0778, 4.0807],
                 [-4.797, -3.6394],
                 [1.9864, 4.4121],
                 [-4.0194, -3.6596],
                 [-3.5245, -4.13],
                 [2.3276, 5.5081],
                 [2.7402, 4.3097],
                 [1.3752, 3.0664],
                 [-1.5404, 5.6972],
                 [1.4154, 3.3806],
                 [0.42705, 3.6297],
                 [2.5492, 1.8254],
                 [2.5842, 2.3835],
                 [-3.5416, -2.2447],
                 [-5.9024, -1.6258],
                 [-4.3864, -4.5051],
                 [2.7947, 5.0437],
                 [4.0486, -2.8225],
                 [-3.1354, 3.0932],
                 [-0.95904, 4.4409],
                 [1.9534, 4.3535],
                 [-2.8075, -5.6848],
                 [-1.9999, -1.7767],
                 [-3.4542, -1.9901],
                 [3.1726, 5.7351],
                 [-4.7562, -3.5957],
                 [1.3344, -3.4159],
                 [-4.1103, -3.7225],
                 [1.7175, 7.5699],
                 [-5.625, -5.5144],
                 [-2.0512, 4.9126],
                 [2.5353, 4.483],
                 [1.8749, -3.989],
                 [-4.2333, -3.5961],
                 [-4.8148, -4.5342],
                 [2.8022, 3.5312],
                 [2.1085, 3.6597],
                 [-1.0716, 3.9958],
                 [-1.2536, 6.561],
                 [-3.2747, -2.3135],
                 [3.202, 3.3393],
                 [3.1952, -2.1111],
                 [3.0692, -0.51318],
                 [2.0825, 3.3961],
                 [1.4841, -5.2285],
                 [-2.5833, -3.9885],
                 [-4.0792, -2.6231],
                 [-3.7343, -4.2351],
                 [-5.4847, -4.3446],
                 [1.7789, 3.4648],
                 [2.2897, 5.1654],
                 [2.1446, 4.669],
                 [-3.032, -3.7299],
                 [-6.4247, -4.2838],
                 [-5.3934, -2.5746],
                 [1.0742, 4.0053],
                 [4.4561, -2.7805],
                 [-2.1149, -4.3068],
                 [3.0945, 4.3212],
                 [2.8886, -1.704],
                 [3.9353, -2.3365],
                 [-4.3017, -5.2004],
                 [-3.5512, -3.9994],
                 [0.47657, 5.5447],
                 [1.1111, 3.0407],
                 [-5.5861, -4.3918],
                 [-0.034027, 2.6714],
                 [1.102, 1.8427],
                 [-4.484, -4.3273],
                 [0.58735, 4.975],
                 [1.6874, 4.3792],
                 [-1.6454, 4.6138],
                 [1.9666, 4.2611],
                 [2.4948, 2.7842],
                 [2.7506, -1.706],
                 [-4.7939, -3.1402],
                 [1.9533, -1.4167],
                 [2.8181, -3.234],
                 [-3.9457, -3.3122],
                 [-3.2453, -4.2919],
                 [-4.5944, -4.4438],
                 [1.1512, 3.9429],
                 [-1.7298, 4.1729],
                 [-2.9542, -4.951],
                 [3.1533, -4.2543],
                 [0.046066, 4.5654],
                 [1.5575, 3.3709],
                 [2.7873, 2.7219],
                 [2.6498, -1.3801],
                 [3.7515, -3.6894],
                 [2.1367, 3.1167],
                 [-4.2752, -3.7589],
                 [1.0629, 4.8352],
                 [-1.1427, 7.7485],
                 [-4.0364, -5.2251],
                 [-0.99429, 5.332],
                 [-0.87455, 5.753],
                 [-3.9169, -3.8422],
                 [-5.9445, -2.4761],
                 [2.7444, -2.2849],
                 [-2.1327, 3.9543],
                 [1.6238, 4.0524],
                 [2.0939, 3.0488],
                 [-4.6344, -3.3702],
                 [-4.8939, -3.9622],
                 [2.8851, -2.9314],
                 [-2.154, 6.275],
                 [1.7707, 4.2623],
                 [-0.49517, 3.0799],
                 [-1.6856, 3.8786],
                 [-3.0724, -4.1102],
                 [-6.0649, -3.4349],
                 [-5.0851, -4.7072],
                 [1.573, 4.0777],
                 [-3.913, -3.8264],
                 [4.7813, -3.0656],
                 [2.523, 4.6768],
                 [-2.1227, -3.3908]])

# plt.scatter(Data[:, 0], Data[:, 1], s=150)

colors = ["g", "r", "c", "b", "k"]


# =============== K-Means Alg =============== #
class KMeans:
    # initializer.
    # Either;
    #   (1) set k by hand (first n)
    #   (2) set k to be randomly chosen n points
    #   (3) choose first k at random and the rest such that it
    #   maximized euc distance from previous points
    #
    # initially set means to 2 and tolerance level
    def __init__(self, k, offset=0.001, max_iterations=100):
        self.k = k
        self.offset = offset
        self.max_iterations = max_iterations
        self.centers = {}  # array to hold our center-points
        self.clusters = {}  # our set of clusters

    # the fit function will generate optimal clusters for our data_set
    def fit(self, data_set):

        for i in range(self.max_iterations):
            # sine we have k means we need k clusters.
            for j in range(self.k):
                self.clusters[j] = []
            # calculate the distance from the mean for every data point
            # and set up an array of distances for every data point
            for point in data_set:
                distances = [np.linalg.norm(point - self.centers[mu]) for mu in self.centers]
                # current feature should lie in this cluster
                clstr = distances.index(min(distances))
                self.clusters[clstr].append(point)
            # Let that be out previous set of center points
            prev_centers = dict(self.centers)
            # enter point for eaech cluster should be the average point in the cluster
            for clstr in self.clusters:
                self.centers[clstr] = np.average(self.clusters[clstr], axis=0)
            # Assume clustering is optimal
            optimized = True
            for c in self.centers:
                init_mu = prev_centers[c]
                curr_mu = self.centers[c]
                # calculate the difference between the two center points and
                # check to see if its within tolerable offset.
                if np.sum((curr_mu - init_mu) / init_mu * 100) > self.offset:
                    optimized = False
            # if it is optimized, break out of fit function loop
            if optimized:
                break

    def init_centers_method(self, m, data_set):
        if m == 1:
            print("first method")
            # Choosing k according to method (1)
            for i in range(self.k):
                self.centers[i] = data_set[i]
        elif m == 2:
            print("second method")
            # Choosing k according to method (2)
            random.shuffle(data_set)
            # random shuffle results in duplicates in the list
            # so i convert list to a set and then back to a list
            data_set = [list(t) for t in set(tuple(feature) for feature in data_set)]
            for i in range(self.k):
                self.centers[i] = data_set[i]
        elif m == 3:
            print("third method")
            # Choosing k according to method (3)
            random.shuffle(data_set)
            data_set = [list(t) for t in set(tuple(feature) for feature in data_set)]
            self.centers[0] = data_set[0]
            for i in range(1, self.k):
                euc_dists = []
                for p in range(i, len(data_set)):
                    distances = [sqrt((data_set[p][0] - data_set[q][0]) ** 2 + (data_set[p][1] - data_set[q][1]) ** 2)
                                 for q in range(p + 1, len(data_set))]
                    euc_dists.append(sum(distances))
                self.centers[i] = data_set[(euc_dists.index(max(euc_dists)) + 1)]
        else:
            return

    # ========================================#
    # create a K-Means obj and pass to it our data_set


clf = KMeans(3)
clf.init_centers_method(2, Data)
clf.fit(Data)

# plot the individual center points
for mu in clf.centers:
    plt.scatter(clf.centers[mu][0], clf.centers[mu][1],
                marker="o", color="k", s=150, linewidths=5)
# plot the all the clusters
for clstr in clf.clusters:
    color = colors[clstr]
    for feature in clf.clusters[clstr]:
        plt.scatter(feature[0], feature[1], marker="x", color=color, s=150, linewidths=5)

plt.show()
