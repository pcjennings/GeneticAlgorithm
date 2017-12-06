"""Toy model to test out prediction routines."""
import numpy as np
import matplotlib.pyplot as plt

from atoml.preprocess.feature_preprocess import standardize
from atoml.preprocess.scale_target import target_standardize
from atoml.regression import GaussianProcess
from atoml.regression.gpfunctions.log_marginal_likelihood import log_marginal_likelihood

from geneticML.algorithm import GeneticAlgorithm


def afunc(x):
    """Define some polynomial function."""
    y = x - 50.
    p = (y + 4.)*(y + 4.)*(y + 1.)*(y - 1.)*(y - 3.5)*(y - 2.)*(y - 1.)
    p += 40. * y + 80. * np.sin(10. * x)
    return 1. / 20. * p + 500.


train_points = 50
test_points = 5000

train = 7.6 * np.random.random_sample((1, train_points)) - 4.2 + 50.
target = []
for i in train:
    target.append(afunc(i))

nt = []
for i in range(train_points):
    nt.append(1.0*np.random.normal())
target += np.array(nt)

stdx = np.std(train)
stdy = np.std(target)
tstd = np.std(target, axis=1)

linex = 8. * np.random.random_sample((1, 5000)) - 4.5 + 50.
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i))

test = 8. * np.random.random_sample((1, test_points)) - 4.5 + 50.
test = np.sort(test)
actual = []
for i in test:
    actual.append(afunc(i))

# Scale the input.
std = standardize(train_matrix=np.reshape(train, (np.shape(train)[1], 1)),
                  test_matrix=np.reshape(test, (np.shape(test)[1], 1)))
# std = {}
# std['train'] = np.reshape(train, (np.shape(train)[1], 1))
# std['test'] = np.reshape(test, (np.shape(test)[1], 1))

td = target_standardize(target[0])
target = np.asarray([td['target']])

actual = (np.asarray(actual[0]) - td['mean']) / td['std']
actual = np.asarray([actual])

# Prediction parameters
sdt1 = 1e-1
w1 = [1.]

big_res = []


def ff2(x):
    """Define the fitness function for the GA."""
    n = x[1][0]
    w = x[0]
    s = 1.  # x[2][0]

    theta = [w, n]  # [w, s, n]
    kdict = {'k1': {'type': 'gaussian', 'width': w, 'scaling': s}}
    try:
        score = -log_marginal_likelihood(
            theta=theta, train_matrix=std['train'], targets=target[0],
            kernel_dict=kdict, scale_optimizer=False)
    except ValueError:
        return -1.e100

    big_res.append([score, w[0], n])
    print(score, w[0], n)

    return score


# Setup the GA search.
ga = GeneticAlgorithm(pop_size=50,
                      fit_func=ff2,
                      d_param=[1, 1],
                      pop=None)
ga.search(500)

# Get optimized parameters.
ga_r = ga.pop[0][1][0]
ga_w = ga.pop[0][0]
ga_s = 1.  # ga.pop[0][2]

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': w1, }}  # 'scaling': 0.9}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=True)

# Do the optimized predictions.
optimized = gp.predict(test_fp=std['test'], test_target=actual[0],
                       uncertainty=True, get_validation_error=True)

opt_upper = np.array(optimized['prediction']) + \
 (np.array(optimized['uncertainty']))
opt_lower = np.array(optimized['prediction']) - \
 (np.array(optimized['uncertainty']))

tgp1 = gp.kernel_dict['k1']['width'][0]
tgp2 = gp.regularization
opte = optimized['validation_error']['rmse_average']

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': ga_w, 'scaling': ga_s}}
gp = GaussianProcess(kernel_dict=kdict, regularization=ga_r,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=False, scale_optimizer=False)

# Do the optimized predictions.
optga = gp.predict(test_fp=std['test'], test_target=actual[0],
                   uncertainty=True, get_validation_error=True)

ga_upper = np.array(optga['prediction']) + \
 (np.array(optga['uncertainty']))
ga_lower = np.array(optga['prediction']) - \
 (np.array(optga['uncertainty']))

gae = optga['validation_error']['rmse_average']

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(131)
ax.plot(linex[0], actual[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], optimized['prediction'], 'g-', lw=1, alpha=0.4)
ax.fill_between(test[0], opt_upper, opt_lower, interpolate=True, color='green',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}, e: {2:.3f}'.format(tgp1, tgp2, opte))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(132)
ax.plot(linex[0], actual[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], optga['prediction'], 'r-', lw=1, alpha=0.4)
ax.fill_between(test[0], ga_upper, ga_lower, interpolate=True, color='red',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}, e: {2:.3f}'.format(
    ga_w[0], ga_r, gae))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(133)
ax.plot(test[0], np.array(optimized['uncertainty']), '-', lw=1,
        color='green')
ax.plot(test[0], np.array(optga['uncertainty']), '-', lw=1,
        color='red')
plt.title('Uncertainty Profile')
plt.xlabel('feature')
plt.ylabel('uncertainty')
plt.axis('tight')

r = np.asarray(big_res)
z, x, y, = r[:, :1], r[:, 1:2], r[:, 2:]

# Change color scale to better show minimum.
z_new = np.array(z).copy()
z_new[z_new < -50.] = -50.

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=z_new)
plt.colorbar()

plt.show()
