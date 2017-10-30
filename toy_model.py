"""Toy model to test out prediction routines."""
import numpy as np
import matplotlib.pyplot as plt

from atoml.preprocess.feature_preprocess import standardize
from atoml.regression import GaussianProcess

from genetic import GeneticAlgorithm


def afunc(x):
    """Define some polynomial function."""
    y = x - 50.
    p = (y + 4.)*(y + 4.)*(y + 1.)*(y - 1.)*(y - 3.5)*(y - 2.)*(y - 1.)
    p += 40. * y + 80. * np.sin(10. * x)
    return 1. / 20. * p + 500.


def k_split(train, target, nsplit):
    """Define some k-fold split."""
    train = np.asarray(train)
    target = np.asarray(np.reshape(target, (len(target), 1)))
    data = np.concatenate((train, target), axis=1)
    np.random.shuffle(data)
    data = np.array_split(data, nsplit)

    split_train = []
    split_target = []
    for d in data:
        s = np.hsplit(d, 2)
        split_train.append(s[0])
        split_target.append(s[1])

    return split_train, split_target


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

# Prediction parameters
sdt1 = np.sqrt(1e-1)
w1 = [0.95]


def ff(x):
    """Define the fitness function for the GA."""
    n = x[0]
    w = x[1]

    score = 0.

    split_train, split_target = k_split(std['train'], target[0], 5)
    index = list(range(len(split_train)))
    for i in index:
        ctest, ctestt = split_train[i], split_target[i]
        ctrain, ctraint = None, None
        for j in index:
            if j is not i:
                if ctrain is None:
                    ctrain, ctraint = split_train[j], split_target[j]
                else:
                    ctrain = np.concatenate((ctrain, split_train[j]))
                    ctraint = np.concatenate((ctraint, split_target[j]))

        # Set up the prediction routine.
        kdict = {'k1': {'type': 'gaussian', 'width': w}}
        gp = GaussianProcess(kernel_dict=kdict, regularization=n,
                             train_fp=ctrain, train_target=ctraint,
                             optimize_hyperparameters=False)

        # Do the optimized predictions.
        ga = gp.predict(test_fp=ctest, test_target=ctestt, uncertainty=True,
                        get_validation_error=True)

        score += ga['validation_error']['rmse_average']

    return score / 5.


# Setup the GA search.
ga = GeneticAlgorithm(pop_size=10,
                      fit_func=ff,
                      d_param=[1, 1],
                      pop=None)
ga.search(500)

# Get optimized parameters.
ga_r = ga.pop[0][0][0]
ga_w = ga.pop[0][1]

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': w1}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=True)

# Do the optimized predictions.
optimized = gp.predict(test_fp=std['test'], test_target=actual[0],
                       uncertainty=True, get_validation_error=True)

opt_upper = np.array(optimized['prediction']) + \
 (np.array(optimized['uncertainty'] * tstd))
opt_lower = np.array(optimized['prediction']) - \
 (np.array(optimized['uncertainty'] * tstd))

tgp1 = gp.kernel_dict['k1']['width'][0]*stdx
tgp2 = np.sqrt(gp.regularization)*stdy
opte = optimized['validation_error']['rmse_average']

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': ga_w}}
gp = GaussianProcess(kernel_dict=kdict, regularization=ga_r,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=False)

# Do the optimized predictions.
optga = gp.predict(test_fp=std['test'], test_target=actual[0],
                   uncertainty=True, get_validation_error=True)

ga_upper = np.array(optga['prediction']) + \
 (np.array(optga['uncertainty'] * tstd))
ga_lower = np.array(optga['prediction']) - \
 (np.array(optga['uncertainty'] * tstd))

gae = optga['validation_error']['rmse_average']

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(131)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], optimized['prediction'], 'g-', lw=1, alpha=0.4)
ax.fill_between(test[0], opt_upper, opt_lower, interpolate=True, color='green',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}, e: {2:.3f}'.format(tgp1, tgp2, opte))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(132)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], optga['prediction'], 'r-', lw=1, alpha=0.4)
ax.fill_between(test[0], ga_upper, ga_lower, interpolate=True, color='red',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}, e: {2:.3f}'.format(
    ga_w[0]*stdx, np.sqrt(ga_r)*stdy, gae))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(133)
ax.plot(test[0], np.array(optimized['uncertainty'] * tstd), '-', lw=1,
        color='green')
ax.plot(test[0], np.array(optga['uncertainty'] * tstd), '-', lw=1,
        color='red')
plt.title('Uncertainty Profile')
plt.xlabel('feature')
plt.ylabel('uncertainty')
plt.axis('tight')

plt.show()
