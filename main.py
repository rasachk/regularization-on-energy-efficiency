import numpy as np
import csv
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

np.random.seed(0)


def load_dataset():
    data = []
    with open('energy_efficiency_data.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                features = [float(x) for x in row[:8]]
                target = float(row[8])
                data.append(features + [target])
            except ValueError:
                continue
    data = np.array(data)
    x = data[:, :8]
    y = data[:, 8]
    return x, y


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    train_size = int(len(x) * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def normalize_features(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / (std + 1e-8)


def compute_basis(dataset, degree):
    n_samples, n_features = dataset.shape
    phi = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        combs_d = combinations_with_replacement(range(n_features), d)
        for comb in combs_d:
            new_feature = np.prod(dataset[:, comb], axis=1, keepdims=True)
            phi = np.hstack((phi, new_feature))
    return phi


def compute_rmsd(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def train_cf(phi, y, lambd):
    I = np.eye(phi.shape[1])
    return np.linalg.inv(phi.T @ phi + lambd * I) @ phi.T @ y


def train_gd(phi, y, lambd, lasso=False, lr=1e-6, max_iter=1000):
    w = np.zeros(phi.shape[1])
    for _ in range(max_iter):
        pred = phi @ w
        grad = 2 * phi.T @ (pred - y) / len(y)
        if lasso:
            grad += lambd * np.sign(w)
        else:
            grad += 2 * lambd * w
        # Gradient clipping
        grad = np.clip(grad, -10, 10)
        w -= lr * grad
    return w


def train_sgd(phi, y, lambd, lasso=False, lr=1e-4, max_iter=5000):
    w = np.zeros(phi.shape[1])
    for _ in range(max_iter):
        i = np.random.randint(len(y))
        xi, yi = phi[i], y[i]
        pred = xi @ w
        grad = 2 * xi * (pred - yi)
        if lasso:
            grad += lambd * np.sign(w)
        else:
            grad += 2 * lambd * w
        grad = np.clip(grad, -10, 10)
        w -= lr * grad
    return w


def count_nonzero(w):
    return np.sum(np.abs(w) > 1e-4)


def main():
    x, y = load_dataset()
    x = normalize_features(x)
    x_train, y_train, x_test, y_test = split_data(x, y)
    degree = 5
    lambdas = [0.01, 0.1, 1, 10]

    methods = ['CF', 'GD-R', 'GD-L', 'SGD-R', 'SGD-L']
    rmsd_results = {m: [] for m in methods}
    nnz_results = {'GD-L': [], 'SGD-L': []}

    for lambd in lambdas:
        phi_train = compute_basis(x_train, degree)
        phi_test = compute_basis(x_test, degree)

        w_cf = train_cf(phi_train, y_train, lambd)
        y_pred = phi_test @ w_cf
        rmsd_results['CF'].append(compute_rmsd(y_test, y_pred))

        w_gdr = train_gd(phi_train, y_train, lambd, lasso=False)
        y_pred = phi_test @ w_gdr
        rmsd_results['GD-R'].append(compute_rmsd(y_test, y_pred))

        w_gdl = train_gd(phi_train, y_train, lambd, lasso=True)
        y_pred = phi_test @ w_gdl
        rmsd_results['GD-L'].append(compute_rmsd(y_test, y_pred))
        nnz_results['GD-L'].append(count_nonzero(w_gdl))

        w_sgdr = train_sgd(phi_train, y_train, lambd, lasso=False)
        y_pred = phi_test @ w_sgdr
        rmsd_results['SGD-R'].append(compute_rmsd(y_test, y_pred))

        w_sgdl = train_sgd(phi_train, y_train, lambd, lasso=True)
        y_pred = phi_test @ w_sgdl
        rmsd_results['SGD-L'].append(compute_rmsd(y_test, y_pred))
        nnz_results['SGD-L'].append(count_nonzero(w_sgdl))

    print("RMSD Table")
    print("Method\t\t", "\t".join(f"λ={l}" for l in lambdas))
    for method in methods:
        print(f"{method:10s}\t", "\t".join(f"{r:.3f}" for r in rmsd_results[method]))

    for method in methods:
        plt.plot(np.log10(lambdas), rmsd_results[method], label=method)
    plt.xlabel("log10(lambda)")
    plt.ylabel("RMSD")
    plt.legend()
    plt.title("RMSD vs Regularization Strength")
    plt.grid(True)
    plt.savefig("rmsd_vs_lambda.png")
    plt.show()

    print("\nNon-zero weights (GD-L, SGD-L)")
    print("λ\tGD-L\tSGD-L")
    for i, lambd in enumerate(lambdas):
        print(f"{lambd}\t{nnz_results['GD-L'][i]}\t{nnz_results['SGD-L'][i]}")


if __name__ == "__main__":
    main()
