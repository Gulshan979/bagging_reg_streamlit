import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

plt.style.use('seaborn-bright')

# Function to generate data
def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y

# Generate training and testing data
n_train = 150
n_test = 100
noise = 0.1
np.random.seed(0)

X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

# Streamlit sidebar inputs
st.sidebar.markdown("# Bagging Regressor")

estimator = st.sidebar.selectbox('Select base estimator', ('Decision Tree', 'SVM', 'KNN'))
n_estimators = int(st.sidebar.number_input('Enter number of estimators', min_value=1, max_value=100, value=10))
max_samples = st.sidebar.slider('Max Samples', 1, n_train, n_train, step=1)
bootstrap_samples = st.sidebar.radio("Bootstrap Samples", ('True', 'False'))

# Convert bootstrap_samples to boolean
bootstrap_samples = True if bootstrap_samples == 'True' else False

# Plot initial graph
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color="yellow", edgecolor="black")
st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    # Select base estimator
    if estimator == 'Decision Tree':
        base_estimator = DecisionTreeRegressor()
    elif estimator == 'SVM':
        base_estimator = SVR()
    else:
        base_estimator = KNeighborsRegressor()

    # Fit base estimator and Bagging regressor
    base_reg = base_estimator.fit(X_train, y_train)
    bagging_reg = BaggingRegressor(base_estimator, n_estimators=n_estimators, max_samples=max_samples, bootstrap=bootstrap_samples).fit(X_train, y_train)

    # Predictions
    base_predict = base_reg.predict(X_test)
    bagging_predict = bagging_reg.predict(X_test)

    # R2 scores
    base_r2 = r2_score(y_test, base_predict)
    bagging_r2 = r2_score(y_test, bagging_predict)

    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax.plot(X_test, base_predict, color='red', label=f"{estimator} (R2: {base_r2:.2f})")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax.plot(X_test, bagging_predict, color='blue', label=f"Bagging {estimator} (R2: {bagging_r2:.2f})")
    ax.legend()
    st.pyplot(fig)
