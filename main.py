#%%
import importlib
import numpy as np
import lr_models
import evaluation
import gaussian_models
from matplotlib import pyplot as plt

importlib.reload(lr_models)
importlib.reload(evaluation)
importlib.reload(prml)
importlib.reload(gaussian_models)

#%% md
# Training Data Import
#%%

import prml
D_raw, L = prml.load('data/Train.txt', 11)
prml.plot_D(D_raw, L, prml.hFea)

#%% md
# Data Analysis
#%%

#%%
D = prml.scale(prml.z_score(D_raw))
#%%
prml.plot_D(D, L, prml.hFea)
#%%
prml.plot_in_two_parts_feature(D_raw, L, 4, 0.13)
prml.plot_in_two_parts_feature(D_raw, L, 5, 90)
prml.plot_in_two_parts_feature(D_raw, L, 6, 250)
#%%
D_gaussian, D_rank = prml.gaussianization(D)
#%%
prml.plot_D(D_gaussian, L, prml.hFea)
#%%
prml.plot_pearson_correlation(D_gaussian)
#%%
prml.plot_accumulated_variance_ratio(D_gaussian)
#%% md
# Features Analysis
#%%
DP_pca = prml.compute_pca(D_gaussian, m=9)
#%%
DP_pca10 = prml.compute_pca(D_gaussian, m=10)
#%%
prml.plot_D(DP_pca, L, prml.pcaFeat)

#%% md
# Classification
## Gaussian Methods
#%%
D_formats = {'raw': D_raw, 'std&nor': D, 'gaussianized': D_gaussian,
             'pca': DP_pca, 'pca10': DP_pca10}
pi = 0.5
gaussian_models.try_all_methods(D_formats, L, pi, K=5)
#%%
pi = 0.1
gaussian_models.try_all_methods(D_formats, L, pi, K=5)
#%%
pi = 0.9
gaussian_models.try_all_methods(D_formats, L, pi, K=5)
#%%
pi = 0.333
gaussian_models.try_all_methods(D_formats, L, pi, K=5)
#%%
outliers = gaussian_models.Outliers_IQR(D_gaussian, 1.8)
print(f"Deleting {len(outliers)} outlier")
D_formats = {'IQR': np.delete(D_gaussian, outliers, 1)}
gaussian_models.try_all_methods(D_formats, np.delete(L, outliers, 0), 0.66, K=5)
#%% md
## Discriminatory Methods
### Hyperparameterization
#%%
lambdas, all_min_dcfs_gaussian = lr_models.cross_validate_lambdas(D_gaussian, L, K=5, prior=0.666, pi=0.5)
_, all_min_dcfs_prior_gaussian_01 = lr_models.cross_validate_lambdas(D_gaussian, L, K=5, prior=0.1, pi=0.5)
_, all_min_dcfs_prior__gaussian_09 = lr_models.cross_validate_lambdas(D_gaussian, L, K=5, prior=0.9, pi=0.5)

#%%
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_gaussian, label='prior=0.66')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_prior_gaussian_01, label='prior=0.1')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_prior__gaussian_09, label='prior=0.9')
plt.xlabel('Lambda')
plt.ylabel('minDCF')
plt.legend()
plt.show()

#%%
lambdas, all_min_dcfs_raw = lr_models.cross_validate_lambdas(D_raw, L, K=5, prior=0.666, pi=0.5)
_, all_min_dcfs_raw_09 = lr_models.cross_validate_lambdas(D_raw, L, K=5, prior=0.9, pi=0.5)
_, all_min_dcfs_raw_01 = lr_models.cross_validate_lambdas(D_raw, L, K=5, prior=0.1, pi=0.5)
#%%
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_raw, label='prior=0.66')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_raw_01, label='prior=0.1')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_raw_09, label='prior=0.9')
plt.xlabel('Lambda')
plt.ylabel('minDCF')
plt.legend()
plt.show()
#%% md
## Analysis
#%%
lr_models.run_priors_and_pi(D_gaussian, L)
#%%
lr_models.run_priors_and_pi(D_raw, L)
#%%
D_raw_2 = np.vstack((D_raw, np.power(D_raw, 2)))
#%%
lambdas, all_min_dcfs_quadratic = lr_models.cross_validate_lambdas(D_raw_2, L, K=5, prior=0.666, pi=0.5)
_, all_min_dcfs_quadratic_09 = lr_models.cross_validate_lambdas(D_raw_2, L, K=5, prior=0.9, pi=0.5)
_, all_min_dcfs_quadratic_01 = lr_models.cross_validate_lambdas(D_raw_2, L, K=5, prior=0.1, pi=0.5)

#%%
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_quadratic, label='pi=0.66')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_quadratic_09, label='pi=0.9')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_quadratic_01, label='pi=0.1')
plt.xlabel('Lambda')
plt.ylabel('minDCF')
plt.legend()
plt.show()
#%%
D_gaussian_2 = np.vstack((D_gaussian, np.power(D_gaussian, 2)))
#%%
lambdas, all_min_dcfs_quadratic = lr_models.cross_validate_lambdas(D_gaussian_2, L, K=5, prior=0.666, pi=0.5)
_, all_min_dcfs_quadratic_09 = lr_models.cross_validate_lambdas(D_gaussian_2, L, K=5, prior=0.9, pi=0.5)
_, all_min_dcfs_quadratic_01 = lr_models.cross_validate_lambdas(D_gaussian_2, L, K=5, prior=0.1, pi=0.5)

#%%
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_quadratic, label='pi=0.66')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_quadratic_09, label='pi=0.9')
lr_models.plot_cross_validated_lambdas(lambdas, all_min_dcfs_quadratic_01, label='pi=0.1')
plt.xlabel('Lambda')
plt.ylabel('minDCF')
plt.legend()
plt.show()

#%%
lr_models.run_priors_and_pi(D_raw_2, L)

#%%
lr_models.run_priors_and_pi(D_gaussian_2, L)
#%% md
# Results
#%% md
## Gaussian
#%%
D_Test_raw, L_Test = prml.load('data/Test.txt', 11)
#%%
D_Test_gaussianized = prml.apply_test_gaussianization(D_raw, D_Test_raw)
#%%
prml.plot_D(D_Test_gaussianized, L_Test, prml.hFea)
#%%
gaussian_models.iterate_test_results(
    DTr=D_raw, LTr=L, DTe=D_Test_raw, LTe=L_Test, label="RAW DATA")
#%%
gaussian_models.iterate_test_results(
    DTr=D_gaussian, LTr=L, DTe=D_Test_gaussianized, LTe=L_Test, label="GAUSSIANIZED DATA")
#%%
DP_Test_pca9 = prml.compute_pca(D_Test_gaussianized, m=9)
DP_Test_pca10 = prml.compute_pca(D_Test_gaussianized, m=10)

#%%
gaussian_models.iterate_test_results(
    DTr=DP_pca, LTr=L, DTe=DP_Test_pca9, LTe=L_Test, label="PCA 9 DATA")
#%%
gaussian_models.iterate_test_results(
    DTr=DP_pca10, LTr=L, DTe=DP_Test_pca10, LTe=L_Test, label="PCA 10 DATA")
#%% md
## Linear regression
#%%
lr_models.iterate_test_results(
    Dtr=D_raw, Ltr=L, DTe=D_Test_raw, LTe=L_Test, label='RAW DATA')
#%%
lr_models.iterate_test_results(
    Dtr=D_gaussian, Ltr=L, DTe=D_Test_gaussianized, LTe=L_Test, label='GAUSSINIZED DATA')
#%%
D_Test_raw_2 = np.vstack((D_Test_raw, np.power(D_Test_raw, 2)))
#%%
lr_models.iterate_test_results(
    Dtr=D_raw_2, Ltr=L, DTe=D_Test_raw_2, LTe=L_Test, label='RAW QUADRATIC DATA')
#%%
D_Test_gaussianized_2 = np.vstack((D_Test_gaussianized, np.power(D_Test_gaussianized, 2)))
#%%
lr_models.iterate_test_results(
    Dtr=D_gaussian_2, Ltr=L, DTe=D_Test_gaussianized_2, LTe=L_Test,
    label='GAUSSIANIZED QUADRATIC DATA')
#%%
prediction = lr_models.run_over_test(Dtr=D_gaussian_2, Ltr=L, DTe=D_Test_gaussianized_2, LTe=L_Test,
                                     pi=0.5, prior=0.9)
evaluation.binary_plot_error(prediction, L_Test)
#%%
