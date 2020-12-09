import copy
import pickle
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.special import logsumexp
from skimage.util import view_as_windows as viewW
import time
from skimage.transform import resize

np.set_printoptions(precision=7, suppress=True)
import pandas as pd

INIT_VALUE = 1000000000

INIT_LL_THRESHOLD = 300

EPS = 0.0001

FIRST_ITER = 10

ZERO_MEAN = 0

PATCH_SIZE = 64
NUM_OF_PATCHES = 20000


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w]), columns


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
                noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        print(i)
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original[0] - denoised_images[i]) ** 2))

    plt.figure()
    plt.title("Denoising compraision for 10k photos")
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1, label="Noisy images for std = 0.01 , 0.05 , 0.1 , 0.2")
        plt.imshow(noisy_images[:, :, i], cmap='gray', )
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()

    return denoised_images, noisy_images


from scipy.stats import multivariate_normal
from scipy.stats import norm


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.normal_rv = multivariate_normal(self.mean, self.cov)

    def log_likelihood(self, X, log=True):
        return logsumexp(self.normal_rv.logpdf(X.T))


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """

    def __init__(self, cov, mix, ICA=False,mean=0):
        self.gaussain_list = []
        self.mix_len = len(mix)
        self.mean=mean
        self.cov = cov
        self.gaus_amount = mix.shape[0]
        self.mix = mix

        for i in range(self.gaus_amount):
            if ICA:
                self.gaussain_list.append(norm(0, cov[i]))
            else:
                self.gaussain_list.append(multivariate_normal(np.zeros(PATCH_SIZE), cov[i, :, :]))

    def update_model(self, cov, mix,mean=0, ICA=False,with_mean=False):
        self.mix = mix
        for i in range(self.gaus_amount):
            if ICA:
                self.gaussain_list[i].cov = cov[i]
                if with_mean:
                    self.gaussain_list[i].mean=mean[i]
            else:
                self.gaussain_list[i].cov = cov[i, :, :]

    def log_likelihood(self, X, log=True):
        return logsumexp(self.calc_logpdf(X))

    def calc_logpdf(self, X, ICR=True):
        prbs = np.zeros((self.gaus_amount, max(X.shape)))
        for i, norm_vr in enumerate(self.gaussain_list):
            prbs[i, :] = (norm_vr.logpdf(X.T))
        return self.mix.reshape(self.gaus_amount, 1) + prbs


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th prbs
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix,mean=0):
        self.P = P
        self.vars = vars
        self.mix = mix
        self.mean=mean
        self.num_source = mix.shape[0]
        self.num_guas = mix.shape[1]

        D1_GMMS = []

        for i in range(self.num_source):
            D1_GMMS.append(GSM_Model(vars[i, :], mix[i, :], ICA=True))

        self.D1_GMMS = D1_GMMS

    def calc_pdf(self, X):
        prbs = np.zeros((self.num_source, self.num_guas, np.max(X.shape)))
        for i, D1_GMM in enumerate(self.D1_GMMS):
            prbs[i, :, :] = D1_GMM.log_likelihood(X[i, :])

        return ((self.mix.reshape(PATCH_SIZE, self.num_guas, 1)) + prbs)

    def log_likelihood(self, X):
        return logsumexp(self.calc_pdf(X))


def MVN_log_likelihood(X, model: MVN_Model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return (model.log_likelihood(X))


def GSM_log_likelihood(X, model: GSM_Model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return model.log_likelihood(X)


def ICA_log_likelihood(X, model: ICA_Model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return model.log_likelihood(X)


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    cov = np.cov(X)
    mean = np.mean(X, axis=1)
    return MVN_Model(mean, cov)


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """

    cov_base, cur_model = create_start_model(X, k)
    converges = False
    ll_last = INIT_VALUE
    em_exp = []

    # Run the loop until convergences , calcaulte the likelihhod , save the values for plots and check if the conv
    # condition is upheld.
    while converges == False:
        cov_base_curr = np.copy(cov_base)
        new_m, exepcted_curr = one_iter_EM(X, cov_base_curr, cur_model)

        cur_model = new_m

        distance_ll = np.abs((exepcted_curr) - (ll_last))
        converges = distance_ll < EPS

        ll_last = exepcted_curr
        print(exepcted_curr)
        em_exp.append(exepcted_curr)

    graph_training(em_exp)

    return cur_model


def graph_training(em_exp):
    plt.plot(np.arange(len(em_exp)), em_exp)
    plt.xlabel("Iter")
    plt.ylabel("Log likelihood ")
    plt.title("Convergence of EM for GSM  with epsilion :" + str(EPS))
    plt.show()


def create_start_model(X, k):
    """
    Initialize the model , chose some parametrs not to make it absoutly random and give it a head start ,
    those parameters worked by empirical tests and could mean nothing.
    @param X:
    @param k:
    @return:
    """
    cov_base = np.cov(X)

    scaling_scalars = np.random.rand(k)
    phi = (np.random.rand(k))
    phi /= phi.sum()

    scaling_reshaped = scaling_scalars[:, np.newaxis,
                       np.newaxis]  # Prepare for multiplying each cov matrix in the model
    
    extend = np.repeat(cov_base[np.newaxis, :, :], k, axis=0)
    cov_base_r_mult = extend * scaling_reshaped

    cur_model = GSM_Model(cov_base_r_mult, phi)
    return cov_base, cur_model


def one_iter_EM(X, cov_base, cur_model: GSM_Model):
    # Expec step
    cdf_mixed = cur_model.calc_logpdf(X)
    # In this function when we finisth the cijs calculate the maximzation begin
    cijs, n_phi = calculate_phis_cjis(cdf_mixed)
    d = cov_base.shape[0]
    logged_ver_xsx = Calculate_xsigmax(X, cov_base)
    refactor_rjs = calculate_rjs(cijs, logged_ver_xsx, d)
    rjs_cov = (cov_base * (refactor_rjs))  # broad casting the rjs and multiyplting them by each cov base matrix to get
    # kxDxD matrix with the correpsonding sizes covs.
    cur_model = update_model_GSM(cur_model, n_phi, rjs_cov)

    curr_logpdf = cur_model.calc_logpdf(X)
    new_expected = (logsumexp(curr_logpdf))

    return cur_model, new_expected


def Calculate_xsigmax(X, cov_base):
    # Used X@inv_cov_base to create a 50,4 shape matrix that every row should be mulitpled by the corrspoding vector
    # in X again , so the next time it is done by element wise mult instead of matrix mult , made the process
    # Much more efficent.
    create_vectors = X.T @ np.linalg.inv(cov_base)
    vectorized_mult = create_vectors * X.T  # Element wise mult gives us the vectors we want.
    vectorized_xi_sigma_xi_t = (vectorized_mult).sum(axis=1)
    # I think i must move int oexp since the sum is sinide the log this time
    logged_ver_xsx = np.log(vectorized_xi_sigma_xi_t)
    return logged_ver_xsx


def calculate_rjs(cijs, const_xtx, d):
    logged_ver_xsx = const_xtx
    inter_cjis_xsx = (cijs) + logged_ver_xsx  # Broadcasting into cijs shape
    a = np.array(logsumexp(inter_cjis_xsx, axis=1))
    b = np.array(logsumexp(cijs + np.log(d), axis=1))
    stable_rjs_calc = a - b
    rjs = np.exp(stable_rjs_calc)
    # When increasgin dims use np.newaxis to force the mlt.
    refactor_rjs = rjs[:, np.newaxis, np.newaxis]  # Prepare for broadcast into the original cov
    return refactor_rjs


def calculate_phis_cjis(cdf_mixed):
    cijs = normalize_log_likelihoods(cdf_mixed)
    sum_ci_by_examples = (logsumexp(cijs, axis=1))
    #  the multiplying by N is actually reducing beacuse we're in log space.
    n_phi = sum_ci_by_examples - np.array(np.log(NUM_OF_PATCHES))
    return cijs, n_phi


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    P = np.linalg.eig(np.cov(X))[1]


    trained_vars = []
    trained_mix = []
    trained_mean=[]
    s_domain_x = P.T @ X
    # At each iteration we perpare the total vars and covs for the final ICA model.
    for i in range(PATCH_SIZE):
        curr_loss, curr_model = train_each_channel(i, k, s_domain_x)

        trained_vars.append(curr_model.cov)
        trained_mix.append((curr_model.mix).T)
        trained_mean.append(curr_model.mean)

    plt_convergence()
    all_mean=np.vstack(trained_mean)
    all_vars = np.vstack(trained_vars)
    all_mix = np.vstack(trained_mix)
    # Init the model with the total amount of variables.
    return ICA_Model(P, all_vars, all_mix,all_mean)


def plt_convergence():
    plt.title("Convergence the different componenets using GMM 1D. \n with epsilon : " + str(EPS) +" Patch num : "+str(NUM_OF_PATCHES))
    plt.xlabel("Iter")
    plt.ylabel("Log Likelihood")
    plt.show()


def train_each_channel(i, k, s_domain_x):
    """
    Almost identitcal to the training of GMS , the different is we need to slice each xi now and use the
    one_iter_em_ICA function instaed of the GMS since the calculation are a bit different.
    @param i: current pixel in patch
    @param k: number of compoenents for the currnet model
    @param s_domain_x:
    @return:
    """
    converges = False
    ll_last = 10
    curr_model = init_model(k)
    curr_loss = []
    dist_values=[]
    while converges == False:
        curr_model, new_exp = one_iter_em_ICA(s_domain_x[i, :], curr_model)

        distance_ll = np.abs((new_exp) - (ll_last))
        converges = distance_ll < EPS

        dist_values.append(distance_ll)
        curr_loss.append(new_exp)

        ll_last = new_exp
        print(str(new_exp) + "  " + str(i))
    plt.plot(np.arange(len(curr_loss)), curr_loss)
    return curr_loss, curr_model


def init_model(k):
    scaling_scalars = np.random.rand(k)
    phi = (np.random.rand(k))
    phi /= phi.sum()
    mean=np.random.rand(k)
    curr_model = GSM_Model(scaling_scalars, np.log(phi), True,mean)
    return curr_model


def one_iter_em_ICA(X, cur_model):
    # Expectation step
    cdf_mixed = cur_model.calc_logpdf(X)
    # In this function when we finisth the cijs calculate the maximzation begin
    cijs, n_phi = calculate_phis_cjis_ica(cdf_mixed)
    # Maximization step



    vars = calculate_vars_ica(X, cijs)
    cur_model = update_model_ICA(cur_model, n_phi, vars)
    new_expected = (logsumexp(cur_model.calc_logpdf(X)))

    return cur_model, new_expected


def update_model_GSM(cur_model, n_phi, vars,ICA=False):
    """
    Update the parameters of the GSM
    @param cur_model:
    @param n_phi:
    @param vars:
    @return:
    """
    cur_model.cov = vars
    cur_model.mix = (n_phi).reshape(-1, 1)
    if ICA:
        cur_model.update_model(vars, n_phi,ICA=ICA,with_mean=False)
    else:
        cur_model.update_model(vars, n_phi, ICA=ICA)
    return cur_model


def update_model_ICA(cur_model, n_phi, vars,mu=0):
    """
    Update the parmaters of ICA
    @param cur_model:
    @param n_phi:
    @param vars:
    @return:
    """
    cur_model.cov = np.exp(vars)
    cur_model.mix = (n_phi).reshape(-1, 1)
    cur_model.mean=mu
    cur_model.update_model(np.exp(vars), n_phi.reshape(-1, 1), ICA=True)
    return cur_model


def calculate_vars_ica(X, cijs):
    # xi dim is 1 therefor the outer product is avoided and we can just mult regulary and then sum,  which avoids
    # a lot of for loops.

    xtx_vectorized = X * X
    # Math equation behind the update of vars in logspace
    logged_ver_xsx = np.log(xtx_vectorized)

    # Vectorized + expanding logged_ver_xsx into cijs dimension and then  adding them up , which is mult in original .
    inter_cjis_xsx = (cijs) + logged_ver_xsx

    # Logsumexp it in the dimension of the examples in order to get the sums of them over the exmaple
    a = np.array(logsumexp(inter_cjis_xsx, axis=1))
    b = np.array(logsumexp(cijs, axis=1))
    # Decrease them to get the final vars.
    stable_rjs_calc = a - b
    vars = (stable_rjs_calc)  # Exponent over it is done in update_model function.
    return vars


def calculate_phis_cjis_ica(cdf_mixed):
    cijs = normalize_log_likelihoods(cdf_mixed)
    sum_ci_by_examples = (logsumexp(cijs, axis=1))
    # Remember the multiplying by N is actually reducing beacuse we're in log space.
    n_phi = sum_ci_by_examples - np.array(np.log(NUM_OF_PATCHES))  # A scalar so its a vectorize decrease.
    return cijs, n_phi


def MVN_Denoise(Y, mvn_model, noise_std, zero_mean=False, ICR=False):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    if ICR:
        return filter_1D(Y, mvn_model, noise_std)

    else:
        return filter_2D_or_more(Y, mvn_model, noise_std, zero_mean)


def filter_2D_or_more(Y, mvn_model, noise_std, zero_mean):
    """
    Do the needed calculation for the 2D filter , only math here.
    @param Y:
    @param mvn_model:
    @param noise_std:
    @param zero_mean:
    @return:
    """
    cov = mvn_model.cov
    sig_inv = np.linalg.inv(cov)
    noise_std = np.square(noise_std)
    left = (1 / noise_std) * np.eye(cov.shape[0]) + sig_inv
    left = np.linalg.inv(left)
    if zero_mean:
        right = (1 / noise_std) * Y

        return left @ right

    else:
        right = (sig_inv @ mvn_model.mean).reshape(64, 1) + (1 / noise_std) * Y
        return left @ right


def filter_1D(Y, mvn_model, noise_std):
    """
    Do the needed calc for the 1D filter , only math here.
    @param Y:
    @param mvn_model:
    @param noise_std:
    @return:
    """
    noise_std = np.square(noise_std)
    cov_inv = 1 / mvn_model.cov
    inv_noise = 1 / noise_std
    left = inv_noise + cov_inv
    left = 1 / left
    right = inv_noise* Y
    return left * right


def GSM_Denoise(Y, gsm_model: GSM_Model, noise_std, ICA=False):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """

    orig_model_cov = copy.deepcopy(gsm_model.cov)

    # Create guassian dist with noise
    if ICA:
        gsm_model.cov += np.square(noise_std)
        update_model_GSM(gsm_model,gsm_model.mix,gsm_model.cov,ICA=True)
    else:
        gsm_model.cov += (np.square(noise_std) * np.eye(PATCH_SIZE))[np.newaxis, :, :]
        update_model_GSM(gsm_model,gsm_model.mix,gsm_model.cov)

    num_guassains = gsm_model.cov.shape[0]
    per_gaus_cdf = (gsm_model.calc_logpdf(Y))

    cdf_mixed = per_gaus_cdf
    cijs = normalize_log_likelihoods(cdf_mixed)

    # perpare img
    if ICA:
        pre_interpolation_img = np.zeros((num_guassains, Y.shape[0]))
    else:
        pre_interpolation_img = np.zeros((num_guassains, max(Y.shape), PATCH_SIZE))

    # denoise img for each gaussian , afterwards interoplate
    for i in range(num_guassains):
        if ICA:
            curr_model = MVN_Model(0, orig_model_cov[i])
            Y_denoised_curr = MVN_Denoise(Y, curr_model, noise_std, zero_mean=True, ICR=True)
            pre_interpolation_img[i, :] = Y_denoised_curr.T
        else:
            curr_model = MVN_Model(np.zeros(PATCH_SIZE), orig_model_cov[i, :, :])
            Y_denoised_curr = MVN_Denoise(Y, curr_model, noise_std, zero_mean=True, ICR=False)
            pre_interpolation_img[i, :, :] = Y_denoised_curr.T

    # Multiply cjis for each picture to allow the summation over noramzlaiton afterwards.
    # Cijs - prbbly for the picture to come from this guass

    if ICA:
        # Prbbly of pic to come from certain guass mult by filters , and takes the interoplation based on cjis
        cijs_mult_pics = cijs + pre_interpolation_img
    else:
        cijs_mult_pics = cijs[:, :, np.newaxis] + pre_interpolation_img

    final_pic_log_space = logsumexp(cijs_mult_pics, axis=0)

    return final_pic_log_space.T


def ICA_Denoise(Y, ica_model: ICA_Model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    # Prepare the pictures in the correct space
    P = ica_model.P
    pic_in_s_dom = P.T @ Y

    Y_Denoised_in_s_dom = np.zeros_like(Y)
    # For every source possible , which mean the number of pixels in the patch (64 usually) , use the GMS denoising
    # Function to denoise each pixel , we extract the vars mix  from the trained model of ICA ,create a 1D GMS model
    # And then denoise him
    for j in range(ica_model.num_source):
        curr_varinces = ica_model.vars[j, :]
        curr_phi_prbs = ica_model.mix[j, :]
        curr_mu=ica_model.mean[j,:]
        curr_1D_GSM = GSM_Model(curr_varinces, curr_phi_prbs, ICA=True,mean=curr_mu)
        current_componenet = pic_in_s_dom[j, :]
        Y_Denoised_in_s_dom[j, :] = GSM_Denoise(current_componenet, curr_1D_GSM, noise_std, ICA=True)

    # Return us to the original space
    Y_final = P @ Y_Denoised_in_s_dom

    return Y_final


def comapre_ICA_model():
    patches = pd.read_pickle("train_images.pickle")
    test = pd.read_pickle("test_images.pickle")


    patches = sample_patches(patches, n=NUM_OF_PATCHES)

    sample_test = sample_patches(test, n=100)
    std_imgs = grayscale_and_standardize(test)
    smal_pic = resize(std_imgs[0], (400, 400))

    s = time.time()
    ica_model_10 = learn_ICA(patches, 5)
    print("ICA learning time : ", time.time() - s)
    s = time.time()

    ica_model_50 = learn_ICA(patches, 10)
    print("ICA learning time : ", time.time() - s)

    s = time.time()
    Ica_model_80 = learn_ICA(patches, 15)


    print("ICA learning time : ", time.time() - s)

    den_10, noisy1 = test_denoising((smal_pic), ica_model_10, ICA_Denoise)
    den_50, noisy2 = test_denoising((smal_pic), ica_model_50, ICA_Denoise)
    den_80, noisy3 = test_denoising((smal_pic), Ica_model_80, ICA_Denoise)

    print("likelihood over test set with ICA 10  model  ", ica_model_10.log_likelihood(sample_test))
    print("likelihood over test set with ICA 50 model ", ica_model_50.log_likelihood(sample_test))
    print("likelihhod over test set with ICA 80 model ", Ica_model_80.log_likelihood(sample_test))

    for i in range(4):
        fig, ax = plt.subplots(2)
        ax[0].imshow(den_10[i], cmap='gray')
        ax[0, 1].imshow(den_50[i], cmap='gray')
        ax[0, 2].imshow(den_80[i], cmap='gray')
        ax[1].imshow(noisy1[:, :, i], cmap='gray')
        ax[1, 1].imshow(noisy2[:, :, i], cmap='gray')
        ax[1, 2].imshow(noisy3[:, :, i], cmap='gray')
        fig_name = "ica_noise_" + str(i) + ".jpg"
        plt.savefig(fig_name)

    plt.show()


def comapre_gms_model():
    patches = pd.read_pickle("train_images.pickle")
    test = pd.read_pickle("test_images.pickle")

    patches = sample_patches(patches, n=NUM_OF_PATCHES)
    sample_test = sample_patches(test, n=100)
    std_imgs = grayscale_and_standardize(test)
    smal_pic = resize(std_imgs[0], (100, 100))

    s = time.time()
    model_gsm_10 = learn_GSM(patches, 10)
    print("GSM learning time : ", time.time() - s)
    s = time.time()

    model_gsm_50 = learn_GSM(patches, 20)
    print("GSM learning time : ", time.time() - s)

    s = time.time()
    model_gsm_80 = learn_GSM(patches, 30)

    print("GSM learning time : ", time.time() - s)

    print("likelihood over test set with GSM 10  model  ", model_gsm_10.log_likelihood(sample_test))
    print("likelihood over test set with GSM 20 model ", model_gsm_50.log_likelihood(sample_test))
    print("likelihhod over test set with GSM 30 model ", model_gsm_80.log_likelihood(sample_test))
    exit()
    den_10, noisy1 = test_denoising((smal_pic), model_gsm_10, GSM_Denoise)

    den_50, noisy2 = test_denoising((smal_pic), model_gsm_50, GSM_Denoise)

    den_80, noisy3 = test_denoising((smal_pic), model_gsm_80, GSM_Denoise)

    for i in range(4):
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(den_10[i], cmap='gray')
        ax[0, 1].imshow(den_50[i], cmap='gray')
        ax[0, 2].imshow(den_80[i], cmap='gray')
        ax[1, 0].imshow(noisy1[:, :, i], cmap='gray')
        ax[1, 1].imshow(noisy2[:, :, i], cmap='gray')
        ax[1, 2].imshow(noisy3[:, :, i], cmap='gray')
        plt.savefig("gsm_noise" + str(i) + ".jpg")

    plt.show()


def save_pickle(name,object):
    with open(name, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    patches = pd.read_pickle("train_images.pickle")
    test = pd.read_pickle("test_images.pickle")


    patches = sample_patches(patches, n=NUM_OF_PATCHES)
    sample_test = sample_patches(test, n=100)
    std_imgs = grayscale_and_standardize(test)
    smal_pic = resize(std_imgs[0], (200, 200))

    s = time.time()
    model_mvn = learn_MVN(patches)
    print("MVN learning time : ", time.time() - s)

    s = time.time()
    model_gsm = learn_GSM(patches,5)

    print("GSM learning time : ", time.time() - s)

    s = time.time()
    model_ica=learn_ICA(patches,3)

    print("ICA learning time : ", time.time() - s)

    print("likelihood over test set with gsm model  ", model_gsm.log_likelihood(sample_test))
    print("likelihood over test set with ica model ", model_ica.log_likelihood(sample_test))
    print("likelihhod over test set with mvn model ", model_mvn.log_likelihood(sample_test))


    denmv, noisy1 = test_denoising((smal_pic), learn_MVN(patches), MVN_Denoise,noise_range=[0.01,0.3])
    dengms, noisy2 = test_denoising((smal_pic), model_gsm, GSM_Denoise,noise_range=[0.01,0.3])
    denica, noisy3 = test_denoising((smal_pic), model_ica, ICA_Denoise,noise_range=[0.01,0.3])


    noise = ["0.01", "0.05", "0.1", "0.2"]
    for i in range(1):
        print(noise[i])
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(denmv[i], cmap='gray')
        ax[0, 1].imshow(dengms[i], cmap='gray')
        ax[0,2].imshow(denica[i], cmap='gray')
        ax[1, 0].imshow(noisy1[:, :, i], cmap='gray')
        ax[1, 1].imshow(noisy2[:, :, i], cmap='gray')
        ax[1,2].imshow(noisy3[:, :, i], cmap='gray')
        plt.savefig("compare_noise" + str(i) + ".jpg")

    plt.show()


if __name__ == '__main__':
    main()