def f_theta(params, score, x, t):

    sigma_data = 0.5

    c_skip = sigma_data**2 / ((t - eps) ** 2 + sigma_data**2)
    c_out = sigma_data * (t - eps) / np.sqrt(sigma_data**2 + t**2)

    x_out = score.apply(params, x, t)

    return x * c_skip + x_out * c_out
