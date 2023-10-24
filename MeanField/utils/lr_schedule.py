import numpy as np


def cosine_linear(n, n_steps, amplitude=1, mean_lr=1., factor=0.1):
	cos_periodicity = n_steps - 2 * factor * n_steps
	lin_periodicity = factor * n_steps
	if n < lin_periodicity:
		lr_curr = mean_lr + amplitude * n / lin_periodicity
	elif n_steps - lin_periodicity < n:
		lr_curr = mean_lr - amplitude + amplitude * (n - lin_periodicity - cos_periodicity) / lin_periodicity
	else:
		lr_curr = mean_lr + amplitude * np.cos(np.pi * (n - lin_periodicity) / cos_periodicity)
	return lr_curr
