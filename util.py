import tensorflow as tf
import numpy as np
from cleverhans.tf2.attacks import fast_gradient_method, basic_iterative_method, projected_gradient_descent, momentum_iterative_method

def gen_tf2_fgsm_attack(org_model, x_test, epsilon):
    """ This method creates adversarial examples with fgsm """
    logits_model = tf.keras.Model(org_model.input, org_model.layers[-1].output)

    adv_fgsm_x = fast_gradient_method.fast_gradient_method(logits_model,
                                      x_test,
                                      epsilon,
                                      np.inf,
                                      targeted=False)
    return adv_fgsm_x

def gen_tf2_bim(org_model, x_test, epsilon):
    """ This method creates adversarial examples with bim """
    logits_model = tf.keras.Model(org_model.input, org_model.layers[-1].output)

    adv_bim_x = basic_iterative_method.basic_iterative_method(logits_model,
                                       x_test,
                                       epsilon,
                                       0.5,
                                       nb_iter=20,
                                       norm=np.inf,
                                       targeted=False)
    return adv_bim_x

def gen_tf2_pgm(org_model, x_test, epsilon):
    """ This method creates adversarial examples with bim """
    logits_model = tf.keras.Model(org_model.input, org_model.layers[-1].output)

    adv_bim_x = projected_gradient_descent.projected_gradient_descent(logits_model,
                                       x_test,
                                       epsilon,
                                       0.5,
                                       nb_iter=30,
                                       norm=np.inf,
                                       targeted=False)
    return adv_bim_x

def gen_tf2_mim(org_model, x_test, epsilon):
    """ This method creates adversarial examples with bim """
    logits_model = tf.keras.Model(org_model.input, org_model.layers[-1].output)

    adv_mim_x = momentum_iterative_method.momentum_iterative_method(logits_model,
                                       x_test,
                                       epsilon,
                                       0.5,
                                       nb_iter=10,
                                       norm=np.inf,
                                       targeted=False)
    return adv_mim_x