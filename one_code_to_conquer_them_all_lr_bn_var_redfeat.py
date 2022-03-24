import sys
import time

import tensorflow as tf
#tf.random.set_seed(1234)
import numpy as np
import os
from tensorflow.keras import layers
import sklearn
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt

os.chdir('/home/labuser1/Hazar')

from hazar_preprocessing_combined import Subject_UCI, Subject_OPPO,Subject_PAMAP,Subject_LISSI, load_data
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input,LeakyReLU, Input, BatchNormalization
from sklearn.utils import shuffle
import tensorflow_addons as tfa
from tensorflow.keras import activations

def pull_away_term(gen_feat):
    nsample = gen_feat.shape[0]
    gen_feat_norm = gen_feat / tf.broadcast_to(tf.norm(gen_feat, ord=2, axis=1), gen_feat.shape)
    cosine = tf.linalg.matmul(gen_feat_norm, tf.transpose(gen_feat_norm))
    mask = tf.ones(cosine.shape) - tf.linalg.tensor_diag(tf.ones(nsample))
    pt_loss = 0.8 * tf.math.reduce_sum((cosine * mask) ** 2) / (nsample * (nsample-1))
    return pt_loss

# Fonction pour calculer le nombre du filtres du générateur en foncyion de base
def get_filters(window_length):
    if window_length % 2 == 0:
        f1 = window_length / 2
    else:
        f1 = (window_length + 1) / 2

    if f1 % 2 == 0:
        f2 = f1 / 2
    else:
        f2 = (f1 + 1) / 2

    if f2 % 2 == 0:
        f3 = f2 / 2
    else:
        f3 = (f2 + 1) / 2

    if f3 % 2 == 0:
        f4 = f3 / 2
    else:
        f4 = (f3 + 1) / 2
    
    filters = [int(f1),int(f2), int(f3), int(f4)]

    crop_value = int(((f4 * 16) - window_length)/2)

    return filters, crop_value


def make_generator_model(data_shape, filter_list, crop_value ):

    in_noise = Input(shape=(100,))

    gen = layers.Dense(filter_list[3] * data_shape[1] * 128, activation='relu')(in_noise)
    gen = BatchNormalization()(gen)
    
    gen = layers.Reshape((filter_list[3], data_shape[1], 128))(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(128, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(128, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.Cropping2D((crop_value, 0))(gen)

    out_layer = tfa.layers.WeightNormalization(layers.Conv2D(1, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='tanh'))(gen)

    model = Model(in_noise, out_layer)

    return model

def make_generator_model_2(data_shape, filter_list, crop_value ):

    in_noise = Input(shape=(100,))

    gen = layers.Dense(filter_list[3] * data_shape[1] * 128, activation='relu')(in_noise)
    gen = BatchNormalization()(gen)
    
    gen = layers.Reshape((filter_list[3], data_shape[1], 128))(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(128, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(128, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.UpSampling2D(size=(2, 1))(gen)
    gen = layers.Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(gen)
    gen = BatchNormalization()(gen)

    gen = layers.Cropping2D((crop_value, 0))(gen)

    out_layer = tfa.layers.WeightNormalization(layers.Conv2D(1, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='tanh'))(gen)

    model = Model(in_noise, out_layer)

    return model

def make_discriminator_model(data_shape, label_count):

    in_window = Input(shape=data_shape)
    x = layers.Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same')(in_window)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Lambda(tf.nn.local_response_normalization, arguments={'alpha': 0.0001, 'beta': 0.75})(x)
    #x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    #x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(128, kernel_size=(5, 1), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(128, kernel_size=(5, 1), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128)(x)
    feature_layer = layers.LeakyReLU(alpha=0.2)(x)

    out_layer = layers.Dense(label_count)(feature_layer)
    
    model = Model(in_window, [out_layer,feature_layer])

    return model


cross_entropy_supervised = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
cross_entropy_unsupervised_real = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
cross_entropy_unsupervised_fake = tf.keras.losses.BinaryCrossentropy()

def weighted_sparse_categorical_crossentropy(labels, logits, weights):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    class_weights = tf.gather(weights, labels)
    loss = tf.cast(loss, tf.float64)
    class_weights = tf.cast(class_weights, tf.float64)
    return tf.reduce_mean(class_weights * loss)

cross_entropy_gen = tf.keras.losses.BinaryCrossentropy()

def improved_loss(logits):
    z_x = tf.math.reduce_sum(tf.math.exp(logits), axis=1)
    loss_disc = z_x / (z_x + 1)
    return loss_disc

def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))

def log_sum_exp(x, axis=1):
    m = tf.math.reduce_max(x, axis=axis)
    return m + tf.math.log(tf.math.reduce_sum(tf.math.exp(x - tf.expand_dims(m, 1)), axis=axis))

def discriminator_loss(real_output_logits, real_labels, unlabeled_output_logits, fake_output_logits):

    #supervised_loss = weighted_sparse_categorical_crossentropy(real_labels, real_output_logits, class_w)
    
    supervised_loss = cross_entropy_supervised(real_labels, real_output_logits)
    #unsupervised_real_loss = cross_entropy_unsupervised_real(tf.ones_like(unlabeled_output_logits[:,0]), improved_loss(unlabeled_output_logits))
    #unsupervised_fake_loss = cross_entropy_unsupervised_fake(tf.zeros_like(fake_output_logits[:,0]), improved_loss(fake_output_logits))
    
    l_unl = log_sum_exp(unlabeled_output_logits)
    l_gen = log_sum_exp(fake_output_logits)
    loss_unl = -0.5*tf.math.reduce_mean(l_unl) + 0.5*tf.math.reduce_mean(tf.math.softplus(l_unl)) + 0.5*tf.math.reduce_mean(tf.math.softplus(l_gen))
    
    #supervised_loss= tf.cast(supervised_loss, tf.float32)
    loss_unl = tf.cast(loss_unl, tf.float32)

    total_loss = supervised_loss + loss_unl
    
    return total_loss


def generator_loss_interm(fake_interm, real_interm):
    #labeled_mean = tf.math.reduce_mean(labeled_interm, axis=0)
    real_mean = tf.math.reduce_mean(real_interm, axis=0)
    fake_mean = tf.math.reduce_mean(fake_interm, axis=0)
    #g_loss_l = tf.math.reduce_mean(tf.math.abs(labeled_mean-fake_mean))
    g_loss = tf.math.reduce_mean(tf.math.abs(real_mean-fake_mean))
    #g_loss = 0.5*(g_loss_l + g_loss_ul)
    #pt_loss = pull_away_term(fake_interm)
    #tf.print("pt_loss: ", pt_loss)
    return g_loss
    
generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.9, beta_2=0.99)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9, beta_2=0.99)

noise_dim = 100
#num_examples_to_generate = 16

#seed = tf.random.normal([num_examples_to_generate, noise_dim])

def label_noise_generator(BATCH=128, repeats=8):
    noise = tf.random.normal([BATCH, noise_dim])
    labels = np.arange(12)
    labels = np.repeat(labels, repeats)
    np.random.shuffle(labels)
    return noise,labels

@tf.function
def train_step():
    noise, _ = label_noise_generator()
    unlabeled_data = next(iter(unlabeled_dataset))[0]
    images, labels = next(iter(labeled_dataset))
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
               
        generated_images = generator(noise, training=True)
        disc_real_output, _ = discriminator(images, training=True)
        disc_unlabeled_output, interm_output_real = discriminator(unlabeled_data, training=True)
        disc_fake_output, interm_output_fake = discriminator(generated_images, training=True)
        #y, idx, count = tf.unique_with_counts(tf.math.argmax(disc_fake_output, axis=1))
        #tf.print("Labels: ", y)
        #tf.print("Label counts: ", count)
        
        disc_loss = discriminator_loss(disc_real_output, labels, disc_unlabeled_output, disc_fake_output)
        tf.print("disc_loss: ", disc_loss)

        gen_loss = generator_loss_interm(interm_output_fake, interm_output_real)
        tf.print("gen_loss: ", gen_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(grads_and_vars=zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(grads_and_vars=zip(gradients_of_discriminator, discriminator.trainable_variables))


    #for i in range(5):
    #    with tf.GradientTape() as gen_tape: 
    #        generated_images = generator(noise, training=True)
    #        disc_unlabeled_output, interm_output_real = discriminator(unlabeled_data, training=True)
    #        disc_fake_output, interm_output_fake = discriminator(generated_images, training=True)
    #        gen_loss = generator_loss_interm(interm_output_fake, interm_output_real)
    #        tf.print("gen_loss: ", gen_loss)
    #   
    #    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    #    generator_optimizer.apply_gradients(grads_and_vars=zip(gradients_of_generator, generator.trainable_variables))


def train(epochs):
    best_val_loss = 0
    i = 0
    val_monitor = 0
    lr_multiplier = 1
    val_monitor_iter = 0
    file_count = 0
    reset_threshold = 0
    iter_count = 100000
    n_batches = len(X_train_ul_f_s_r) // BATCH_SIZE
    for iters in range(iter_count):
            print(iters,end='\r')
            print(n_batches)
            train_step()

            #out_file = 'gensigs_' + current_time + '.txt'
            #c_reshaped = c.reshape(c.shape[0], -1)
            #with open(out_file, "ab") as f:
                #f.write(b"\n")
                #np.savetxt(f, c_reshaped)
            #predictions, _ = discriminator.predict(X_val_f_s_r)
            #disc_accuracy = sklearn.metrics.accuracy_score(y_test_f_s, predictions.argmax(1))
            #disc_test_loss = cross_entropy_supervised(y_test_f_s, predictions)
            #print("Discriminator test accuracy is " + str(disc_accuracy))
            #print("Discriminator test loss is " + str(disc_test_loss))
            #f1 = sklearn.metrics.f1_score(y_val_f_s, predictions.argmax(1), average='weighted')
            #print('Val iter F1 score is: ' + str(f1))

            if iters % n_batches == 0 and iters != 0:
                val_monitor_iter += 1
                print('Epoch is')
                print(iters / n_batches, end='\r')
                predictions, _  = discriminator.predict(X_val_f_s_r)
                disc_accuracy = sklearn.metrics.accuracy_score(y_val_f_s, predictions.argmax(1))
                disc_val_loss = cross_entropy_supervised(y_val_f_s, predictions)
                print("Discriminator validation accuracy is " + str(disc_accuracy))
                print("Discriminator validation loss is " + str(disc_val_loss))
                val_f1 = sklearn.metrics.f1_score(y_val_f_s, predictions.argmax(1), average='weighted')
                print('Validation F1 score is: ' + str(val_f1))

                print(sklearn.metrics.confusion_matrix(y_val_f_s, predictions.argmax(1)))
                # if val_monitor_iter == 5 + reset_threshold:
                #     reset_threshold += 3
                    
                #     print('creating new gen')
                #     reset_weights(generator)
                #     new_lr = 2e-3 * lr_multiplier
                #     val_monitor_iter = 0
                #     for var in generator_optimizer.variables():
                #         var.assign(tf.zeros_like(var))
                #     generator_optimizer.lr.assign(new_lr)
                #     lr_multiplier = lr_multiplier * 2
                noise,_ = label_noise_generator()
                generated_sig = generator(noise, training=True)
                c = generated_sig.numpy()
                for n in range(20):
                    plt.close()
                    fig = plt.figure(dpi=600)
                    ax1 = plt.subplot(311)
                    plt.plot(c[n,:,0])
                    ax2 = plt.subplot(312)
                    plt.plot(c[n,:,2])
                    plt.plot(c[n,:,3])
                    plt.plot(c[n,:,4])
                    ax3 = plt.subplot(313)
                    plt.plot(c[n,:,8])
                    plt.plot(c[n,:,9])
                    plt.plot(c[n,:,10])
                    #ax4 = plt.subplot(414)
                    #plt.plot(c[n,:,11])
                    #plt.plot(c[n,:,12])
                    #plt.plot(c[n,:,13])

                    #ax4 = plt.subplot(414, sharex = ax1, sharey = ax1)

                    plt.setp(ax1.get_xticklabels(), visible=False)
                    plt.setp(ax2.get_xticklabels(), visible=False)
                    #plt.setp(ax3.get_xticklabels(), visible=False)

                    ax1.set_title('Heart Rate (bpm)', fontsize=10)
                    ax2.set_title('Hand 3D-acceleration data', fontsize=10)
                    ax3.set_title('Hand 3D-gyroscope data', fontsize=10)
                    #ax4.set_title('Hand 3D-magnetometer data', fontsize=10)

                    ax1.set_ylim([-0.5, 0.5])
                    ax2.set_ylim([-0.5, 0.5])
                    ax3.set_ylim([-0.5, 0.5])
                    #ax4.set_ylim([-0.5, 0.5])

                    fig.text(0.5, 0.03, 'Time step (33.3Hz)', ha='center', va='center', fontsize=12)
                    fig.text(0.03, 0.5,'Signal values (normalized)' , ha='center', va='center', rotation='vertical', fontsize=12)
                    
                    #plt.ylim([-1, 1])
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
                    #plt.legend(['X', 'Y', 'Z'])
                    ax2.legend(['X', 'Y', 'Z'], loc="lower right", prop={"size":7})
                    ax3.legend(['X', 'Y', 'Z'], loc="lower right", prop={"size":7})
                    #ax4.legend(['X', 'Y', 'Z'], loc="lower right", prop={"size":7})
                    fig_folder_name = "plots/" + "iter_" + str(iters // n_batches) + "/" + str(n) + ".png"
                    os.makedirs("plots/" + "iter_" + str(iters // n_batches), exist_ok=True)
                    plt.savefig(fig_folder_name, format='png', dpi=600)
                    plt.close()

                if val_f1 > best_val_loss:
                    print('New best')
                    print('Epoch is')
                    print(iters / n_batches, end='\r')
                    #model_filename =  dataset_name + '_'  + 'bs' + str(BATCH_SIZE) + '_' + 'e' + str(EPOCHS) + '_' + current_time + '_discriminator_model_long_train'
                    #discriminator.save(model_filename)
                    best_val_loss = val_f1
                    val_monitor_iter = 0

                    predictions, _ = discriminator.predict(X_train_f_s_r)
                    disc_accuracy = sklearn.metrics.accuracy_score(y_train_f_s, predictions.argmax(1))
                    disc_train_loss = cross_entropy_supervised(y_train_f_s, predictions)
                    print("Discriminator train accuracy is " + str(disc_accuracy))
                    print("Discriminator train loss is " + str(disc_train_loss))
                    f1 = sklearn.metrics.f1_score(y_train_f_s, predictions.argmax(1), average='weighted')
                    print('Training F1 score is: ' + str(f1))

                    print(sklearn.metrics.confusion_matrix(y_train_f_s, predictions.argmax(1)))

                    predictions, _  = discriminator.predict(X_train_ul_f_s_r)
                    disc_accuracy = sklearn.metrics.accuracy_score(y_train_ul_f_s, predictions.argmax(1))
                    disc_val_loss = cross_entropy_supervised(y_train_ul_f_s, predictions)
                    print("Ul validation accuracy is " + str(disc_accuracy))
                    print("Ul validation loss is " + str(disc_val_loss))
                    val_f1 = sklearn.metrics.f1_score(y_train_ul_f_s, predictions.argmax(1), average='weighted')
                    print('Ul F1 score is: ' + str(val_f1))

                    print(sklearn.metrics.confusion_matrix(y_train_ul_f_s, predictions.argmax(1)))
        
                    predictions, _ = discriminator.predict(X_test_f_s_r)
                    disc_accuracy = sklearn.metrics.accuracy_score(y_test_f_s, predictions.argmax(1))
                    disc_test_loss = cross_entropy_supervised(y_test_f_s, predictions)
                    print("Discriminator test accuracy is " + str(disc_accuracy))
                    print("Discriminator test loss is " + str(disc_test_loss))
                    f1 = sklearn.metrics.f1_score(y_test_f_s, predictions.argmax(1), average='weighted')
                    print('Test F1 score is: ' + str(f1))

                    print(sklearn.metrics.confusion_matrix(y_test_f_s, predictions.argmax(1)))

              

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size")
    parser.add_argument("epochs")
    parser.add_argument("dataset")
    args = parser.parse_args()

    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
   
    dataset = str(args.dataset)
    dataset_name = str(args.dataset)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    filename =  dataset_name + '_'  + 'bs' + 'e' + str(EPOCHS) + '_' + current_time + '_orig' + '.log'
    results_folder = os.path.join(os.getcwd(), 'results_var', filename)

    sys.stdout = open(results_folder , 'w')
    print("Started.")

    def sel_columns(data_x, data_y_idx):
        if data_y_idx == 'orig':
            return data_x
        else:
            X = data_x[:, :, data_y_idx]
        return X

    #PAMAP2: hand chest ankle
    #labels_to_remove_1 = [i for i in range(40)][:14]
    #labels_to_remove_2 = [[i for i in range(40)][:-13][0]] + [i for i in range(40)][:-13][-13:]
    #labels_to_remove_3 = [[i for i in range(40)][:-13][0]] + [i for i in range(40)][-13:]
    #cols_to_remove_4 =  [i for i in range(40)][:14] + [i for i in range(40)][:-13][-13:] #hand chest 
    #cols_to_remove_5 = [i for i in range(40)][:14] + [i for i in range(40)][-13:] #hand ankle 
    #cols_to_remove_6 =  [[i for i in range(40)][:-13][0]] + [i for i in range(40)][:-13][-13:] + [i for i in range(40)][-13:] # chest ankle

    #LISSI: arm chest ankle
    labels_to_remove_0 = 'orig'
    #labels_to_remove_1 = [i for i in range(50)][0:10] + [i for i in range(50)][30:40] #arm
    #labels_to_remove_2 = [i for i in range(50)][10:20] #chest
    #labels_to_remove_3 = [i for i in range(50)][20:30] + [i for i in range(50)][40:50] #ankle
    #cols_to_remove_4 = [i for i in range(50)][0:10] + [i for i in range(50)][30:40] + [i for i in range(50)][10:20] #arm chest 
    #cols_to_remove_5 = [i for i in range(50)][0:10] + [i for i in range(50)][30:40] + [i for i in range(50)][20:30] + [i for i in range(50)][40:50] #arm ankle 
    #cols_to_remove_6 =[i for i in range(50)][10:20] + [i for i in range(50)][20:30] + [i for i in range(50)][40:50]

    #OPPO
    #cols_to_remove_0 = 'orig'
    #cols_to_remove_1 = [i for i in range(113)][54:63] + [i for i in range(113)][72:81] # lower arms
    #cols_to_remove_2 = [i for i in range(113)][-77:][:9] #back
    #cols_to_remove_3 = [i for i in range(113)][-77:][-32:] #shoe
    #cols_to_remove_4 = [i for i in range(113)][-77:][:9] + [i for i in range(113)][54:63] + [i for i in range(113)][72:81] #backarms
    #cols_to_remove_5 = [i for i in range(113)][54:63] + [i for i in range(113)][72:81] + [i for i in range(113)][-77:][-32:] # armsshoe
    #cols_to_remove_6 = [i for i in range(113)][-77:][:9] + [i for i in range(113)][-77:][-32:] #backshoe
    #cols_to_remove_7 = [i for i in range(113)][-77:][:9] + [i for i in range(113)][54:63] + [i for i in range(113)][72:81] +  [i for i in range(113)][-77:][-32:] #backarmsshoe

    cols_to_remove = labels_to_remove_0

    print("Columns are:")
    print(cols_to_remove)

    print("Loading data.")
    X_train_f_s_r, y_train_f_s, X_train_ul_f_s_r, y_train_ul_f_s, X_val_f_s_r, y_val_f_s, X_test_f_s_r, y_test_f_s = load_data(dataset)
    X_train_f_s_r = sel_columns(X_train_f_s_r, cols_to_remove)
    X_train_ul_f_s_r = sel_columns(X_train_ul_f_s_r, cols_to_remove)
    X_val_f_s_r = sel_columns(X_val_f_s_r, cols_to_remove)
    X_test_f_s_r = sel_columns(X_test_f_s_r, cols_to_remove)

    print(X_train_f_s_r[:10])
        
    #from sklearn.model_selection import train_test_split
    #X_train_f_s_r, _, y_train_f_s, _ = train_test_split(X_train_f_s_r, y_train_f_s, test_size=0.70, random_state=42)

    print(X_train_f_s_r.shape)
    labeled_dataset = tf.data.Dataset.from_tensor_slices((X_train_f_s_r, y_train_f_s)).shuffle(len(X_train_f_s_r)).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
    unlabeled_dataset = tf.data.Dataset.from_tensor_slices((X_train_ul_f_s_r, y_train_ul_f_s)).shuffle(len(X_train_ul_f_s_r)).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)

    label_count = int(len(set(y_val_f_s)))
    dataset_shape = X_train_f_s_r.shape[1:]

    class_w = sklearn.utils.class_weight.compute_class_weight(class_weight ='balanced',
                                            classes = np.unique(y_train_f_s),
                                            y =y_train_f_s)

    print("L shape")
    print(X_train_f_s_r.shape)
    print("Ul shape")
    print(X_train_ul_f_s_r.shape)
    print("Starting training.")

    filter_list, crop = get_filters(dataset_shape[0])

    discriminator = make_discriminator_model(dataset_shape, label_count )
    generator = make_generator_model(dataset_shape,filter_list,  crop )
    print(discriminator.summary())
    print(generator.summary())
    print("Starting training.")
    train(EPOCHS)
    sys.stdout.close()
    

