import model
import numpy as np
import tensorflow as tf
import random
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from target_lstm import TARGET_LSTM

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
SEQ_LENGTH = 20
START_TOKEN = 0

EPOCH_NUM = 300
SEED = 88
BATCH_SIZE = 64

positive_file = 'target_generate/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'

generated_num = 10000


##############################################################################################


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    #  Generated Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch(sess, trainable_model, data_loader, curriculum_rate):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch, curriculum_rate)
        supervised_g_losses.append(g_loss)

    print '>>>> generator train loss:', np.mean(supervised_g_losses)
    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE)
    vocab_size = 5000

    best_score = 9.5

    generator = get_trainable_model(vocab_size)
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer)

    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('log/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start scheduled sampling training...'
    log.write('scheduled sampling training...\n')
    curriculum_rate = 1.0
    for epoch in xrange(EPOCH_NUM):
        curriculum_rate = max(0.0, curriculum_rate - 0.002)
        loss = pre_train_epoch(sess, generator, gen_data_loader, curriculum_rate)
        generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
        likelihood_data_loader.create_batches(eval_file)
        test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        print 'pre-train epoch ', epoch, 'curriculum rate:', curriculum_rate, 'test_loss ', test_loss
        buffer = str(epoch) + ' ' + str(curriculum_rate) + ' ' + str(test_loss) + '\n'
        log.write(buffer)

        if test_loss < best_score:
            best_score = test_loss
            print 'best score: ', test_loss
            generate_samples(sess, generator, BATCH_SIZE, 100000, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            significance_test(sess, target_lstm, likelihood_data_loader, 'significance/schedule_sampling.txt')

    log.close()


if __name__ == '__main__':
    main()
