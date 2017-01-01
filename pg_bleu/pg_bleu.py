import model
import numpy as np
import tensorflow as tf
import random
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
SEQ_LENGTH = 20
START_TOKEN = 0

PRE_EPOCH_NUM = 200
TRAIN_ITER = 1  # generator
SEED = 88
BATCH_SIZE = 64

##########################################################################################

TOTAL_BATCH = 800

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


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


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


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def load_references(file):
    ref = []
    with open(file)as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            line = [int(x) for x in line]
            if random.random() < 0.4:
                ref.append(line)
    return ref


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE)
    vocab_size = 5000

    best_score = 9.1
    generator = get_trainable_model(vocab_size)
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer)

    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    ################################################################
    gen_data_loader.create_batches(positive_file)
    references = load_references(positive_file)

    log = open('log/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = str(epoch) + ' ' + str(test_loss) + '\n'
            log.write(buffer)

    generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    buffer = 'After pre-training:' + ' ' + str(test_loss) + '\n'
    log.write(buffer)

    generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    significance_test(sess, target_lstm, likelihood_data_loader, 'significance/supervise.txt')

    rollout = ROLLOUT(generator, references)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'
    log.write('Reinforcement Training...\n')

    for total_batch in range(TOTAL_BATCH):
        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            print 'start calculating BLEU...'
            rewards = rollout.get_reward(sess, samples, 1, (1.0 / 3, 1.0 / 3, 1.0 / 3))
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = str(total_batch) + ' ' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)

            if test_loss < best_score:
                best_score = test_loss
                print 'best score: ', test_loss
                significance_test(sess, target_lstm, likelihood_data_loader, 'significance/pg_bleu.txt')

        rollout.update_params()

    log.close()


if __name__ == '__main__':
    main()
