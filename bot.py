import requests
import json
url='https://hooks.slack.com/services/secret_key'



def keeping_track_count(count_no,accuracy_count,loss_count,epoch_count):
    payload = {
    "attachments": [
        {
            "title": "dummy_data",

            "text" : 'dummy_data',


    },{
            "title": "dummy_data",

            "text" : 'dummy_data',

    }]}
    payload["attachments"][0]["title"] = "epoch" + "  "+ str(epoch_count)
    payload["attachments"][1]["title"] = "iteration_no" + "  "+ str(count_no)

    payload["attachments"][0]["text"] = "loss" + "  " + str(loss_count)
    payload["attachments"][1]["text"] = "accuracy" + " " + str(accuracy_count)





    r = requests.post(url, data=json.dumps(payload))


def rand_exe(model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in tqdm(range(iteration)):
                # batch = vectors[j * batch_size:(j + 1) * batch_size]
                # tweets_data = np.array(padding_data([aa for aa, bb in batch])['input'])
                # labels_data = np.array([labels_datr.index(bb) for aa, bb in batch])

                labess, tweetsr = getTrainBatch()

                out_a = sess.run(model.out, feed_dict={model.placeholder['input']: labess,
                                                       model.placeholder['output']: tweetsr})

                keeping_track_count(j,out_a['accuracy'],out_a['loss'],i)




        saver.save(sess, '/home/ayodhyankit/sentimnt_aadi/training_data/')

