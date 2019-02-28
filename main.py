import os
import math
import numpy
import tensorflow
from tensorflow.contrib import lite
import json
import re
import matplotlib.pyplot as plot
import requests
import time


class Analyser(object):

    CURRENT_DIR = os.path.dirname(__file__) + '/'
    REEL_AUTO_ARCHIVE_CATEGORY = ["unset", "on", "off"]
    RELATIONSHIP_CATEGORY = ["none", "mutual", "fans", "unrequited"]

    USERNAME_MATCHING_1 = re.compile(r"([a-z]+)(\.|_)([a-z0-9]+)")
    USERNAME_MATCHING_2 = re.compile(r"([a-z]+)(\.|_)?([0-9]+)")
    USERNAME_MATCHING_3 = re.compile(r"[a-z]+")

    def __init__(self):
        if Analyser.CURRENT_DIR is '/':
            Analyser.CURRENT_DIR = ''

        return None

    def load_data(self, file_name):
        item_data = None

        with open(self.CURRENT_DIR + "data/" + file_name, "r") as file:
            item_data = json.load(file)

        return item_data

    def save_data(self, file_name, data):
        with open(self.CURRENT_DIR + "data/" + file_name, "w") as file:
            json.dump(data, file)

        return True

    def vectorize(self, data_list):
        """
        - Make sure to test with ghost followers -- they're more likely to be 
        """

        features = []
        response = []

        for key in data_list:
            profile = data_list[key]

            if 'username' not in profile:
                print 'Username not found -- ' + key
                continue

            is_valid = True
            if 'is_authentic' in profile:
                is_valid = profile['is_authentic']
                del profile['is_authentic']

            features.append([
                self.encode_username(profile['username'].lower()),
                len(re.split(r"_|\.| ", profile['username'])),
                len(profile['full_name']),
                len(re.split(r"_|\.| ", profile['full_name'])),
                profile['full_name'].count('?'),
                len(re.split(r"_|\.| ", profile['biography'])),
                profile['biography'].count('?'),
                profile['follower_count'] / (profile['following_count'] * 1.0) if profile['following_count'] > 0 else 0,
                profile['follower_count'],
                profile['following_count'],
                profile['media_count'],
                profile['usertags_count'],
                profile['is_private'],
                profile['is_verified'],
                profile['has_anonymous_profile_picture'],
                Analyser.REEL_AUTO_ARCHIVE_CATEGORY.index(profile['reel_auto_archive']) # imp
            ])
            response.append(is_valid)

        return {
            'features': features,
            'response': response
        }

    def export_model(self, model, output_name):
        model.save(output_name)

    def encode_username(self, username):
        if Analyser.USERNAME_MATCHING_1.match(username):
            return 1
        if Analyser.USERNAME_MATCHING_2.match(username):
            return 2
        if Analyser.USERNAME_MATCHING_3.match(username):
           return 3

        return 0

    def segment_data(self, data_list):
        print 'Segmenting data...'

        # Determine the split index for the training and test data.
        segment_index = int(math.floor(0.8 * len(data_list)))

        training_data = numpy.asarray(data_list[0:segment_index])
        test_data = numpy.asarray(data_list[segment_index:])

        print "Training set: {}".format(training_data.shape)
        print "Test set: {}".format(test_data.shape)

        return [training_data, test_data]

    def build_model(self, training_data):
        model = tensorflow.keras.Sequential([
            # Select 16 units for the input layer to encapsulate all the inputs.
            tensorflow.keras.layers.Dense(16,
                                          activation=tensorflow.nn.relu,
                                          input_shape=(training_data.shape[1],)),
            # Map to a single unit and normalize to constrain our output to a binary value.
            tensorflow.keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)
        ])

        # Since this is a binary classification problem and the model outputs a
        # probability we'll use the binary_crossentropy loss function.
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        return model

    def normalize(self, predictions):
        normalized = []

        for prediction in predictions:
            if prediction > 1:
                normalized.append(1)
            elif prediction < 0:
                normalized.append(0)
            elif prediction > 0.5:
                normalized.append(1)
            else:
                normalized.append(0)

        return normalized


if __name__ == '__main__':
    # Create an instane of the analyser
    instance = Analyser()

    follower_data = instance.load_data(file_name='classified_profiles.json')
    follower_data = instance.vectorize(data_list=follower_data)

    (training_data, test_data) = instance.segment_data(data_list=follower_data['features'])
    training_data_size = len(training_data)

    print 'Found training data -- ' + str(training_data_size)

    model = instance.build_model(training_data=training_data)
    model.summary()

    history = model.fit(training_data,
                        follower_data['response'][0:training_data_size],
                        epochs=200)

    instance.export_model(model=model, output_name="model.h5")

    [loss, acc] = model.evaluate(test_data,
                                 follower_data['response'][training_data_size:],
                                 verbose=0)

    print "Testing set accuracy: {:.2f}".format(acc)

    responses = numpy.asarray(follower_data['response'][training_data_size:])
    test_predictions = instance.normalize(model.predict(test_data).flatten())

    print 'Done'

    total = len(responses)
    tt = 0.0
    tf = 0.0
    ft = 0.0
    ff = 0.0

    for i in range(0, total):
        j = responses[i]
        k = test_predictions[i]

        if (j == 1 and k == 1):
            tt = tt + 1.0
        elif (j == 1 and k == 0):
            tf = tf + 1.0
        elif (j == 0 and k == 1):
            ft = ft + 1.0
        elif (j == 0 and k == 0):
            ff = ff + 1.0

    print ''
    print 'Confusion matrix for profile authenticity (Y = authentic).'
    print '         Pred.               |   Total'
    print '           Y        N        |'
    print "Act  Y   {:.2f}  |   {:.2f}  |   {:.2f}".format(tt, tf, tt + tf)
    print "     N   {:.2f}  |   {:.2f}  |   {:.2f}".format(ft, ff, ft + ff)
    print "-----------------------------------"
    print "Total    {:.2f} | {:.2f} |  {:.2f}".format(tt+ft, tf+ff, total)

    print ''
    print 'Accuracy: {}'.format((tt + ff) / total)
    print 'True positive: {}'.format(tt/(tt + tf))
    print 'False positive: {}'.format(tf/(tt + tf))
    print 'Precision (Y): {}'.format(tt/(tt + ft))
    print 'Precision (N): {}'.format(ff/(tf + ff))
