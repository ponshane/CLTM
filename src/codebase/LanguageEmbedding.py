import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.linear_model import LogisticRegression
import operator
import tensorflow as tf
import time
import re
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from scipy.spatial.distance import cosine

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class LanguageEmbedding:
    """
    need to be written.
    """

    def __init__(self, vectors_file=None):

        """Read in word vectors in gensim format"""
        #read gensim's word2vec model
        self.model = Word2Vec.load(vectors_file)
        self.model.init_sims()

        print('This word embedding matrix has shape: {0}'.format(self.model.wv.syn0norm.shape))


    def export_vocab_count(self, out_vocab_file_location = None):
        chinese_dict = dict()

        for key, value in self.model.wv.vocab.items():
            if value.count > 100:
                chinese_dict[key] = value.count

        # sort dictionary
        sort_chinese_dict = sorted(chinese_dict.items(), key=operator.itemgetter(1), reverse=True)

        new_file = open(out_vocab_file_location, 'w')
        for each_tuple in sort_chinese_dict:
            line = each_tuple[0] +","+ str(each_tuple[1]) + "\n"
            new_file.writelines(line)

        new_file.close()

        print("Your vocabulary count csv is exported to {0}".format(out_vocab_file_location))

    def apply_transform(self, translation_matrix):
        self.transform_embedding = np.dot(self.model.wv.syn0norm, translation_matrix)

    # can be deprecated
    def do_MiniBatchKmeans(self, cluster_size = 100, batch_size = 200, n_init = 10, use_transformed_embedding = False):

        if use_transformed_embedding:
            input_matrix = self.transform_embedding
        else:
            input_matrix = self.model.wv.syn0norm

        print("Your embedding has shape: {0}".format(input_matrix.shape))

        start = time.time() # Start time

        # Set "k" (num_clusters) to be 1/100th of the vocabulary size, or an
        # average of 100 words per cluster

        num_clusters = input_matrix.shape[0] // cluster_size
        print("You will get {0} clusters".format(num_clusters))

        # Initalize a k-means object and use it to extract centroids
        self.kmeans_clustering = MiniBatchKMeans(n_clusters = num_clusters, batch_size=batch_size, n_init = n_init, verbose=True)
        self.cluster_idx = self.kmeans_clustering.fit_predict(input_matrix)

        # Get the end time and print how long the process took
        end = time.time()
        elapsed = end - start
        print("Time taken for MiniBatchKMeans clustering: ", elapsed, "seconds.")

    def find_words_not_in_translation_pair(self, word_vector_list):
        not_in_list = list()
        for word in self.model.wv.vocab.keys():
            if word not in word_vector_list.keys():
                not_in_list.append(word)
        return not_in_list


    @classmethod
    def check_word_in_model(cls, translation_pairs_csv, chinese_model, english_model):

        f = open(translation_pairs_csv)
        top_translation_pairs = dict()

        for line in f.readlines():
            line = line.rstrip("\n").split(",")
            top_translation_pairs[line[0]] = line[2].lower()

        f.close()

        # pre-cleaning for translation pair, there are some chinese terms have the same translation
        real_distinc_translated_pair = dict()
        for chinese_word, english_word in top_translation_pairs.items():
            if chinese_word not in real_distinc_translated_pair.keys() and \
            english_word not in real_distinc_translated_pair.values():

                real_distinc_translated_pair[chinese_word] = english_word

        chinese_word_vector_list = dict()
        english_word_vector_list = dict()

        # check if pairs exist in both language model.
        for chinese_word, english_word in zip(real_distinc_translated_pair.keys(), real_distinc_translated_pair.values()):
            if chinese_word in chinese_model.model.wv.index2word and english_word in english_model.model.wv.index2word:
                chinese_word_vector_list[chinese_word] = chinese_model.model.wv.word_vec(chinese_word, use_norm=True)
                english_word_vector_list[english_word] = english_model.model.wv.word_vec(english_word, use_norm=True)

        assert len(chinese_word_vector_list) == len(english_word_vector_list), "Got different length in wordlist!"
        return((chinese_word_vector_list, english_word_vector_list))

    @classmethod
    def split_evaluation_set(cls, chinese_word_vector_list, englsih_word_vector_list, evaluation_set_ratio = 0.1):
        assert len(chinese_word_vector_list) == len(englsih_word_vector_list), "Got different length in wordlist!"

        total_data_amounts = len(chinese_word_vector_list)
        stop_amounts = int(total_data_amounts*(1 - evaluation_set_ratio))
        print("{0} data will be remained.".format(stop_amounts))

        split_chinese_word_vector_list = dict()
        split_english_word_vector_list = dict()
        training_set = dict()
        test_set = dict()

        index = 0
        for chinses_word, english_word in zip(chinese_word_vector_list.keys(), englsih_word_vector_list.keys()):
            if index <= stop_amounts:
                split_chinese_word_vector_list[chinses_word] = chinese_word_vector_list[chinses_word]
                split_english_word_vector_list[english_word] = englsih_word_vector_list[english_word]

                training_set[chinses_word] = english_word
                index += 1
            else:
                test_set[chinses_word] = english_word

        return(split_chinese_word_vector_list, split_english_word_vector_list, training_set, test_set)

    @classmethod
    def build_pre_mapping_matrix(cls, word_vector_list, num_features=300):
        # Initialize a counter
        counter = 0
        #
        # Preallocate a 2D numpy array, for speed
        FeatureMatrix = np.zeros((len(word_vector_list),num_features),dtype="float32")
        #
        for word, vector in word_vector_list.items():
            FeatureMatrix[counter] = vector
           #
           # Increment the counter
            counter = counter + 1

        return FeatureMatrix

    @classmethod
    def learn_translation_matrix(cls, Source_Feature_Matrix, Target_Feature_Matrix, num_features, early_stop_val):
        W = tf.Variable(tf.random_uniform([num_features, num_features], -1.0, 1.0))

        transform_X = tf.matmul(Source_Feature_Matrix, W)

        # 我們的目標是要讓 loss（MSE）最小化
        loss = tf.reduce_mean(tf.square(transform_X - Target_Feature_Matrix))
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.5)
        train = optimizer.minimize(loss)

        # 初始化
        init = tf.global_variables_initializer()

        # 將神經網絡圖畫出來
        sess = tf.Session()
        sess.run(init)

        for step in range(50001):
            _, loss_val = sess.run([train, loss])
            if step % 10 == 0:
                print("Step {0}, and loss = {1}".format(step, loss_val))
            if loss_val < early_stop_val:
                transform_matrix = sess.run(W)
                break
            if step == 50000:
                print(sess.run(W))
                transform_matrix = sess.run(W)

        # 關閉 Session
        sess.close()

        return(transform_matrix)

    @classmethod
    def evaluate_translation_precision(cls, concatenate_word_vector, test_set, print_match_pair = False, topn = 30):
        test_n = len(test_set)

        precision_at_1 = np.zeros((test_n,), dtype=int)
        precision_at_5 = np.zeros((test_n,), dtype=int)
        precision_at_10 = np.zeros((test_n,), dtype=int)
        precision_at_30 = np.zeros((test_n,), dtype=int)

        # this index indicates how many data has looped.
        index = 0
        for chinese_word, english_word in test_set.items():
            for idx, each_tuple in enumerate(concatenate_word_vector.wv.most_similar(english_word, topn=topn)):

                # if we natch the translation word
                if chinese_word in each_tuple[0]:
                    if print_match_pair:
                        print(idx, english_word, chinese_word)

                    if idx < 1:
                        precision_at_1[index] += 1
                        precision_at_5[index] += 1
                        precision_at_10[index] += 1
                        precision_at_30[index] += 1
                    elif idx < 5 and idx >= 1:
                        precision_at_5[index] += 1
                        precision_at_10[index] += 1
                        precision_at_30[index] += 1
                    elif idx < 10 and idx >= 5:
                        precision_at_10[index] += 1
                        precision_at_30[index] += 1
                    elif idx < 30 and idx >=10:
                        precision_at_30[index] += 1

            index +=1

        print("Precision @1: {:0.3f}, Precision @5: {:0.3f}, Precision @10: {:0.3f}, Precision @30 {:0.3f}".format(np.mean(precision_at_1),np.mean(precision_at_5),np.mean(precision_at_10),np.mean(precision_at_30)))

    # # can be deprecated
    @classmethod
    def merge_clusters_across_languages(cls, chinese_model, english_model, metric="euclidean_distance"):
        '''
        It is possible different chinese clusters map the same english cluster
        '''
        chinese_cluster_id = list()
        english_cluster_id = list()
        distance = list()
        # chinese_kmeans_clustering.cluster_centers_.shape
        # It is arguable we use chinese cluster as index
        if metric == "euclidean_distance":
            for chi_idx in range(chinese_model.kmeans_clustering.cluster_centers_.shape[0]):
                chinese_center = chinese_model.kmeans_clustering.cluster_centers_[chi_idx]
                min_distance = float("inf")
                min_index = None

                for eng_idx in range(english_model.kmeans_clustering.cluster_centers_.shape[0]):
                    english_center = english_model.kmeans_clustering.cluster_centers_[eng_idx]
                    distance_between_cluster = np.linalg.norm(chinese_center - english_center)

                    if distance_between_cluster < min_distance:
                        min_distance = distance_between_cluster
                        min_index = eng_idx

                chinese_cluster_id.append(chi_idx)
                english_cluster_id.append(min_index)
                distance.append(min_distance)

        elif metric == "cosine_similarity":
            for chi_idx in range(chinese_model.kmeans_clustering.cluster_centers_.shape[0]):
                chinese_center = chinese_model.kmeans_clustering.cluster_centers_[chi_idx]
                min_distance = float("inf")
                min_index = None

                for eng_idx in range(english_model.kmeans_clustering.cluster_centers_.shape[0]):
                    english_center = english_model.kmeans_clustering.cluster_centers_[eng_idx]
                    distance_between_cluster = cosine(chinese_center, english_center)

                    if distance_between_cluster < min_distance:
                        min_distance = distance_between_cluster
                        min_index = eng_idx

                chinese_cluster_id.append(chi_idx)
                english_cluster_id.append(min_index)
                distance.append(min_distance)

        cluster_dict = {"chinese_cluster_id": chinese_cluster_id,
           "english_cluster_id": english_cluster_id,
           "distance": distance}


        cluster_df = pd.DataFrame(cluster_dict)

        return(cluster_df)

    # can be deprecated
    @classmethod
    def create_centroid_word_map_dictionary(cls, chinese_model, english_model, select_cluster_df):
        chinese_idx_check_list = []
        english_idx_check_list = []

        concatenate_word_centroid_map = dict()

        chinese_word_centroid_map = dict(zip( chinese_model.model.wv.index2word, chinese_model.cluster_idx ))
        english_word_centroid_map = dict(zip( english_model.model.wv.index2word, english_model.cluster_idx ))

        count = 0
        # https://stackoverflow.com/questions/22219004/grouping-rows-in-list-in-pandas-groupby
        group_df = select_cluster_df.groupby('english_cluster_id')['chinese_cluster_id'].apply(list)

        for english_cluster_id, chineses_id_list in group_df.iteritems():

            words = []
            for english_key, english_idx in english_word_centroid_map.items():
                if(english_idx == english_cluster_id):
                    words.append(english_key)

            for each_chinese_cluster_id in chineses_id_list:
                for chinese_key, chinese_idx in chinese_word_centroid_map.items():
                        if(chinese_idx == each_chinese_cluster_id):
                            words.append(chinese_key)
                chinese_idx_check_list.append(each_chinese_cluster_id)

            english_idx_check_list.append(english_cluster_id)

            concatenate_word_centroid_map[count] = words
            count +=1
            if (count % 10 ==0):
                print("Already grouped {0} cross-lingaul clusters".format(count))

        print("We will return you a {0} size cross-language dictionary".format(len(concatenate_word_centroid_map)))
        print("The cross-language dictionary is composed by {0} english clusters and {1} chinese clusters".format(len(english_idx_check_list), len(chinese_idx_check_list)))
        print()
        print("Now add non-group chinese clusters")
        chinese_start_count = count
        chinese_cluster_size = chinese_model.kmeans_clustering.cluster_centers_.shape[0]
        for idx in range(chinese_cluster_size):
            words = []
            if idx not in chinese_idx_check_list:
                for chinese_key, chinese_idx in chinese_word_centroid_map.items():
                        if(chinese_idx == idx):
                            words.append(chinese_key)

                chinese_idx_check_list.append(idx)
                concatenate_word_centroid_map[count] = words
                count +=1

            #if (count % 10 ==0):
            #    print("Already added {0} single chinese clusters".format(count - chinese_start_count))
        print("Already added {0} single chinese clusters".format(count - chinese_start_count))
        print("Now add non-group english clusters")
        english_start_count = count
        english_cluster_size = english_model.kmeans_clustering.cluster_centers_.shape[0]
        for idx in range(english_cluster_size):
            words = []
            if idx not in english_idx_check_list:
                for english_key, english_idx in english_word_centroid_map.items():
                    if(english_idx == idx):
                        words.append(english_key)

                english_idx_check_list.append(idx)
                concatenate_word_centroid_map[count] = words
                count +=1

            #if (count % 10 ==0):
            #    print("Already added {0} single english clusters".format(count - english_start_count))

        print("Already added {0} single english clusters".format(count - english_start_count))
        print("Totally we got {0} size cluster dictionary.".format(count))
        return(concatenate_word_centroid_map)

    @classmethod
    def export_to_KeyedVectors(cls, source_model, target_model, file_path):
        print("transform_source_model.shape = ", source_model.transform_embedding.shape)
        print("target_model.shape = ", target_model.model.wv.syn0norm.shape)

        two_languages_index2word = source_model.model.wv.index2word + target_model.model.wv.index2word
        normalised_transform_embedding = LanguageEmbedding.normalised(source_model.transform_embedding)
        print("Normalization is finished!")
        tw_languages_word_vector = np.concatenate((normalised_transform_embedding,\
                                              target_model.model.wv.syn0norm), axis=0)
        print("concatenate_model.shape = ", tw_languages_word_vector.shape)

        out = open(file_path,'w')
        out.write(str(tw_languages_word_vector.shape[0]) + " " + str(tw_languages_word_vector.shape[1]))
        out.write("\n")
        for each_word in two_languages_index2word:
            out.write(each_word + " ")
            out.write(' '.join(map(str, tw_languages_word_vector[two_languages_index2word.index(each_word)])) + "\n")
        out.close()

        print("KeyedVectors have exported to", file_path)

    @classmethod
    def read_from_KeyedVectors(cls, file_path):
        word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=False)  # C text format
        return(word_vectors)

    @classmethod
    def normalised(cls, mat, axis=-1, order=2):
        """Utility function to normalise the rows of a numpy array."""
        norm = np.linalg.norm(
            mat, axis=axis, ord=order, keepdims=True)
        norm[norm == 0] = 1
        return mat / norm

    @classmethod
    def check_vocabulary_type_of_space(cls, embedding_space):
        """
        this function is used for checking whether there are
        words not belong to english use encode to test

        Args:
            cls: just classmethod required
            embedding_space: a word2vec object created by gensim

        Returns:
            en_num (int): # of english words in the space
            chi_num (int): # of chinese/second language words in the space
            error_word_idx (list): a list of word_id for words which
                do not belong english or chinese
        """

        en_num = 0
        chi_num = 0
        error_word_idx = []

        for word in embedding_space.vocab.keys():
            try:
                # 可以解碼為 ascii 的為英文單字
                word.encode(encoding='ascii')
                en_num +=1
            except UnicodeEncodeError:
                # 為什麼還有一正規判斷?
                chi_num +=1
                ''' this section only designs for Chinese characters
                if re.search(u'[\u4e00-\u9fff]', word):
                    chi_num +=1
                else:
                    # there are some words contains emphasized accents
                    print(word)
                    error_word_idx.append(embedding_space.vocab[word].index)
                    continue
                '''

        return en_num, chi_num, error_word_idx

    @classmethod
    def estimate_coefficient_of_language_dimension(cls, embedding_space, en_num,
     chi_num, classifier="logistic"):
        """
        this function is used to train a classifier for dsicriminating language.

        Args:
            cls: just classmethod required
            embedding_space: a word2vec object created by gensim
            en_num (int): # of english words in the space
            chi_num (int): # of chinese words in the space

        Returns:
            classifier.coef_ (list): list of dimensions' coefficient which
            estimated by classfier
        """
        samples = embedding_space.wv.vectors
        target = [0] * en_num + [1] * chi_num
        assert samples.shape[0] == len(target)

        if classifier == "logistic":
            classifier = LogisticRegression()  # default value
            classifier.fit(samples, target)  # do not return

            x = classifier.predict(samples)

            true_list = []
            for idx in range(0, len(target)):
                if x[idx] == target[idx]:
                    true_list.append(True)

            print("Precision of Language dimension::", len(true_list) / len(target))
            return classifier.coef_

        elif classifier == "xgb":
            classifier = XGBClassifier()  # default value
            classifier.fit(samples, target)  # do not return

            x = classifier.predict(samples)
            predictions = [round(value) for value in x]
            accuracy = accuracy_score(target, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            return classifier.feature_importances_

    @classmethod
    def mask_language_dim_and_export_to_KeyedVectors(cls, coef_list, topRemovedN,
     id2word, input_matrix, file_path, classifier="logistic"):
        """
        this function is used to mask dimensions which are most contributed
        in language classifier and re-generate mask word space.

        Args:
            cls: just classmethod required
            coef_list (list): list of dimensions' coefficient which
                estimated by classfier
            topRemovedN (int): # of dimensions we want to mask, output shape
                will be masked by 2*topRemovedN
            id2word (list): list of words
            input_matrix (numpy array): word space which row_id correspond to
                index in id2word
            file_path (str): the location of exported word space

        Returns:
            None but export files
        """
        if classifier == "logistic":
            removal_dimension_list = list(np.argsort(coef_list)[0][:topRemovedN]) + list(np.argsort(coef_list)[0][-topRemovedN:])
        elif classifier == "xgb":
            removal_dimension_list = list(np.argsort(coef_list)[(-2)*topRemovedN:])

        matrix = np.delete(input_matrix, removal_dimension_list, 1)

        out = open(file_path,'w')
        out.write(str(matrix.shape[0]) + " " + str(matrix.shape[1]))
        out.write("\n")

        for idx, val in enumerate(id2word):
            out.write(val + " ")
            out.write(' '.join(map(str, matrix[idx])) + "\n")
        out.close()

        print("KeyedVectors have exported to", file_path)
    
    @classmethod
    def mask_language_dim_and_export_to_CLTM(cls, coef_list, topRemovedN,
     id2word, input_matrix, file_path, classifier="logistic"):
        """
        this function is used to mask dimensions which are most contributed
        in language classifier and re-generate mask word space.

        Args:
            cls: just classmethod required
            coef_list (list): list of dimensions' coefficient which
                estimated by classfier
            topRemovedN (int): # of dimensions we want to mask, output shape
                will be masked by 2*topRemovedN
            id2word (list): list of words
            input_matrix (numpy array): word space which row_id correspond to
                index in id2word
            file_path (str): the location of exported word space

        Returns:
            None but export files
        """
        if classifier == "logistic":
            removal_dimension_list = list(np.argsort(coef_list)[0][:topRemovedN]) + list(np.argsort(coef_list)[0][-topRemovedN:])
        elif classifier == "xgb":
            removal_dimension_list = list(np.argsort(coef_list)[(-2)*topRemovedN:])

        matrix = np.delete(input_matrix, removal_dimension_list, 1)

        out = open(file_path,'w')

        for idx, val in enumerate(id2word):
            out.write(val + " ")
            out.write(' '.join(map(str, matrix[idx])) + "\n")
        out.close()

        print("KeyedVectors have exported to", file_path)