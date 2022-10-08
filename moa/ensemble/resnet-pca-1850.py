import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras import layers,regularizers,Sequential,backend,callbacks,optimizers,metrics,losses
from keras.models import Model
import tensorflow as tf
import sys
# sys.path.append('../input/iterative-stratification/iterative-stratification-master')
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from tensorflow_addons.optimizers import AdamW
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3

# Import train data, drop sig_id, cp_type

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
non_ctl_idx = train_features.loc[train_features['cp_type']!='ctl_vehicle'].index.to_list()
train_features = train_features.drop(['sig_id'],axis=1)
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets_scored = train_targets_scored.drop('sig_id',axis=1)
labels_train = train_targets_scored.values

# Drop training data with ctl vehicle

train_features = train_features.iloc[non_ctl_idx].reset_index(drop=True)
labels_train = labels_train[non_ctl_idx]

# Import test data

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
test_features = test_features.drop('sig_id',axis=1)

# Label Encoder for categorical cp_dose
def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 1, 'D2': 2})
    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 1, 48: 2, 72: 3})
    return df

train_features = preprocess(train_features)
test_features = preprocess(test_features)

aa = 'g-0 g-7 g-8 g-10 g-13 g-17 g-20 g-22 g-24 g-26 g-28 g-29 g-30 g-31 g-32 g-34 g-35 g-36 g-37 g-38 g-39 g-41 g-46 g-48 g-50 g-51 g-52 g-55 g-58 g-59 g-61 g-62 g-63 g-65 g-66 g-67 g-68 g-70 g-72 g-74 g-75 g-79 g-83 g-84 g-85 g-86 g-90 g-91 g-94 g-95 g-96 g-97 g-98 g-100 g-102 g-105 g-106 g-112 g-113 g-114 g-116 g-121 g-123 g-126 g-128 g-131 g-132 g-134 g-135 g-138 g-139 g-140 g-142 g-144 g-145 g-146 g-147 g-148 g-152 g-155 g-157 g-158 g-160 g-163 g-164 g-165 g-170 g-173 g-174 g-175 g-177 g-178 g-181 g-183 g-185 g-186 g-189 g-192 g-194 g-195 g-196 g-197 g-199 g-201 g-202 g-206 g-208 g-210 g-213 g-214 g-215 g-220 g-226 g-228 g-229 g-235 g-238 g-241 g-242 g-243 g-244 g-245 g-248 g-250 g-251 g-254 g-257 g-259 g-261 g-266 g-270 g-271 g-272 g-275 g-278 g-282 g-287 g-288 g-289 g-291 g-293 g-294 g-297 g-298 g-301 g-303 g-304 g-306 g-308 g-309 g-310 g-311 g-314 g-315 g-316 g-317 g-320 g-321 g-322 g-327 g-328 g-329 g-332 g-334 g-335 g-336 g-337 g-339 g-342 g-344 g-349 g-350 g-351 g-353 g-354 g-355 g-357 g-359 g-360 g-364 g-365 g-366 g-367 g-368 g-369 g-374 g-375 g-377 g-379 g-385 g-386 g-390 g-392 g-393 g-400 g-402 g-406 g-407 g-409 g-410 g-411 g-414 g-417 g-418 g-421 g-423 g-424 g-427 g-429 g-431 g-432 g-433 g-434 g-437 g-439 g-440 g-443 g-449 g-458 g-459 g-460 g-461 g-464 g-467 g-468 g-470 g-473 g-477 g-478 g-479 g-484 g-485 g-486 g-488 g-489 g-491 g-494 g-496 g-498 g-500 g-503 g-504 g-506 g-508 g-509 g-512 g-522 g-529 g-531 g-534 g-539 g-541 g-546 g-551 g-553 g-554 g-559 g-561 g-562 g-565 g-568 g-569 g-574 g-577 g-578 g-586 g-588 g-590 g-594 g-595 g-596 g-597 g-599 g-600 g-603 g-607 g-615 g-618 g-619 g-620 g-625 g-628 g-629 g-632 g-634 g-635 g-636 g-638 g-639 g-641 g-643 g-644 g-645 g-646 g-647 g-648 g-663 g-664 g-665 g-668 g-669 g-670 g-671 g-672 g-673 g-674 g-677 g-678 g-680 g-683 g-689 g-691 g-693 g-695 g-701 g-702 g-703 g-704 g-705 g-706 g-708 g-711 g-712 g-720 g-721 g-723 g-724 g-726 g-728 g-731 g-733 g-738 g-739 g-742 g-743 g-744 g-745 g-749 g-750 g-752 g-760 g-761 g-764 g-766 g-768 g-770 g-771 c-0 c-1 c-2 c-3 c-4 c-5 c-6 c-7 c-8 c-9 c-10 c-11 c-12 c-13 c-14 c-15 c-16 c-17 c-18 c-19 c-20 c-21 c-22 c-23 c-24 c-25 c-26 c-27 c-28 c-29 c-30 c-31 c-32 c-33 c-34 c-35 c-36 c-37 c-38 c-39 c-40 c-41 c-42 c-43 c-44 c-45 c-46 c-47 c-48 c-49 c-50 c-51 c-52 c-53 c-54 c-55 c-56 c-57 c-58 c-59 c-60 c-61 c-62 c-63 c-64 c-65 c-66 c-67 c-68 c-69 c-70 c-71 c-72 c-73 c-74 c-75 c-76 c-77 c-78 c-79 c-80 c-81 c-82 c-83 c-84 c-85 c-86 c-87 c-88 c-89 c-90 c-91 c-92 c-93 c-94 c-95 c-96 c-97 c-98 c-99'
tops2 = aa.split(' ')

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

g_feats = [col for col in train_features.columns if col.startswith('g-')]
c_feats = [col for col in train_features.columns if col.startswith('c-')]
g_feats2 = tops2[:347]
c_feats2 = tops2[347:]


# Function to extract pca features
def fe_pca(train, test, n_components_g=60, n_components_c=10, SEED=123, inputs='input1'):
    if inputs == 'input1':
        features_g = g_feats
        features_c = c_feats
    if inputs == 'input2':
        features_g = g_feats2
        features_c = c_feats2

    def create_pca(train, test, features, kind='g', n_components=n_components_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        pca = PCA(n_components=n_components, random_state=SEED)
        data = pca.fit_transform(data)
        print('Explained variance for PCA', pca.explained_variance_ratio_.sum())
        columns = [f'pca_{kind}{i + 1}' for i in range(n_components)]
        data = pd.DataFrame(data, columns=columns)
        train_ = data.iloc[:train.shape[0]]
        test_ = data.iloc[train.shape[0]:].reset_index(drop=True)
        train = pd.concat([train, train_], axis=1)
        test = pd.concat([test, test_], axis=1)
        return train, test

    train, test = create_pca(train, test, features_g, kind='g', n_components=n_components_g)
    train, test = create_pca(train, test, features_c, kind='c', n_components=n_components_c)
    return train, test


# Function to extract kmeans features
def fe_cluster(train, test, n_clusters_g=35, n_clusters_c=5, SEED=123):
    features_g = g_feats
    features_c = c_feats

    def create_cluster(train, test, features, kind='g', n_clusters=n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns=[f'clusters_{kind}'])
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(train, test, features_g, kind='g', n_clusters=n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind='c', n_clusters=n_clusters_c)
    return train, test


# Function to extract common stats features
def fe_stats(train, test, inputs='input1'):
    if inputs == 'input1':
        features_g = list(train.columns[4:776])
        features_c = list(train.columns[776:876])
    if inputs == 'input2':
        features_g = g_feats2
        features_c = c_feats2

    for df in train, test:
        # df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis=1)
        # df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis=1)
        df['g_skew'] = df[features_g].skew(axis=1)
        # df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis=1)
        # df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis=1)
        df['c_skew'] = df[features_c].skew(axis=1)
        '''
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        '''
    return train, test


def c_squared(train, test):
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f'{feature}_squared'] = df[feature] ** 2
    return train, test

print('train_features.shape: ', train_features.shape)

train_features, test_features = fe_pca(train_features, test_features, n_components_g = 70, n_components_c = 30, SEED = 42)
# train_features, test_features = fe_cluster(train_features, test_features, n_clusters_g = 35, n_clusters_c = 5, SEED = 42)
train_features, test_features = fe_stats(train_features, test_features)
train_features, test_features = c_squared(train_features, test_features)

train_features2 = train_features.loc[:,tops2]
test_features2 = test_features.loc[:,tops2]
train_features2, test_features2 = fe_pca(train_features2, test_features2, n_components_g = 200, n_components_c = 30, SEED = 42,inputs='input2')
train_features2, test_features2 = fe_stats(train_features2, test_features2, inputs='input2')

from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(0.8)  #<-- Update
data = train_features.append(test_features)
print(data.shape)
data_transformed = var_thresh.fit_transform(data.iloc[:, 876:]) # data.iloc[:, 876:]

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]

top_feats = [  1,   2,   3,   4,   5,   6,   7,   9,  11,  14,  15,  16,  17,
        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,
        32,  33,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  46,
        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58,  59,  60,
        61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
        74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,
        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
       115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128,
       129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143,
       144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,
       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197,
       198, 199, 200, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212,
       213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226,
       227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
       240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
       254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
       267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294,
       295, 296, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
       310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
       337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
       350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
       363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,
       378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391,
       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
       405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418,
       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
       432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446,
       447, 448, 449, 450, 453, 454, 456, 457, 458, 459, 460, 461, 462,
       463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,
       476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489,
       490, 491, 492, 493, 494, 495, 496, 498, 500, 501, 502, 503, 505,
       506, 507, 509, 510, 511, 512, 513, 514, 515, 518, 519, 520, 521,
       522, 523, 524, 525, 526, 527, 528, 530, 531, 532, 534, 535, 536,
       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 549, 550, 551,
       552, 554, 557, 559, 560, 561, 562, 565, 566, 567, 568, 569, 570,
       571, 572, 573, 574, 575, 577, 578, 580, 581, 582, 583, 584, 585,
       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599,
       600, 601, 602, 606, 607, 608, 609, 611, 612, 613, 615, 616, 617,
       618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,
       631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 644,
       645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 658, 659,
       660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
       673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
       686, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 699, 700,
       701, 702, 704, 705, 707, 708, 709, 710, 711, 713, 714, 716, 717,
       718, 720, 721, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,
       733, 734, 735, 737, 738, 739, 740, 742, 743, 744, 745, 746, 747,
       748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761,
       762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774,
       775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788,
       789, 790, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803,
       804, 805, 806, 808, 809, 811, 813, 814, 815, 816, 817, 818, 819,
       821, 822, 823, 825, 826, 827, 828, 829, 830, 831, 832, 834, 835,
       837, 838, 839, 840, 841, 842, 845, 846, 847, 848, 850, 851, 852,
       854, 855, 856, 858, 859, 860, 861, 862, 864, 866, 867, 868, 869,
       870, 871, 872, 873, 874]
print(len(top_feats))

train_features = train_features.iloc[:,top_feats]
# train_features = train_features.iloc[:,[1,2]]

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
print('train shape is:', train_features.shape)

test_features = test_features.iloc[:,top_feats]
# test_features = test_features.iloc[:,[1,2]]
test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)
print('test shape is:', test_features.shape)

# Min Max Scaler for numerical values

# Fit scaler to joint train and test data
# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(train_features.append(test_features))

# Scale train data
data_train = scaler.transform(train_features)

# Scale test data
data_test = scaler.transform(test_features)

# Scale extract train data
# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(train_features2.append(test_features2))
data_train2 = scaler.transform(train_features2)
data_test2 = scaler.transform(test_features2)


def create_model(ncol_X, ncol_Y, ncol_secondX):
    input1 = layers.Input(shape=(ncol_X,))
    input2 = layers.Input(shape=(ncol_secondX,))

    # Define NN Model
    x = layers.BatchNormalization()(input1)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='elu')(x)
    x = layers.BatchNormalization()(x)
    output1 = layers.Dense(256, activation="elu")(x)

    x = layers.concatenate([output1, input2], axis=1)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="elu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    ans1 = layers.Dense(256, activation="elu")(x)

    x = layers.Average()([output1, ans1])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(ncol_Y, kernel_initializer='lecun_normal', activation='selu')(x)
    x = layers.BatchNormalization()(x)
    ans2 = layers.Dense(ncol_Y, activation="sigmoid")(x)

    model = Model(inputs=[input1, input2], outputs=ans2)
    optimizer = AdamW(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(label_smoothing=0.0005), metrics=logloss)
    return model


n_labels = train_targets_scored.shape[1]
n_features = data_train.shape[1]
n_features2 = data_train2.shape[1]
# n_features = 2+350+50
n_train = data_train.shape[0]
n_test = data_test.shape[0]

# Prediction Clipping Thresholds

p_min = 0.0005  # 0.001
p_max = 0.9995


# Evaluation Metric with clipping and no label smoothing

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, p_min, p_max)
    return -backend.mean(y_true * backend.log(y_pred) + (1 - y_true) * backend.log(1 - y_pred))


# Generate Seeds

n_seeds = 6

## I seed the np rng with a fixed seed and then use it to generate the random seeds for the MSKF.
## Keep the same seed for different models that you plan to ensemble/blend so that their OOF performance is comparable.
## Verify that the array "seeds" has unique integers. For eg. if np.random is seeded with 0, n_seeds = 6 results in 5 unique seeds.
np.random.seed(1)
seeds = np.random.randint(0, 100, size=n_seeds)

# Training Loop

n_folds = 10
y_pred = np.zeros((n_test, n_labels))
oof = tf.constant(0.0)
val_smooth_loss = tf.constant(0.0)
hists = []
for seed in seeds:
    fold = 0
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in mskf.split(data_train, labels_train):
        print(fold)
        X_train = data_train[train]
        X_test = data_train[test]
        y_train = labels_train[train]
        y_test = labels_train[test]
        X_train2 = data_train2[train]
        X_test2 = data_train2[test]

        # Define NN Model
        model = create_model(n_features, n_labels, n_features2)
        # ocp = one_cycle_scheduler.OneCycleScheduler(verbose=0, **scheduler_params)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_logloss', factor=0.3, patience=4, mode='min', min_lr=1E-5)
        # patience 5
        early_stopping = callbacks.EarlyStopping(monitor='val_logloss', min_delta=1E-5, patience=10, mode='min',
                                                 restore_best_weights=True)
        # patience 15
        hist = model.fit((X_train, X_train2), y_train, batch_size=128, epochs=50, verbose=0,
                         validation_data=((X_test, X_test2), y_test), callbacks=[reduce_lr, early_stopping])
        # 192 epochs
        hists.append(hist)
        print("train", model.evaluate((X_train, X_train2), y_train, verbose=0, batch_size=128))
        print("val", model.evaluate((X_test, X_test2), y_test, verbose=0, batch_size=128))
        val_smooth_loss += model.evaluate((X_test, X_test2), y_test, verbose=0, batch_size=128)[0] / (n_folds * n_seeds)
        # Save Model
        model.save('LabelSmoothed_seed_' + str(seed) + '_fold_' + str(fold))

        # OOF Score
        y_val = model.predict((X_test, X_test2))
        oof += logloss(tf.constant(y_test, dtype=tf.float32), tf.constant(y_val, dtype=tf.float32)) / (
                    n_folds * n_seeds)

        # Run prediction
        y_pred += model.predict((data_test, data_test2)) / (n_folds * n_seeds)

        fold += 1

# Analysis of Training

tf.print('OOF score is ', oof)
tf.print('val smooth loss score is ', val_smooth_loss)
print(model.summary())
print(model.count_params())

# Generate submission file, Clip Predictions
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
test_features = test_features.drop('sig_id',axis=1)
sub2 = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
sub2.iloc[:,1:] = np.clip(y_pred,p_min,p_max)

# Set ctl_vehicle to 0
sub2.iloc[test_features['cp_type'] == 'ctl_vehicle',1:] = 0

# Save Submission
sub2.to_csv('submission2.csv', index=False)