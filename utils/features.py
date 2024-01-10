import numpy as np
import matplotlib.pyplot as plt


def filter_horizontal_matches(kp1, kp2, matches):
    '''
    Keeps matches that are horizontally aligned.

    In a Stereo Vision System, we know features need to be horizontally alligned
    since both cameras are identically and just displaces by a baseline.
    '''
    good_matches = []
    for match in matches:
        pt1 = tuple(int(x) for x in kp1[match.queryIdx].pt)
        pt2 = tuple(int(x) for x in kp2[match.trainIdx].pt)
        if np.abs(pt1[1] - pt2[1]) <= 5:
            good_matches.append(match)
    return good_matches


def filter_matches_by_distance_ratio(knn_matches):
    '''
    Ratio test as per Lowe's paper: If the ratio is close to 1, both matches are
    equally good and choosing one would give you an outlier in around 50%. Therefore
    it is usually better to discard both matches.
    https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    '''
    good_matches = []
    for i, knn_match in enumerate(knn_matches):
        m1, m2 = knn_match[0], knn_match[1]
        if m1.distance < 0.7*m2.distance: 
            good_matches.append(m1)
    return good_matches


def filter_best_matches(matches, keep_n_best):
    '''Keep the n best matches (smallest distance) and reject all other matches.'''
    # fig, ax = plt.subplots()
    # ax.hist([m.distance for m in matches], bins=100)
    # plt.show()
    n_matches = min(keep_n_best, len(matches))
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    good_matches = sorted_matches[:n_matches]
    return good_matches