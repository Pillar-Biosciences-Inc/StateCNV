import numpy as np

def _extract_ratio_list(
    list_of_contiguous_segments_sample,
    list_of_contiguous_segments_normal,
    control_sample,
    control_normal,
):
    """ """
    n_segments = len(list_of_contiguous_segments_sample)
    ratio_list = []
    upweight = np.nanmean(control_sample / control_normal)

    for i in range(n_segments):
        myRatio = list_of_contiguous_segments_sample[i] / ( 
            list_of_contiguous_segments_normal[i] + 1 
        )   
        ratio_list.append(myRatio / upweight)
    return ratio_list
