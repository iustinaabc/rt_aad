import numpy as np
import math

# A helper function to classify given samples into their classes, returns a dictionary with keys 'class_x' and a list
# of corresponding samples as values.
def group_by_class(samples, sample_classes, timefr):
    class_one = []
    class_two = []
    for i in range(len(samples)):
        if sample_classes[math.floor((i*timefr)/60)] == 1:
            class_one.append(np.transpose(samples[i]))
        else:
            class_two.append(np.transpose(samples[i]))
    class_one = np.array(class_one)
    class_two = np.array(class_two)
    return class_one, class_two
