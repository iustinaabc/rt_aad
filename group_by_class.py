import numpy as np

# A helper function to classify given samples into their classes, returns a dictionary with keys 'class_x' and a list
# of corresponding samples as values.
def group_by_class(samples, sample_classes):
    class_one = []
    class_two = []
    for i in range(len(samples)):
        if sample_classes[i] == 1:
            class_one.append(np.transpose(samples[i]))
        else:
            class_two.append(np.transpose(samples[i]))
    class_one = np.array(class_one)
    class_two = np.array(class_two)
    return class_one, class_two