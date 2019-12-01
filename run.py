import json
import subprocess
import statistics


def color_to_index(target_color: str):
    colors = ['Red', 'Orange', 'Yellow', 'Green', 'Purple']
    for i, color in enumerate(colors):
        if color == target_color:
            return i
    return None


if __name__ == '__main__':
    with open('../data/labels.json') as file_handle:
        json_object = json.load(file_handle)
    labels = json_object['labels']
    relative_errors = []
    for instance_id in labels:
        counters = [0] * 5
        output = subprocess.check_output(['./count', instance_id], encoding='UTF-8')
        for line in output.split('\n'):
            if line.startswith('Object'):
                counters[color_to_index(line.split(' ')[-1][1:-2])] += 1

        ground_truth = labels[instance_id]
        error = sum(abs(counter - ground_truth[i]) for i, counter in enumerate(counters))
        relative_error = error / sum(ground_truth)
        relative_errors.append(relative_error)
        error_message = 'Error = {}, R. Error = {:.3f}.'.format(error, relative_error)
        print('{} - {} : {} - {}'.format(instance_id, counters, ground_truth, error_message))
    print('Average relative error: {:.3f}.'.format(statistics.mean(relative_errors)))
