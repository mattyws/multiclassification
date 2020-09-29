import enum

from resources import functions
from resources.data_generators import LengthLongitudinalDataGenerator


class GeneratorEnum(enum.Enum):
    array_dg = 1
    bert_dg = 2
    length_longitudinal_dg = 3
    mixed_length_dg = 4


class GeneratorFactory():

    def create(self, type:GeneratorEnum, data, classes, batch_size, params:dict=None):
        functions.print_with_time("Creating generator")
        if type == GeneratorEnum.length_longitudinal_dg:
            self.__create_length_longitudinal_generator(data, classes, batch_size, params=params)


    def __create_length_longitudinal_generator(self, data, classes, batch_size, params:dict=None):
        events_sizes_file_path = None
        events_sizes_labels_file_path = None
        if params is not None:
            if 'sizes_path' in params.keys():
                events_sizes_file_path = params['sizes_path']
            if 'sizes_labels_path' in params.keys():
                events_sizes_labels_file_path = params['sizes_labels_path']
            if events_sizes_file_path is None or events_sizes_labels_file_path is None:
                events_sizes_file_path = None
                events_sizes_labels_file_path = None
        train_sizes, train_labels = functions.divide_by_events_lenght(data, classes,
                                                                      sizes_filename=events_sizes_file_path,
                                                                      classes_filename=events_sizes_labels_file_path)
        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                             max_batch_size=batch_size)
        dataTrainGenerator.create_batches()
        return dataTrainGenerator