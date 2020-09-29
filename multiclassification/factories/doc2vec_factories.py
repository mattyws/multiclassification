from adapter import Doc2VecTrainer
from data_generators import TaggedNoteeventsDataGenerator
from resources.data_representation import TransformClinicalTextsRepresentations
from resources.functions import print_with_time, whitespace_tokenize_text, train_representation_model
from multiclassification.dag.dag import Node
from multiclassification.dag.resource import Resource
import os


class Doc2VecTrainingNode(Node):

    def prepare(self, resource:Resource):
        pass

    def run(self, resource:Resource):
        print_with_time("Training/Loading representation model")
        embedding_size = resource.parameters['textual_embedding_size']
        min_count = resource.parameters['textual_min_count']
        workers = resource.parameters['textual_workers']
        window = resource.parameters['textual_window']
        iterations = resource.parameters['textual_iterations']
        hs = resource.parameters['textual_doc2vec_hs'],
        dm = resource.parameters['textual_doc2vec_dm'],
        negative = resource.parameters['textual_doc2vec_negative']
        textual_input_shape = (None, embedding_size)
        preprocessing_pipeline = [whitespace_tokenize_text]


        texts_hourly_merged_dir = resource.parameters['multiclassification_base_path'] + "textual_hourly_merged/"
        representation_model_data = [texts_hourly_merged_dir + x for x in os.listdir(texts_hourly_merged_dir)]
        textual_representation_path = os.path.join(resource.parameters['textual_representation_model_path'], str(embedding_size))
        textual_representation_model_path = os.path.join(textual_representation_path,
                                                         resource.parameters['textual_representation_model_filename'])
        if not os.path.exists(textual_representation_path):
            os.makedirs(textual_representation_path)
        noteevents_iterator = TaggedNoteeventsDataGenerator(representation_model_data, preprocessing_pipeline=preprocessing_pipeline)
        model_trainer = Doc2VecTrainer(min_count=min_count, size=embedding_size, workers=workers, window=window, iter=iterations,
                                       hs=hs, dm=dm, negative=negative)

        if os.path.exists(textual_representation_model_path):
            representation_model = model_trainer.load_model(textual_representation_model_path)
        else:
            model_trainer.train(noteevents_iterator)
            model_trainer.save(textual_representation_model_path)
            representation_model = model_trainer.model
        resource.artifacts['doc2vec_model'] = representation_model
        resource.artifacts['textual_input_shape'] = textual_input_shape

class Doc2VecRepresentationTransformNode(Node):

    def prepare(self, resource:Resource):
        pass

    def run(self, resource:Resource):
        print_with_time("Transforming/Retrieving representation")
        embedding_size = resource.parameters['textual_embedding_size']
        window = resource.parameters['textual_window']
        preprocessing_pipeline = [whitespace_tokenize_text]
        problem = resource.problem

        textual_representation_path = os.path.join(resource.parameters['textual_representation_model_path'], str(embedding_size))
        notes_textual_representation_path = os.path.join(textual_representation_path, problem,
                                                         resource.parameters['notes_textual_representation_directory'])
        if not os.path.exists(notes_textual_representation_path):
            os.makedirs(notes_textual_representation_path)
        representation_model = resource.artifacts['doc2vec_model']
        texts_transformer = TransformClinicalTextsRepresentations(representation_model, embedding_size=embedding_size,
                                                                  window=window,
                                                                  representation_save_path=notes_textual_representation_path,
                                                                  is_word2vec=False)
        representation_model = None
        new_paths = texts_transformer.transform(resource.artifacts['data_csv'], 'textual_path', preprocessing_pipeline=preprocessing_pipeline,
                                                remove_temporal_axis=resource.parameters['remove_temporal_axis'],
                                                remove_no_text_constant=resource.parameters['remove_no_text_constant'])
        resource.artifacts['textual_training_data'] = new_paths