import keras
import numpy as np
from keras.utils.vis_utils import plot_model

from model_creators import NoteeventsClassificationModelCreator

data = [
    [
        [
            [1, 2, 3],
            [3, 2, 1],
            [4, 5, 6]
        ],
        [
            [3, 2, 1],
            [1, 2, 3],
            [6, 5, 4]
        ]
    ],
    [
        [
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ],
        [
            [9, 8, 7],
            [12, 11, 10],
            [15, 14, 13]
        ]
    ]
]
classes = [0, 1]

input_shape = (None, None, 3)
model_creator = NoteeventsClassificationModelCreator(input_shape, [64], 1, embedding_size=3,
                                                         loss='binary_crossentropy', layersActivations=["relu"],
                                                         gru=True, use_dropout=True,
                                                         dropout=.5,
                                                         metrics=[keras.metrics.binary_accuracy])
model = model_creator.create().model
print(model.summary())
plot_model(model)
data = np.array(data)
model.fit(data, classes)

