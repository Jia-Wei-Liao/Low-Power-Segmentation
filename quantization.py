import os
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path


def representative_dataset():
    for data in Path(qualification_path).glob('*.jpg'):
        img = cv2.imread(str(data))
        img = np.expand_dims(img,0)
        img = img.astype(np.float32)
        yield [img]


if __name__ == '__main__':
  input_path = 'model_weight.pd'
  qualification_path = 'dataset_test/'
  output_path = 'quantized_model.tflite'
  converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=input_path,
        input_arrays=['Placeholder'],
        input_shapes={'Placeholder': [1, 1080, 1920, 3]},
        output_arrays=['ArgMax']
    )
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  tflite_quant_model = converter.convert()
  open(output_path, 'wb').write(tflite_quant_model)
