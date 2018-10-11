import tensorflow as tf
import numpy as np
import librosa.core
import shutil
import os


INPUT_PATH = 'C:\\Users\\tkrokots\\Downloads\\VCTK-Corpus\\wav48'
OUTPUT_PATH = 'C:\\Users\\tkrokots\\Music\\Dataset_voice'
SEQ_LEN = 10000


def convert(input_path=INPUT_PATH, output_path=OUTPUT_PATH, seq_len=SEQ_LEN):
    files = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        files.extend([os.path.join(dirpath, p) for p in filenames])
    print('Found %d files' % len(files))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i, f in enumerate(files):
        print('Process file %s' % f)
        audio, _ = librosa.core.load(f, sr=16000)
        file_writer = tf.python_io.TFRecordWriter(
            os.path.join(output_path, repr(i) + '.tfrecord'))
        prefix_len = audio.shape[0] % seq_len
        audio = audio[prefix_len:]
        for j in range(audio.shape[0] // seq_len):
            snippet = audio[(j * seq_len):((j+1) * seq_len)]
            if np.all(snippet == 0.0):
                continue
            ex = tf.train.SequenceExample()
            ex.context.feature['seq_len'].int64_list.value.append(seq_len)
            fl_wave = ex.feature_lists.feature_list['wave']
            for sample in snippet:
                fl_wave.feature.add().float_list.value.append(sample)
            file_writer.write(ex.SerializeToString())
        print('Extracted %d samples' % j)


if __name__ == '__main__':
    convert()
