# This module prepares midi file data 
# and feeds it to the NN for training

# imports
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# train NN to generate music
def train_network():
    notes = get_notes()

    n_vocab = len(set(notes))

    n_input, n_output = prepare_sequences(notes, n_vocab)

    model = create_network(n_input, n_vocab)

    train(model, n_input, n_output)

# get all notes and chords from midi files in data/midi
# exception:
# file has instrument parts
# file has notes in flat structure
def get_notes():
    notes = []

    for file in glob.glob("midi/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

# prep the sequences used by NN
# get all pitch names
# create dictionary to map pitches to integers
# create input sequences and corresponding outputs
# reshape input into format compatible with LSTM layers
# normalize input

def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    n_input = []
    n_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        n_input.append([note_to_int[char] for char in sequence_in])
        n_output.append(note_to_int[sequence_out])

    n_patterns = len(n_input)

    n_input = numpy.reshape(n_input, (n_patterns, sequence_length, 1))
    n_input = n_input / float(n_vocab)

    n_output = np_utils.to_categorical(n_output)

    return (n_input, n_output)

# create the structure of NN
def create_network(n_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(n_input.shape[1], n_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

# train NN
def train(model, n_input, n_output):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(n_input, n_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()