from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists, basename, splitext
import librosa
from glob import glob
from os.path import join
import subprocess
import csv

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    src_files = sorted(glob(join(in_dir, "*.wav")))
    for wav_path in src_files:
        futures.append(executor.submit(
            partial(_process_song, out_dir, index, wav_path, "dummy")))
        index += 1
    return [future.result() for future in tqdm(futures)]


def _process_song(out_dir, index, wav_path, text):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Trim begin/end silences
    # NOTE: the threshold was chosen for clean signals
    wav, _ = librosa.effects.trim(wav, top_db=60, frame_length=2048, hop_length=512)

    if hparams.highpass_cutoff > 0.0:
        wav = audio.low_cut_filter(wav, hparams.sample_rate, hparams.highpass_cutoff)

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # Trim silences in mul-aw quantized domain
        silence_threshold = 0
        if silence_threshold > 0:
            # [0, quantize_channels)
            out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
            start, end = audio.start_and_end_indices(out, silence_threshold)
            wav = wav[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels - 1)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        constant_values = P.mulaw(0.0, hparams.quantize_channels - 1)
        out_dtype = np.float32
    else:
        # [-1, 1]
        constant_values = 0.0
        out_dtype = np.float32


    #### CLAIRE Work here
    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
    os.makedirs('./pwavs', exist_ok=True)
    pwav_path = './pwavs/{0}.wav'.format(wav_name)
    audio.save_wav(wav, pwav_path)
    # make the chord directory if it does not exist
    chord_dir = "chord_dir"
    os.makedirs(chord_dir, exist_ok=True)

    # create xml file with notes and timestamps
    #subprocess.check_call(['./extract_chord_notes.sh', wav_path, chord_dir], shell=True)
    #os.system('./extract_chord_notes.sh {0} {1}'.format(pwav_path, chord_dir))
    os.system('./extract_chord_notes.sh {0} {1} > /dev/null 2>&1'.format(pwav_path, chord_dir))


    note_filename = '{0}/{1}.csv'.format(chord_dir, wav_name)

    #### Instead of computing the Mel Spectrogram, here return a time series of one hot encoded chords.
    # vector with 1 in row for each note played
    # 1000 samples per second
    note_samples = int((len(wav) / hparams.sample_rate) * 1000)
    # 12 notes per octave
    chords_time_series = np.zeros((12, note_samples))

    #print(np.shape(chords_time_series))

    with open(note_filename, newline='\n') as csvfile:
        #chordreader = csv.reader(csvfile, delimeter=',')
        chordreader = csvfile.readlines()
        #print(chordreader)
        for row in chordreader:
            row = row.split(",")
            start_time = float(row[0])
            end_time = float(row[1]) + start_time
            note = int(row[2]) % 12
            start_sample = min(note_samples-1, int(start_time * 1000))
            end_sample = min(note_samples, int(end_time * 1000))
            try:
                chords_time_series[note][start_sample:end_sample]=1
                print('wav {0} start {1} end {2} note {3} num_notes {4}'.format(wav_name, start_sample, end_sample, note, note_samples))
            except Exception as e:
                print(np.shape(chords_time_series))
                print('wav {0} start {1} end {2} note {3} num_notes {4}'.format(wav_name, start_sample, end_sample, note, note_samples))


    chords_time_series = chords_time_series.T

    if hparams.global_gain_scale > 0:
        wav *= hparams.global_gain_scale

    # Time domain preprocessing
    if hparams.preprocess is not None and hparams.preprocess not in ["", "none"]:
        f = getattr(audio, hparams.preprocess)
        wav = f(wav)

    wav = np.clip(wav, -1.0, 1.0)

    # Set waveform target (out)
    if is_mulaw_quantize(hparams.input_type):
        out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        out = P.mulaw(wav, hparams.quantize_channels - 1)
    else:
        out = wav

    # zero pad
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.pad_lr(out, hparams.fft_size, audio.get_hop_size())
    if l > 0 or r > 0:
        out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = chords_time_series.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    # Write the spectrograms to disk:
    name = splitext(basename(wav_path))[0]
    audio_filename = '%s-wave.npy' % (name)
    chords_filename = '%s-feats.npy' % (name)
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, chords_filename),
            chords_time_series.astype(np.int16), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_filename, chords_filename, N, text)
