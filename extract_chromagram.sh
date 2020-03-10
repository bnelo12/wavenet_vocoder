#!/usr/bin/env bash

sonic-annotator -s vamp:nnls-chroma:nnls-chroma:bothchroma > chordnotes.n3

sample=$1
chord_dir=$2

sample_name=${sample##*/}

sample_csv="$chord_dir"/"${sample_name%.*}".csv

sonic-annotator -t chordnotes.n3 "$sample" -w csv 

mv "${sample%.*}"_vamp_nnls-chroma_nnls-chroma_bothchroma.csv "$sample_csv"


