#!/bin/bash

url=http://nlp.stanford.edu/data/glove.6B.zip
file=`basename $url`

curl -LO $url
mkdir data
unzip $fname -d data/glove
