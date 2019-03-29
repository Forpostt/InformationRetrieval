#!/usr/bin/env bash

mkdir data
cd data
kaggle competitions download -c text-relevance-competition-ir-2-ts-spring-2019
tar -xf content.tar.gz
unzip urls.numerate.txt.zip
rm content.tar.gz
rm urls.numerate.txt.zip
cd ../

mkdir processed_data
mkdir processed_data/content
