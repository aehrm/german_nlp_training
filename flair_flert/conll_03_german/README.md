Place the CONLL-03 German NER train and test datasets, i.e. `deu.testa`, `deu.testb`, `deu.train`, encoded in UTF-8,
here before building the Docker images.
You can verify the integrity of the files with the attached hashsums.

You probably need to adjust the original build script to obtain the German datasets.
Assuming you have access to the ECI Multilingual Text Corpus (<https://catalog.ldc.upenn.edu/LDC94T5>), you can generate the datasets with the following commands:

```
export CORPUS=/path/to/eci_multilang_txt/data/eci1/ger03/ger03b05.eci


wget 'https://www.clips.uantwerpen.be/conll2003/ner.tgz'
tar xvf ner.tgz
patch -p0 -b -i make.deu.patch
cd ner
cp etc.2006/tags.deu etc/tags.deu  # use the 2006 revised dataset
bin/make.deu
cd ..

# convert to UTF-8
for f in deu.testa deu.testb deu.train; do
    iconv -f ISO-8859-1 -t UTF-8 ner/$f > $f
done

sha256sum --check german_revised_conll_dataset.sha256
```
