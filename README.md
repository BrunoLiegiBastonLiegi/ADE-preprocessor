# ADE-preprocessor

Little python script for preparing the ADE-Corpus dataset for NER and RE tasks.

Original annotation:
10082597|Naproxen, the most common offender, has been associated with a dimorphic clinical pattern: a PCT-like presentation and one simulating erythropoietic protoporphyria in the pediatric population.|erythropoietic protoporphyria|1612|1641|Naproxen|1478|1486

After processing:
Naproxen, the most common offender, has been associated with a dimorphic clinical pattern: a PCT-like presentation and one simulating erythropoietic protoporphyria in the pediatric population.|S-DRUG|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|O|B-AE|I-AE|I-AE|I-AE|I-AE|I-AE|E-AE|O|O|O|O|O|