## GxTheta - A tool to identify sites with variable effect size due to ancestry

Gene by Ancestry association based on PyLMM for use in identifying individual polymorphisms whose effect size vary based on individual-level relatedness to a founder individual/line/population

### An Example Command:

```
python pylmmGxTheta.py -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos --covfile Ancestry.cov --Ancestry out.foo
```

The GxTheta pylmmGxTheta.py reads PLINK formated input files (BED or TPED only).  There is also an option to use "EMMA" formatted files as in regular PyLMM.  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

Ancestry values (eg % of genome derived from individual/line/population A) should be present as the last covariate in the covariate file

## Installation 
You will need to have numpy and scipy installed on your current system.
You can install pylmmGxT using pip by doing the following 

```
   pip install git+https://github.com/ChristophRau/GxTheta
```
This should make the module pylmmGxT available as well as the two scripts pylmmGxTheta.py and pylmmKinship.py.

You can also clone the repository and do a manual install.
```
   git clone https://github.com/ChristophRau/GxTheta
   python setup.py install
```
