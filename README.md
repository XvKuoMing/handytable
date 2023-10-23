# handytable
Package for excel-like table manipulations in Jupiter notebook &amp; google.colab

Check docs.ipynb or [docs](https://colab.research.google.com/drive/1ssm8jMPkIN-1iZPETTP1sg7xSAkPEufP?usp=sharing) for usage

# Installation

Currently handytable could not be downloaded using pip.
Use the following steps instead:

## Install svn
```
  apt install subversion
```

## download subdirectory using svn 

```
  svn export https://github.com/XvKuoMing/handytable/trunk/handytable
```

## import as python module

```
from handytable.tables impoort HandyTable
```

# Recomendation
Due to difference in Jypiter notebook and Google colab API, the lead-js feature of HandyTable does not work in Jypiter. I will implement it soon.
<br />
For now I recommend to use handytable in google.colab.
<br />
Just paste the following code:
```
%%capture

!apt install subversion
!svn export https://github.com/XvKuoMing/handytable/trunk/handytable

from handytable.tables import HandyTable
```
