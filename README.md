[![CI](https://github.com/smeinecke/image-match/actions/workflows/check.yml/badge.svg)](https://github.com/smeinecke/image-match/actions/workflows/check.yml)
[![PyPI](https://img.shields.io/pypi/v/image-match.svg)](https://pypi.python.org/pypi/image-match)
[![Release](https://img.shields.io/github/v/release/smeinecke/image-match)](https://github.com/smeinecke/image-match/releases)
[![Documentation Status](https://github.com/smeinecke/image-match/actions/workflows/docs.yml/badge.svg)](https://smeinecke.github.io/image-match/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.13-blue)](https://pypi.org/project/image-match/)

![image-match](https://cloud.githubusercontent.com/assets/6517700/17741093/41040a64-649b-11e6-8499-48b78ddca56b.png)

image-match is a simple (now Python 3!) package for finding approximate image matches from a
corpus.  It is similar, for instance, to [pHash](http://www.phash.org/), but
includes a database backend that easily scales to billions of images and
supports sustained high rates of image insertion: up to 10,000 images/s on our
cluster!

**PLEASE NOTE:** This algorithm is intended to find nearly duplicate images -- think copyright
violation detection.  It is **NOT** intended to find images that are conceptually similar.
For more explanation, see [this issue](https://github.com/edjo-labs/image-match/issues/62) or
[this video](https://www.youtube.com/watch?v=DfWLBzArzKE).

Based on the paper [_An image signature for any kind of image_, Wong et
al](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps).  There is an existing
[reference implementation](https://github.com/jedisct1/libpuzzle) which
may be more suited to your needs.

The folks over at [Pavlov](https://usepavlov.com/) have released an excellent
[containerized version of image-match](https://github.com/pavlovml/match) for
easy scaling and deployment.

## Quick start

### [Install and setup image-match](https://smeinecke.github.io/image-match/start.html)

Once you're up and running, read these two (short) sections of the documentation to get a feel
for what image-match is capable of:

### [Image signatures](https://smeinecke.github.io/image-match/signatures.html)
### [Storing and searching images](https://smeinecke.github.io/image-match/searches.html)
