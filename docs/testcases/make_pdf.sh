exec docker run --rm -v `pwd`:/docs -it sphinx make latex
exit $