AC_INIT(networkBMA)
CXX=`"${R_HOME}/bin/R" CMD config CXX` -std=c++11
AC_CONFIG_HEADERS([src/config.h])
AC_LANG(C++)
AC_CHECK_HEADERS([unordered_set tr1/unordered_set])
AC_CONFIG_FILES([src/Makevars])
## BLAS.
# I did not find any BLAS libraries, (R is probably not well configured)
#if test "${BLAS_LIBS}" = ""; then
  BLAS_LIBS="-lgslcblas"
#fi
AC_OUTPUT()
