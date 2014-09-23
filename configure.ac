AC_INIT(networkBMA)
#CXX="$( "${R_HOME}/bin/R" CMD config CXX ) $( "${R_HOME}/bin/R" CMD config CXX1XSTD )"
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
AC_CONFIG_HEADERS([src/config.h])
AC_LANG(C++)
AC_CHECK_HEADERS([unordered_set tr1/unordered_set])
AC_OUTPUT