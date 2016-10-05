AC_INIT(networkBMA)
#CXX1X=`"${R_HOME}/bin/R" CMD config CXX11X`
#CXX1XSTD=`"${R_HOME}/bin/R" CMD config CXX11XSTD`
#CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXX11XFLAGS`
CXX=`"${R_HOME}/bin/R" CMD config CXX` -std=c++11
#CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
AC_CONFIG_HEADERS([src/config.h])
AC_LANG(C++)
AC_CHECK_HEADERS([unordered_set tr1/unordered_set])
AC_OUTPUT
