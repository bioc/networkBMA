AC_INIT(networkBMA)
CXX=`"${R_HOME}/bin/R" CMD config CXX` -std=c++11
AC_CONFIG_HEADERS([src/config.h])
AC_LANG(C++)
AC_CHECK_HEADERS([unordered_set tr1/unordered_set])

AC_PATH_PROG(PKG_CONFIG, pkg-config)
LAPACK_VERSION="`${PKG_CONFIG} lapack --version || echo ''`"
if test -z "${LAPACK_VERSION}" ; then
   AC_MSG_ERROR(['pkg-config lapack' not found; install liblapack-dev])
fi
PKG_CFLAGS="`${PKG_CONFIG} lapack --cflags`"
PKG_LIBS="`${PKG_CONFIG} lapack --libs`"

AC_CONFIG_FILES([src/Makevars])
AC_SUBST(PKG_CFLAGS)
AC_SUBST(PKG_LIBS)

AC_OUTPUT()
