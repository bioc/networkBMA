\name{gControl}
\alias{gControl}
\title{
  Control parameters for using Zellner's g-prior in \code{ScanBMA} 
}
\description{
  Assigns default control parameters for the use of Zellner's g-prior in
  \code{ScanBMA}, and allows setting control parameter values.
}
\usage{
gControl( optimize = TRUE, optMethod = "perTarget", g0 = NULL,
          iterlim = 100, epsilon = 0.1 ) 
}
\arguments{
  \item{optimize}{
      A logical value indicating whether to optimze g using an iterative
      EM algorithm or use a fixed value of g.
    }
  \item{optMethod}{
      A character string indicating how to optimize g. Currently, only
      perTarget is supported, indicating that g should be optimized
      individually for each target.
    }
  \item{g0}{
      An initial value of g to use if optimize is TRUE, or the fixed
      value to use without optimization.
    }
 \item{iterlim}{
     If optimize is TRUE, the maximum number of iterations of the EM
     algorithm to use. Ignored otherwise.
   }
 \item{epsilon}{
     If optimize is TRUE, the precision with which to find g using the
     EM algorithm. Ignored otherwise.
   }
}
\value{
    A list of values for the named control parameters to be passed 
    to \code{ScanBMAcontrol} and \code{ScanBMA}.
}
\section{References}{
  A. Zellner (1986), On assessing prior distributions and Bayesian
  regression analysis with g-prior distributions, Bayesian inference and
  decision techniques: Essays in Honor of Bruno De Finetti, 6:233-243.

  M. Clyde and E.I. George (2004), Model Uncertainty, Statistical
  Science, 81-94.
}
\seealso{
 \code{\link{ScanBMAcontrol}},
 \code{\link{ScanBMA}},
 \code{\link{networkBMA}}
}
\examples{
data(dream4)

network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                          nTimePoints = nTimePoints,
                          control = ScanBMAcontrol(gCtrl =
                          gControl(optimize = TRUE)) )
}
\keyword{models}
