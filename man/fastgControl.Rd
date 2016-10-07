\name{fastgControl}
\alias{fastgControl}
\title{
  Control parameters for using Zellner's g-prior in fastBMA algorithm in \code{networkBMA} 
}
\description{
  Assigns default control parameters for the use of Zellner's g-prior in
  fastBMA algorithm in \code{networkBMA}, and allows setting control parameter values.
}
\usage{
fastgControl( optimize = 0, g0 = NULL, iterlim = 20 ) 
}
\arguments{
  \item{optimize}{
      optimize bits - an int value determines how accurate the optimization
	  of g is. 0 means not optimize.
    }
  \item{g0}{
      An initial value of g to use if optimize is TRUE, or the fixed
      value to use without optimization.
    }
 \item{iterlim}{
     If optimize is non-zero, the maximum number of iterations of the
	 optimization algorithm to use. Ignored otherwise.
   }
}
\value{
    A list of values for the named control parameters to be passed 
    to \code{fastBMAcontrol} and \code{networkBMA} using fastBMA algorithm.
}
\section{References}{
  A. Zellner (1986), On assessing prior distributions and Bayesian
  regression analysis with g-prior distributions, Bayesian inference and
  decision techniques: Essays in Honor of Bruno De Finetti, 6:233-243.

  M. Clyde and E.I. George (2004), Model Uncertainty, Statistical
  Science, 81-94.
  
  L. H. Hong, M. Wu1, A. Lee, W. C. Young, A. E. Raftery and K. Y. Yeung,
  FastBMA and Transitive Reduction for Gene Network Inference. [in preparation]
}
\seealso{
 \code{\link{fastBMAcontrol}},
 \code{\link{networkBMA}}
}
\examples{
data(dream4)

network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA(data = dream4ts10[[network]][,-(1:2)],
                         nTimePoints = nTimePoints,
						 control=fastBMAcontrol(fastgCtrl=
						 fastgControl(optimize=4)))
}
\keyword{models}
