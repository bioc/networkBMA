\name{ScanBMAcontrol}
\alias{ScanBMAcontrol}
\title{
  Control parameters for \code{ScanBMA} 
}
\description{
  Assigns default control parameters for \code{ScanBMA}, and
  allows setting control parameter values.
}
\usage{
ScanBMAcontrol( OR = 100, useg = TRUE, gCtrl = gControl(), thresProbne0 = 1 ) 
}
\arguments{
  \item{OR}{
    A number specifying the maximum ratio for excluding models in
    Occam's window.
  }
  \item{useg}{
      A logical value indicating whether to use Zellner's g-prior in
      model likelihood evaluation. If set to FALSE, \code{ScanBMA} will
      use BIC to approximate the likelihood.
   }
  \item{gCtrl}{
      A list of control variables affecting ScanBMA computations when
      using Zellner's g-prior in model likelihhod evaluation. A function
      called \code{gControl} is provided to facilitate this setting, and
      the default is \code{gControl()}.
   }
 \item{thresProbne0}{
     Threshold (in percent) for the posterior probability that
     each variable is has a non-zero coefficient (in percent). 
     Variables with posterior probability less than \code{thresProbne0}
     are removed in future BMA iterations. The default value is 1 percent.
 }
}
\value{
    A list of values for the named control parameters to be passed 
    to \code{ScanBMA}.
}
\section{References}{
  K. Lo, A. E. Raftery, K. M. Dombek, J. Zhu, E. E. Schadt, R. E. Bumgarner
  and K. Y. Yeung (2011), Integrating External Biological 
  Knowledge in the Construction of Regulatory Networks from Time-series 
  Expression Data, unpublished manuscript, University of Washington.

  K. Y. Yeung, K. M. Dombek, K. Lo, J. E. Mittler, J. Zhu,  E. E. Schadt,
  R. E. Bumgarner and A. E. Raftery (2011), Construction of 
  regulatory networks using expression time-series data of a genotyped 
  population, Proceedings of the National Academy of Sciences, 
  108(48):19436-41.

  K. Y. Yeung, R. E. Bumgarner and A. E. Raftery (2005).
  Bayesian Model Averaging: Development of an improved multi-class, 
  gene selection and classification tool for microarray data. 
  Bioinformatics 21:2394-2402. 

  J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky (1999).
  Bayesian Model Averaging: a tutorial, 
  Statistical Science 14(4): 382-417. 
}
\seealso{
 \code{\link{gControl}},
 \code{\link{ScanBMA}},
 \code{\link{networkBMA}}
}
\examples{
data(dream4)

network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                          nTimePoints = nTimePoints,
                          control = ScanBMAcontrol(thresProbne0 = 1) )
}
\keyword{models}
