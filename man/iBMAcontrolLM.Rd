\name{iBMAcontrolLM}
\alias{iBMAcontrolLM}
\title{
  Control parameters for \code{iterateBMAlm} 
}
\description{
  Assigns default control parameters for \code{iterateBMAlm}, and
  allows setting control parameter values.
}
\usage{
iBMAcontrolLM( OR = 20, nbest = 10, maxNvar = 30, thresProbne0 = 1,
               keepModels = FALSE, maxIter = 200000) 
}
\arguments{
  \item{OR}{
    A number specifying the maximum ratio for excluding models in
    Occam's window.
  }
  \item{nbest}{
      A positive integer specifying the number of models of each size
      to be considered by leaps-and-bounds in determining the model space for 
      Bayesian Model Averaging. The default value is 10.
   }
 \item{maxNvar}{
      A positive integer specifying the maximum number of variables 
      (excluding the intercept) used in each iteration of BMA. The default 
      value is 30.
 }
 \item{thresProbne0}{
     Threshold (in percent) for the posterior probability that
     each variable is has a non-zero coefficient (in percent). 
     Variables with posterior probability less than \code{thresProbne0}
     are removed in future BMA iterations. The default value is 1 percent.
 }
 \item{keepModels}{
      A logical value indicating whether or not to keep the BMA models
      from all of the iterations and apply Occam's window using \code{OR} at
      the end,  or to apply Occam's window in all BMA iterations and return 
      the final model. The default is not to keep the models. Setting the
      argument to \code{TRUE} requires more memory and may slow the 
      computation as a result.
  }
 \item{maxIter}{
       A  positive integer giving a limit on the number of iterations of
      \code{iterateBMAlm}. The default value is 20000. \code{iterateBMAlm} 
      will terminate in fewer than \code{maxIter} iterations if the iterative
      BMA modeling process has seen all available variables.
  }
}
\value{
    A list of values for the named control parameters to be passed 
    to a version of the function \code{bicreg} from the \code{BMA}
    package that has been modified to handle prior probabilities.
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

  K. Y. Yeung (with contributions from A. E. Raftery and I. Painter), 
  iterativeBMA: The Iterative Bayesian Model Averaging (BMA) algorithm,
  Bioconductor R package, version 1.8.0 posted in 2009.

  K. Y. Yeung, R. E. Bumgarner and A. E. Raftery (2005).
  Bayesian Model Averaging: Development of an improved multi-class, 
  gene selection and classification tool for microarray data. 
  Bioinformatics 21:2394-2402. 

  A. E. Raftery, J. A. Hoeting, C. T. Volinsky, I. Painter and K. Y. Yeung 
 (2005), BMA: Bayesian Model Averaging, Comnprehensive R Archhive Network 
 (CRAN),  package version 3.15.1 posted in 2012.

  J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky (1999).
  Bayesian Model Averaging: a tutorial, 
  Statistical Science 14(4): 382-417. 
}
\seealso{
 \code{\link{iterateBMAlm}},
 \code{\link{networkBMA}}
  \code{\link{do.call}}
}
\examples{
data(dream4)

network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                          nTimePoints = nTimePoints,
                          control = iBMAcontrolLM(thresProbne0 = 1))
}
\keyword{models}
