\name{fastBMAcontrol}
\alias{fastBMAcontrol}
\title{
  Control parameters for \code{networkBMA} when using fastBMA algorithm
}
\description{
  Assigns default control parameters for \code{networkBMA} when using fastBMA 
  algorithm, and allows setting control parameter values.
}
\usage{
fastBMAcontrol(OR = 10000, timeSeries = TRUE, rankOnly = FALSE, noPrune = FALSE,
               edgeMin = 0.5, edgeTol = -1, nThreads = 1,
			   selfie = FALSE, showPrune = FALSE, nVars = 0,
			   fastgCtrl = fastgControl(), start = -1,
			   end = -1, pruneFilterMin = 0, timeout = 0) 
}
\arguments{
  \item{OR}{
    A number specifying the maximum ratio for excluding models in
    Occam's window.
  }
  \item{timeSeries}{
    A logical value indicating whether the input the input data set is a time series data or static data.
  }
  \item{rankOnly}{
    A logical value indicating use priors to rank variables but uniform prior otherwise
  }
  \item{noPrune}{
    A logical value indicating whether not applying transitive reduction on the output edges or not
  }
  \item{edgeMin}{
    Threshold for the posterior probability to be shown
  }
  \item{edgeTol}{
    the error tolerance for determining whether an indirect path is as good as a direct path
  }
  \item{nThreads}{
    The number of threads used in the parallel computing of fastBMA 
  }
  \item{selfie}{
    A logical value indicating whether showing self-loop edges or not
  }
  \item{showPrune}{
    A logical value indicating whether showing removed edges in transitive reduction or not. Ignored if noPrune is TRUE
  }
  \item{nVars}{
    the number of variables analyzed
  }
  \item{fastgCtrl}{
      A list of control variables affecting fastBMA computations when
      using Zellner's g-prior in model likelihhod evaluation. A function
      called \code{fastgCtrl} is provided to facilitate this setting, and
      the default is \code{fastgCtrl()}.
  }
  \item{start}{
	start point of eval subset.
  }
  \item{end}{
	end point of eval subset.
  }
  \item{pruneFilterMin}{
    minimum posterior prob (0-1) before an edge will be included in the network to be pruned.
  }
  \item{timeout}{
    maximum number of seconds for the regression before it stops the search. 0 if not apply.
  }
   
 
}
\value{
    A list of values for the named control parameters to be passed 
    to \code{fastBMA}.
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
  
  L. H. Hong, M. Wu1, A. Lee, W. C. Young, A. E. Raftery and K. Y. Yeung,
  FastBMA and Transitive Reduction for Gene Network Inference. [in preparation]
}
\seealso{
 \code{\link{fastgControl}},
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
