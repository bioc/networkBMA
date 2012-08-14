\name{varord}
\alias{varord}
\title{
  Variable orderings for linear regression.
}
\description{
   Gives variable orderings for linear regression in high-dimensional data
   by various approaches.
}
\usage{
varord(x, y, prior.prob = NULL, ordering = c("bic1","prior","bic1+prior"))
}
\arguments{
  \item{x}{
     A matrix whose columns correspond to variables or genes and whose
     rows correspond to the observations.
  }
  \item{y}{
    A vector of response values for fitting a linear model to a subset of
    the variables in \code{x}.
   }
 \item{prior.prob}{
     A vector of prior probabilities corresponding to the variables (genes).
  }
  \item{ordering}{
      A character string specifying the method to be used for the 
     ordering the variables.
     \describe{
         \item{"bic1"}{
           Variables are ordered in by the BIC for the univariate linear model 
           with that variable as predictor. (Yeung et~al. 2011)
        }
        \item{"prior"}{
           The variables are ordered according to the log odds of their prior
           values. (Lo et~al. 2011)
        }
       \item{"bic1+prior"}{
           The variables are ordered according to sum of the univarite BIC values
           and the log odds of their prior values. 
        }
     }
  }
}
\value{
 A vector of integers corresponding to the variable indexes in the specified 
 order.
}
\section{References}{
  K. Lo, A. E. Raftery, K. M. Dombek, J. Zhu, E. E. Schadt, R. E. Bumgarner 
  and K. Y. Yeung (2011), Integrating External Biological 
  Knowledge in the Construction of Regulatory Networks from Time-series 
  Expression Data, unpublished manuscript, University of Washington.

  K. Y. Yeung, K. M. Dombek, K. Lo, J. E. Mittler, J. Zhu, E. E. Schadt,
  R. E. Bumgarner and A. E. Raftery (2011), Construction of 
  regulatory networks using expression time-series data of a genotyped 
  population, Proceedings of the National Academy of Sciences, 
  108(48):19436-41.
}
\seealso{
  \code{\link{networkBMA}},
  \code{\link{iterateBMAlm}}
}
\examples{
data(dream4)

network <- 1

Time <- as.numeric(dream4ts100[[network]]$time)

xIndex <- which(Time != max(Time))
yIndex <- which(Time != min(Time))

gene <- "G1"

x <- dream4ts100[[network]][xIndex,-(1:2)]
y <- dream4ts100[[network]][yIndex,gene]

nvar <- 50
ord <- varord( x, y, ordering = "bic1")[1:nvar]

\dontrun{
result <- iterateBMAlm( x = x[,ord], y = y)
}
}
\keyword{models}
