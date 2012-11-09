\name{iterateBMAlm}
\alias{iterateBMAlm}
\title{
  Iterative BMA for linear modeling with prior variable probabilities.
}
\description{
   An iterative version of Bayesian Model Averaging (BMA) for linear models
   with many variables. Incorporates prior probabilities for inclusion of 
   variables in models.
}
\usage{
iterateBMAlm( x, y, prior.prob = NULL, control = iBMAcontrolLM(), 
             verbose = FALSE)
}
\arguments{
  \item{x}{
     A matrix of real-valued predictor variables. Rows correspond to
     observations and columns to variables.
  }
  \item{y}{
     A real-valued response vector.
   }
 \item{prior.prob}{
    An optional vector of prior probabilities for each predictor 
    variable belonging to a linear model for the data. If not specified,
    predictor variables are assumed to have equal prior probability.
  }
 \item{control}{
    A list of values controling the underlying algorithm.
    The default is given by \code{iBMAcontrolLM()}.
  }
 \item{verbose}{
     A logical variable indicating whether or not the details of the
     method's progress shoul be printed during computation. The default
     value is \code{FALSE}.
   }
}
\value{
A list with the following components, similar to the output of function
\code{bicreg} in the \code{BMA} package:
  \item{bic}{values of BIC for the models}
  \item{postprob}{the posterior probabilities of the models selected}
  \item{priorprob}{the prior probabilities of the variables in the models}
  \item{namesx}{the names of the variables}
  \item{label}{labels identifying the models selected}
  \item{r2}{R2 values for the models}
  \item{size}{the number of independent variables in each of the models}
  \item{which}{a logical matrix with one row per model and one column per 
  variable indicating whether that variable is in the model}
  \item{probne0}{the posterior probability that each variable is non-zero 
                 (in percent)}
  \item{postmean}{the posterior mean of each coefficient (from model averaging)}
  \item{condpostmean}{the posterior mean of each coefficient conditional on the 
   variable being included in the model}
  \item{condpostsd}{the posterior standard deviation of each coefficient 
  conditional on the variable being included in the model}
  \item{ols}{matrix with one row per model and one column per variable giving the
OLS estimate of each coefficient for each model}
  \item{mle}{the same as \code{ols}}
  \item{n.models}{the number of models}
  \item{n.vars}{the number of variables}
}
\details{
    \code{iterateBMAlm} is intended for datasets that have more variables
    (e.g. gene expression values) than observations (e.g. subjects).
    There is currently no mechanism for handling factor variables in
    \code{iterateBMAlm}, as there is in the underlying function
    \code{bicreg} in the \code{BMA} package. However factors can be encoded
    by users and included with other variables as input.
}
\section{References}{
  K. Y. Yeung, R. E. Bumgarner and A. E. Raftery (2005), 
  Bayesian model averaging: development of an improved multi-class, 
  gene selection and classification tool for microarray data,
  Bioinformatics 21(10) 2394-2402.

  K. Lo, A. E. Raftery, K. M. Dombek, J. Zhu, E. E. Schadt, R. E. Bumgarner 
  and K. Y. Yeung (2012), Integrating External Biological 
  Knowledge in the Construction of Regulatory Networks from Time-series 
  Expression Data, BMC Systems Biology, 6:101.

  K. Y. Yeung, K. M. Dombek, K. Lo, J. E. Mittler, J. Zhu, E. E. Schadt,
  R. E. Bumgarner and A. E. Raftery (2011), Construction of 
  regulatory networks using expression time-series data of a genotyped 
  population, Proceedings of the National Academy of Sciences, 
  108(48):19436-41.
}
\seealso{
    \code{\link{iBMAcontrolLM}},
    \code{\link{varord}},
    \code{\link{bicreg}}
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

result <- iterateBMAlm( x = x[,ord], y = y)
}
\keyword{models}










