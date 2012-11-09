\name{vignette}
\alias{vignette}
\alias{brem.data}
\alias{referencePairs}
\alias{reg.known}
\alias{reg.prob}
\alias{timeSeries}

\title{
   The subsets of the yeast-rapamycin time-series data from 
   Yeung et al. (2011) and  Lo et al. (2011), and of the static yeast 
   gene-expression data from Brem et al. (2002, 2005), that are used in the 
  \code{networkBMA}  package vignette.
}

\description{
 Contains four data objects used to illustrate the \code{networkBMA} package in
 the accompanying vignette.
 \describe{
    \item{brem.data}{
An 85 by 111 subset of the data used for network inference in
yeast in Brem et~al. (2002) and Brem and Kruglyak (2005).
The rows correspond to genes and the columns to experiments.
Provided courtesy of Rachel Brem.
    }
    \item{referencePairs}{
       A 2-column data frame giving 287 regulator-gene pairs among the
       selected set of genes reported from the literature.
       The known regulatory relationships are not necessarily included 
       in the reference  network.
     }
    \item{reg.known}{
       A 2-column matrix of known (hard-coded) regulatory 
       relationships among the 100 gene subset.
       The first column gives the name of the regulator, 
       and the second column gives the name of the target gene. 
       The gene names should be consistent with the data.
     } 
    \item{reg.prob}{
     A 100 by 100 matrix in which the (i,j) entry is the estimated prior 
     probability that gene i regulates gene j (from Lo et al. 2011).
    }
     \item{timeSeries}{
       A data frame in which the first two columns are factors identifying
       the replicate and time (in minutes) after drug perturbation,
       and the remaining 100 columns are the expression measurements
      for a subset of 100 genes from the 
       yeast-rapamycin experiment described in Yeung et al. (2011).
     }
 }
}

\usage{data(vignette)}


\section{References}{
R. B. Brem and G. Yvert and R. Clinton and L. Kruglyak (2002),
Genetic dissection of transcriptional regulation in budding yeast, 
Science 296(5568):752-755.

R. B. Brem and L. Kruglyak (2005), The landscape of genetic complexity across 
5,700 gene expression traits in yeast, Proceedings of the National Academy of
Sciences 102(5):1572-1577.

  K. Lo, A. E. Raftery, K. M. Dombek, J. Zhu, E. E. Schadt, R. E. Bumgarner
  and K. Y. Yeung (2012), Integrating External Biological
  Knowledge in the Construction of Regulatory Networks from Time-series
  Expression Data, BMC Systems Biology, 6:101.

  K. Y. Yeung, K. M. Dombek, K. Lo, J. E. Mittler, J. Zhu, E. E. Schadt,
  R. E. Bumgarner and A. E. Raftery (2011), Construction of
  regulatory  networks  using expression time-series data of a genotyped
  population, Proceedings of the National Academy of Sciences,
  108(48):19436-41.
}

\seealso{\code{iterateBMAlm}}

\examples{
data(vignette)

gene <- "YNL037C"                                                              
variables <- which(rownames(brem.data) != gene)                               
control <- iBMAcontrolLM(OR = 50, nbest = 20, thresProbne0 = 5)               
iBMAmodel.YNL037C <- iterateBMAlm( x = t(brem.data[variables,]),               
                           y = unlist(brem.data[gene,]), control = control)   
}

\keyword{datasets}