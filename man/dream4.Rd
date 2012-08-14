\name{dream4}
\alias{dream4gold10}
\alias{dream4gold100}
\alias{dream4ts10}
\alias{dream4ts100}
\alias{dream4wild10}
\alias{dream4wild100}

\title{
  DREAM 4 (Stolovitsky et al. 2007) `gold standard' reference network 
  specifications, simulated time series perturbation subsets,
  and steady state (wild-type) gene expression levels.
}

\description{
 Contains six objects: \code{dream4gold10}, \code{dream4gold100}, 
 \code{dream4ts10}, \code{dream4ts100}, \code{dream4wild10},
 \code{dream4wild100}, each in the form of a list of 
  matrices, in which each element corresponds to one of 5 different datasets.
 \cr
 \code{dream4gold10} and  \code{dream4gold100} are lists of gold standard
  network specifications for the 10-gene and 100-gene simulated networks.
  The list elements are three-column data frames, in which the first and 
  second column represent all possible regulator-gene pairs in the network, 
  excluding self edges. The third column is a boolean variable indicating 
  whether or not the genes in the first two columns are a regulatory pair 
  and thus consitute an edge in the network.\cr
  \code{dream4ts10} and  \code{dream4ts100} are lists of time course datasets
  showing how the simulated network responds to a perturbation and how it 
  relaxes upon removal of the perturbation.  For both size 10 and size 100, 
  5 different datasets are available. For networks of size 10, each of the 5 
  datasets consists of 5 different time series replicates, and for networks 
  of size 100, each of the 5 datasets consists of 10 time series replicates. 
  Each time series has 21 time points.
  In both cases, there are 5 list components corresponding to the datasets,
  each of which is a data frame in which the first two columns give
  the replicate and time, and the remaining columns give the measured
  expression levels. \cr
  The initial condition always corresponds to a steady-state measurement of 
  the wild-type. At t=0, a perturbation is applied to the network as 
  described below. The first half of the time series (until t=500) shows the 
  response of the network to the perturbation. At t=500, the perturbation is 
  removed (the wild-type network is restored). The second half of the time 
  series (until t=1000) shows how the gene expression levels go back from 
  the perturbed to the wild-type state. \cr
 \code{dream4wild10} and  \code{dream4wild100} are lists giving the
  steady state (wild-type) expression levels for the genes in each dataset.
}

\usage{data(dream4)}

\details{
   The perturbations applied here only affect about a third of all genes, 
   but basal activation of these genes can be strongly increased or 
   decreased. For example, these experiments could correspond to physical or
   chemical perturbations applied to the cells, which would cause (via 
   regulatory mechanisms not explicitly modeled here) some genes to have an 
   increased or decreased basal activation. The genes that are directly 
   targeted by the perturbation may then cause a change in the expression 
   level of their downstream target genes. \cr
 The time series datasets included in this package are a subset of the DREAM 4
 data available for the reference networks.
}

\section{References}{
  G. Stolovitzky, D. Monroe and A. Califano (2007),
  Dialogue on Reverse-Engineering Assessment and Methods.
  Annals of the New York Academy of Sciences,1115(1),1--22.

  G. Stolovitzky, A. Califano, R. Prill and J. Saez Rodriguez (2009),
  DREAM: Dialogue for Reverse Engineering Assessments and Methods,
  http://wiki.c2b2.columbia.edu/dream/
}

\seealso{\code{networkBMA}}

\keyword{datasets}