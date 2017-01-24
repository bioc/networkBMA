// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include "fastBMA.hpp"
#include "bmaNoMPI.hpp"

using namespace Rcpp;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
const List fastBMA_g(
		NumericMatrix x,
		int nTimePoints,
		NumericMatrix priorProbs_,
		double oddsRatio,
		int nVars,
		double g,
		CharacterVector geneNames,
		int start,
		int end,
		int nThreads,
		int optimizeBits,
		int maxOptimizeCycles,
		bool timeSeries,
		bool rankOnly,
		bool noPrune,
		double edgeMin,
		double edgeTol,
		double pruneFilterMin, //new??
		double timeout //new?
		) {
  //Rcpp::Rcout << "you got here!\nthe cpp fastBMA_g\n";
  int colNum = x.ncol();
  
	if (x.ncol() == 0 || x.nrow() == 0) {
		Rcpp::stop("x row or col 0\n");
	}
	
	//malloc data********************************************
	double **data = new double*[x.ncol()];
	for (int i = 0; i < x.ncol(); i++) {
		data[i] = new double[x.nrow()];
	}
	for (int i = 0; i < x.ncol(); i++) {
		for (int j = 0; j < x.nrow(); j++) {
			data[i][j] = x(j, i);
		}
	}
	
	/*cerr << "x Data:\n" << x.ncol() << "," << x.nrow() << "\n";
	for (int i = 0; i < x.ncol(); i++) {
		for (int j = 0; j < x.nrow(); j++) {
			cerr << data[i][j] << ", ";
		}
		cerr << endl;
	}*/

	if (priorProbs_.ncol() == 0 || priorProbs_.nrow() == 0) {
		Rcpp::stop("priorProbs_ row or col 0\n");
	}
	
	//malloc rProbs********************************************
	double **rProbs = new double*[priorProbs_.ncol()];
	for (int i = 0; i < priorProbs_.ncol(); i++) {
		rProbs[i] = new double[priorProbs_.nrow()];
	}
	// change i,j here if vertical
	for (int i = 0; i < priorProbs_.ncol(); i++) {
		for (int j = 0; j < priorProbs_.nrow(); j++) {
			rProbs[i][j] = priorProbs_(j,i);
		}
	}

	
	/*cerr << "priorProbs_ Data:" << priorProbs_.ncol() << "," << priorProbs_.nrow() << "\n";
	for (int i = 0; i < priorProbs_.ncol(); i++) {
		for (int j = 0; j < priorProbs_.nrow(); j++) {
			cerr << rProbs[i][j] << ", ";
		}
		cerr << endl;
	}*/
	

	//new headers********************************************
	vector<string> *headers = new vector<string>();
	for (int i = 0; i < geneNames.size(); i++)
		headers->push_back(as<string>(geneNames[i]));

	/*cout << "headers Data:\n";
	vector<string>::iterator itr = headers->begin();
	for (int i = 0; i < geneNames.size(); i++) {
		cout << *itr << endl;
		itr++;
	}*/
	
	//pre set param
	double uPrior = UNIFORM_PRIOR;
	
	/*
	cerr << "flags:" << endl;
	cerr << "nTimePoints: " << nTimePoints << endl;
	cerr << "g: " << g << endl;
	cerr << "optimizeBits: " << optimizeBits << endl;
	cerr << "maxOptimizeCycles: " << maxOptimizeCycles << endl;
	cerr << "uPrior: " << uPrior << endl;
	cerr << "oddsRatio: " << oddsRatio << endl;
	cerr << "nGenes: " << x.ncol() << endl;
	cerr << "nSamples: " << x.nrow() << endl;
	cerr << "nThreads: " << nThreads << endl;
	cerr << "noPrune: " << noPrune << endl;
	cerr << "edgeMin: " << edgeMin << endl;
	cerr << "edgeTol: " << edgeTol << endl;
	*/
	
	const float twoLogOR=2.0 * log(oddsRatio);//new logOR
	
	if(!pruneFilterMin){
		pruneFilterMin=edgeMin/4.0;
	}
	if(noPrune){
		pruneFilterMin=edgeMin;
	}
	//cerr << "bma_result create" << endl;
	//bool useResiduals = false;
	BMARetStruct *bma_result = new BMARetStruct;
	//cerr << "before no MPI call" << endl;
	findEdges(
		bma_result,
		data, //data
		rProbs, //rProbs
		*headers, //headers
		x.ncol(),//nGenes
		x.nrow(),//nSamples
		nTimePoints,
		timeSeries, //isTimeSeries
		rankOnly, //new***
		noPrune,
		nVars,
		nThreads,
		optimizeBits,
		maxOptimizeCycles,
		twoLogOR, //new?***
		g,
		pruneFilterMin, //new***
		edgeMin,
		edgeTol,
		uPrior,
		start - 1, // new, negative number if not used
		end - 1 // new, negative number if not used
	);

	//bma_result
	float *edgeWeights = bma_result->edgeWeights;
	int *parents = bma_result->parents;
	int *nParents = bma_result->nParents;
	int nNodes = bma_result->nNodes;
	int edgeNum = bma_result->edgeNum;
	//edges
	
	if (nNodes == 0 || edgeNum == 0) {
		Rcpp::stop("Rcpp length 0\n");
	}

	NumericVector edgeWeightsRet(edgeNum);
	IntegerVector parentsRet(edgeNum);
	IntegerVector nParentsRet(nNodes);
	
	for (int i = 0; i < nNodes; i++) {
		nParentsRet[i] = nParents[i];
	}
	
	for (int i = 0; i < edgeNum; i++) {
		edgeWeightsRet[i] = edgeWeights[i];
		parentsRet[i] = parents[i];
	}
	
	/*cerr << nNodes << endl;
	cerr << "nParents" << endl;
	for (int i = 0; i < nNodes; i++) {
		cerr << nParentsRet[i] << ", ";
	}
	cerr << endl;
	
	cerr << "parents" << endl;
	for (int i = 0; i < edgeNum; i++) {
		cerr << parentsRet[i] << ", ";
	}
	cerr << endl;
	cerr << "edgeWeights" << endl;
	for (int i = 0; i < edgeNum; i++) {
		cerr << edgeWeightsRet[i] << ", ";
	}
	cerr << endl;
	int edgeIndex = 0;
	cerr << "check point4" << endl;

	cerr << "check point5" << endl;
	for (int i = 0; i < nNodes; i++)
		nParentsRet[i] = nParents[i];
	cerr << "check point6" << endl;*/
	
	List ret;
	ret["edgeWeights"] = edgeWeightsRet;
	ret["nParents"] = nParentsRet;
	ret["parents"] = parentsRet;
	
	if (bma_result->nParents) delete []bma_result->nParents;
	if (bma_result->parents) delete []bma_result->parents;
	if (bma_result->edgeWeights) delete []bma_result->edgeWeights;
	if (bma_result) delete bma_result;
	
	return ret;

}
