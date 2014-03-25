#include <Rcpp.h>

using namespace Rcpp;

// ScanBMA_g
const List ScanBMA_g(NumericVector y, NumericMatrix x, NumericVector priorProbs_, double oddsRatio, double g);
RcppExport SEXP networkBMA_ScanBMA_g(SEXP ySEXP, SEXP xSEXP, SEXP priorProbs_SEXP, SEXP oddsRatioSEXP, SEXP gSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    NumericVector y = Rcpp::as<NumericVector >(ySEXP);
    NumericMatrix x = Rcpp::as<NumericMatrix >(xSEXP);
    NumericVector priorProbs_ = Rcpp::as<NumericVector >(priorProbs_SEXP);
    double oddsRatio = Rcpp::as<double >(oddsRatioSEXP);
    double g = Rcpp::as<double >(gSEXP);
    const List __result = ScanBMA_g(y, x, priorProbs_, oddsRatio, g);
    return Rcpp::wrap(__result);
END_RCPP
}

// BMA_Diff_g
const List BMA_Diff_g(NumericVector y, NumericMatrix x, NumericVector priorProbs_,
                        double g, IntegerVector bestModel, bool diff100, bool diff0,
                        IntegerVector pred100, IntegerVector pred0, double minprob, double epsilon);
RcppExport SEXP networkBMA_BMA_Diff_g(SEXP ySEXP, SEXP xSEXP, SEXP priorProbs_SEXP, SEXP gSEXP,
                        SEXP bestModelSEXP, SEXP diff100SEXP, SEXP diff0SEXP,
                        SEXP pred100SEXP, SEXP pred0SEXP, SEXP minprobSEXP, SEXP epsilonSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    NumericVector y = Rcpp::as<NumericVector >(ySEXP);
    NumericMatrix x = Rcpp::as<NumericMatrix >(xSEXP);
    NumericVector priorProbs_ = Rcpp::as<NumericVector >(priorProbs_SEXP);
    double g = Rcpp::as<double >(gSEXP);
    IntegerVector bestModel = Rcpp::as<IntegerVector >(bestModelSEXP);
    bool diff100 = Rcpp::as<bool >(diff100SEXP);
    bool diff0 = Rcpp::as<bool >(diff0SEXP);
    IntegerVector pred100 = Rcpp::as<IntegerVector >(pred100SEXP);
    IntegerVector pred0 = Rcpp::as<IntegerVector >(pred0SEXP);
    double minprob = Rcpp::as<double >(minprobSEXP);
    double epsilon = Rcpp::as<double >(epsilonSEXP);
    const List __result = BMA_Diff_g(y, x, priorProbs_, g, bestModel, diff100, diff0,
                                     pred100, pred0, minprob, epsilon);
    return Rcpp::wrap(__result);
END_RCPP
}

// ScanBMA_BIC
const List ScanBMA_BIC(NumericVector y, NumericMatrix x, NumericVector priorProbs_, double oddsRatio);
RcppExport SEXP networkBMA_ScanBMA_BIC(SEXP ySEXP, SEXP xSEXP, SEXP priorProbs_SEXP, SEXP oddsRatioSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    NumericVector y = Rcpp::as<NumericVector >(ySEXP);
    NumericMatrix x = Rcpp::as<NumericMatrix >(xSEXP);
    NumericVector priorProbs_ = Rcpp::as<NumericVector >(priorProbs_SEXP);
    double oddsRatio = Rcpp::as<double >(oddsRatioSEXP);
    const List __result = ScanBMA_BIC(y, x, priorProbs_, oddsRatio);
    return Rcpp::wrap(__result);
END_RCPP
}

// BMA_Diff_BIC
const List BMA_Diff_BIC(NumericVector y, NumericMatrix x, NumericVector priorProbs_,
                        IntegerVector bestModel, bool diff100, bool diff0,
                        IntegerVector pred100, IntegerVector pred0, double minprob, double epsilon);
RcppExport SEXP networkBMA_BMA_Diff_BIC(SEXP ySEXP, SEXP xSEXP, SEXP priorProbs_SEXP,
                        SEXP bestModelSEXP, SEXP diff100SEXP, SEXP diff0SEXP,
                        SEXP pred100SEXP, SEXP pred0SEXP, SEXP minprobSEXP, SEXP epsilonSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    NumericVector y = Rcpp::as<NumericVector >(ySEXP);
    NumericMatrix x = Rcpp::as<NumericMatrix >(xSEXP);
    NumericVector priorProbs_ = Rcpp::as<NumericVector >(priorProbs_SEXP);
    IntegerVector bestModel = Rcpp::as<IntegerVector >(bestModelSEXP);
    bool diff100 = Rcpp::as<bool >(diff100SEXP);
    bool diff0 = Rcpp::as<bool >(diff0SEXP);
    IntegerVector pred100 = Rcpp::as<IntegerVector >(pred100SEXP);
    IntegerVector pred0 = Rcpp::as<IntegerVector >(pred0SEXP);
    double minprob = Rcpp::as<double >(minprobSEXP);
    double epsilon = Rcpp::as<double >(epsilonSEXP);
    const List __result = BMA_Diff_BIC(y, x, priorProbs_, bestModel, diff100, diff0,
                                     pred100, pred0, minprob, epsilon);
    return Rcpp::wrap(__result);
END_RCPP
}

