// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// case_path_nonsmooth
Rcpp::List case_path_nonsmooth(arma::vec a, arma::mat B, arma::vec c, double lam, double alpha_0, double alpha_1, int j, double beta_0_w0, arma::vec beta_w0, arma::vec theta_w0);
RcppExport SEXP _quantileShanshan_case_path_nonsmooth(SEXP aSEXP, SEXP BSEXP, SEXP cSEXP, SEXP lamSEXP, SEXP alpha_0SEXP, SEXP alpha_1SEXP, SEXP jSEXP, SEXP beta_0_w0SEXP, SEXP beta_w0SEXP, SEXP theta_w0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type a(aSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    Rcpp::traits::input_parameter< double >::type lam(lamSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_0(alpha_0SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_1(alpha_1SEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    Rcpp::traits::input_parameter< double >::type beta_0_w0(beta_0_w0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_w0(beta_w0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_w0(theta_w0SEXP);
    rcpp_result_gen = Rcpp::wrap(case_path_nonsmooth(a, B, c, lam, alpha_0, alpha_1, j, beta_0_w0, beta_w0, theta_w0));
    return rcpp_result_gen;
END_RCPP
}
// LamPath
arma::mat LamPath(arma::vec y, arma::mat X, double tau);
RcppExport SEXP _quantileShanshan_LamPath(SEXP ySEXP, SEXP XSEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    rcpp_result_gen = Rcpp::wrap(LamPath(y, X, tau));
    return rcpp_result_gen;
END_RCPP
}
// solution_for_given_lambda
arma::mat solution_for_given_lambda(arma::vec y, arma::mat X, double tau, arma::vec lambda_list);
RcppExport SEXP _quantileShanshan_solution_for_given_lambda(SEXP ySEXP, SEXP XSEXP, SEXP tauSEXP, SEXP lambda_listSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_list(lambda_listSEXP);
    rcpp_result_gen = Rcpp::wrap(solution_for_given_lambda(y, X, tau, lambda_list));
    return rcpp_result_gen;
END_RCPP
}
// Simulation_LamPath
arma::vec Simulation_LamPath(arma::vec y, arma::mat X, double tau, arma::vec lam_list);
RcppExport SEXP _quantileShanshan_Simulation_LamPath(SEXP ySEXP, SEXP XSEXP, SEXP tauSEXP, SEXP lam_listSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lam_list(lam_listSEXP);
    rcpp_result_gen = Rcpp::wrap(Simulation_LamPath(y, X, tau, lam_list));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_hello_world
List rcpp_hello_world();
RcppExport SEXP _quantileShanshan_rcpp_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpp_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// w_path_simulation
double w_path_simulation(arma::vec a, arma::mat B, arma::vec c, double alpha_0, double alpha_1, arma::vec lam_list, arma::vec beta_0_w0_list, arma::mat beta_w0_mat, arma::mat theta_w0_mat);
RcppExport SEXP _quantileShanshan_w_path_simulation(SEXP aSEXP, SEXP BSEXP, SEXP cSEXP, SEXP alpha_0SEXP, SEXP alpha_1SEXP, SEXP lam_listSEXP, SEXP beta_0_w0_listSEXP, SEXP beta_w0_matSEXP, SEXP theta_w0_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type a(aSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_0(alpha_0SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_1(alpha_1SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lam_list(lam_listSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_0_w0_list(beta_0_w0_listSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta_w0_mat(beta_w0_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta_w0_mat(theta_w0_matSEXP);
    rcpp_result_gen = Rcpp::wrap(w_path_simulation(a, B, c, alpha_0, alpha_1, lam_list, beta_0_w0_list, beta_w0_mat, theta_w0_mat));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_quantileShanshan_case_path_nonsmooth", (DL_FUNC) &_quantileShanshan_case_path_nonsmooth, 10},
    {"_quantileShanshan_LamPath", (DL_FUNC) &_quantileShanshan_LamPath, 3},
    {"_quantileShanshan_solution_for_given_lambda", (DL_FUNC) &_quantileShanshan_solution_for_given_lambda, 4},
    {"_quantileShanshan_Simulation_LamPath", (DL_FUNC) &_quantileShanshan_Simulation_LamPath, 4},
    {"_quantileShanshan_rcpp_hello_world", (DL_FUNC) &_quantileShanshan_rcpp_hello_world, 0},
    {"_quantileShanshan_w_path_simulation", (DL_FUNC) &_quantileShanshan_w_path_simulation, 9},
    {NULL, NULL, 0}
};

RcppExport void R_init_quantileShanshan(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
