#' Lambda-path for quantile regression
#'
#' @description
#' Compute lambda-path for regularized quantile regression
#'
#' @param X A \eqn{n \times p} feature matrix
#' @param y A \eqn{n \times 1} response vector
#' @param tau The parameter for quantile regression, ranging between 0 and 1.
#'
#' @return enter_E_index Keep track which index enters set E at each step
#' @return lambda_vec A list of breakout points
#' @return beta_0 True values of beta_0 at breakout points
#' @return theta True values of theta at breakout points
#' @export

lam_path <- function(X, y, tau){
  .Call(`_quantileShanshan_LamPath`, y, X, tau)
}
