// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// Update or downdate the inverse matrix by changine only one point
arma::mat update(arma::mat A_inv, arma::mat X, arma::uvec E, int E_size) {
       
    arma::uword k = E_size;
    arma::uword index_insert = E[k-1];
    arma::mat A_inv_new(k+1, k+1);
    arma::uvec E_prev = E.subvec(0, k-2);
    arma::mat X_E_prev = X.rows(E_prev);
    
    arma::vec v = (X.row(index_insert)).t();
    arma::vec u1(k, arma::fill::ones);
    u1.tail(k-1) = X_E_prev * v;
    arma::vec u2 = A_inv * u1;
    double d = 1.0/(arma::dot(v, v) - dot(u1, u2));
    arma::vec u3 = d * u2;
    arma::mat F11_inv = A_inv + d * u2 * u2.t();
    A_inv_new(k, k) = d;
    A_inv_new(arma::span(0, k-1), k) = -u3;
    A_inv_new(k, arma::span(0, k-1)) = -u3.t();
    A_inv_new(arma::span(0, k-1), arma::span(0, k-1)) = F11_inv;

    return A_inv_new;
}

arma::mat downdate(arma::mat A_inv, arma::mat X, arma::uvec E_prev, int E_prev_size, int index_remove) {

    arma::uword k = E_prev_size;
    arma::mat A_inv_new(k, k);
    if (index_remove < k-1) {
        arma::rowvec tmpv1 = A_inv.row(index_remove+1);
        A_inv.rows(index_remove+1, k-1) = A_inv.rows(index_remove+2, k);
        A_inv.row(k) = tmpv1;
        arma::vec tmpv2 = A_inv.col(index_remove+1);
        A_inv.cols(index_remove+1, k-1) = A_inv.cols(index_remove+2, k);
        A_inv.col(k) = tmpv2;
    }
    arma::mat F11_inv = A_inv.submat(0, 0, k-1, k-1);
    double d = A_inv(k, k);
    arma::vec u = - A_inv(arma::span(0, k-1), k) / d;
    A_inv_new = F11_inv - d * u * u.t();
    return A_inv_new;
}

arma::mat Compute_A_inv(arma::uvec E, arma::uword E_size, arma::mat X){

    arma::mat X_elb = X.rows(E);
    arma::mat A(E_size + 1, E_size + 1, arma::fill::ones);
    A(0,0) = 0;
    A(1,1,arma::size(E_size,E_size)) = X_elb * X_elb.t();
    arma::mat A_inv = arma::inv(A);
    return A_inv;
}

//' Lambda path for quantile regression
//'
//' @description
//' Path-following algorithm to exactly solve 
//' 	(beta_{0,w}, beta_{w}) = argmin_{beta_0, beta} \sum_{i \neq j} f(g_i(beta_0, beta)) + w*f(g_{j}(beta_0, beta)) + lambda / 2 * \|beta\|_2^2
//' for 0 <= w <= 1, where g_i(beta_0, beta) = a_i beta_0 + b_i^T beta + c_i and f(r) = alpha_0 max(r, 0) + alpha_1 max(-r, 0)
//'
//' @param y A \eqn{n \times 1} vector
//' @param X A \eqn{n \times p} matrix 
//' @param tau A scalar between 0 and 1
//'
//' @details
//'
//' @return enter_E_index Keep track which index enters set E at each step
//' @return lambda_vec A list of breakout points
//' @return beta_0 True values of beta_0 at breakout points
//' @return theta True values of theta at breakout points
// [[Rcpp::export(LamPath)]]
arma::mat LamPath(arma::vec y, arma::mat X, double tau) {
    arma::uword n = y.n_elem, p = X.n_cols;

    // Declare three elbow sets
    arma::uvec E;
    arma::uvec L;
    arma::uvec R;
    arma::uword E_size = 0;
    arma::uword L_size = 0;
    arma::uword R_size = 0;
    arma::uvec index_insert(1);

    // Store breakpoints and solutions
    const arma::uword N = 10000;
    arma::vec bkpts_lam(N);
    arma::vec bkpts_beta0(N);
    arma::vec Empty_E(N, arma::fill::zeros);
    arma::mat bkpts_theta;
    arma::mat bkpts_fitted;
    arma::vec h_vec(n);
    arma::mat elbow_record;

    //initialization
    arma::uvec indices = sort_index(y);
    arma::uword L_R_size;
    arma::mat X_R;
    arma::mat X_L;
    double lambda; 
    double beta0;
    arma::vec theta = arma::linspace(tau - 1, tau - 1, n);
    arma::vec fitted(n);
    arma::vec elbow(n, arma::fill::zeros); // -1: L, 0: E, 1: R
    arma::rowvec a; 
    arma::mat candidate;
    arma::uvec linear_ind;
    arma::umat subscript;
    arma::uword max_L_ind = 0;
    arma::uword min_R_ind = 0;

    // Declare variables in determining next breakpoints
    arma::vec diff_bkpts;
    double diff_bkpts_final = 0.0;
    arma::uvec diff_bkpts_index(2);
    arma::uword leaving_E_size = 0;
    double lam_leaving_E = 0.0;
    arma::uword flag_leaving_E = 0;
    //arma::uword flag_entering_R = 0;
    const double INF = 1e8;

    arma::vec ratio_bkpts;
    arma::rowvec temp_h_fixed(p);
    double temp_h;
    double ratio_bkpts_final = 0.0;
    arma::uword ratio_bkpts_index = 0;
    arma::uword flag_entering_E = 0;
    arma::uword flag_leaving_L = 0;
    double lam_entering_E = 0.0;

    arma::vec b;
    arma::mat A_inv;
    arma::uword s = 0;

    // Initialization
    int key_index = int(n*tau);
    if (n*tau != key_index){ 
        // initialize three sets
        index_insert(0) = indices(key_index);
        E.insert_rows(E_size, index_insert); E_size += 1;

        for (arma::uword i = 0; i < key_index; i++) {
            index_insert(0) = indices(i);
            L.insert_rows(L_size, index_insert); L_size += 1;
            elbow(indices(i)) = -1;
        }
        for (arma::uword i = key_index+1; i < n; i++) {
            index_insert(0) = indices(i);
            R.insert_rows(R_size, index_insert); R_size += 1;
            elbow(indices(i)) = 1;
        }

        // Compute next breakpoint
        L_R_size = L_size + R_size;
        arma::vec candidate_vec(L_R_size);
        X_R = X.rows(R); X_L = X.rows(L);
        arma::rowvec x_E = X.row(E[0]); 
        double y_E = y(E[0]); 
        arma::rowvec a_E = tau * arma::sum(X_R) - (1.0 - tau) * arma::sum(X_L) +
            ((1.0 - tau) * L_size - R_size * tau) * x_E;
        for (arma::uword i = 0, index = 0; i < L_R_size; i++){
            if (i < L_size){
                index = L[i];
            } else {
                index = R[i - L_size];
            }
            candidate_vec(i) =  double(1.0/(y_E - y(index)) * 
                    dot(x_E - X.row(index), a_E));
        } 
        if (max(candidate_vec) <= 0.0){
            std::cout << "The first entered lambda is nonpositive!." << "\n";
        }
        arma::uword index_max_lam = index_max(candidate_vec);
        lambda = candidate_vec(index_max_lam);
        bkpts_lam(0) = lambda;       

        //beta0, theta, fitted values
        beta0 = y_E - arma::dot(x_E, a_E/lambda);
        bkpts_beta0(0) = beta0;
        theta(R) = arma::linspace(tau, tau, R_size);
        theta(E[0]) = (1.0 - tau) * L_size - R_size * tau;
        bkpts_theta.insert_cols(bkpts_theta.n_cols,theta);  // Double check
        fitted = arma::linspace(beta0, beta0, n) + X * a_E.t() / lambda;
        bkpts_fitted.insert_cols(bkpts_fitted.n_cols, fitted);
        elbow_record.insert_cols(elbow_record.n_cols, elbow);

        // change of the three sets: either L or R hits E
        if (index_max_lam < L_size){
            index_insert(0) = L[index_max_lam];
            elbow(L[index_max_lam]) = 0;
            E.insert_rows(E_size, index_insert); E_size += 1;
            L.shed_row(index_max_lam); L_size -= 1;
        } else {
            index_insert(0) = R[index_max_lam - L_size];
            elbow(R[index_max_lam - L_size]) = 0;
            E.insert_rows(E_size, index_insert); E_size += 1;
            R.shed_row(index_max_lam - L_size); R_size -= 1;
        }
    } else {
        // initialize three sets
        for (arma::uword i = 0; i < key_index;i++){
            index_insert(0) = indices(i);
            L.insert_rows(L_size, index_insert);
            L_size += 1;
            elbow(indices(i)) = -1;
         }
        for (arma::uword i = key_index; i < n;i++){
            index_insert(0) = indices(i);
            R.insert_rows(R_size, index_insert);
            R_size += 1;
            elbow(indices(i)) = 1;
        } 

        // Compute next breakpoint
        candidate.set_size(L_size, R_size);
        X_R = X.rows(R); X_L = X.rows(L);
        a = tau * arma::sum(X_R) - (1.0 - tau) * arma::sum(X_L);
        for (arma::uword i=0; i < L_size; i++){
            for(arma::uword j=0; j < R_size; j++){
                candidate(i,j) = arma::dot(X.row(L[i]) - X.row(R[j]),
                        a)/(y(L[i]) - y(R[j]));
            }
        }
        lambda = arma::max(arma::max(candidate));
        bkpts_lam(0) = lambda;

        linear_ind = arma::find(candidate == lambda);
        subscript = arma::ind2sub(arma::size(candidate),linear_ind);
        max_L_ind = L[subscript(0,0)];
        min_R_ind = R[subscript(1,0)];

        // beta0, theta, fitted values
        beta0  = y(max_L_ind) - arma::dot(X.row(max_L_ind), a) / lambda; 
        bkpts_beta0(0) = beta0;
        theta(R) = arma::linspace(tau, tau, R_size);
        bkpts_theta.insert_cols(bkpts_theta.n_cols, theta); // double check
        fitted = arma::linspace(beta0, beta0, n) + X * a.t() / lambda;
        bkpts_fitted.insert_cols(bkpts_fitted.n_cols, fitted);
        //enter_E_index(0) = max_L_ind;
        Empty_E(0) = 1;
        elbow_record.insert_cols(elbow_record.n_cols, elbow);


        // change of the three sets: L and R hit E simultaneously
        index_insert(0) = max_L_ind;
        elbow(max_L_ind) = 0;
        E.insert_rows(E_size, index_insert); E_size += 1;
        index_insert(0) = min_R_ind;
        elbow(min_R_ind) = 0;
        E.insert_rows(E_size, index_insert); E_size += 1;
        L.shed_row(subscript(0,0)); L_size -= 1;
        R.shed_row(subscript(1,0)); R_size -= 1;
    } // End initialization

    A_inv = Compute_A_inv(E, E_size, X);
    arma::vec y_et(E_size + 1, arma::fill::zeros);
    y_et.tail(E_size) = y.elem(E);
    b = A_inv * y_et;

    // Regularization Path (Check while condition)
    while (lambda > 1e-8) {
/*        if (s < 10) {
           std::cout << "###########################################" << std::endl;
           std::cout << "Current lambda is " << lambda << std::endl; 
           std::cout << "Current E is " << E << std::endl;
        }*/
        
        if (E_size > 0) {
            //E -> L or R, choose the max bkpt less than the current bkpt
            diff_bkpts.set_size(E_size);
            diff_bkpts.zeros();
            diff_bkpts_final = -INF;
            leaving_E_size = 0;
            flag_leaving_E = 0;
            lam_leaving_E = 0.0;
            for(arma::uword i=0; i < E_size; i++){
                if(b(i+1) > 0){
                    diff_bkpts(i) = (tau - 1 - bkpts_theta(E[i], s))/b(i+1);
                } else if(b(i+1) < 0) {
                    diff_bkpts(i) = (tau  - bkpts_theta(E[i], s))/b(i+1);
                }
                if (diff_bkpts(i) < 0 && diff_bkpts(i) > diff_bkpts_final + 1e-8) {
                    diff_bkpts_final = diff_bkpts(i);
                    diff_bkpts_index(0) = i;
                    flag_leaving_E = 1; leaving_E_size = 1;
                } else if (diff_bkpts(i) < 0 && fabs(diff_bkpts(i) - diff_bkpts_final) < 1e-8) {
                    //std::cout << "Two cases leave E" << std::endl;
                    diff_bkpts_index(1) = i;
                    leaving_E_size = 2;
                }
            }
            if (flag_leaving_E){
                lam_leaving_E = bkpts_lam[s] + diff_bkpts_final;            
            }

            // L or R -> E, choose the max bkpt less than the current bkpt
            L_R_size = L_size + R_size;
            ratio_bkpts.set_size(L_R_size);
            ratio_bkpts.zeros();
            ratio_bkpts_final = 0;
            ratio_bkpts_index = 0;
            flag_entering_E = 0, flag_leaving_L = 0;
            lam_entering_E = 0.0;
            temp_h_fixed =  b.tail(E_size).t() * X.rows(E);
            for(arma::uword i = 0, index = 0; i < L_R_size; i++){
                index = (i < L_size) ? L[i] : R[i - L_size];
                temp_h = b(0) + arma::dot(X.row(index), temp_h_fixed);
                ratio_bkpts(i) = (bkpts_fitted(index, s) 
                        - temp_h) / (y(index) - temp_h); 
                // Check this condition
                if (ratio_bkpts(i) > 0 && ratio_bkpts(i) < 1 && ratio_bkpts(i) > ratio_bkpts_final) {
                    ratio_bkpts_final = ratio_bkpts(i);
                    ratio_bkpts_index = i;
                    flag_entering_E = 1;
                }
            }
            if (flag_entering_E) {
                lam_entering_E = bkpts_lam[s] * ratio_bkpts_final;
                if (ratio_bkpts_index < L_size) flag_leaving_L = 1;      
            }

            // determin the next lamda
            if (flag_entering_E && flag_leaving_E){
                lambda = std::max(lam_leaving_E, lam_entering_E);
            } else if (!flag_leaving_E && flag_entering_E){
                lambda = lam_entering_E;
            } else if (flag_leaving_E && !flag_entering_E){
                lambda = lam_leaving_E;
            } else { 
                std::cout << "flag_leaving_E = 0 and flag_entering_E = 0" << "\n";
            }

            //update beta0, theta and fitted in the piecewise linear path
            beta0 = b(0) + bkpts_lam[s]/lambda *(bkpts_beta0[s] - b(0));
            bkpts_beta0[s+1] = beta0;
            theta = bkpts_theta(arma::span::all, s);
            theta(E) += b.tail(E_size) * (lambda - bkpts_lam[s]);
            bkpts_theta.insert_cols(bkpts_theta.n_cols, theta);
            
            h_vec = arma::linspace(b(0), b(0), n) + X * temp_h_fixed.t();
            fitted = h_vec + (bkpts_fitted(arma::span::all, s) - h_vec)
                * bkpts_lam[s] / lambda;
            bkpts_fitted.insert_cols(bkpts_fitted.n_cols, fitted);
            bkpts_lam[s+1] = lambda;
            elbow_record.insert_cols(elbow_record.n_cols, elbow);


            // update the three sets
            if (lambda == lam_leaving_E){
                for (arma::uword i = 0; i < leaving_E_size; i++) {
                    index_insert(0) = E[diff_bkpts_index(i)];
                    if (b(diff_bkpts_index(i) + 1) < 0) {
                        elbow(E[diff_bkpts_index(i)]) = 1;
                        R.insert_rows(R_size, index_insert); R_size += 1;
                    } else {
                        elbow(E[diff_bkpts_index(i)]) = -1;
                        L.insert_rows(L_size, index_insert); L_size += 1;
                    }
                    bkpts_fitted(E[diff_bkpts_index(i)], s+1) = y[E[diff_bkpts_index(i)]];  // Double check
                }
                //inverse Downdate K
                for (arma::uword i = 0; i < leaving_E_size; i++) {
                    if (E_size > 1) {
                        A_inv = downdate(A_inv, X, E, E_size, arma::uword (diff_bkpts_index(i)-i));
                    }
                    E.shed_row(arma::uword (diff_bkpts_index(i)-i)); E_size -= 1;
                }                
            } else {
                if (flag_leaving_L){
                    index_insert(0) = L[ratio_bkpts_index];
                    elbow(L[ratio_bkpts_index]) = 0;
                    E.insert_rows(E_size, index_insert); E_size += 1;
                    bkpts_theta(L[ratio_bkpts_index], s+1) = tau - 1;  // Double check
                    L.shed_row(ratio_bkpts_index); L_size -= 1;
                } else {
                    index_insert(0) = R[ratio_bkpts_index - L_size];
                    elbow(R[ratio_bkpts_index - L_size]) = 0;
                    E.insert_rows(E_size, index_insert); E_size += 1;
                    bkpts_theta(R[ratio_bkpts_index - L_size], s+1) = tau;  // Double check
                    R.shed_row(ratio_bkpts_index - L_size); R_size -= 1;
                }
                A_inv = update(A_inv, X, E, E_size);
            }

            if (E_size > 0) {
                y_et.set_size(E_size+1);
                y_et[0] = 0;
                y_et.tail(E_size) = y.elem(E);
                b = A_inv * y_et;
            }
            s += 1;
        } else {
            // Compute next breakpoint
            candidate.set_size(L_size, R_size);
            X_R = X.rows(R); X_L = X.rows(L);
            a = tau * arma::sum(X_R) - (1.0 - tau) * arma::sum(X_L);
            for (arma::uword i=0; i < L_size; i++){
                for(arma::uword j=0; j < R_size; j++){
                    candidate(i,j) = arma::dot(X.row(L[i]) - X.row(R[j]),
                            a)/(y(L[i]) - y(R[j]));
                    if (candidate(i, j) > bkpts_lam[s] - 1e-8) {
                        candidate(i, j) = -INF;
                    }
                }
            }
            lambda = arma::max(arma::max(candidate));

            linear_ind = arma::find(candidate == lambda);
            subscript = arma::ind2sub(arma::size(candidate),linear_ind);
            max_L_ind = L[subscript(0,0)];
            min_R_ind = R[subscript(1,0)];

            // beta0, theta, fitted values
            beta0  = y(max_L_ind) - arma::dot(X.row(max_L_ind), a) / lambda; 
            bkpts_beta0(s+1) = beta0;
            theta = arma::linspace(tau - 1, tau - 1, n);  // Double check 
            theta(R) = arma::linspace(tau, tau, R_size);  // Double check
            bkpts_theta.insert_cols(bkpts_theta.n_cols, theta); // double check
            fitted = arma::linspace(beta0, beta0, n) + X * a.t() / lambda;
            bkpts_fitted.insert_cols(bkpts_fitted.n_cols, fitted);
            bkpts_lam[s+1] = lambda;
            //enter_E_index(s+1) = max_L_ind;
            Empty_E(s+1) = 1;
            s += 1;
            elbow_record.insert_cols(elbow_record.n_cols, elbow);


            // change of the three sets: L and R hit E simultaneously
            index_insert(0) = max_L_ind;
            elbow(max_L_ind) = 0;
            E.insert_rows(E_size, index_insert); E_size += 1;
            index_insert(0) = min_R_ind;
            elbow(min_R_ind) = 0;
            E.insert_rows(E_size, index_insert); E_size += 1;
            L.shed_row(subscript(0,0)); L_size -= 1;
            R.shed_row(subscript(1,0)); R_size -= 1;

            A_inv = Compute_A_inv(E, E_size, X);
            y_et.set_size(E_size+1);
            y_et[0] = 0;
            y_et.tail(E_size) = y.elem(E);
            b = A_inv * y_et;
        }
        
    }   

    arma::vec beta_0_output(s+1);
    beta_0_output = bkpts_beta0.subvec(0, s);
    arma::vec lambda_vec_output(s+1);
    lambda_vec_output = bkpts_lam.subvec(0, s);
    arma::vec Empty_E_output(s+1);
    Empty_E_output = Empty_E.subvec(0, s);

    arma::mat output_temp = join_cols(Empty_E_output.t(), lambda_vec_output.t(), beta_0_output.t(), bkpts_theta);
    arma::mat output = join_cols(output_temp, elbow_record);
    return output;
}

// Binary search 
arma::uword binary_search(arma::uword lo, arma::uword hi, arma::vec lambda_list, double lambda) {
    // base case
    if (hi == lo + 1 || hi == lo) {
        return hi;
    }

    arma::uword mid = lo + (hi - lo)/2;
    if (lambda_list(mid) > lambda) {
        return binary_search(mid, hi, lambda_list, lambda);
    } else if (lambda_list(mid) < lambda) {
        return binary_search(lo, mid, lambda_list, lambda);
    } else {
        return mid;
    }
}

//' Find solutions at any given lambda vector for regularized quantile regression
//'
//' @description
//'
//' @param y A \eqn{n \times 1} vector
//' @param X A \eqn{n \times p} matrix 
//' @param tau A scalar between 0 and 1
//' @param lambda_list A list of lambda, in descending order
//'
//' @details
//'
//' @return output First row is the beta_0, next p rows are beta and the last n rows are theta
// [[Rcpp::export(solution_for_given_lambda)]]
arma::mat solution_for_given_lambda(arma::vec y, arma::mat X, double tau, arma::vec lambda_list) {
    arma::uword n = y.n_elem, p = X.n_cols;
    arma::uword length_lam = lambda_list.n_elem;
    arma::uvec indices = sort_index(y);

    arma::mat LamPath_out = LamPath(y, X, tau);
    //arma::vec enter_E_index = LamPath_out.row(0).t();
    arma::vec Empty_E = LamPath_out.row(0).t();
    arma::vec lambda_bkpts = LamPath_out.row(1).t();
    arma::vec beta0_list = LamPath_out.row(2).t();
    arma::mat theta_mat = LamPath_out(arma::span(3, n+2), arma::span::all);
    arma::mat elbow_record = LamPath_out(arma::span(n+3, LamPath_out.n_rows-1), arma::span::all);
    arma::mat output(2*n+p+1, length_lam, arma::fill::zeros);

    arma::vec theta, beta;
    arma::vec elbow;
    double beta0;
    int key_index;
    arma::uword index;
    arma::uword search_status = 0;
    arma::uword current_lo = 0;
    arma::uword prev_lo = current_lo;
    arma::uword lambda_index = 0;
    double lambda_m1, lambda_m;
    double beta0_m1, beta0_m;
    arma::vec theta_m1, theta_m;
    double lb_beta0, ub_beta0;

    for (arma::uword i = 0; i < length_lam; i++) {
        if (lambda_list[i] > lambda_bkpts[0]) { // go back to initial cases
            key_index = int(n * tau);
            index = indices[key_index];
            theta = theta_mat.col(0);
            beta = 1 / lambda_list[i] * (X.t() * theta);
            elbow = elbow_record.col(0);

            if (n*tau != key_index) {
                beta0 = y[index] - arma::dot(X.row(index), beta.t());
            } else {
                //elbow = elbow_record.col(0);
                arma::uvec L_ind = arma::find(elbow == -1);
                arma::uvec R_ind = arma::find(elbow == 1);
                lb_beta0 = arma::max(y.elem(L_ind) - X.rows(L_ind) * beta);
                ub_beta0 = arma::min(y.elem(R_ind) - X.rows(R_ind) * beta);
                beta0 = (lb_beta0 + ub_beta0) / 2.0;
                /*beta0_bkpt = beta0_list[0];
                k = (y[index] - beta0_bkpt) * lambda_bkpts[0];
                beta0 = y[index] - k/lambda_list[i];*/
            }
        } else {
            // Binary search
            if (!search_status) {
                current_lo = binary_search(0, lambda_bkpts.n_elem-1, lambda_bkpts, lambda_list[i]);
                search_status = 1;
            } else {
                current_lo = binary_search(prev_lo-1, lambda_bkpts.n_elem-1, lambda_bkpts, lambda_list[i]);
            }
            prev_lo = current_lo;
            lambda_index = current_lo;
            lambda_m1 = lambda_bkpts[lambda_index];
            lambda_m = lambda_bkpts[lambda_index-1];
            beta0_m1 = beta0_list[lambda_index];
            beta0_m = beta0_list[lambda_index-1];
            theta_m1 = theta_mat.col(lambda_index);
            theta_m = theta_mat.col(lambda_index-1);
            elbow = elbow_record.col(lambda_index);
            //Check if E is empty between lambda_m1 and lambda_m. 
            //If not, do the linear interpolation, else
            //find beta, theta by similar process as the initial case
            //if (enter_E_index[lambda_index] < 0)
            if (Empty_E[lambda_index] == 0) {
                beta0 = ((lambda_list[i]-lambda_m)/(lambda_m1-lambda_m)*(lambda_m1*beta0_m1) + (lambda_m1-lambda_list[i])/(lambda_m1-lambda_m)*(lambda_m*beta0_m)) / lambda_list[i];
                theta = (lambda_list[i]-lambda_m)/(lambda_m1-lambda_m)*theta_m1 + (lambda_m1 - lambda_list[i])/(lambda_m1-lambda_m)*theta_m;
                beta = 1 / lambda_list[i] * (X.t() * theta);
            } else {
                theta = theta_mat.col(lambda_index);
                beta = 1 / lambda_list[i] * (X.t() * theta);

                //elbow = elbow_record.col(lambda_index);
                arma::uvec L_ind = arma::find(elbow == -1);
                arma::uvec R_ind = arma::find(elbow == 1);
                lb_beta0 = arma::max(y.elem(L_ind) - X.rows(L_ind) * beta);
                ub_beta0 = arma::min(y.elem(R_ind) - X.rows(R_ind) * beta);
                beta0 = (lb_beta0 + ub_beta0) / 2.0;
                /*k = (beta0_m1 - beta0_m) / (1/lambda_m - 1/lambda_m1);
                a = beta0_m + k/lambda_m;
                beta0 = a - k/lambda_list[i];*/
            }
        }
        output(0, i) = beta0;
        output(arma::span(1, p), i) = beta;
        output(arma::span(p+1, n+p), i) = theta;
        output(arma::span(n+p+1, 2*n+p), i) = elbow;
    } //end for loop
    return output;
}

// Simulation for LOO using Lambda-path (lam_list in descending order)
// [[Rcpp::export(Simulation_LamPath)]]
arma::vec Simulation_LamPath(arma::vec y, arma::mat X, double tau, arma::vec lam_list) {
    arma::uword n = y.n_elem, p = X.n_cols;
    arma::uword lam_length= lam_list.n_elem;
    arma::uvec selected(n-1);
    clock_t begin, end;
    clock_t begin_path, end_path;
    double time_path = 0.0;

    arma::mat solutions_at_grids;
    double lam, beta0;
    arma::vec beta(p);
    arma::vec Lam_rcv(lam_length, arma::fill::zeros);
    arma::mat predict; predict.zeros(n, lam_length);

    arma::vec zeros(n, arma::fill::zeros);
    arma::vec r(n, arma::fill::zeros);

    begin = clock();
    for (arma::uword i = 0; i < n; i++) {
        if (i == 0) {
            selected = arma::conv_to<arma::uvec>::from(arma::linspace(1, n-1, n-1));
        } else if (i == n-1) {
            selected = arma::conv_to<arma::uvec>::from(arma::linspace(0, n-2, n-1));
        } else {
            selected = arma::join_cols(arma::conv_to<arma::uvec>::from(arma::linspace(0, i-1, i)), arma::conv_to<arma::uvec>::from(arma::linspace(i+1, n-1, n-i-1)));
        }
        arma::mat X_loo = X.rows(selected);
        arma::vec y_loo = y(selected);
       
        // Linear interpolation
        begin_path = clock();
        solutions_at_grids = solution_for_given_lambda(y_loo, X_loo, tau, lam_list);
        end_path = clock();
        time_path += double(end_path - begin_path)/CLOCKS_PER_SEC;
        for (arma::uword ind_lam = 0; ind_lam < lam_length; ind_lam++) {
            lam = lam_list[ind_lam];
            beta0 = solutions_at_grids(0, ind_lam);
            beta = solutions_at_grids(arma::span(1, p), ind_lam);
            predict(i, ind_lam) = arma::dot(X.row(i), beta.t()) + beta0;
        }
    }
    end = clock();
    double time_total = double(end - begin)/CLOCKS_PER_SEC;

    for (arma::uword k = 0; k < lam_length; k++) {
        //Lam_rcv[k] = arma::sum(arma::square(y - predict.col(k)))/n;

        r = y - predict.col(k);
        Lam_rcv[k] = arma::sum(tau * arma::max(r, zeros) + (1 - tau) * arma::max(-r, zeros) )/n;
    }
    std::cout<< "lam_opt is "<< lam_list(Lam_rcv.index_min()) << "\n";
    std::cout<< "lam_opt_mse is "<< Lam_rcv.min() << "\n";

    arma::vec output(2);
    output(0) = time_total;
    output(1) = time_path;
    return output;
}
